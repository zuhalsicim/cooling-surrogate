"""The surrogate: a small MLP from the six process parameters to the solver's outputs.

Targets are scaled so that the network never has to learn units:
  * cooling time as log(seconds), standardised;
  * the profiles as theta = (T - T_mould) / (T_melt - T_mould), which runs from 1 (melt) to 0 (mould);
  * the heat removed as a fraction of the heat a wall can give off between melt and mould temperature.
The conversion back to physical units uses the inputs, so bounds such as 'never colder than the
mould' are easy to express but are not forced by the architecture; the gate checks them.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import data, physics

N_FRACTIONS = physics.FRACTIONS.size
N_NODES = physics.N_CELLS + 1
N_OUT = 1 + N_FRACTIONS * N_NODES + N_FRACTIONS


@dataclass
class ModelMeta:
    """Everything needed to rebuild and use a saved model, stored next to its weights."""

    hidden: int
    depth: int
    box_lo: list[float]
    box_hi: list[float]
    log_t_mean: float
    log_t_std: float
    dataset_version: str
    quantized: bool = False

    def save(self, folder: Path) -> None:
        (folder / "model.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, folder: Path) -> ModelMeta:
        return cls(**json.loads((folder / "model.json").read_text()))


class Surrogate(nn.Module):
    def __init__(self, hidden: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        width = len(physics.PARAM_NAMES)
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, N_OUT))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def scale_inputs(params: np.ndarray, meta: ModelMeta) -> np.ndarray:
    lo, hi = np.asarray(meta.box_lo), np.asarray(meta.box_hi)
    return 2.0 * (params - lo) / (hi - lo) - 1.0


def make_targets(split: data.Split, meta: ModelMeta) -> np.ndarray:
    case = physics.Case.from_params(split.inputs)
    log_t = (np.log(split.t_cool) - meta.log_t_mean) / meta.log_t_std
    span = (case.t_melt - case.t_mould)[:, None, None]
    theta = (split.field - case.t_mould[:, None, None]) / span
    heat = split.heat / physics.removable_heat(case)[:, None]
    return np.concatenate([log_t[:, None], theta.reshape(len(theta), -1), heat], axis=1)


@dataclass(frozen=True)
class Prediction:
    t_cool: np.ndarray  # (B,) s
    field: np.ndarray  # (B, F, N) C
    heat: np.ndarray  # (B, F) J/m^2


def to_physical(raw: np.ndarray, params: np.ndarray, meta: ModelMeta) -> Prediction:
    case = physics.Case.from_params(params)
    t_cool = np.exp(raw[:, 0] * meta.log_t_std + meta.log_t_mean)
    theta = raw[:, 1 : 1 + N_FRACTIONS * N_NODES].reshape(-1, N_FRACTIONS, N_NODES)
    field = case.t_mould[:, None, None] + theta * (case.t_melt - case.t_mould)[:, None, None]
    heat = raw[:, 1 + N_FRACTIONS * N_NODES :] * physics.removable_heat(case)[:, None]
    return Prediction(t_cool=t_cool, field=field, heat=heat)


def predict(model: nn.Module, meta: ModelMeta, params: np.ndarray) -> Prediction:
    params = np.atleast_2d(np.asarray(params, dtype=np.float64))
    x = torch.as_tensor(scale_inputs(params, meta), dtype=torch.float32)
    with torch.inference_mode():
        raw = model(x).double().numpy()
    return to_physical(raw, params, meta)


def quantize_int8(model: nn.Module, hidden_only: bool = False, per_channel: bool = False) -> nn.Module:
    """Dynamic int8 quantisation: int8 weights, activations quantised on the fly in every call.

    The default (every Linear layer, one scale per weight tensor) is the registered int8 version.
    `hidden_only` keeps the output layer in float32; `per_channel` gives each output row its own scale.

    torch.ao.quantization is deprecated in favour of torchao, but still ships with torch 2.15 and
    runs on any CPU without extra packages; the warnings are silenced here and noted in the README.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torch.ao.quantization import default_dynamic_qconfig, per_channel_dynamic_qconfig, quantize_dynamic

        qconfig = per_channel_dynamic_qconfig if per_channel else default_dynamic_qconfig
        linear = [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]
        if hidden_only:
            linear = linear[:-1]
        return quantize_dynamic(model, {name: qconfig for name in linear}, dtype=torch.qint8)


def save(model: nn.Module, meta: ModelMeta, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "weights.pt")
    meta.save(folder)


def load(folder: Path) -> tuple[nn.Module, ModelMeta]:
    meta = ModelMeta.load(folder)
    model = Surrogate(meta.hidden, meta.depth)
    if meta.quantized:
        model = quantize_int8(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.load_state_dict(torch.load(folder / "weights.pt", weights_only=True))
    model.eval()
    return model, meta
