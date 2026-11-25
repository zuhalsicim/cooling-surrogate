"""Train one surrogate from a config and a dataset version. The config and the data are all it sees."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import data, metrics
from .model import N_FRACTIONS, N_NODES, ModelMeta, Surrogate, make_targets, predict, save, scale_inputs


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: dict) -> torch.Tensor:
    sq = (pred - target) ** 2
    n_field = N_FRACTIONS * N_NODES
    return (
        weights["t_cool"] * sq[:, 0].mean()
        + weights["field"] * sq[:, 1 : 1 + n_field].mean()
        + weights["heat"] * sq[:, 1 + n_field :].mean()
    )


def train(dataset_dir: Path, config: dict, out_dir: Path, log=print) -> dict:
    """Fit a model, save it to `out_dir`, and return the training record (losses, timings, history)."""
    manifest = data.verify(dataset_dir)
    torch.manual_seed(int(config["seed"]))
    rng = np.random.default_rng(int(config["seed"]))

    train_split = data.load_split(dataset_dir, "train")
    keep = round(config.get("train_fraction", 1.0) * len(train_split.inputs))
    idx = np.sort(rng.permutation(len(train_split.inputs))[:keep])
    train_split = data.Split(*(getattr(train_split, f)[idx] for f in data.ARRAYS))
    val_split = data.load_split(dataset_dir, "val")

    lo, hi = data.box_arrays(data.TRAIN_BOX)
    log_t = np.log(train_split.t_cool)
    meta = ModelMeta(
        hidden=int(config["hidden"]),
        depth=int(config["depth"]),
        box_lo=lo.tolist(),
        box_hi=hi.tolist(),
        log_t_mean=float(log_t.mean()),
        log_t_std=float(log_t.std()),
        dataset_version=manifest["version"],
    )
    x = torch.as_tensor(scale_inputs(train_split.inputs, meta), dtype=torch.float32)
    y = torch.as_tensor(make_targets(train_split, meta), dtype=torch.float32)

    model = Surrogate(meta.hidden, meta.depth)
    epochs, batch = int(config["epochs"]), int(config["batch_size"])
    steps_per_epoch = int(np.ceil(len(x) / batch))
    opt = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(config["lr"]), total_steps=epochs * steps_per_epoch, pct_start=0.05
    )
    weights = config["loss_weights"]
    log_every = max(1, epochs // 10)
    history: list[tuple[int, dict]] = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        for i in range(steps_per_epoch):
            sel = order[i * batch : (i + 1) * batch]
            loss = _weighted_mse(model(x[sel]), y[sel], weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(sel)
        if epoch % log_every == 0 or epoch == epochs:
            model.eval()
            val = metrics.errors(predict(model, meta, val_split.inputs), val_split)
            point = {
                "train_loss": total / len(x),
                "val_t_cool_rel_error_mean": float(val["t_cool_rel_error"].mean()),
                "val_field_rmse_k_mean": float(val["field_rmse_k"].mean()),
            }
            history.append((epoch, point))
            log(
                f"    epoch {epoch:5d}  loss {point['train_loss']:.2e}  val cooling time "
                f"{100 * point['val_t_cool_rel_error_mean']:.3f} %  val profile RMSE "
                f"{point['val_field_rmse_k_mean']:.3f} K"
            )
    seconds = time.perf_counter() - start
    model.eval()
    save(model, meta, out_dir)
    return {
        "name": config["name"],
        "config": config,
        "dataset_version": manifest["version"],
        "dataset_content_sha256": manifest["content_sha256"],
        "n_train": len(x),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "train_seconds": round(seconds, 1),
        "final": history[-1][1],
        "history": history,
    }
