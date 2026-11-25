"""Datasets of solver runs, versioned by content.

A dataset is a folder `data/<version>/` with one sub-folder per split and a `manifest.json`. The
manifest records the recipe (sampling config and the SHA-256 of the solver and generator code) and
the SHA-256 of every array file. The version is derived from those file hashes, so two folders with
the same version hold the same bytes, and any change to the solver or the sampling gives a new one.
Loading a split checks the hashes first.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from . import physics

ARRAYS = ("inputs", "t_cool", "field", "heat")

# The training box. Everything the surrogate learns comes from inside it.
TRAIN_BOX: dict[str, tuple[float, float]] = {
    "melt_temp_c": (200.0, 260.0),
    "mould_temp_c": (20.0, 60.0),
    "thickness_mm": (1.0, 3.0),
    "diffusivity_mm2_s": (0.08, 0.12),
    "htc_w_m2k": (1500.0, 4000.0),
    "latent_heat_kj_kg": (0.0, 80.0),
}

# Out-of-distribution groups: one parameter leaves the box, the others stay inside it.
OOD_SHIFTS: dict[str, tuple[str, tuple[float, float]]] = {
    "thicker_wall": ("thickness_mm", (3.0, 5.0)),
    "hotter_melt": ("melt_temp_c", (260.0, 300.0)),
    "warmer_mould": ("mould_temp_c", (60.0, 80.0)),
    "weaker_contact": ("htc_w_m2k", (600.0, 1500.0)),
}

_CODE_FILES = ("physics.py", "data.py")


def box_arrays(box: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([box[name][0] for name in physics.PARAM_NAMES])
    hi = np.array([box[name][1] for name in physics.PARAM_NAMES])
    return lo, hi


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash() -> str:
    """SHA-256 over the solver and generator source, line endings normalised."""
    digest = hashlib.sha256()
    here = Path(__file__).parent
    for name in _CODE_FILES:
        digest.update(name.encode())
        digest.update((here / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sample_inputs(config: dict) -> dict[str, np.ndarray]:
    """Parameter sets for every split. Training uses a Latin hypercube; the others plain uniform draws."""
    lo, hi = box_arrays(TRAIN_BOX)
    seed = int(config["seed"])
    lhs = qmc.LatinHypercube(d=lo.size, seed=seed)
    splits = {"train": lo + (hi - lo) * lhs.random(int(config["n_train"]))}
    rng = np.random.default_rng(seed + 1)
    splits["val"] = lo + (hi - lo) * rng.random((int(config["n_val"]), lo.size))
    splits["test"] = lo + (hi - lo) * rng.random((int(config["n_test"]), lo.size))
    n_ood = int(config["n_ood_per_shift"])
    for group, (name, (a, b)) in OOD_SHIFTS.items():
        x = lo + (hi - lo) * rng.random((n_ood, lo.size))
        x[:, physics.PARAM_NAMES.index(name)] = rng.uniform(a, b, n_ood)
        splits[f"ood_{group}"] = x
    return splits


def generate(config: dict, data_root: Path, log=print) -> Path:
    """Run the solver over every split, write the arrays and the manifest. Returns the dataset folder.

    If a dataset with the same recipe already exists and its hashes check out, it is reused.
    """
    recipe = {"sampling": config, "code_sha256": code_hash(), "box": TRAIN_BOX, "ood_shifts": OOD_SHIFTS}
    recipe_hash = hashlib.sha256(_canonical(recipe)).hexdigest()
    data_root.mkdir(parents=True, exist_ok=True)
    for manifest_path in sorted(data_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("recipe_sha256") == recipe_hash:
            verify(manifest_path.parent)
            log(f"dataset {manifest['version']} already built from this recipe, hashes verified")
            return manifest_path.parent

    staging = data_root / f".staging-{recipe_hash[:12]}"
    shutil.rmtree(staging, ignore_errors=True)
    files: dict[str, dict] = {}
    seconds: dict[str, float] = {}
    for split, inputs in sample_inputs(config).items():
        start = time.perf_counter()
        sol = physics.solve(inputs)
        seconds[split] = round(time.perf_counter() - start, 2)
        log(f"  {split}: {inputs.shape[0]} solver runs in {seconds[split]:.1f} s")
        folder = staging / split
        folder.mkdir(parents=True)
        arrays = {"inputs": inputs, "t_cool": sol.t_cool, "field": sol.field, "heat": sol.heat}
        for name in ARRAYS:
            path = folder / f"{name}.npy"
            np.save(path, np.ascontiguousarray(arrays[name], dtype=np.float64))
            files[f"{split}/{name}.npy"] = {"sha256": sha256_file(path), "shape": list(arrays[name].shape)}

    content_hash = hashlib.sha256(_canonical({k: v["sha256"] for k, v in files.items()})).hexdigest()
    version = f"ds-{content_hash[:12]}"
    manifest = {
        "version": version,
        "content_sha256": content_hash,
        "recipe_sha256": recipe_hash,
        "recipe": recipe,
        "solver": {
            "n_cells": physics.N_CELLS,
            "dfo": physics.DFO,
            "steps_to_eject": physics.STEPS_TO_EJECT,
            "fractions": physics.FRACTIONS.tolist(),
            "t_eject_c": physics.T_EJECT,
        },
        "splits": {s: int(np.load(staging / s / "inputs.npy").shape[0]) for s in sample_inputs(config)},
        "files": files,
        "solver_seconds": seconds,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    target = data_root / version
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    return target


def verify(dataset_dir: Path) -> dict:
    """Check every file against the manifest. Raises on any mismatch; returns the manifest."""
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    for rel, meta in manifest["files"].items():
        actual = sha256_file(dataset_dir / rel)
        if actual != meta["sha256"]:
            raise ValueError(f"{dataset_dir.name}/{rel}: SHA-256 {actual[:12]} does not match the manifest")
    content = hashlib.sha256(_canonical({k: v["sha256"] for k, v in manifest["files"].items()})).hexdigest()
    if f"ds-{content[:12]}" != manifest["version"]:
        raise ValueError(f"{dataset_dir.name}: the file hashes do not add up to version {manifest['version']}")
    return manifest


@dataclass(frozen=True)
class Split:
    inputs: np.ndarray  # (B, 6)
    t_cool: np.ndarray  # (B,)
    field: np.ndarray  # (B, F, N)
    heat: np.ndarray  # (B, F)


def load_split(dataset_dir: Path, split: str) -> Split:
    arrays = {name: np.load(dataset_dir / split / f"{name}.npy") for name in ARRAYS}
    return Split(**arrays)


def splits_of(manifest: dict, prefix: str = "") -> list[str]:
    return [s for s in manifest["splits"] if s.startswith(prefix)]
