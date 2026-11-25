"""The whole loop, from solver to served model, driven by one config file.

    python -m cooling_surrogate.pipeline --config configs/full.json

1. verify the solver against the exact solution             -> results/solver-verification.json
2. build (or reuse) the dataset version                     -> data/<version>/, results/dataset.json
3. train every candidate, track it and register a version   -> state/mlflow.db, results/training.json
4. register an int8 copy of one candidate as its own version
5. gate every version; a passing version becomes production -> results/gate-v<N>.json
6. measure accuracy in and out of the training box          -> results/evaluation.json
7. measure speed, and int8 options on the validation split -> results/speed.json, int8-variants.json
8. export the production version as a serving bundle        -> serving/bundle/
9. replay traffic through the service and read its log      -> results/monitor.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from . import checks, data, gate, metrics, model, monitor, physics, speed, train, verify
from .registry import MODEL_NAME, PRODUCTION, Registry


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _evaluate(net, meta, dataset_dir: Path, manifest: dict, n_boot: int, seed: int) -> dict:
    out = {}
    for split in ["test", *data.splits_of(manifest, "ood_")]:
        truth = data.load_split(dataset_dir, split)
        pred = model.predict(net, meta, truth.inputs)
        summary = metrics.summarise(metrics.errors(pred, truth), n_boot, seed)
        violated = checks.run_all(pred, truth.inputs)
        summary["physics_violations"] = {k: int(v.sum()) for k, v in violated.items()}
        summary["cases_failing_any_check"] = int(np.any(list(violated.values()), axis=0).sum())
        out[split] = summary
    return out


INT8_VARIANTS = {
    "float32 (no quantisation)": None,
    "int8, all layers, per-tensor (the registered recipe)": {"hidden_only": False, "per_channel": False},
    "int8, all layers, per-channel": {"hidden_only": False, "per_channel": True},
    "int8, hidden layers only, per-tensor": {"hidden_only": True, "per_channel": False},
    "int8, hidden layers only, per-channel": {"hidden_only": True, "per_channel": True},
}


def _int8_variants(net, meta, dataset_dir: Path, repeats: int, work: Path) -> dict:
    """Quantisation options measured on the validation split (the test split stays for the gate)."""
    val = data.load_split(dataset_dir, "val")
    out = {}
    with speed.one_thread():
        for label, options in INT8_VARIANTS.items():
            m = net if options is None else model.quantize_int8(net, **options)
            pred = model.predict(m, meta, val.inputs)
            errs = metrics.errors(pred, val)
            folder = work / "int8-variants" / str(len(out))
            model.save(m, meta, folder)
            out[label] = {
                "cases": len(val.inputs),
                "t_cool_rel_error_mean": float(errs["t_cool_rel_error"].mean()),
                "t_cool_rel_error_p95": float(metrics.p95(errs["t_cool_rel_error"])),
                "field_rmse_k_mean": float(errs["field_rmse_k"].mean()),
                "cases_failing_any_check": int(np.any(list(checks.run_all(pred, val.inputs).values()), axis=0).sum()),
                "weights_bytes": (folder / "weights.pt").stat().st_size,
                **speed.time_model(m, meta, val.inputs, repeats),
            }
    return out


def export_bundle(registry: Registry, version: str, gate_report: dict, dest: Path, work: Path) -> None:
    """Copy the production version out of the registry into a folder the service image can ship."""
    folder = registry.download(version, work / f"export-v{version}")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(folder, dest / "model")
    tags = registry.tags(version)
    _write(
        dest / "bundle.json",
        {
            "registered_model": MODEL_NAME,
            "model_version": version,
            "alias": PRODUCTION,
            "name": tags.get("name"),
            "quantized": tags.get("quantized") == "True",
            "dataset_version": tags.get("dataset_version"),
            "dataset_content_sha256": tags.get("dataset_content_sha256"),
            "weights_sha256": tags.get("weights_sha256"),
            "gate_decision": gate_report["decision"],
            "gate_dataset_version": gate_report["dataset_version"],
        },
    )


def _replay_traffic(bundle: Path, dataset_dir: Path, manifest: dict, cfg: dict, log_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # starlette's note on its test client
        from fastapi.testclient import TestClient

    from .service import create_app

    rng = np.random.default_rng(int(cfg["seed"]))
    inside = data.load_split(dataset_dir, "test").inputs
    outside = np.concatenate([data.load_split(dataset_dir, s).inputs for s in data.splits_of(manifest, "ood_")])
    rows = np.concatenate(
        [
            inside[rng.choice(len(inside), int(cfg["n_in_box"]), replace=False)],
            outside[rng.choice(len(outside), int(cfg["n_out_of_box"]), replace=False)],
        ]
    )
    rows = rows[rng.permutation(len(rows))]
    log_path.unlink(missing_ok=True)
    client = TestClient(create_app(bundle, request_log=log_path))
    for row in rows:
        resp = client.post("/v1/cooling-time", json=dict(zip(physics.PARAM_NAMES, row.tolist(), strict=True)))
        resp.raise_for_status()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/full.json"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--state", type=Path, default=Path("state"))
    ap.add_argument("--bundle", type=Path, default=Path("serving/bundle"))
    args = ap.parse_args(argv)

    cfg = json.loads(args.config.read_text())
    results, work = args.results, args.state / "work"
    shutil.rmtree(args.state, ignore_errors=True)  # a fresh registry per run keeps version numbers stable
    work.mkdir(parents=True)
    torch.set_num_threads(int(cfg.get("train_threads", 4)))
    stage_seconds: dict[str, float] = {}
    t_run = time.perf_counter()

    def stage(name: str):
        print(f"\n== {name}", flush=True)
        stage_seconds[name] = time.perf_counter()

    def done(name: str):
        stage_seconds[name] = round(time.perf_counter() - stage_seconds[name], 1)

    stage("verify solver")
    _write(results / "solver-verification.json", verify.run(**cfg.get("verify", {})))
    done("verify solver")

    stage("dataset")
    dataset_dir = data.generate(cfg["dataset"], args.data)
    manifest = data.verify(dataset_dir)
    _write(results / "dataset.json", manifest)
    print(f"  dataset {manifest['version']} ({sum(manifest['splits'].values())} solver runs)")
    done("dataset")

    stage("train and register")
    registry = Registry(args.state)
    trained: dict[str, dict] = {}
    by_name: dict[str, str] = {}
    for cand in cfg["candidates"]:
        print(f"  {cand['name']}: {cand['note']}")
        out = work / cand["name"]
        rec = train.train(dataset_dir, cand, out)
        run_id = registry.log_run(
            cand["name"],
            params={k: v for k, v in cand.items() if k != "note"},
            metrics=rec["final"],
            tags={"dataset_version": rec["dataset_version"], "dataset_content_sha256": rec["dataset_content_sha256"]},
            model_dir=out,
            history=rec["history"],
        )
        version = registry.register(
            run_id,
            out,
            tags={
                "name": cand["name"],
                "note": cand["note"],
                "quantized": False,
                "dataset_version": rec["dataset_version"],
                "dataset_content_sha256": rec["dataset_content_sha256"],
            },
        )
        rec.update(version=version, run_id=run_id, note=cand["note"])
        trained[version] = rec
        by_name[cand["name"]] = version
        print(f"  registered {MODEL_NAME} version {version}")

    if cfg.get("quantize"):
        base = by_name[cfg["quantize"]]
        net, meta = model.load(registry.download(base, work / f"base-v{base}"))
        meta.quantized = True
        out = work / f"{cfg['quantize']}-int8"
        model.save(model.quantize_int8(net), meta, out)
        name = f"{cfg['quantize']}-int8"
        note = f"version {base} with dynamic int8 quantisation of every Linear layer, no retraining"
        run_id = registry.log_run(
            name,
            params={"quantized_from_version": base, "method": "torch dynamic int8, Linear layers"},
            metrics={},
            tags={"dataset_version": meta.dataset_version},
            model_dir=out,
        )
        version = registry.register(
            run_id,
            out,
            tags={
                "name": name,
                "note": note,
                "quantized": True,
                "quantized_from_version": base,
                "dataset_version": meta.dataset_version,
                "dataset_content_sha256": manifest["content_sha256"],
            },
        )
        trained[version] = {"name": name, "note": note, "version": version, "quantized_from_version": base}
        print(f"  registered {MODEL_NAME} version {version} ({note})")
    _write(results / "training.json", trained)
    done("train and register")

    stage("gate")
    ev = cfg["evaluation"]
    reports: dict[str, dict] = {}
    promotions = []
    for version in sorted(trained, key=int):
        net, meta = model.load(registry.download(version, work / f"gate-v{version}"))
        report = gate.run(net, meta, dataset_dir, cfg["gate"], int(ev["bootstrap"]), int(ev["seed"]))
        report.update(model_version=version, name=trained[version]["name"])
        if report["decision"] == "promote":
            registry.promote(version)
            promotions.append(version)
        report["production_after"] = registry.production_version()
        registry.set_tags(version, {"gate_decision": report["decision"], "gate_failed": ",".join(report["failed"])})
        reports[version] = report
        _write(results / f"gate-v{version}.json", report)
        why = f" (failed: {', '.join(report['failed'])})" if report["failed"] else ""
        print(f"  version {version} {trained[version]['name']}: {report['decision']}{why}")
    production = registry.production_version()
    print(f"  production is version {production}")
    done("gate")

    stage("evaluate")
    evaluation = {}
    loaded = {}
    for version in sorted(trained, key=int):
        net, meta = model.load(registry.download(version, work / f"eval-v{version}"))
        loaded[version] = (net, meta)
        evaluation[version] = {
            "name": trained[version]["name"],
            **_evaluate(net, meta, dataset_dir, manifest, int(ev["bootstrap"]), int(ev["seed"])),
        }
    _write(results / "evaluation.json", evaluation)
    done("evaluate")

    stage("speed")
    test_inputs = data.load_split(dataset_dir, "test").inputs
    timed = {f"v{v} {trained[v]['name']}": loaded[v] for v in sorted(trained, key=int)}
    sp = speed.measure(timed, test_inputs, cfg["speed"])
    for v in sorted(trained, key=int):
        sp["models"][f"v{v} {trained[v]['name']}"]["weights_bytes"] = (
            (work / f"eval-v{v}" / "model" / "weights.pt").stat().st_size
        )
    _write(results / "speed.json", sp)
    if cfg.get("quantize"):
        base = by_name[cfg["quantize"]]
        variants = _int8_variants(*loaded[base], dataset_dir, int(cfg["speed"]["repeats"]), work)
        _write(results / "int8-variants.json", {"base_version": base, "split": "val", "variants": variants})
    done("speed")

    stage("export and serve")
    if production is None:
        print("  no version passed the gate; nothing to serve")
        monitor_report = None
    else:
        export_bundle(registry, production, reports[production], args.bundle, work)
        log_path = args.state / "requests.jsonl"
        _replay_traffic(args.bundle, dataset_dir, manifest, cfg["traffic"], log_path)
        monitor_report = monitor.report(
            log_path, int(cfg["traffic"]["shadow_sample"]), int(ev["seed"]), int(ev["bootstrap"]), cfg["monitor"]
        )
        monitor_report["model_version"] = production
        _write(results / "monitor.json", monitor_report)
        for alert in monitor_report["alerts"]:
            print(f"  ALERT: {alert}")
    done("export and serve")

    _write(
        results / "run.json",
        {
            "config": str(args.config.as_posix()),
            "dataset_version": manifest["version"],
            "production_version": production,
            "promotions_in_order": promotions,
            "stage_seconds": stage_seconds,
            "total_seconds": round(time.perf_counter() - t_run, 1),
            "environment": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "torch": torch.__version__,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
            },
        },
    )
    print(f"\ndone in {time.perf_counter() - t_run:.0f} s; production is version {production}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
