"""HTTP service for the promoted model, with the solver as fallback.

Every request goes through two guards before the surrogate's answer is returned:
  1. inputs outside the training box  -> the solver answers (`fallback_reason: out_of_box`);
  2. an answer that fails a physics check -> the solver answers (`fallback_reason: physics_check`).
Inputs outside physical sense (mould hotter than the ejection temperature, negative thickness, ...)
are rejected with 422 before either.

Run: COOLING_BUNDLE=serving/bundle uvicorn --factory cooling_surrogate.service:create_app
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

from . import checks, model, monitor, physics


class CoolingRequest(BaseModel):
    melt_temp_c: float = Field(gt=physics.T_EJECT, le=400.0, description="melt temperature, C")
    mould_temp_c: float = Field(ge=0.0, lt=physics.T_EJECT, description="mould wall temperature, C")
    thickness_mm: float = Field(gt=0.0, le=20.0, description="wall thickness of the part, mm")
    diffusivity_mm2_s: float = Field(gt=0.0, le=1.0, description="thermal diffusivity, mm^2/s")
    htc_w_m2k: float = Field(gt=0.0, le=100_000.0, description="part-to-mould heat-transfer coefficient")
    latent_heat_kj_kg: float = Field(ge=0.0, le=300.0, description="latent heat of crystallisation, kJ/kg")

    @model_validator(mode="after")
    def melt_above_mould(self) -> CoolingRequest:
        if self.melt_temp_c <= self.mould_temp_c:
            raise ValueError("melt temperature must be above mould temperature")
        return self

    def as_array(self) -> np.ndarray:
        return np.array([[getattr(self, name) for name in physics.PARAM_NAMES]])


class Profile(BaseModel):
    positions_mm: list[float] = Field(description="distance from the mid-plane")
    times_s: list[float]
    temperatures_c: list[list[float]] = Field(description="one row per time, one value per position")


class CoolingResponse(BaseModel):
    cooling_time_s: float
    heat_removed_kj_m2: float = Field(description="heat that left through each face by the cooling time")
    source: str
    fallback_reason: str | None
    out_of_box: list[str]
    failed_checks: list[str]
    model_version: str
    dataset_version: str
    profile: Profile
    latency_ms: float


def create_app(bundle_dir: Path | None = None, request_log: Path | None = None) -> FastAPI:
    bundle_dir = Path(bundle_dir or os.environ.get("COOLING_BUNDLE", "serving/bundle"))
    log_env = os.environ.get("COOLING_REQUEST_LOG")
    request_log = request_log or (Path(log_env) if log_env else None)
    torch.set_num_threads(int(os.environ.get("COOLING_TORCH_THREADS", "1")))

    info = json.loads((bundle_dir / "bundle.json").read_text())
    net, meta = model.load(bundle_dir / "model")
    lo, hi = np.asarray(meta.box_lo), np.asarray(meta.box_hi)
    counters: Counter[str] = Counter()
    lock = threading.Lock()

    app = FastAPI(title="cooling-surrogate", version=info["model_version"])

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "model_version": info["model_version"]}

    @app.get("/v1/model")
    def model_info() -> dict:
        box = {name: [float(lo[i]), float(hi[i])] for i, name in enumerate(physics.PARAM_NAMES)}
        return {**info, "training_box": box}

    @app.get("/metrics")
    def metrics() -> dict:
        with lock:
            return dict(counters)

    @app.post("/v1/cooling-time", response_model=CoolingResponse)
    def cooling_time(req: CoolingRequest) -> CoolingResponse:
        start = time.perf_counter()
        params = req.as_array()
        outside = monitor.out_of_box(params, lo, hi)[0]
        pred = model.predict(net, meta, params)
        failed = [name for name, bad in checks.run_all(pred, params).items() if bad[0]]

        reason = "out_of_box" if outside else ("physics_check" if failed else None)
        if reason:
            sol = physics.solve(params)
            t_cool, field, heat = sol.t_cool[0], sol.field[0], sol.heat[0]
        else:
            t_cool, field, heat = pred.t_cool[0], pred.field[0], pred.heat[0]
        source = "solver" if reason else "surrogate"
        latency = 1000.0 * (time.perf_counter() - start)

        positions = np.linspace(0.0, req.thickness_mm / 2.0, model.N_NODES)
        response = CoolingResponse(
            cooling_time_s=float(t_cool),
            heat_removed_kj_m2=float(heat[-1] / 1e3),
            source=source,
            fallback_reason=reason,
            out_of_box=outside,
            failed_checks=failed,
            model_version=info["model_version"],
            dataset_version=meta.dataset_version,
            profile=Profile(
                positions_mm=positions.round(4).tolist(),
                times_s=(physics.FRACTIONS * t_cool).round(4).tolist(),
                temperatures_c=np.round(field, 3).tolist(),
            ),
            latency_ms=round(latency, 3),
        )
        record = {
            "inputs": req.model_dump(),
            "source": source,
            "fallback_reason": reason,
            "out_of_box": outside,
            "failed_checks": failed,
            "t_cool_s": float(t_cool),
            "surrogate_t_cool_s": float(pred.t_cool[0]),
            "model_version": info["model_version"],
            "latency_ms": round(latency, 3),
        }
        with lock:
            counters["requests"] += 1
            counters[f"source_{source}"] += 1
            if reason:
                counters[f"fallback_{reason}"] += 1
            if request_log is not None:
                with open(request_log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
        return response

    return app
