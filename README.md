# cooling-surrogate

**Kurz auf Deutsch.** Ein kleines, vollständiges Beispiel für die Plattform um ein neuronales Physikmodell:
Ein Finite-Volumen-Solver berechnet, wie die Wand eines Spritzgussteils im Werkzeug abkühlt (mit
Kristallisationswärme) und wann die Mitte die Entformungstemperatur erreicht. Der Solver ist gegen die
exakte Reihenlösung geprüft. Seine Läufe bilden einen per SHA-256 versionierten Datensatz; ein kleines MLP
lernt daraus Kühlzeit, Temperaturprofile und abgeführte Wärme. Jedes Training wird in MLflow erfasst und als
Modellversion registriert. Ein Gate lässt eine Version nur in Produktion, wenn Fehler und Physik-Checks
(Grenzen, monotone Abkühlung, Energiebilanz) auch am oberen Ende des 95-%-Intervalls halten. Ein
FastAPI-Dienst liefert die Produktionsversion aus und gibt Eingaben außerhalb des Trainingsbereichs oder
Antworten, die einen Physik-Check verletzen, an den Solver weiter; das Monitoring liest das Request-Log und
misst den Live-Fehler mit dem Solver nach. Alle Zahlen unten erzeugt `scripts/make_tables.py` aus den Dateien
in `results/`. Läuft komplett auf der CPU in wenigen Minuten.

---

## The question

A neural surrogate can answer a simulation question a thousand times faster than the solver it learned
from. The hard part is everything around it: knowing which data a model saw, refusing to ship a model
that is worse or that breaks physics, noticing when users ask questions the model was never trained on,
and still giving them a correct answer. This repository builds that loop end to end, small enough to read
in one sitting, around a physics problem whose exact answer is known, so every claim can be checked.

The physics: a moulded part's wall, thickness *s*, starts at melt temperature and cools against the mould.
How long until its mid-plane is cold enough to eject the part, and what do the temperature profiles look
like on the way?

## How it works

```
 physics.py ─► data.py ─────────► train.py ─► registry.py (MLflow) ─► gate.py ─► alias "production"
 solver,       dataset version     config +     run + version +          errors and           │
 exact series  = SHA-256 of the    dataset      lineage tags             physics checks       ▼
 (verify.py)   files; recipe hash  version                               on the test set   bundle ─► service.py
                                                                                                       │  inside the box and
     monitor.py ◄─ request log ◄───────────────────────────────────────────────────────────────────────┘  checks pass: surrogate
     out-of-box share, shadow re-runs with the solver, error avoided by the fallback                       otherwise: solver
```

**Solver** (`physics.py`). Half the wall is solved (symmetry at the mid-plane) with vertex-centred finite
volumes, 32 cells, Crank-Nicolson in time after four backward-Euler start-up steps. The face loses heat to
the mould through a contact coefficient *h* (a Robin condition). Latent heat of crystallisation enters as
an enthalpy *H(T)* with a Gaussian release band around 120 °C; each time step iterates on the chord
(*H(T_new) − H(T_old)) / (T_new − T_old)*, which makes the step conserve enthalpy exactly even across the
peak. All cases of a batch are one banded linear system, so the same code runs one request or thousands.
Two passes: the first finds the cooling time, the second re-runs with 400 steps up to it and records the
profiles at fixed fractions of it (5 % to 100 %).

**Verification** (`verify.py`). Without latent heat the problem has an exact series solution,
θ = Σ Cₙ cos(λₙ ξ) exp(−λₙ² Fo) with λₙ tan λₙ = Bi. The solver is refined against it (grid and time step
together) and must show second-order convergence; with latent heat it is compared with a four times finer
run, and the energy balance is checked to round-off.

**Inputs** (six): melt temperature, mould temperature, wall thickness, thermal diffusivity, part-to-mould
heat-transfer coefficient, latent heat. Density and heat capacity are fixed (a polypropylene-like
polymer); the ejection temperature is 100 °C.

**Data** (`data.py`). Training inputs come from a Latin hypercube over the training box; validation and test
are independent uniform draws from the same box; four out-of-distribution sets each move one parameter
outside it. Every array is a `.npy` file whose SHA-256 is in `manifest.json`; the dataset version is a hash
of those hashes. A recipe hash (sampling config plus the source of the solver and generator) lets the
pipeline reuse a dataset only when nothing that made it has changed. Loading for training or gating checks
every hash first.

**Surrogate** (`model.py`, `train.py`). An MLP from the six inputs to 1 + 12×33 + 12 outputs: log cooling
time, the profiles in θ = (T − T_mould)/(T_melt − T_mould), and the heat removed as a fraction of what the
wall can give off. Units come back through the inputs, so the network never learns them. The architecture
does not force physical bounds; the gate and the service check them.

**Registry** (`registry.py`). MLflow with a local SQLite store. Each training is a run (config, loss curve,
dataset version, model files); each candidate is a registered version tagged with its dataset version,
dataset content hash and the SHA-256 of its weights. Promotion is the MLflow alias `production`, and only
the gate moves it. Downloads check the weights against the registered hash.

**Gate** (`gate.py`, `checks.py`). On the in-box test set: 95th percentile of the cooling-time error and mean
profile RMSE, each judged by the upper end of its 95 % bootstrap interval; five physics checks, each judged
by the upper end of the 95 % Wilson interval of its violation rate:

| check | a case fails when |
|---|---|
| within bounds | any temperature is more than 1 K below the mould or above the melt |
| cools monotonically | any point warms by more than 1 K between two output times |
| coolest at the face | any profile rises by more than 1 K from mid-plane towards the mould |
| energy balance | heat removed and drop in stored enthalpy differ by more than 2 % of the heat the wall can give off |
| ejects at cooling time | the predicted profile at the predicted cooling time has its mid-plane more than 2 K from 100 °C |

and a lineage check: the model must have been trained on the dataset version the test set belongs to.
The thresholds were set in `configs/full.json` before the first run and not changed afterwards.

**Service** (`service.py`). FastAPI, serving the version exported from the registry (`serving/bundle/`, built
into the Docker image). Physically meaningless input gets 422. Input outside the training box, or an answer
that fails any physics check, is answered by the solver instead, and the response says so. Each request is
logged as one JSON line.

**Monitoring** (`monitor.py`). Reads the request log: share of traffic outside the box and along which
parameter, a shadow re-run of a sample of the surrogate's answers with the solver, and the error the
fallback avoided (the service logs the surrogate's answer for fallback requests too). It raises an alert when
the out-of-box share is above 5 % with confidence.

**Pipeline** (`pipeline.py`). One command runs all of it from one config: verify, data, train and register
three versions (the intended model, a deliberately weak one, an int8 copy of the first), gate them in
order, evaluate, time, export, replay traffic through the service, monitor.

## Results

Produced by `python -m cooling_surrogate.pipeline --config configs/full.json`, written to `results/*.json`,
turned into the tables below by `python scripts/make_tables.py`. CI fails if the tables and the files
disagree.

<!-- results:begin -->
### Headline

- **Solver:** cooling time within 0.025 % of the exact series at the dataset's grid; error falls at order 2.02 as the grid is refined.
- **Surrogate v1, inside the training box:** cooling time error 0.055 % [0.052, 0.058] on average, 0.138 % at the 95th percentile; profile RMSE 0.164 K.
- **Outside the box** it degrades, worst for thicker wall, 3-5 mm: 95th percentile 16.2 %. The service sends such inputs to the solver.
- **Gate:** v1 promoted, v2 blocked, v3 blocked; production is v1.
- **Speed, one CPU thread:** one request 0.271 ms against 0.342 s for the solver (1,266x); in batches 813x.

### 1. Solver against the exact solution

40 random cases from the training box, latent heat set to zero so the exact series applies. Grid and time step are refined together.

| cells across the half wall | time step (Fourier units) | cooling time error, max | order | profile error, max | order |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.02000 | 0.4007 % | - | 4.536 K | - |
| 16 | 0.01000 | 0.1023 % | 1.97 | 1.048 K | 2.11 |
| 32 (dataset) | 0.00500 | 0.0253 % | 2.02 | 0.263 K | 1.99 |
| 64 | 0.00250 | 0.0065 % | 1.97 | 0.067 K | 1.97 |
| 128 | 0.00125 | 0.0015 % | 2.06 | 0.017 K | 1.99 |

With latent heat (no exact solution), the dataset setting against a run with 128 cells and a 4x smaller time step: cooling time differs by at most 0.0269 %, profiles by at most 0.403 K (largest at 5 % of the cooling time, near the mould face). Energy balance residual: at most 6.8e-10 of the heat the wall can give off.

### 2. Dataset

Version `ds-b17ba686b995` (content SHA-256 `b17ba686b9954847...`, recipe SHA-256 `151f97a71272c092...`), 6500 solver runs in 96 s.

| split | runs |
|---|---:|
| train | 4000 |
| val | 500 |
| test | 1000 |
| ood_thicker_wall | 250 |
| ood_hotter_melt | 250 |
| ood_warmer_mould | 250 |
| ood_weaker_contact | 250 |

| parameter | training box |
|---|---|
| melt_temp_c | 200 to 260 |
| mould_temp_c | 20 to 60 |
| thickness_mm | 1 to 3 |
| diffusivity_mm2_s | 0.08 to 0.12 |
| htc_w_m2k | 1500 to 4000 |
| latent_heat_kj_kg | 0 to 80 |

### 3. Gate decisions

Every registered version is gated on the in-box test set, in the order it was registered. Values are point estimates with the upper end of the 95 % interval in parentheses; the gate compares the upper end with the threshold.

| version | model | cooling time error, p95 | profile RMSE, mean | worst physics check | decision |
|---|---|---:|---:|---|---|
| 1 | mlp-256x3 | 0.14 % (0.15 %) | 0.164 K (0.168 K) | coolest_at_face: 1 of 1000 | **promote** |
| 2 | mlp-32x1-quick | 39.70 % (41.20 %) | 26.132 K (26.579 K) | cools_monotonically: 1000 of 1000 | **block** (t_cool_rel_error_p95, field_rmse_k_mean, physics:within_bounds, physics:cools_monotonically, physics:coolest_at_face, physics:energy_balance, physics:ejects_at_cooling_time) |
| 3 | mlp-256x3-int8 | 1.21 % (1.30 %) | 0.712 K (0.726 K) | coolest_at_face: 677 of 1000 | **block** (physics:within_bounds, physics:cools_monotonically, physics:coolest_at_face, physics:energy_balance, physics:ejects_at_cooling_time) |

Thresholds: cooling time error p95 at most 2.0 %, profile RMSE at most 1.5 K, each physics check violated by at most 1 % of cases (Wilson upper bound). Production after the run: version 1 (promotions in order: 1).

- v1 `mlp-256x3`: the intended model: 3 hidden layers of 256, all 4000 training runs, 400 epochs.
- v2 `mlp-32x1-quick`: a quick experiment registered by mistake: 1 hidden layer of 32, 10 % of the training runs, 50 epochs.
- v3 `mlp-256x3-int8`: version 1 with dynamic int8 quantisation of every Linear layer, no retraining.

### 4. Accuracy in and out of the training box (v1 `mlp-256x3`, the model as trained)

Error against the solver; 95 % bootstrap intervals in brackets. Out-of-box rows are what the surrogate would answer if nothing stopped it; the service sends those inputs to the solver instead.

| inputs | cases | cooling time error, mean | cooling time error, p95 | profile RMSE, mean | cases failing a physics check |
|---|---:|---:|---:|---:|---:|
| in the box (test) | 1000 | 0.055 % [0.052, 0.058] | 0.138 % [0.130, 0.153] | 0.164 K [0.160, 0.168] | 1 |
| thicker wall, 3-5 mm | 250 | 5.176 % [4.557, 5.805] | 16.175 % [13.813, 17.169] | 1.566 K [1.443, 1.696] | 204 |
| hotter melt, 260-300 C | 250 | 0.189 % [0.171, 0.209] | 0.490 % [0.464, 0.567] | 0.607 K [0.566, 0.650] | 88 |
| warmer mould, 60-80 C | 250 | 1.266 % [1.128, 1.400] | 3.376 % [3.169, 3.753] | 0.520 K [0.486, 0.560] | 24 |
| weaker contact, 600-1500 W/m2K | 250 | 3.558 % [3.126, 3.987] | 10.695 % [9.219, 11.456] | 1.724 K [1.552, 1.896] | 59 |

### 5. Speed and int8 quantisation

One CPU thread each. Single case: median over repeated calls (solver: 20 cases). Batch: all 1000 test cases in one call. Surrogate times include input scaling and the conversion back to temperatures.

| | single case | per case in a batch | speed-up, single | speed-up, batch | weights |
|---|---:|---:|---:|---:|---:|
| solver | 0.342 s | 9.391 ms | 1x | 1x | - |
| v1 mlp-256x3 | 0.271 ms | 0.012 ms | 1,266x | 813x | 935 KiB |
| v2 mlp-32x1-quick | 0.181 ms | 0.003 ms | 1,894x | 2,796x | 56 KiB |
| v3 mlp-256x3-int8 | 0.929 ms | 0.009 ms | 369x | 1,057x | 242 KiB |

int8 (v3) against float32 (v1) on the in-box test set:

| | float32 | int8 |
|---|---:|---:|
| cooling time error, mean | 0.055 % [0.052, 0.058] | 0.468 % [0.446, 0.491] |
| cooling time error, p95 | 0.138 % [0.130, 0.153] | 1.208 % [1.123, 1.301] |
| profile RMSE, mean | 0.164 K [0.160, 0.168] | 0.712 K [0.699, 0.726] |
| cases failing a physics check | 1 of 1000 | 749 of 1000 |

Other int8 options for v1, measured on the validation split (not used by the gate):

| option | cooling time error, mean / p95 | profile RMSE | cases failing a check | single case | per case in a batch | weights |
|---|---:|---:|---:|---:|---:|---:|
| float32 (no quantisation) | 0.057 % / 0.144 % | 0.158 K | 1 of 500 | 0.244 ms | 0.013 ms | 935 KiB |
| int8, all layers, per-tensor (the registered recipe) | 0.465 % / 1.134 % | 0.679 K | 359 of 500 | 0.898 ms | 0.009 ms | 242 KiB |
| int8, all layers, per-channel | 0.459 % / 1.090 % | 0.615 K | 289 of 500 | 0.938 ms | 0.012 ms | 262 KiB |
| int8, hidden layers only, per-tensor | 0.461 % / 1.131 % | 0.348 K | 32 of 500 | 0.729 ms | 0.011 ms | 548 KiB |
| int8, hidden layers only, per-channel | 0.450 % / 1.131 % | 0.335 K | 33 of 500 | 0.731 ms | 0.011 ms | 562 KiB |

### 6. Monitoring a replayed day of traffic (production v1)

400 requests: test-set inputs mixed with out-of-box inputs, sent through the service. Answered by the surrogate: 300; by the solver: 100 (out_of_box: 100).

| measure | value |
|---|---|
| out-of-box share | 25.0 % [21.0, 29.5] |
| out of box, by parameter | melt_temp_c: 30, thickness_mm: 27, mould_temp_c: 25, htc_w_m2k: 18 |
| shadow check: cooling time error on 100 answered requests, mean | 0.053 % [0.045, 0.061] |
| same, p95 | 0.138 % [0.106, 0.171] |
| error avoided by the out_of_box fallback (100 requests), mean / p95 | 2.31 % / 12.24 % |
| median latency | surrogate: 1.04 ms, solver: 335.13 ms |

Alerts raised:
- out-of-box traffic 25.0 % (95 % CI from 21.0 %) is above 5 %; most often outside: melt_temp_c (30), thickness_mm (27), mould_temp_c (25)

Whole run: 246 s (verify solver 11 s, dataset 96 s, train and register 78 s, gate 1 s, evaluate 2 s, speed 18 s, export and serve 39 s) with `configs/full.json` on AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD, 16 logical cores, Python 3.12.16, torch 2.15.0+cpu, numpy 2.5.5.
<!-- results:end -->

## What the results say

- Inside the training box the surrogate is well within the gate. Its cooling-time error is measured, not
  blurred by the solver: the solver's largest cooling-time error against the exact series is below the
  surrogate's average error. For the profiles the two are of the same order. Close to the mould face, early
  in the cooling, the solver itself is off by a few tenths of a kelvin, so profile differences of that size
  say how well the surrogate copies the solver, not how well it matches the physics.
- Outside the box the error grows quickly, and it grows most for an input nobody would think unusual: a
  thicker wall. The box guard catches these by construction. The physics checks alone would catch many of
  them, but not all: an extrapolated answer can be smooth, monotone and energy-conserving and still be wrong.
- The deliberately weak version fails everything; the gate's job there is easy.
- The int8 version is the interesting one. Its error metrics are within the thresholds, so a gate that only
  looked at accuracy would have promoted it. Quantisation noise of a fraction of a kelvin is enough to push
  late temperatures at the mould face below the mould temperature and to make the flat early profiles
  wobble, and the physics checks block it. It is also slower for a single request: for a network this
  small, quantising the activations on every call costs more than the int8 matrix products save. Only in
  large batches is it somewhat faster. Keeping the output layer in float32 cuts the violations by an order of
  magnitude, but no option comes near the 1 % limit, and most of the batch speed-up goes with it. For this model,
  int8 buys a smaller file and little else.
- The fallback is what makes out-of-box traffic safe, at the solver's latency. The monitor sees the shift
  (here by construction, a quarter of the replayed traffic) and names the parameters that left the box.

## Limits

- The physics is one-dimensional, with constant conductivity and a constant contact coefficient. Latent
  heat is released in a fixed temperature band; real crystallisation depends on cooling rate, and real
  parts detach from the mould as they shrink. Cadmould solves 3D geometry; this solves a slab.
- Without latent heat the problem reduces to two dimensionless numbers and has an exact solution, so a
  surrogate is not needed physically. That is on purpose: the point is the machinery around the model, and
  an exact answer lets every number be checked.
- The surrogate is an MLP, not a transformer, and it is trained once with one seed. The intervals cover
  the spread over test cases, not over retraining.
- The out-of-box guard is a per-parameter box. It does not flag combinations that are rare inside the box,
  and it says nothing about a model that is bad inside the box; that is the gate's and the shadow check's
  job.
- The speed-up is measured against this repository's Python solver, not against a production solver, on one
  CPU thread. Batching changes the picture for both sides.
- Every version is gated on the same test set. The int8 options were compared on the validation set so
  that no choice is made on the test set.
- The registry is MLflow on a local SQLite file, and the service runs in a container on one machine. Moving to
  AWS means an S3 artifact store, the image in ECR and a managed endpoint; none of that is set up here.
- `torch.ao.quantization` (dynamic int8) is deprecated in favour of torchao; it still ships with torch 2.15.
- The traffic for the monitoring section is replayed, not real: a quarter of it is out of the box by
  construction.

## Run it

Python 3.12, CPU only.

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt        # Windows: .venv\Scripts\pip
.venv/bin/python -m cooling_surrogate.pipeline --config configs/full.json
.venv/bin/python scripts/make_tables.py
.venv/bin/python -m pytest -q
```

Serve the promoted version:

```sh
docker build -t cooling-surrogate .                  # needs serving/bundle from the pipeline
docker run -p 8000:8000 cooling-surrogate
curl -X POST localhost:8000/v1/cooling-time -H "content-type: application/json" -d '{"melt_temp_c": 230,
  "mould_temp_c": 40, "thickness_mm": 2.0, "diffusivity_mm2_s": 0.1, "htc_w_m2k": 2500, "latent_heat_kj_kg": 40}'
```

`GET /v1/model` shows the served version, its dataset version and hashes, and the training box;
`GET /metrics` counts requests and fallbacks. The MLflow store is in `state/` (`mlflow ui --backend-store-uri
sqlite:///state/mlflow.db` with the full `mlflow` package).

CI (`.github/workflows/ci.yml`) runs lint, the tests and the README check, then the whole pipeline on a
tiny config, builds the image from the version it promoted, and sends it one request inside and one outside
the training box.

## Reading order

1. `cooling_surrogate/physics.py`: the solver, the enthalpy, the exact series.
2. `cooling_surrogate/verify.py` and `tests/test_physics.py`: how the solver is checked.
3. `cooling_surrogate/data.py`: dataset versions, manifest, hash checks.
4. `cooling_surrogate/checks.py` and `cooling_surrogate/gate.py`: what blocks a model.
5. `cooling_surrogate/registry.py`: versions, lineage tags, the `production` alias.
6. `cooling_surrogate/service.py` and `cooling_surrogate/monitor.py`: guards, fallback, request log.
7. `cooling_surrogate/pipeline.py`: the order in which it all runs.
8. `cooling_surrogate/model.py` and `cooling_surrogate/train.py`: the model itself, deliberately plain.

MIT licence.
