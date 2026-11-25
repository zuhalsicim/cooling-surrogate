"""Write the results tables in README.md from the files in results/.

    python scripts/make_tables.py          # rewrite the block between the markers
    python scripts/make_tables.py --check  # exit 1 if the README is out of date (run in CI)

Every number in the README's results section comes from here; none is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"

SPLIT_LABEL = {
    "test": "in the box (test)",
    "ood_thicker_wall": "thicker wall, 3-5 mm",
    "ood_hotter_melt": "hotter melt, 260-300 C",
    "ood_warmer_mould": "warmer mould, 60-80 C",
    "ood_weaker_contact": "weaker contact, 600-1500 W/m2K",
}


def load(results: Path, name: str):
    return json.loads((results / name).read_text(encoding="utf-8"))


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f} %"


def ci(d: dict, scale: float = 1.0, digits: int = 2, unit: str = "") -> str:
    return f"{scale * d['value']:.{digits}f}{unit} [{scale * d['low']:.{digits}f}, {scale * d['high']:.{digits}f}]"


def ms(seconds: float) -> str:
    return f"{1000 * seconds:.3f} ms" if seconds < 0.1 else f"{seconds:.3f} s"


def solver_section(results: Path) -> list[str]:
    v = load(results, "solver-verification.json")
    out = [
        "### 1. Solver against the exact solution",
        "",
        f"{v['cases']} random cases from the training box, latent heat set to zero so the exact series applies. "
        "Grid and time step are refined together.",
        "",
        "| cells across the half wall | time step (Fourier units) | cooling time error, max | order "
        "| profile error, max | order |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in v["against_exact_without_latent_heat"]:
        mark = " (dataset)" if row["n_cells"] == v["dataset_setting"]["n_cells"] else ""
        order_t = f"{row['order_t_cool']:.2f}" if "order_t_cool" in row else "-"
        order_f = f"{row['order_field']:.2f}" if "order_field" in row else "-"
        out.append(
            f"| {row['n_cells']}{mark} | {row['dfo']:.5f} | {pct(row['t_cool_rel_error_max'], 4)} "
            f"| {order_t} | {row['field_error_k_max']:.3f} K | {order_f} |"
        )
    lat = v["with_latent_heat_against_finer_run"]
    out += [
        "",
        f"With latent heat (no exact solution), the dataset setting against a run with "
        f"{lat['reference']['n_cells']} cells and a 4x smaller time step: cooling time differs by at most "
        f"{pct(lat['t_cool_rel_diff_max'], 4)}, profiles by at most {lat['field_diff_k_max']:.3f} K "
        f"(largest at 5 % of the cooling time, near the mould face). Energy balance residual: at most "
        f"{lat['energy_residual_max']:.1e} of the heat the wall can give off.",
        "",
    ]
    return out


def dataset_section(results: Path) -> list[str]:
    m = load(results, "dataset.json")
    box = m["recipe"]["box"]
    out = [
        "### 2. Dataset",
        "",
        f"Version `{m['version']}` (content SHA-256 `{m['content_sha256'][:16]}...`, recipe SHA-256 "
        f"`{m['recipe_sha256'][:16]}...`), {sum(m['splits'].values())} solver runs in "
        f"{sum(m['solver_seconds'].values()):.0f} s.",
        "",
        "| split | runs |",
        "|---|---:|",
    ]
    out += [f"| {s} | {n} |" for s, n in m["splits"].items()]
    out += ["", "| parameter | training box |", "|---|---|"]
    out += [f"| {k} | {lo:g} to {hi:g} |" for k, (lo, hi) in box.items()]
    return [*out, ""]


def gate_section(results: Path) -> list[str]:
    training = load(results, "training.json")
    run = load(results, "run.json")
    out = [
        "### 3. Gate decisions",
        "",
        "Every registered version is gated on the in-box test set, in the order it was registered. "
        "Values are point estimates with the upper end of the 95 % interval in parentheses; the gate compares "
        "the upper end with the threshold.",
        "",
        "| version | model | cooling time error, p95 | profile RMSE, mean | worst physics check | decision |",
        "|---|---|---:|---:|---|---|",
    ]
    policy = None
    for version in sorted(training, key=int):
        g = load(results, f"gate-v{version}.json")
        policy = g["policy"]
        crit = {c["name"]: c for c in g["criteria"]}
        t, f = crit["t_cool_rel_error_p95"], crit["field_rmse_k_mean"]
        phys = [c for c in g["criteria"] if c["name"].startswith("physics:")]
        worst = max(phys, key=lambda c: c["failures"])
        worst_txt = (
            f"{worst['name'][8:]}: {worst['failures']} of {g['test_cases']}" if worst["failures"] else "none fails"
        )
        decision = "**promote**" if g["decision"] == "promote" else f"**block** ({', '.join(g['failed'])})"
        out.append(
            f"| {version} | {training[version]['name']} | {pct(t['estimate'])} ({pct(t['upper95'])}) "
            f"| {f['estimate']:.3f} K ({f['upper95']:.3f} K) | {worst_txt} | {decision} |"
        )
    out += [
        "",
        f"Thresholds: cooling time error p95 at most {pct(policy['max_t_cool_rel_error_p95'], 1)}, profile RMSE "
        f"at most {policy['max_field_rmse_k_mean']} K, each physics check violated by at most "
        f"{pct(policy['max_violation_rate'], 0)} of cases (Wilson upper bound). Production after the run: version "
        f"{run['production_version']} (promotions in order: {', '.join(run['promotions_in_order']) or 'none'}).",
        "",
    ]
    for version in sorted(training, key=int):
        out.append(f"- v{version} `{training[version]['name']}`: {training[version]['note']}.")
    return [*out, ""]


def accuracy_section(results: Path) -> list[str]:
    ev = load(results, "evaluation.json")
    training = load(results, "training.json")
    main = next(v for v in sorted(training, key=int) if not training[v].get("quantized_from_version"))
    e = ev[main]
    out = [
        f"### 4. Accuracy in and out of the training box (v{main} `{e['name']}`, the model as trained)",
        "",
        "Error against the solver; 95 % bootstrap intervals in brackets. Out-of-box rows are what the surrogate "
        "would answer if nothing stopped it; the service sends those inputs to the solver instead.",
        "",
        "| inputs | cases | cooling time error, mean | cooling time error, p95 | profile RMSE, mean "
        "| cases failing a physics check |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, label in SPLIT_LABEL.items():
        if split not in e:
            continue
        s = e[split]
        fails = s["cases_failing_any_check"]
        out.append(
            f"| {label} | {s['n']} | {ci(s['t_cool_rel_error_mean'], 100, 3, ' %')} "
            f"| {ci(s['t_cool_rel_error_p95'], 100, 3, ' %')} | {ci(s['field_rmse_k_mean'], 1, 3, ' K')} | {fails} |"
        )
    return [*out, ""]


def speed_quant_section(results: Path) -> list[str]:
    sp = load(results, "speed.json")
    ev = load(results, "evaluation.json")
    training = load(results, "training.json")
    sol = sp["solver"]
    out = [
        "### 5. Speed and int8 quantisation",
        "",
        f"One CPU thread each. Single case: median over repeated calls (solver: {sol['single_case_runs']} cases). "
        f"Batch: all {sp['batch_cases']} test cases in one call. Surrogate times include input scaling and "
        "the conversion back to temperatures.",
        "",
        "| | single case | per case in a batch | speed-up, single | speed-up, batch | weights |",
        "|---|---:|---:|---:|---:|---:|",
        f"| solver | {ms(sol['single_case_s_median'])} | {ms(sol['batched_s_per_case'])} | 1x | 1x | - |",
    ]
    for name, m in sp["models"].items():
        out.append(
            f"| {name} | {ms(m['single_case_s_median'])} | {ms(m['batched_s_per_case'])} "
            f"| {m['speedup_single_vs_solver']:,.0f}x | {m['speedup_batched_vs_solver']:,.0f}x "
            f"| {m['weights_bytes'] / 1024:,.0f} KiB |"
        )
    q = next((v for v in training if training[v].get("quantized_from_version")), None)
    if q:
        base = training[q]["quantized_from_version"]
        a, b = ev[base]["test"], ev[q]["test"]
        rows = [
            ("cooling time error, mean", "t_cool_rel_error_mean", 100, " %"),
            ("cooling time error, p95", "t_cool_rel_error_p95", 100, " %"),
            ("profile RMSE, mean", "field_rmse_k_mean", 1, " K"),
        ]
        out += ["", f"int8 (v{q}) against float32 (v{base}) on the in-box test set:", ""]
        out += ["| | float32 | int8 |", "|---|---:|---:|"]
        out += [f"| {label} | {ci(a[k], f, 3, u)} | {ci(b[k], f, 3, u)} |" for label, k, f, u in rows]
        out.append(
            f"| cases failing a physics check | {a['cases_failing_any_check']} of {a['n']} "
            f"| {b['cases_failing_any_check']} of {b['n']} |"
        )
    path = results / "int8-variants.json"
    if path.exists():
        iv = load(results, "int8-variants.json")
        out += [
            "",
            f"Other int8 options for v{iv['base_version']}, measured on the validation split (not used by the gate):",
            "",
            "| option | cooling time error, mean / p95 | profile RMSE | cases failing a check | single case "
            "| per case in a batch | weights |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, v in iv["variants"].items():
            out.append(
                f"| {label} | {pct(v['t_cool_rel_error_mean'], 3)} / {pct(v['t_cool_rel_error_p95'], 3)} "
                f"| {v['field_rmse_k_mean']:.3f} K | {v['cases_failing_any_check']} of {v['cases']} "
                f"| {ms(v['single_case_s_median'])} | {ms(v['batched_s_per_case'])} "
                f"| {v['weights_bytes'] / 1024:,.0f} KiB |"
            )
    return [*out, ""]


def monitor_section(results: Path) -> list[str]:
    path = results / "monitor.json"
    if not path.exists():
        return ["### 6. Monitoring", "", "No version reached production, so no traffic was served.", ""]
    m = load(results, "monitor.json")
    served = m["served_by"]
    out = [
        f"### 6. Monitoring a replayed day of traffic (production v{m['model_version']})",
        "",
        f"{m['requests']} requests: test-set inputs mixed with out-of-box inputs, sent through the service. "
        f"Answered by the surrogate: {served.get('surrogate', 0)}; by the solver: {served.get('solver', 0)} "
        f"({', '.join(f'{k}: {v}' for k, v in m['fallback_reasons'].items()) or 'no fallbacks'}).",
        "",
        "| measure | value |",
        "|---|---|",
        f"| out-of-box share | {ci(m['out_of_box_rate'], 100, 1, ' %')} |",
        f"| out of box, by parameter | {', '.join(f'{k}: {v}' for k, v in m['out_of_box_by_feature'].items())} |",
    ]
    sh = m["shadow_on_answered"]
    if sh["sampled"]:
        out += [
            f"| shadow check: cooling time error on {sh['sampled']} answered requests, mean "
            f"| {ci(sh['t_cool_rel_error_mean'], 100, 3, ' %')} |",
            f"| same, p95 | {ci(sh['t_cool_rel_error_p95'], 100, 3, ' %')} |",
        ]
    for reason, d in m["surrogate_error_avoided_by_fallback"].items():
        out.append(
            f"| error avoided by the {reason} fallback ({d['n']} requests), mean / p95 "
            f"| {pct(d['t_cool_rel_error_mean']['value'], 2)} / {pct(d['t_cool_rel_error_p95']['value'], 2)} |"
        )
    lat = m["median_latency_ms"]
    out.append(f"| median latency | {', '.join(f'{k}: {v:.2f} ms' for k, v in lat.items())} |")
    out += ["", "Alerts raised:" if m["alerts"] else "No alerts.", *[f"- {a}" for a in m["alerts"]], ""]
    return out


def run_section(results: Path) -> list[str]:
    r = load(results, "run.json")
    env = r["environment"]
    stages = ", ".join(f"{k} {v:.0f} s" for k, v in r["stage_seconds"].items())
    return [
        f"Whole run: {r['total_seconds']:.0f} s ({stages}) with `{r['config']}` on {env['processor'] or 'CPU'}, "
        f"{env['cpu_count']} logical cores, Python {env['python']}, torch {env['torch']}, numpy {env['numpy']}.",
        "",
    ]


def headline_section(results: Path) -> list[str]:
    v = load(results, "solver-verification.json")
    training = load(results, "training.json")
    ev = load(results, "evaluation.json")
    sp = load(results, "speed.json")
    run = load(results, "run.json")
    main = next(k for k in sorted(training, key=int) if not training[k].get("quantized_from_version"))
    dataset_row = next(r for r in v["against_exact_without_latent_heat"] if r["n_cells"] == 32)
    t = ev[main]["test"]
    ood = {s: e for s, e in ev[main].items() if s.startswith("ood_")}
    worst = max(ood, key=lambda s: ood[s]["t_cool_rel_error_p95"]["value"])
    model_speed = next(m for name, m in sp["models"].items() if name.startswith(f"v{main} "))
    decisions = []
    for version in sorted(training, key=int):
        g = load(results, f"gate-v{version}.json")
        decisions.append(f"v{version} {'promoted' if g['decision'] == 'promote' else 'blocked'}")
    out = [
        "### Headline",
        "",
        f"- **Solver:** cooling time within {pct(dataset_row['t_cool_rel_error_max'], 3)} of the exact series at "
        f"the dataset's grid; error falls at order {dataset_row['order_t_cool']:.2f} as the grid is refined.",
        f"- **Surrogate v{main}, inside the training box:** cooling time error "
        f"{ci(t['t_cool_rel_error_mean'], 100, 3, ' %')} on average, "
        f"{pct(t['t_cool_rel_error_p95']['value'], 3)} at the 95th percentile; "
        f"profile RMSE {t['field_rmse_k_mean']['value']:.3f} K.",
        f"- **Outside the box** it degrades, worst for {SPLIT_LABEL[worst]}: 95th percentile "
        f"{pct(ood[worst]['t_cool_rel_error_p95']['value'], 1)}. The service sends such inputs to the solver.",
        f"- **Gate:** {', '.join(decisions)}; production is v{run['production_version']}.",
        f"- **Speed, one CPU thread:** one request {ms(model_speed['single_case_s_median'])} against "
        f"{ms(sp['solver']['single_case_s_median'])} for the solver ({model_speed['speedup_single_vs_solver']:,.0f}x); "
        f"in batches {model_speed['speedup_batched_vs_solver']:,.0f}x.",
        "",
    ]
    return out


def render(results: Path) -> str:
    lines: list[str] = headline_section(results)
    for section in (solver_section, dataset_section, gate_section, accuracy_section, speed_quant_section):
        lines += section(results)
    lines += monitor_section(results)
    lines += run_section(results)
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = ap.parse_args()
    text = args.readme.read_text(encoding="utf-8")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{BEGIN}\n{render(args.results)}{END}{tail}"
    if args.check:
        if new != text:
            print("README.md results are out of date; run python scripts/make_tables.py", file=sys.stderr)
            return 1
        return 0
    args.readme.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
