"""Preregistered analysis for docs/preregistration.md (commit 3fc5105).

Reads runs/confirm (written by scripts/confirmatory_run.sh) and writes
reports/confirmatory.md, reports/confirmatory.json and paper/figures/*.
Stages that have not finished are reported as missing, never imputed.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from persona_holonomy.decomposition import decompose, seed_estimates, summarize_values

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "confirm"
REPORT, FIGS = ROOT / "reports", ROOT / "paper" / "figures"
GENRES = ("gothic", "pirate", "noir")
ALPHA = 0.05


def rows_of(*stages: str) -> list[dict] | None:
    out = []
    for stage in stages:
        path = RUN / stage / "rows.jsonl"
        if not (RUN / f"{stage}.done").exists() or not path.exists():
            return None
        out += [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return out


def pooled(by_group: dict, variant: str, phase: str, name: str) -> dict[int, float] | None:
    series = [by_group.get((g, variant, phase), {}).get(name) for g in GENRES]
    if any(s is None for s in series):
        return None
    seeds = sorted(set.intersection(*(set(s) for s in series)))
    return {s: statistics.mean(x[s] for x in series) for s in seeds}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[key] = running
    return adjusted


def placebo_gate(rows: list[dict] | None) -> dict:
    if rows is None:
        return {"status": "missing"}
    cells: dict[tuple, set] = {}
    for r in rows:
        key = (r["genre"], r["system_variant"], r["seed"], r["probe_id"])
        cells.setdefault(key, set()).add(round(r["copy_free_margin"], 12))
    spread = [len(v) for v in cells.values()]
    return {"status": "pass" if max(spread) == 1 else "fail", "cells": len(cells),
            "cells_with_condition_differences": sum(s > 1 for s in spread)}


def length_gate(rows: list[dict]) -> dict:
    c = {(r["genre"], r["system_variant"], r["phase"], r["seed"], r["probe_id"], r["condition"]):
         r["prefix_tokens"] for r in rows}
    keys = {k[:5] for k in c}
    bad = sum((c[k + ("ordered",)] - c[k + ("neutral",)]) !=
              (c[k + ("lexical_match",)] - c[k + ("lexical_neutral",)]) for k in keys)
    return {"status": "pass" if bad == 0 else "fail", "cells": len(keys), "mismatched": bad}


def sweep_specificity(path: Path, *, layer: int, mode: str) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r["layer"] == layer and r["mode"] == mode and r["alpha"] == 1.0]
    if not rows:
        return None
    cells: dict[tuple, dict] = {}
    for r in rows:
        cells.setdefault((r["seed"], r["probe_id"]), {})[r["control"]] = r["effect"]
    by_seed: dict[int, list[float]] = {}
    for (seed, _), v in cells.items():
        by_seed.setdefault(seed, []).append(v["candidate"] - v["orthogonal"])
    values = [statistics.mean(v) for _, v in sorted(by_seed.items())]
    out = summarize_values(values)
    out["raw_candidate_mean"] = statistics.mean(v["candidate"] for v in cells.values())
    out["raw_null_mean"] = statistics.mean(v["orthogonal"] for v in cells.values())
    out["self_patch_max_abs"] = max(abs(r["self_patch_delta"] or 0) for r in rows)
    return out


def q5_rates(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    table = {}
    for genre in GENRES:
        for cond in ("ordered", "neutral", "lexical_match", "lexical_neutral"):
            chosen = [r for r in rows if r.get("genre") == genre and r["condition"] == cond]
            if chosen:
                table[f"{genre}/{cond}"] = {
                    "genre_intrusion_rate": statistics.mean(r["genre_intrusion_rate"] for r in chosen),
                    "theatrical_rate": statistics.mean(r["theatrical_rate"] for r in chosen),
                    "n_cases": len(chosen)}
    # DiD on theatrical rate, pooled over genres and probes, per seed.
    for r in rows:
        r["theatrical"] = r["theatrical_rate"]
        r.setdefault("system_variant", "default")
    est = seed_estimates(rows, metric="theatrical")
    did = {}
    for name in ("P", "L", "R"):
        series = pooled(est, "default", "exit", name)
        if series:
            did[name] = summarize_values(list(series.values()))
    return {"rates": table, "theatrical_rate_contrasts": did,
            "total_samples": sum(r["n_samples"] for r in rows)}


def per_probe(rows: list[dict] | None, *, metric: str = "copy_free_margin",
              phase: str = "exit", variant: str = "default") -> dict | None:
    """Exploratory: estimands per probe family, averaged over genres and seeds."""
    if not rows:
        return None
    c = {(r["genre"], r["seed"], r["probe_id"], r["condition"]): r[metric] for r in rows
         if r["phase"] == phase and r["system_variant"] == variant}
    out = {}
    for probe in sorted({k[2] for k in c}):
        keys = sorted({k[:2] for k in c if k[2] == probe})
        vals = {"R": [], "L": [], "F": [], "P": []}
        for g, s in keys:
            v = {cond: c[(g, s, probe, cond)] for cond in
                 ("ordered", "neutral", "lexical_match", "lexical_neutral")}
            vals["R"].append(v["ordered"] - v["neutral"])
            vals["L"].append(v["lexical_match"] - v["lexical_neutral"])
            vals["F"].append(v["lexical_neutral"] - v["neutral"])
            vals["P"].append(vals["R"][-1] - vals["L"][-1])
        out[probe] = {k: statistics.mean(v) for k, v in vals.items()}
    return out


def item_level(tables: dict[str, dict]) -> dict:
    """Probe family as the unit: sign-flip over families, and cross-model agreement."""
    out = {}
    for name, table in tables.items():
        if table:
            out[name] = {e: summarize_values([v[e] for v in table.values()]) for e in "RLFP"}
    a, b = tables.get("4b_confirmatory"), tables.get("17b_confirmatory")
    if a and b:
        common = sorted(set(a) & set(b))
        xs, ys = [a[p]["P"] for p in common], [b[p]["P"] for p in common]
        out["P_cross_model_pearson"] = statistics.correlation(xs, ys) if len(common) > 2 else None
    return out


def figures(results: dict) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    FIGS.mkdir(parents=True, exist_ok=True)
    made = []
    # Figure 1: P and L at exit (default system), per genre and pooled, both models.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, est_name in zip(axes, ("P", "L")):
        labels, means, lows, highs, colors = [], [], [], [], []
        for model, color in (("4b", "#2f5d8a"), ("17b", "#c26a2b")):
            groups = results.get(f"{model}_decomposition", {})
            for g in GENRES + ("pooled",):
                key = f"{g}/default/exit" if g != "pooled" else "default/exit"
                src = groups.get("pooled_over_genres" if g == "pooled" else "groups", {})
                x = src.get(key, {}).get(est_name)
                if x:
                    labels.append(f"{model} {g}")
                    means.append(x["mean"]); lows.append(x["mean"] - x["ci95"][0])
                    highs.append(x["ci95"][1] - x["mean"]); colors.append(color)
        ys = range(len(labels))
        ax.errorbar(means, ys, xerr=[lows, highs], fmt="none", ecolor="#555", capsize=2)
        ax.scatter(means, ys, c=colors, zorder=3)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_yticks(list(ys)); ax.set_yticklabels(labels)
        ax.set_title({"P": "P = enacted − quoted (DiD)", "L": "L = quoted − quoted neutral"}[est_name])
        ax.set_xlabel("copy-free margin, nats/token")
    fig.tight_layout(); fig.savefig(FIGS / "fig1_provenance.png", dpi=200); plt.close(fig)
    made.append("paper/figures/fig1_provenance.png")
    # Figure 2: time course of P (4B, default system).
    groups = results.get("4b_decomposition", {}).get("groups", {})
    phases = ("peak", "exit", "washout_1", "washout_2")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for g, color in zip(GENRES, ("#2f5d8a", "#c26a2b", "#4a8a3f")):
        pts = [(i, groups[f"{g}/default/{ph}"]["P"]) for i, ph in enumerate(phases)
               if "P" in groups.get(f"{g}/default/{ph}", {})]
        if pts:
            ax.errorbar([i for i, _ in pts], [x["mean"] for _, x in pts],
                        yerr=[[x["mean"] - x["ci95"][0] for _, x in pts],
                              [x["ci95"][1] - x["mean"] for _, x in pts]],
                        marker="o", color=color, capsize=2, label=g)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(range(len(phases))); ax.set_xticklabels(phases)
    ax.set_ylabel("P, nats/token"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "fig2_timecourse.png", dpi=200); plt.close(fig)
    made.append("paper/figures/fig2_timecourse.png")
    return made


def cell(x: dict | None) -> str:
    if not x:
        return "missing"
    return (f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}], "
            f"{x['positive_seeds']}/{x['n_seeds']} seeds > 0, p = {x['signflip']['p_value']:.2g}")


def main() -> None:
    results: dict = {"preregistration_commit": "3fc5105", "gates": {}, "hypotheses": {}}
    results["gates"]["placebo"] = placebo_gate(rows_of("4b-placebo"))
    main4 = rows_of(*(f"4b-main-{g}" for g in GENRES))
    wash4 = rows_of(*(f"4b-washout-{g}" for g in GENRES))
    main17 = rows_of(*(f"17b-main-{g}" for g in GENRES))
    raw: dict[str, tuple] = {}
    if main4:
        results["gates"]["length_4b"] = length_gate(main4)
        est = seed_estimates(main4, metric="copy_free_margin")
        for name, (variant, phase, key) in {"H1": ("default", "exit", "P"),
                                            "H2": ("default", "exit", "L")}.items():
            raw[name] = (summarize_values(list(pooled(est, variant, phase, key).values())), name)
        plain, default = pooled(est, "plain", "peak", "P"), pooled(est, "default", "peak", "P")
        raw["H3"] = (summarize_values([plain[s] - default[s] for s in sorted(plain)]), "H3")
        results["4b_decomposition"] = decompose(main4 + (wash4 or []),
                                                copy_metric="copyable_margin")
    h4 = sweep_specificity(RUN / "q6-sweep.jsonl", layer=18, mode="erase") \
        if (RUN / "q6-sweep.done").exists() else None
    if h4:
        raw["H4"] = (h4, "H4")
    pvals = {k: v[0]["signflip"]["p_value"] for k, v in raw.items()}
    adjusted = holm(pvals) if len(pvals) == 4 else {}
    predictions = {"H1": -1, "H2": 1, "H3": 0, "H4": 1}
    for key in ("H1", "H2", "H3", "H4"):
        if key not in raw:
            results["hypotheses"][key] = {"status": "missing"}
            continue
        x = raw[key][0]
        p_adj = adjusted.get(key)
        sign_ok = predictions[key] == 0 or (x["mean"] * predictions[key] > 0)
        status = ("confirmed" if p_adj is not None and p_adj < ALPHA and sign_ok else
                  "not confirmed" if p_adj is not None else "pending other hypotheses")
        results["hypotheses"][key] = {**x, "holm_p": p_adj, "status": status,
                                      "small_effect": abs(x["mean"]) < 0.10}
    if (RUN / "q6-sweep.done").exists():
        results["q6_secondary"] = {
            f"block{b}/{m}": sweep_specificity(RUN / "q6-sweep.jsonl", layer=b, mode=m)
            for b in (9, 18, 27) for m in ("erase", "inject")}
    if main17:
        results["gates"]["length_17b"] = length_gate(main17)
        results["17b_decomposition"] = decompose(main17, copy_metric="copyable_margin")
    results["q5"] = q5_rates(RUN / "q5-sample.jsonl") if (RUN / "q5-sample.done").exists() else None
    cal = []
    for g in GENRES:
        path = ROOT / "reports" / "cal9200" / g / "rows.jsonl"
        if path.exists():
            cal += [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    tables = {"4b_confirmatory": per_probe(main4), "17b_confirmatory": per_probe(main17),
              "4b_calibration_9200": per_probe(cal or None),
              "4b_origprobes_confirmatory_seeds": per_probe(
                  rows_of(*(f"x-4b-origprobes-{g}" for g in GENRES)))}
    results["exploratory_per_probe"] = tables
    results["exploratory_item_level"] = item_level(tables)
    results["figures"] = figures(results)
    (REPORT / "confirmatory.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = ["# Confirmatory results (preregistration `3fc5105`)", "",
             "Generated by `scripts/confirmatory_analysis.py` from `runs/confirm`. "
             "Units: copy-free margin, nats/token; seeds are the test unit.", "",
             "## Gates", ""]
    for name, gate in results["gates"].items():
        lines.append(f"- **{name}:** " + ", ".join(f"{k} = {v}" for k, v in gate.items()))
    lines += ["", "## Confirmatory hypotheses (Holm over H1–H4)", "",
              "| | Hypothesis | Estimate [95% CI], seeds, raw p | Holm p | Status |",
              "|---|---|---|---|---|"]
    text = {"H1": "P < 0 (4B, exit, default, pooled genres)",
            "H2": "L > 0 (same cell)",
            "H3": "ΔP = P(plain) − P(default) ≠ 0 (4B, peak, pooled)",
            "H4": "erasure specificity > 0 (4B, gothic, block 18)"}
    for key in ("H1", "H2", "H3", "H4"):
        h = results["hypotheses"][key]
        if h.get("status") == "missing":
            lines.append(f"| {key} | {text[key]} | missing | – | missing |")
        else:
            hp = f"{h['holm_p']:.2g}" if h["holm_p"] is not None else "–"
            lines.append(f"| {key} | {text[key]} | {cell(h)} | {hp} | {h['status']}"
                         + (" (small)" if h["small_effect"] else "") + " |")
    for model in ("4b", "17b"):
        dec = results.get(f"{model}_decomposition")
        if dec:
            from persona_holonomy.decomposition import markdown
            lines += ["", f"## Decomposition, {'Qwen3-4B-Instruct (bf16)' if model == '4b' else 'Qwen3-1.7B (float32, no thinking)'}", "",
                      markdown(dec)]
            if dec.get("copying"):
                lines += ["Copying share (genre-specific copyable target minus copy-free), at exit, default:", ""]
                for g in GENRES:
                    x = dec["copying"].get(f"{g}/default/exit", {})
                    lines.append(f"- {g}: R {cell(x.get('R'))}; P {cell(x.get('P'))}")
    if results.get("q6_secondary"):
        lines += ["", "## Q6 direction sweep (raw: candidate effect − mean of 8 nulls)", "",
                  "| block / mode | specificity | candidate | null | self-patch max |", "|---|---|---|---|---|"]
        for key, x in results["q6_secondary"].items():
            if x:
                lines.append(f"| {key} | {cell(x)} | {x['raw_candidate_mean']:+.3f} | "
                             f"{x['raw_null_mean']:+.3f} | {x['self_patch_max_abs']:.1e} |")
    if results.get("q5"):
        q5 = results["q5"]
        lines += ["", f"## Q5 sampled behaviour ({q5['total_samples']} samples, T = 1)", "",
                  "| genre / condition | genre intrusion | theatrical markers |", "|---|---|---|"]
        for key, v in q5["rates"].items():
            lines.append(f"| {key} | {v['genre_intrusion_rate']:.3f} | {v['theatrical_rate']:.3f} |")
        for name, x in q5["theatrical_rate_contrasts"].items():
            lines.append(f"\nTheatrical-rate contrast {name}: {cell(x)}")
    lines += ["", "## Exploratory (not preregistered): probe family as the unit", "",
              "Seed-level tests treat the probe families as fixed. Below, each family is one "
              "unit (values averaged over genres and seeds; exit, default system).", ""]
    for name, table in results["exploratory_per_probe"].items():
        if not table:
            lines += [f"- {name}: missing", ""]
            continue
        lines += [f"**{name}**", "", "| probe family | R | L | F | P |", "|---|---|---|---|---|"]
        for probe, v in table.items():
            lines.append(f"| {probe} | {v['R']:+.3f} | {v['L']:+.3f} | {v['F']:+.3f} | {v['P']:+.3f} |")
        item = results["exploratory_item_level"].get(name)
        if item:
            lines.append("")
            lines.append("Across families: " + "; ".join(
                f"{e} mean {item[e]['mean']:+.3f}, {item[e]['positive_seeds']}/{item[e]['n_seeds']} "
                f"families > 0, sign-flip p = {item[e]['signflip']['p_value']:.2g}" for e in "LP"))
        lines.append("")
    r = results["exploratory_item_level"].get("P_cross_model_pearson")
    if r is not None:
        lines.append(f"Per-family P, 4B vs 1.7B: Pearson r = {r:+.2f} over the six families.")
    (REPORT / "confirmatory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT / 'confirmatory.md'} and confirmatory.json; figures: {results['figures']}")


if __name__ == "__main__":
    main()
