"""Paired analysis of persona carryover over history and recovery checkpoints."""
from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .scoring import score_response


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {lineno}") from exc
        if not isinstance(row, dict) or "response" not in row:
            raise ValueError(f"Missing response object on line {lineno}")
        records.append(row)
    return records


def _bootstrap_ci(values: list[float], *, iterations: int = 2000) -> list[float] | None:
    if len(values) < 5:
        return None
    rng = random.Random(20261022)
    n = len(values)
    estimates = sorted(
        statistics.mean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(iterations)
    )
    return [round(estimates[int(iterations * 0.025)], 4),
            round(estimates[int(iterations * 0.975)], 4)]


def paired_effect(
    rows: list[dict[str, Any]], *,
    phase: str, left: str, right: str, metric: str = "persona_index"
) -> dict[str, Any] | None:
    # Match by prompt and seed, then average within seed before bootstrapping.
    l = {(row["seed"], row["probe_id"]): row["scores"][metric]
         for row in rows if row["condition"] == left and row["phase"] == phase}
    r = {(row["seed"], row["probe_id"]): row["scores"][metric]
         for row in rows if row["condition"] == right and row["phase"] == phase}
    shared = sorted(l.keys() & r.keys())
    if not shared:
        return None
    by_seed: dict[int, list[float]] = defaultdict(list)
    for seed, probe_id in shared:
        by_seed[seed].append(l[seed, probe_id] - r[seed, probe_id])
    clustered = [statistics.mean(v) for _, v in sorted(by_seed.items())]
    return {
        "phase": phase,
        "contrast": f"{left} minus {right}",
        "metric": metric,
        "n_paired_probes": len(shared),
        "n_seeds": len(clustered),
        "mean_difference": round(statistics.mean(clustered), 4),
        "bootstrap_95pct_ci": _bootstrap_ci(clustered),
    }


def _matched_scores(
    rows: list[dict[str, Any]], *, condition: str, phase: str,
    metric: str = "persona_index",
) -> dict[tuple[int, str], float]:
    return {(row["seed"], row["probe_id"]): row["scores"][metric]
            for row in rows if row["condition"] == condition and row["phase"] == phase}


def closure_effect(
    rows: list[dict[str, Any]], *, condition: str, phase: str,
    metric: str = "persona_index",
) -> dict[str, Any] | None:
    """Post-history change relative to each seed/probe's own empty baseline.

    This behavioral recovery measure alone is NOT geometric holonomy.
    """
    baseline = _matched_scores(rows, condition=condition, phase="baseline", metric=metric)
    current = _matched_scores(rows, condition=condition, phase=phase, metric=metric)
    shared = sorted(baseline.keys() & current.keys())
    if not shared:
        return None
    by_seed: dict[int, list[float]] = defaultdict(list)
    for key in shared:
        by_seed[key[0]].append(current[key] - baseline[key])
    clustered = [statistics.mean(values) for _, values in sorted(by_seed.items())]
    return {
        "condition": condition,
        "phase": phase,
        "reference_phase": "baseline",
        "metric": metric,
        "n_paired_probes": len(shared),
        "n_seeds": len(clustered),
        "mean_change_from_baseline": round(statistics.mean(clustered), 4),
        "bootstrap_95pct_ci": _bootstrap_ci(clustered),
    }


def recovery_did(
    rows: list[dict[str, Any]], *, phase: str,
    left: str = "ordered", right: str = "permuted",
    metric: str = "persona_index",
) -> dict[str, Any] | None:
    """Paired difference-in-differences vs pre-induction baseline."""
    lb = _matched_scores(rows, condition=left, phase="baseline", metric=metric)
    lt = _matched_scores(rows, condition=left, phase=phase, metric=metric)
    rb = _matched_scores(rows, condition=right, phase="baseline", metric=metric)
    rt = _matched_scores(rows, condition=right, phase=phase, metric=metric)
    shared = sorted(lb.keys() & lt.keys() & rb.keys() & rt.keys())
    if not shared:
        return None
    by_seed: dict[int, list[float]] = defaultdict(list)
    for key in shared:
        by_seed[key[0]].append((lt[key] - lb[key]) - (rt[key] - rb[key]))
    clustered = [statistics.mean(values) for _, values in sorted(by_seed.items())]
    return {
        "phase": phase,
        "contrast": f"({left} - baseline) minus ({right} - baseline)",
        "metric": metric,
        "n_paired_probes": len(shared),
        "n_seeds": len(clustered),
        "mean_difference_in_differences": round(statistics.mean(clustered), 4),
        "bootstrap_95pct_ci": _bootstrap_ci(clustered),
    }


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot analyze an empty run")
    rows = []
    seen: set[str] = set()
    for row in records:
        identifier = row["case_id"]
        if identifier in seen:
            raise ValueError(f"Duplicate case_id in run: {identifier}")
        seen.add(identifier)
        new_row = dict(row)
        new_row["scores"] = score_response(row["response"])
        rows.append(new_row)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["phase"])].append(row)
    summaries = []
    for (condition, phase), subset in sorted(groups.items()):
        summaries.append({
            "condition": condition,
            "phase": phase,
            "n": len(subset),
            "mean_persona_index": round(statistics.mean(
                row["scores"]["persona_index"] for row in subset), 4),
            "genre_intrusion_rate": round(statistics.mean(
                row["scores"]["genre_intrusion"] for row in subset), 4),
            "mean_history_chars": round(statistics.mean(
                row["history_chars"] for row in subset), 1),
        })

    comparisons = []
    for phase in ("peak", "exit", "washout_1", "washout_2"):
        for right in ("tail_matched", "permuted", "neutral"):
            effect = paired_effect(rows, phase=phase, left="ordered", right=right)
            if effect is not None:
                comparisons.append(effect)

    closures = []
    for condition in ("ordered", "tail_matched", "permuted", "neutral"):
        for phase in ("peak", "exit", "washout_1", "washout_2"):
            effect = closure_effect(rows, condition=condition, phase=phase)
            if effect is not None:
                closures.append(effect)

    difference_in_differences = []
    for phase in ("peak", "exit", "washout_1", "washout_2"):
        for right in ("tail_matched", "permuted", "neutral"):
            effect = recovery_did(rows, phase=phase, right=right)
            if effect is not None:
                difference_in_differences.append(effect)

    synthetic = any(row.get("backend") == "mock" or
                    row.get("generation_meta", {}).get("synthetic")
                    for row in rows)
    return {
        "schema_version": 1,
        "n_records": len(rows),
        "synthetic": synthetic,
        "limitations": [
            "Scripted assistant turns: tests transcript conditioning, not autonomous multi-turn rollouts.",
            "Lexical features are exploratory proxies requiring blind human validation.",
            "Bootstrap intervals resample seeds; fewer than five seeds yield no interval.",
            "The setup cannot directly identify GPT-4.1-like internals or geometric holonomy.",
            "Context length, tokenization, and recency remain possible confounders.",
            "Baseline-to-recovery changes and difference-in-differences are behavioral, not geometric holonomy.",
        ],
        "groups": summaries,
        "comparisons": comparisons,
        "closure_effects": closures,
        "recovery_difference_in_differences": difference_in_differences,
    }


def to_markdown(report: dict[str, Any]) -> str:
    parts = ["# Persona Holonomy Bench: run report", ""]
    parts.append(f"Recorded generations: **{report['n_records']}**")
    if report["synthetic"]:
        parts.append("\n**SYNTHETIC MOCK DATA. No empirical model finding is implied.**")
    parts.extend([
        "", "## Behavior by condition and phase", "",
        "| Condition | Phase | N | Persona index | Genre intrusion | History chars |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for g in report["groups"]:
        parts.append(
            f"| {g['condition']} | {g['phase']} | {g['n']} | "
            f"{g['mean_persona_index']:.3f} | {g['genre_intrusion_rate']:.3f} | "
            f"{g['mean_history_chars']:.0f} |"
        )
    parts.extend([
        "", "## Paired comparisons", "",
        "Positive means the ordered trajectory had a larger persona index.", "",
        "| Phase | Contrast | Paired probes | Seeds | Δ mean | Seed-bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---|",
    ])
    for c in report["comparisons"]:
        ci = c["bootstrap_95pct_ci"]
        ci_label = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "insufficient seeds"
        parts.append(f"| {c['phase']} | {c['contrast']} | "
                     f"{c['n_paired_probes']} | {c['n_seeds']} | "
                     f"{c['mean_difference']:+.3f} | {ci_label} |")
    parts.extend(["", "## Behavioral closure: change from baseline", "",
        "These estimates pair each probe and seed with its own empty-history baseline.", "",
        "| Condition | Phase | Paired probes | Seeds | Change | Seed-bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---|",
    ])
    for c in report.get("closure_effects", []):
        ci = c["bootstrap_95pct_ci"]
        ci_label = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "insufficient seeds"
        parts.append(f"| {c['condition']} | {c['phase']} | {c['n_paired_probes']} | "
                     f"{c['n_seeds']} | {c['mean_change_from_baseline']:+.3f} | {ci_label} |")
    parts.extend(["", "## Recovery difference-in-differences", "",
        "Changes within condition are compared to changes in the control condition.", "",
        "| Phase | Contrast | Paired probes | Seeds | DiD | Seed-bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---|",
    ])
    for c in report.get("recovery_difference_in_differences", []):
        ci = c["bootstrap_95pct_ci"]
        ci_label = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "insufficient seeds"
        parts.append(f"| {c['phase']} | {c['contrast']} | {c['n_paired_probes']} | "
                     f"{c['n_seeds']} | {c['mean_difference_in_differences']:+.3f} | {ci_label} |")
    parts.extend(["", "## Interpretation limits", ""])
    parts.extend(f"- {item}" for item in report["limitations"])
    parts.append("")
    return "\n".join(parts)
