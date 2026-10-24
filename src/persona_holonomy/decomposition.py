"""Enacted-vs-quoted residue decomposition (docs/paper-spec.md, Q1-Q4).

Input rows carry one forced-continuation readout per case (seed, probe, phase,
condition, genre, system_variant), e.g. from scripts/h1_calibration.py. Every
estimand is a paired contrast averaged over probes within a seed; seeds are the
randomization unit for the sign-flip test and the bootstrap.

    R = ordered - neutral                      (residue after a persona history)
    L = lexical_match - lexical_neutral        (quoted persona text, format fixed)
    F = lexical_neutral - neutral              (counting-format effect alone)
    P = R - L                                  (enacted minus quoted, difference-in-differences)

P is a contrast decomposition, not an identified causal one: quoting bundles
provenance, quotation and task, and token positions differ beyond length.
"""
from __future__ import annotations

import random
import statistics
from typing import Any, Iterable

from .inference import signflip_pvalue

CONDITIONS = ("ordered", "neutral", "lexical_match", "lexical_neutral")
ESTIMANDS = {
    "R": (("ordered", 1), ("neutral", -1)),
    "L": (("lexical_match", 1), ("lexical_neutral", -1)),
    "F": (("lexical_neutral", 1), ("neutral", -1)),
    "P": (("ordered", 1), ("neutral", -1), ("lexical_match", -1), ("lexical_neutral", 1)),
}


def _bootstrap_ci(values: list[float], *, draws: int = 10000, seed: int = 20261025):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(draws))
    return [means[int(.025 * draws)], means[int(.975 * draws) - 1]]


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("No seed-level values to summarize")
    return {
        "mean": statistics.mean(values),
        "sd_seed": statistics.stdev(values) if len(values) > 1 else None,
        "ci95": _bootstrap_ci(values) if len(values) > 1 else None,
        "positive_seeds": sum(v > 0 for v in values),
        "n_seeds": len(values),
        "signflip": signflip_pvalue(values),
    }


def _index(rows: Iterable[dict]) -> dict[tuple, dict]:
    cells: dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("genre", "gothic"), row.get("system_variant", "default"),
               row["phase"], row["seed"], row["probe_id"], row["condition"])
        if key in cells:
            raise ValueError(f"Duplicate cell {key}")
        cells[key] = row
    return cells


def seed_estimates(rows: Iterable[dict], *, metric: str) -> dict[tuple, dict[str, dict[int, float]]]:
    """{(genre, system_variant, phase): {estimand: {seed: probe-mean contrast}}}."""
    cells = _index(rows)
    groups = sorted({k[:3] for k in cells})
    out: dict[tuple, dict[str, dict[int, float]]] = {}
    for genre, variant, phase in groups:
        seeds = sorted({k[3] for k in cells if k[:3] == (genre, variant, phase)})
        probes = sorted({k[4] for k in cells if k[:3] == (genre, variant, phase)})
        result: dict[str, dict[int, float]] = {}
        for name, terms in ESTIMANDS.items():
            needed = [(s, p, c) for s in seeds for p in probes for c, _ in terms]
            if any((genre, variant, phase, s, p, c) not in cells or
                   cells[(genre, variant, phase, s, p, c)].get(metric) is None
                   for s, p, c in needed):
                continue
            result[name] = {
                s: statistics.mean(
                    sum(sign * cells[(genre, variant, phase, s, p, c)][metric]
                        for c, sign in terms)
                    for p in probes)
                for s in seeds}
        out[(genre, variant, phase)] = result
    return out


def decompose(rows: list[dict], *, metric: str = "copy_free_margin",
              copy_metric: str | None = "margin") -> dict[str, Any]:
    """Estimands per group, genre-pooled, the system-prompt effect on P, and copying share."""
    by_group = seed_estimates(rows, metric=metric)
    report: dict[str, Any] = {"metric": metric, "groups": {}, "pooled_over_genres": {},
                              "system_plain_minus_default": {}, "copying": {}}
    for (genre, variant, phase), estimands in by_group.items():
        report["groups"][f"{genre}/{variant}/{phase}"] = {
            name: summarize_values(list(values.values())) for name, values in estimands.items()}
    # Pool genres within (variant, phase): seed value = mean over genres.
    pooled_keys = sorted({(v, ph) for _, v, ph in by_group})
    for variant, phase in pooled_keys:
        genres = [g for g, v, ph in by_group if (v, ph) == (variant, phase)]
        entry = {}
        for name in ESTIMANDS:
            series = [by_group[(g, variant, phase)].get(name) for g in genres]
            if any(s is None for s in series):
                continue
            seeds = set.intersection(*(set(s) for s in series))
            entry[name] = summarize_values(
                [statistics.mean(s[seed] for s in series) for seed in sorted(seeds)])
        report["pooled_over_genres"][f"{variant}/{phase}"] = entry
    # Q4(a): does dropping the fiction-ending clause change P? Paired by seed.
    for genre, variant, phase in by_group:
        if variant != "plain" or (genre, "default", phase) not in by_group:
            continue
        plain = by_group[(genre, "plain", phase)].get("P")
        default = by_group[(genre, "default", phase)].get("P")
        if plain and default:
            seeds = sorted(set(plain) & set(default))
            report["system_plain_minus_default"][f"{genre}/{phase}"] = summarize_values(
                [plain[s] - default[s] for s in seeds])
    # Q3: copying share = copyable-target contrast minus copy-free contrast, per estimand.
    if copy_metric:
        copyable = seed_estimates(rows, metric=copy_metric)
        for key, estimands in copyable.items():
            entry = {}
            for name, values in estimands.items():
                base = by_group.get(key, {}).get(name)
                if base:
                    seeds = sorted(set(values) & set(base))
                    entry[name] = summarize_values([values[s] - base[s] for s in seeds])
            report["copying"]["/".join(key)] = entry
    return report


def markdown(report: dict[str, Any]) -> str:
    def cell(x):
        if not x:
            return "n/a"
        p = x["signflip"]["p_value"]
        return (f"{x['mean']:+.3f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}] "
                f"{x['positive_seeds']}/{x['n_seeds']} p={p:.2g}")
    lines = [f"Metric: `{report['metric']}` (nats/token; positive = more theatrical).", "",
             "| group | R | L | F | P |", "|---|---|---|---|---|"]
    for key, est in report["groups"].items():
        lines.append(f"| {key} | " + " | ".join(cell(est.get(n)) for n in "RLFP") + " |")
    lines += ["", "Pooled over genres:", "", "| variant/phase | R | L | F | P |", "|---|---|---|---|---|"]
    for key, est in report["pooled_over_genres"].items():
        lines.append(f"| {key} | " + " | ".join(cell(est.get(n)) for n in "RLFP") + " |")
    if report["system_plain_minus_default"]:
        lines += ["", "P(system_plain) − P(default):", "", "| genre/phase | ΔP |", "|---|---|"]
        for key, x in report["system_plain_minus_default"].items():
            lines.append(f"| {key} | {cell(x)} |")
    return "\n".join(lines) + "\n"
