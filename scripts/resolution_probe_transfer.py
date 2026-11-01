"""Offline audit of PHB's probe-family sign reversals.

Reads frozen calibration summaries and the published confirmatory report.
No inference, model downloads, network access, or new experimental samples.

The two strata used different seed ranges and probe families. The audit
reports descriptive transfer diagnostics, not a pooled causal estimand.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from statistics import mean

GENRES = ("gothic", "pirate", "noir")
CALIBRATION_PROBES = ("flaky_ci", "parser_validator", "stale_cache")
CONFIRMATORY_PROBES = (
    "feature_flag", "log_rotation", "memory_leak", "rate_limit",
    "retry_backoff", "schema_migration",
)
READOUT = "copy_free_margin"


def exact_signflip_p(values: list[float]) -> float:
    """Two-sided exact paired sign-flip p-value, families as the units."""
    if not values or len(values) > 20:
        raise ValueError("Sign-flip requires between 1 and 20 independent units")
    observed = abs(sum(values))
    exceed = sum(
        abs(sum(sign * v for sign, v in zip(signs, values))) >= observed - 1e-12
        for signs in product((-1, 1), repeat=len(values))
    )
    return exceed / (1 << len(values))


def calibration_probe_matrix(root: Path) -> dict[str, dict[str, float]]:
    """Per-genre, per-probe P = (ordered-neutral) - (quoted-quote-neutral)."""
    results: dict[str, dict[str, float]] = {}
    for genre in GENRES:
        path = root / "reports" / "cal9200" / genre / "summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        group = data["groups"][f"{genre}/default"]
        if group["n_seeds"] != 16:
            raise ValueError(f"Unexpected calibration seed count in {path}")
        contrasts = group["contrasts"]
        r = contrasts["exit/ordered-neutral"][READOUT]["per_probe_mean"]
        l = contrasts["exit/lexical_match-lexical_neutral"][READOUT]["per_probe_mean"]
        if set(r) != set(CALIBRATION_PROBES) or set(l) != set(CALIBRATION_PROBES):
            raise ValueError(f"Unexpected calibration probes in {path}")
        results[genre] = {probe: r[probe] - l[probe] for probe in CALIBRATION_PROBES}
    return results


def confirmatory_probe_effects(root: Path) -> dict[str, float]:
    """Read Qwen3-4B confirmatory table, validating published R - L = P."""
    path = root / "reports" / "confirmatory.md"
    doc = path.read_text(encoding="utf-8")
    if "**4b_confirmatory**" not in doc or "- 17b_confirmatory:" not in doc:
        raise ValueError("Missing or changed 4b_confirmatory report boundary")
    section = doc.split("**4b_confirmatory**", 1)[1].split("- 17b_confirmatory:", 1)[0]
    results: dict[str, float] = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) != 5 or fields[0] not in CONFIRMATORY_PROBES:
            continue
        probe, r, l, f, p = fields
        if probe in results:
            raise ValueError(f"Duplicate confirmatory probe: {probe}")
        r_value, l_value, _, p_value = map(float, (r, l, f, p))
        if abs((r_value - l_value) - p_value) > 0.002:
            raise ValueError(f"Reported P != R - L (rounding tolerance) for {probe}")
        results[probe] = p_value
    if set(results) != set(CONFIRMATORY_PROBES):
        raise ValueError(f"Missing confirmatory probes: {set(CONFIRMATORY_PROBES) - set(results)}")
    return results


def describe(values: list[float]) -> dict:
    return {
        "n_probe_families": len(values),
        "mean": mean(values),
        "n_positive": sum(v > 0 for v in values),
        "n_negative": sum(v < 0 for v in values),
        "exact_two_sided_family_signflip_p": exact_signflip_p(values),
        "leave_one_family_out_means": [(sum(values) - v) / (len(values) - 1) for v in values]
        if len(values) > 1 else [],
    }


def audit(root: Path) -> dict:
    matrix = calibration_probe_matrix(root)
    calibration = {
        probe: mean(matrix[genre][probe] for genre in GENRES)
        for probe in CALIBRATION_PROBES
    }
    confirmatory = confirmatory_probe_effects(root)
    cal_values = [calibration[k] for k in CALIBRATION_PROBES]
    conf_values = [confirmatory[k] for k in CONFIRMATORY_PROBES]
    cal_sign = 1 if mean(cal_values) > 0 else -1 if mean(cal_values) < 0 else 0
    same_sign = sum(v * cal_sign > 0 for v in conf_values)
    return {
        "schema": "phb.resolution_probe_transfer.v1",
        "readout": READOUT,
        "contrast": "P=(ordered-neutral)-(lexical_match-lexical_neutral)",
        "calibration_9200": {"per_probe": calibration, "per_genre": matrix, **describe(cal_values)},
        "confirmatory_4b": {"per_probe": confirmatory, **describe(conf_values)},
        "cross_stratum": {
            "confirmatory_families_matching_calibration_mean_sign": same_sign,
            "confirmatory_families_total": len(conf_values),
            "mean_shift_confirmatory_minus_calibration": mean(conf_values) - mean(cal_values),
        },
        "limitations": [
            "Both strata concern theatrical style; no orthogonal persona trait was measured.",
            "Calibration and confirmatory families and seed ranges differ; no pooled causal estimate.",
            "The 32-seed confirmatory p-value holds probe families fixed; family-level p uses six units.",
            "Per-genre confirmatory raw rows and activations are absent from this offline audit.",
            "Sign reversal in a scalar likelihood margin does not establish latent-frame holonomy.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Write JSON report; otherwise print to stdout")
    args = parser.parse_args()
    report = json.dumps(audit(args.repo_root), sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
