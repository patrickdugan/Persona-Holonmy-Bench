"""Offline, causal one-step temporal diagnostic; Python standard library only.

Exploratory diagonal autoregression, not Temporal SAE or a persona model.
Input consists solely of explicit in-memory numeric histories; no model/data loaders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class History:
    seed: int
    split: str
    states: tuple[tuple[float, ...], ...]
    events: tuple[int, ...] = ()
    probe_id: str = "default"

    def __post_init__(self):
        if self.split not in ("train", "test"):
            raise ValueError("split must be train/test")
        if not isinstance(self.probe_id, str) or not self.probe_id:
            raise ValueError("probe_id must be nonempty")
        if len(self.states) < 2 or not self.states[0]:
            raise ValueError("history needs >=2 nonempty states")
        dimension = len(self.states[0])
        if any(len(x) != dimension or any(not math.isfinite(v) for v in x)
               for x in self.states):
            raise ValueError("ragged or nonfinite states")
        if len(set(self.events)) != len(self.events) or any(
                t < 1 or t >= len(self.states) for t in self.events):
            raise ValueError("duplicate or invalid event index")


@dataclass(frozen=True)
class Fit:
    train_seeds: frozenset[int]
    training_fingerprint: str
    dimension: int
    mean: tuple[float, ...]
    mean_difference: tuple[float, ...]
    slope: tuple[float, ...]
    intercept: tuple[float, ...]


def _validate(histories):
    if not histories:
        raise ValueError("empty histories")
    seen, seed_splits = set(), {}
    dimension = len(histories[0].states[0])
    for h in histories:
        identity = (h.seed, h.probe_id)
        if identity in seen:
            raise ValueError("duplicate seed/probe: repeated rows would be overweighted")
        seen.add(identity)
        if h.seed in seed_splits and seed_splits[h.seed] != h.split:
            raise ValueError("seed spans train/test splits: group leakage")
        seed_splits[h.seed] = h.split
        if len(h.states[0]) != dimension:
            raise ValueError("dimension mismatch")
    return dimension


def _fingerprint(histories):
    rows = [{"seed": h.seed, "probe_id": h.probe_id, "states": h.states,
             "events": h.events} for h in sorted(histories, key=lambda x: (x.seed, x.probe_id))]
    data = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def fit(histories, *, shuffled=False, shuffle_seed=731, ridge=1e-9):
    """Fit diagonal AR(1) and a static mean-step-difference on training groups.

    Each independent seed has equal total weight, regardless of probe/step count.
    The shuffled control permutes only states within each training history.
    """
    histories = tuple(histories)
    dimension = _validate(histories)
    if any(h.split != "train" for h in histories):
        raise ValueError("held-out-fit rejection: training histories only")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    pairs_by_seed = defaultdict(list)
    for h in sorted(histories, key=lambda x: (x.seed, x.probe_id)):
        states = list(h.states)
        if shuffled:
            random.Random(f"{shuffle_seed}:{h.seed}:{h.probe_id}").shuffle(states)
        pairs_by_seed[h.seed].extend(zip(states[:-1], states[1:]))
    seeds = sorted(pairs_by_seed)
    n = len(seeds)

    def weighted_mean(index, dimension_index):
        return sum(sum(pair[index][dimension_index] for pair in pairs_by_seed[s]) /
                   len(pairs_by_seed[s]) for s in seeds) / n

    means, differences, slopes, offsets = [], [], [], []
    for j in range(dimension):
        mx = weighted_mean(0, j)
        my = weighted_mean(1, j)
        covariance = sum(sum((x[j] - mx) * (y[j] - my) for x, y in pairs_by_seed[s]) /
                         len(pairs_by_seed[s]) for s in seeds)
        variance = sum(sum((x[j] - mx) ** 2 for x, _ in pairs_by_seed[s]) /
                       len(pairs_by_seed[s]) for s in seeds)
        slope = covariance / (variance + ridge)
        means.append(my)
        differences.append(my - mx)
        slopes.append(slope)
        offsets.append(my - slope * mx)
    return Fit(frozenset(seeds), _fingerprint(histories), dimension,
               tuple(means), tuple(differences),
               tuple(slopes), tuple(offsets))


def _paired_bootstrap(comparator, candidate, *, seed=20261031, samples=2000):
    """Fixed-seed paired percentile interval across independent seed-level scores."""
    if len(comparator) != len(candidate):
        raise ValueError("unpaired comparator")
    differences = [b - a for b, a in zip(comparator, candidate)]
    n = len(differences)
    if n < 8:
        return {"status": "insufficient_independent_units", "n": n,
                "mean_improvement": sum(differences) / n if n else None,
                "bootstrap_95_interval": None}
    rng = random.Random(seed)
    draws = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(samples))
    return {"status": "descriptive_only", "n": n,
            "mean_improvement": sum(differences) / n,
            "bootstrap_95_interval": [draws[int(samples * .025)],
                                      draws[int(samples * .975) - 1]]}


def evaluate(model, histories, *, shuffled_control=None):
    """Hold out complete seeds and report one paired score per independent seed.

    At time t, each predictor sees only x[t-1] of the held-out history.
    It is teacher-forced one-step evaluation, not multi-step rollout.
    """
    histories = tuple(histories)
    _validate(histories)
    if any(h.split != "test" or h.seed in model.train_seeds for h in histories):
        raise ValueError("test split overlaps fit or includes training history")
    if any(len(h.states[0]) != model.dimension for h in histories):
        raise ValueError("dimension mismatch")
    if shuffled_control is not None and (
            shuffled_control.train_seeds != model.train_seeds or
            shuffled_control.training_fingerprint != model.training_fingerprint or
            shuffled_control.dimension != model.dimension):
        raise ValueError("control fit provenance fingerprint mismatch")

    raw = []
    for h in sorted(histories, key=lambda x: (x.seed, x.probe_id)):
        squared, static, mean_diff, shuffle, residuals = [], [], [], [], []
        for t in range(1, len(h.states)):
            previous, actual = h.states[t - 1], h.states[t]
            predicted = tuple(model.intercept[j] + model.slope[j] * previous[j]
                              for j in range(model.dimension))
            residual = tuple(actual[j] - predicted[j] for j in range(model.dimension))
            residuals.append(residual)
            squared.append(sum(v * v for v in residual) / model.dimension)
            static.append(sum((actual[j] - model.mean[j]) ** 2
                              for j in range(model.dimension)) / model.dimension)
            mean_diff.append(sum((actual[j] - previous[j] - model.mean_difference[j]) ** 2
                                 for j in range(model.dimension)) / model.dimension)
            if shuffled_control is not None:
                shuffled = tuple(shuffled_control.intercept[j] +
                                 shuffled_control.slope[j] * previous[j]
                                 for j in range(model.dimension))
                shuffle.append(sum((actual[j] - shuffled[j]) ** 2
                                   for j in range(model.dimension)) / model.dimension)

        def event_mean(indices):
            magnitudes = [math.sqrt(sum(v * v for v in residuals[t-1])) for t in indices]
            return sum(magnitudes) / len(magnitudes) if magnitudes else None

        non_events = [t for t in range(1, len(h.states)) if t not in h.events]
        raw.append({"seed": h.seed, "probe_id": h.probe_id, "steps": len(squared),
                    "mse": sum(squared) / len(squared),
                    "static_mean_mse": sum(static) / len(static),
                    "static_mean_difference_mse": sum(mean_diff) / len(mean_diff),
                    "shuffle_mse": sum(shuffle) / len(shuffle) if shuffle else None,
                    "events": len(h.events),
                    "event_residual_norm": event_mean(h.events),
                    "ordinary_residual_norm": event_mean(non_events),
                    "residuals": [list(v) for v in residuals]})

    grouped = defaultdict(list)
    for row in raw:
        grouped[row["seed"]].append(row)
    keys = ("mse", "static_mean_mse", "static_mean_difference_mse", "shuffle_mse",
            "event_residual_norm", "ordinary_residual_norm")
    per_seed = []
    for seed, group in sorted(grouped.items()):
        item = {"seed": seed, "probe_count": len(group),
                "steps": sum(r["steps"] for r in group),
                "events": sum(r["events"] for r in group)}
        for key in keys:
            values = [r[key] for r in group if r[key] is not None]
            item[key] = sum(values) / len(values) if values else None
        per_seed.append(item)

    def average(key):
        values = [row[key] for row in per_seed if row[key] is not None]
        return sum(values) / len(values) if values else None

    candidate = [r["mse"] for r in per_seed]
    paired = {}
    for label, key in (("static_mean_difference", "static_mean_difference_mse"),
                       ("static_mean", "static_mean_mse"),
                       ("time_shuffle", "shuffle_mse")):
        if all(r[key] is not None for r in per_seed):
            paired[label] = _paired_bootstrap([r[key] for r in per_seed], candidate)

    return {"evidence": "synthetic_or_explicitly_authorized_input_only",
            "training_fingerprint_sha256": model.training_fingerprint,
            "independent_test_units": len(per_seed),
            "repeated_steps_are_not_units": True,
            "mean_mse": average("mse"),
            "mean_static_mean_difference_mse": average("static_mean_difference_mse"),
            "mean_static_mse": average("static_mean_mse"),
            "mean_shuffle_mse": average("shuffle_mse"),
            "paired_group_bootstrap": paired,
            "per_seed": per_seed, "per_history": raw}


def synthetic(kind, seeds=range(16), length=12):
    """Deterministic mock trajectories. Planted events occur on held-out seeds only."""
    if kind not in ("stationary", "drifting", "order_sensitive", "event_boundary"):
        raise ValueError("unknown synthetic kind")
    seeds = tuple(seeds)
    if len(set(seeds)) != len(seeds) or len(seeds) < 4 or length < 4:
        raise ValueError("need unique seeds, >=4 seeds, and >=4 states")
    out = []
    for i, seed in enumerate(seeds):
        rng = random.Random(1000 + seed)
        x = rng.uniform(-1, 1)
        states, events = [(x,)], []
        split = "train" if i < len(seeds) // 2 else "test"
        for t in range(1, length):
            jitter = rng.uniform(-.01, .01)
            if kind == "drifting":
                x = .75 * x + .4 + jitter
            elif kind == "order_sensitive":
                x = -.9 * x + jitter
            else:
                x = .85 * x + jitter
                if kind == "event_boundary" and split == "test" and t == length // 2:
                    x += 3.0
                    events.append(t)
            states.append((x,))
        out.append(History(seed, split, tuple(states), tuple(events)))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline synthetic temporal diagnostic")
    parser.add_argument("--kind", choices=["stationary", "drifting", "order_sensitive",
                                           "event_boundary"], default="event_boundary")
    args = parser.parse_args(argv)
    histories = synthetic(args.kind)
    train = [h for h in histories if h.split == "train"]
    report = evaluate(fit(train), [h for h in histories if h.split == "test"],
                      shuffled_control=fit(train, shuffled=True))
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
