"""Offline, causal one-step temporal diagnostic. Python standard library only.

An exploratory linear autoregression, not Temporal SAE or a persona model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass

@dataclass(frozen=True)
class History:
    seed: int
    split: str
    states: tuple[tuple[float, ...], ...]
    events: tuple[int, ...] = ()

    def __post_init__(self):
        if self.split not in ("train", "test"):
            raise ValueError("split must be train/test")
        if len(self.states) < 2 or not self.states[0]:
            raise ValueError("history needs >=2 nonempty states")
        d = len(self.states[0])
        if any(len(x) != d or any(not math.isfinite(v) for v in x) for x in self.states):
            raise ValueError("ragged or nonfinite states")
        if any(t < 1 or t >= len(self.states) for t in self.events):
            raise ValueError("event index out of range")

@dataclass(frozen=True)
class Fit:
    train_seeds: frozenset[int]
    dimension: int
    mean: tuple[float, ...]
    slope: tuple[float, ...]
    intercept: tuple[float, ...]

def _validate(histories):
    if not histories:
        raise ValueError("empty histories")
    seeds = [h.seed for h in histories]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed repeated across histories/splits: unit leakage")
    if len({len(h.states[0]) for h in histories}) != 1:
        raise ValueError("dimension mismatch")

def fit(histories, *, shuffled=False, shuffle_seed=731, ridge=1e-9):
    """Fit diagonal AR(1) from training histories only; no future covariates."""
    histories = tuple(histories)
    _validate(histories)
    if any(h.split != "train" for h in histories):
        raise ValueError("held-out-fit rejection: training histories only")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    rng = random.Random(shuffle_seed)
    d = len(histories[0].states[0])
    pairs = []
    for h in histories:
        states = list(h.states)
        if shuffled:
            rng.shuffle(states)  # destroy order independently within each train history
        pairs.extend(zip(states[:-1], states[1:]))
    xs, ys = zip(*pairs)
    mean = tuple(sum(y[j] for y in ys) / len(ys) for j in range(d))
    slopes, offsets = [], []
    for j in range(d):
        mx = sum(x[j] for x in xs) / len(xs)
        my = mean[j]
        a = sum((x[j]-mx)*(y[j]-my) for x,y in pairs)
        b = sum((x[j]-mx)**2 for x in xs) + ridge
        slope = a/b
        slopes.append(slope)
        offsets.append(my-slope*mx)
    return Fit(frozenset(h.seed for h in histories), d, mean,
               tuple(slopes), tuple(offsets))

def evaluate(model, histories, *, shuffled_control=None):
    """One score per independent test history, never one per repeated probe."""
    histories = tuple(histories)
    _validate(histories)
    if any(h.split != "test" or h.seed in model.train_seeds for h in histories):
        raise ValueError("test split overlaps fit or includes training history")
    if shuffled_control is not None and (shuffled_control.train_seeds != model.train_seeds
                                        or shuffled_control.dimension != model.dimension):
        raise ValueError("control fit provenance mismatch")
    if any(len(h.states[0]) != model.dimension for h in histories):
        raise ValueError("dimension mismatch")
    results = []
    for h in histories:
        err, static, shuffled, residuals = [], [], [], []
        for t in range(1, len(h.states)):
            previous, actual = h.states[t-1], h.states[t]
            prediction = tuple(model.intercept[j] + model.slope[j]*previous[j]
                               for j in range(model.dimension))
            residual = tuple(actual[j]-prediction[j] for j in range(model.dimension))
            residuals.append(residual)
            err.append(sum(v*v for v in residual)/model.dimension)
            static.append(sum((actual[j]-model.mean[j])**2 for j in range(model.dimension))/model.dimension)
            if shuffled_control:
                shuffle_prediction = tuple(shuffled_control.intercept[j]+
                                           shuffled_control.slope[j]*previous[j]
                                           for j in range(model.dimension))
                shuffled.append(sum((actual[j]-shuffle_prediction[j])**2
                                    for j in range(model.dimension))/model.dimension)
        event_steps = [t for t in h.events]
        other_steps = [t for t in range(1,len(h.states)) if t not in h.events]
        def magnitude(t):
            return math.sqrt(sum(v*v for v in residuals[t-1]))
        results.append({"seed":h.seed, "steps":len(err), "mse":sum(err)/len(err),
                        "static_mse":sum(static)/len(static),
                        "shuffle_mse":sum(shuffled)/len(shuffled) if shuffled else None,
                        "event_residual_norm":sum(magnitude(t) for t in event_steps)/len(event_steps)
                        if event_steps else None,
                        "ordinary_residual_norm":sum(magnitude(t) for t in other_steps)/len(other_steps)
                        if other_steps else None,
                        "residuals":[list(v) for v in residuals]})
    def avg(field):
        vals = [row[field] for row in results if row[field] is not None]
        return sum(vals)/len(vals) if vals else None
    return {"evidence":"synthetic_or_user_authorized_input_only",
            "independent_test_units":len(results), "repeated_steps_are_not_units":True,
            "mean_mse":avg("mse"),"mean_static_mse":avg("static_mse"),
            "mean_shuffle_mse":avg("shuffle_mse"),"per_history":results}

def synthetic(kind, seeds=range(16), length=12):
    if kind not in ("stationary", "drifting", "order_sensitive", "event_boundary"):
        raise ValueError("unknown synthetic kind")
    out = []
    for i,seed in enumerate(seeds):
        rng = random.Random(1000+seed)
        x = rng.uniform(-0.2,0.2)
        states=[(x,)]
        events=[]
        for t in range(1,length):
            if kind == "stationary":
                x = 0.8*x + rng.uniform(-0.02,0.02)
            elif kind == "drifting":
                x = 0.8*x + 0.4 + rng.uniform(-0.02,0.02)
            elif kind == "order_sensitive":
                x = -0.9*x + (0.15 if t % 2 else -0.15)
            else:
                x = 0.8*x + rng.uniform(-0.02,0.02)
                if t == length//2:
                    x += 3.0
                    events.append(t)
            states.append((x,))
        out.append(History(seed,"train" if i < len(seeds)//2 else "test",
                           tuple(states),tuple(events)))
    return out

def main(argv=None):
    parser=argparse.ArgumentParser(description="Offline synthetic temporal diagnostic")
    parser.add_argument("--kind",choices=["stationary","drifting","order_sensitive","event_boundary"],
                        default="event_boundary")
    args=parser.parse_args(argv)
    hs=synthetic(args.kind)
    train=[h for h in hs if h.split=="train"]
    result=evaluate(fit(train),[h for h in hs if h.split=="test"],
                    shuffled_control=fit(train,shuffled=True))
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=="__main__":
    main()
