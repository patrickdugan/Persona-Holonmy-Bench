"""Seed-clustered randomization inference for registered causal contrasts.

The unit of permutation is the entire held-out transcript seed. All probes
within a seed have already been averaged before this module is called.
The sign-flip test relies on a symmetry/exchangeability assumption for paired
seed-level null effects; it does not make an observational dataset randomized.
"""
from __future__ import annotations

import itertools
import math
import random
import statistics
from typing import Sequence


def signflip_pvalue(values: Sequence[float], *, seed: int = 20261022,
                    exact_limit: int = 16, permutations: int = 20000) -> dict:
    """Two-sided sign-flip p-value for the seed-mean paired contrast.

    Exact for n<=exact_limit, otherwise deterministic Monte Carlo. In the
    Monte Carlo branch we add one to numerator and denominator, including
    the original unpermuted configuration as a conservative reference.
    """
    data = [float(v) for v in values]
    if not data or any(not math.isfinite(v) for v in data):
        raise ValueError("Sign-flip test requires finite observations")
    if not 1 <= exact_limit <= 20:
        raise ValueError("exact_limit must be in 1..20")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    observed = abs(statistics.mean(data))
    threshold = observed - 1e-12 * max(1.0, observed)
    n = len(data)
    if n <= exact_limit:
        extreme = sum(
            abs(sum(x * sign for x, sign in zip(data, flips)) / n) >= threshold
            for flips in itertools.product((-1, 1), repeat=n)
        )
        total = 2 ** n
        return {"p_value": extreme / total, "method": "exact_sign_flip",
                "n_seed_clusters": n, "null_draws": total}
    rng = random.Random(seed)
    extreme = 0
    for _ in range(permutations):
        sample = sum(v * (1 if rng.getrandbits(1) else -1) for v in data) / n
        if abs(sample) >= threshold:
            extreme += 1
    return {"p_value": (extreme + 1) / (permutations + 1),
            "method": "monte_carlo_sign_flip", "n_seed_clusters": n,
            "null_draws": permutations}


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm familywise step-down p adjustment, preserving original ordering."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("p-values must be finite and in [0,1]")
    n = len(p_values)
    adjusted = [0.0] * n
    floor = 0.0
    for rank, idx in enumerate(sorted(range(n), key=lambda j: (p_values[j], j))):
        floor = max(floor, min(1.0, (n-rank) * p_values[idx]))
        adjusted[idx] = floor
    return adjusted
