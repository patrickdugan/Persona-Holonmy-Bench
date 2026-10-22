"""Exact seed-level nulls and multiplicity adjustment."""
from __future__ import annotations

import pytest

from persona_holonomy.inference import holm_adjust, signflip_pvalue


def test_exact_sign_flip_resolution_and_symmetry():
    # All four independent seeds agree: two-sided p = 2/2**4 = 0.125.
    result = signflip_pvalue([.1, .2, .3, .4])
    assert result == {"p_value": .125, "method": "exact_sign_flip",
                      "n_seed_clusters": 4, "null_draws": 16}
    assert signflip_pvalue([-.1, -.2, -.3, -.4])["p_value"] == .125
    assert signflip_pvalue([0., 0.])["p_value"] == 1
    assert signflip_pvalue([1., -1.])["p_value"] == 1


def test_sign_flip_monte_carlo_is_reproducible_and_validates_inputs():
    a = signflip_pvalue([1.] * 8, exact_limit=4, permutations=100, seed=5)
    b = signflip_pvalue([1.] * 8, exact_limit=4, permutations=100, seed=5)
    assert a == b and a["method"] == "monte_carlo_sign_flip"
    assert 0 < a["p_value"] <= 1 and a["null_draws"] == 100
    for values in ([], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="finite"):
            signflip_pvalue(values)
    with pytest.raises(ValueError, match="exact_limit"):
        signflip_pvalue([1.], exact_limit=0)
    with pytest.raises(ValueError, match="permutations"):
        signflip_pvalue([1.], permutations=0)


def test_holm_stepdown_preserves_order_and_monotonicity():
    assert holm_adjust([.04, .01, .03, 1.]) == pytest.approx([.09, .04, .09, 1.])
    assert holm_adjust([]) == []
    for bad in ([-.1], [float("nan")], [1.001]):
        with pytest.raises(ValueError, match="p-values"):
            holm_adjust(bad)
