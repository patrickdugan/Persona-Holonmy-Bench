"""CPU-only falsification tests for the synthetic connection oracle."""
from math import cos, sin, pi
import pytest

from persona_holonomy.synthetic_manifold_oracle import (
    endpoint_linear, path_difference, rotate, transport,
)

SQUARE = ((0., 0.), (1., 0.), (1., 1.), (0., 1.), (0., 0.))
STRAIGHT = ((0., 0.), (1., 0.), (1., 1.))
BENT = ((0., 0.), (0., 1.), (1., 1.))


def test_zero_dose_is_identity():
    assert transport(SQUARE, dose=0) == 0
    assert path_difference(STRAIGHT, BENT, dose=0) == 0
    assert endpoint_linear(STRAIGHT, dose=0) == 0


def test_closed_path_reversal_changes_sign():
    assert transport(SQUARE) == pytest.approx(1)
    assert transport(tuple(reversed(SQUARE))) == pytest.approx(-1)
    assert transport(SQUARE) + transport(tuple(reversed(SQUARE))) == pytest.approx(0)


def test_matched_endpoints_detect_path_dependence():
    assert path_difference(STRAIGHT, BENT) == pytest.approx(1)
    assert path_difference(BENT, STRAIGHT) == pytest.approx(-1)
    assert endpoint_linear(STRAIGHT) == endpoint_linear(BENT)


@pytest.mark.parametrize("theta", [0., pi / 7, pi / 2, pi, 1.73])
def test_proper_rotation_invariance(theta):
    rotated = rotate(SQUARE, cos(theta), sin(theta))
    assert transport(rotated) == pytest.approx(transport(SQUARE), abs=1e-12)
    assert path_difference(rotate(STRAIGHT, cos(theta), sin(theta)),
                           rotate(BENT, cos(theta), sin(theta))) == pytest.approx(1, abs=1e-12)


def test_linear_case_adds_nothing():
    assert transport(SQUARE, curvature=0) == 0
    assert path_difference(STRAIGHT, BENT, curvature=0) == 0
    assert endpoint_linear(STRAIGHT, vector=(2., -3.)) == endpoint_linear(
        BENT, vector=(2., -3.))
    assert endpoint_linear(SQUARE, vector=(2., -3.)) == 0


def test_dose_scaling_predeclared():
    assert transport(SQUARE, dose=0.5) == pytest.approx(0.5)
    assert transport(SQUARE, dose=-1) == pytest.approx(-1)


@pytest.mark.parametrize("bad", [
    ((0., 0.),),
    ((0., 0.), (float("nan"), 1.)),
    ((0., 0.), (float("inf"), 1.)),
])
def test_invalid_paths_rejected(bad):
    with pytest.raises(ValueError):
        transport(bad)


def test_unmatched_endpoints_rejected():
    with pytest.raises(ValueError):
        path_difference(STRAIGHT, SQUARE)


def test_non_rotation_rejected():
    with pytest.raises(ValueError):
        rotate(SQUARE, 2, 0)


def test_nonfinite_parameters_rejected():
    with pytest.raises(ValueError):
        transport(SQUARE, dose=float("inf"))
    with pytest.raises(ValueError):
        endpoint_linear(SQUARE, dose=float("nan"))
