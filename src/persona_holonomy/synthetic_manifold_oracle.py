"""Deterministic 2D synthetic connection oracle, independent of any model pipeline.

The chosen connection A=(-k*y/2, k*x/2) has constant curvature k.
Its line integral is a toy transport diagnostic, not evidence of a learned
persona manifold. No model weights, external services, or randomness.
"""
from __future__ import annotations

from math import isfinite
from typing import Sequence

Point = tuple[float, float]


def _validated(path: Sequence[Point]) -> tuple[Point, ...]:
    if len(path) < 2:
        raise ValueError("path requires at least two vertices")
    points = tuple((float(x), float(y)) for x, y in path)
    if any(not (isfinite(x) and isfinite(y)) for x, y in points):
        raise ValueError("path coordinates must be finite")
    return points


def transport(path: Sequence[Point], *, dose: float = 1.0,
              curvature: float = 1.0) -> float:
    """Exact segment integral of a constant-curvature planar connection.

    Transport is a scalar line integral, not a physical state update.
    Dose and curvature must be finite. Negative dose represents sign reversal.
    """
    pts = _validated(path)
    if not (isfinite(dose) and isfinite(curvature)):
        raise ValueError("dose and curvature must be finite")
    return 0.5 * dose * curvature * sum(
        x0 * y1 - y0 * x1
        for (x0, y0), (x1, y1) in zip(pts, pts[1:])
    )


def endpoint_linear(path: Sequence[Point], *, vector: Point = (1.0, 0.0),
                    dose: float = 1.0) -> float:
    """Linear endpoint-only comparator; path history has no effect."""
    pts = _validated(path)
    vx, vy = vector
    if not all(map(isfinite, (vx, vy, dose))):
        raise ValueError("linear parameters must be finite")
    return dose * (vx * (pts[-1][0] - pts[0][0])
                   + vy * (pts[-1][1] - pts[0][1]))


def rotate(path: Sequence[Point], cosine: float, sine: float) -> tuple[Point, ...]:
    """Apply a proper orthogonal SO(2) rotation to coordinates."""
    if not all(map(isfinite, (cosine, sine))):
        raise ValueError("rotation coefficients must be finite")
    if abs(cosine * cosine + sine * sine - 1.0) > 1e-10:
        raise ValueError("rotation coefficients must be unit normalized")
    return tuple((cosine*x - sine*y, sine*x + cosine*y)
                 for x, y in _validated(path))


def path_difference(first: Sequence[Point], second: Sequence[Point],
                    *, dose: float = 1.0, curvature: float = 1.0) -> float:
    """Compare paths only when they share their ordered endpoints."""
    a, b = _validated(first), _validated(second)
    if a[0] != b[0] or a[-1] != b[-1]:
        raise ValueError("paths must have identical endpoints")
    return transport(a, dose=dose, curvature=curvature) - transport(
        b, dose=dose, curvature=curvature)
