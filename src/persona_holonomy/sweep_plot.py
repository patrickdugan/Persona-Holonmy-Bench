"""Dependency-free SVG overview of held-out directional specificity by layer."""
from __future__ import annotations

import math
from html import escape
from typing import Any


def render_sweep_svg(report: dict[str, Any], *, source: str = "ordered",
                     phase: str = "exit", alpha: float = 1.) -> str:
    """Plot descriptively, without selecting a best layer or declaring significance."""
    samples = [s for s in report["summaries"] if
               s["direction_source"] == source and s["phase"] == phase
               and abs(s["alpha"] - alpha) < 1e-9]
    if not samples:
        raise ValueError("No sweep rows match requested source, phase, and strength")
    by_key = {(s["layer"], s["mode"]): s for s in samples}
    layers = sorted({s["layer"] for s in samples})
    vals = [0.] + [float(s["mean_paired_specificity"]) for s in samples]
    for s in samples:
        ci = s["seed_bootstrap_95pct_ci"]
        if ci is not None:
            vals.extend(ci)
    extent = max(abs(min(vals)), abs(max(vals)), 1e-4) * 1.20
    width, height = max(600, 125 * len(layers) + 155), 385
    left, right, top, bottom = 90, 40, 85, 85
    graph_w, graph_h = width - left - right, height - top - bottom

    def x(index: int) -> float:
        return left + (index + .5) * graph_w / len(layers)

    def y(value: float) -> float:
        return top + (extent - value) * graph_h / (2 * extent)

    def f(value: float) -> str:
        return f"{value:.2f}"

    title = escape(str(report.get("model", "model")), quote=True)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Causal sweep by layer">',
             f'<rect width="{width}" height="{height}" fill="#fff"/>',
             f'<text x="{left}" y="29" font-size="17" font-family="sans-serif" '
             f'font-weight="600">PHB: held-out direction specificity</text>',
             f'<text x="{left}" y="52" font-size="11" fill="#555" '
             f'font-family="sans-serif">{title} | source={escape(source)} | '
             f'phase={escape(phase)} | α={alpha:g}</text>']
    for fraction in [-1, -.5, 0, .5, 1]:
        value = fraction * extent
        cy = y(value)
        stroke = '#666' if fraction == 0 else '#e3e3e3'
        parts.append(f'<line x1="{left}" y1="{f(cy)}" x2="{width-right}" y2="{f(cy)}" '
                     f'stroke="{stroke}" stroke-width="{1.3 if fraction==0 else .7}"/>')
        parts.append(f'<text x="{left-12}" y="{f(cy+4)}" font-size="10" '
                     f'text-anchor="end" font-family="monospace" fill="#555">{value:+.3f}</text>')
    keys = [("inject", "#205f91", -8), ("erase", "#b25a29", 8)]
    for layer_index, layer in enumerate(layers):
        cx = x(layer_index)
        parts.append(f'<text x="{f(cx)}" y="{height-48}" text-anchor="middle" '
                     f'font-family="monospace" font-size="12">{layer}</text>')
        for mode, color, dx in keys:
            s = by_key.get((layer, mode))
            if s is None:
                continue
            px = cx + dx
            py = y(s["mean_paired_specificity"])
            if s["seed_bootstrap_95pct_ci"] is not None:
                lo, hi = s["seed_bootstrap_95pct_ci"]
                top_ci, bottom_ci = y(hi), y(lo)
                parts.extend([
                    f'<line x1="{f(px)}" y1="{f(top_ci)}" x2="{f(px)}" '
                    f'y2="{f(bottom_ci)}" stroke="{color}" stroke-width="1.8"/>',
                    f'<line x1="{f(px-4)}" y1="{f(top_ci)}" x2="{f(px+4)}" '
                    f'y2="{f(top_ci)}" stroke="{color}"/>',
                    f'<line x1="{f(px-4)}" y1="{f(bottom_ci)}" x2="{f(px+4)}" '
                    f'y2="{f(bottom_ci)}" stroke="{color}"/>',
                ])
            parts.append(f'<circle cx="{f(px)}" cy="{f(py)}" r="4.5" fill="{color}"/>')
    parts.append(f'<text x="{width/2:.0f}" y="{height-21}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="12">Decoder block (zero-indexed)</text>')
    parts.append(f'<text x="18" y="{height/2:.0f}" font-family="sans-serif" font-size="11" '
                 f'transform="rotate(-90 18 {height/2:.0f})" text-anchor="middle">'
                 f'Oriented effect minus orthogonal null (mean log-prob)</text>')
    for i, (mode, color, _) in enumerate(keys):
        legend_x = width - 200 + i * 87
        parts.append(f'<circle cx="{legend_x}" cy="35" r="4" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+9}" y="39" font-family="sans-serif" '
                     f'font-size="11">{mode}</text>')
    parts.append(f'<text x="{width/2:.0f}" y="{height-67}" text-anchor="middle" '
                 f'font-size="10" fill="#666" font-family="sans-serif">'
                 f'Whiskers are seed-bootstrap 95% intervals when ≥5 held-out seeds exist.</text>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'
