"""Dependency-free SVG plot of observed phase trajectories."""
from __future__ import annotations

from html import escape
from typing import Any

from .protocol import CONDITIONS, PHASES

COLORS = {"ordered": "#147d89", "permuted": "#d18c37",
          "tail_matched": "#38844b", "neutral": "#8060b0"}


def render_svg(report: dict[str, Any]) -> str:
    width, height = 900, 440
    left, top, right, bottom = 82, 90, 40, 105
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = {phase: left + plot_w * idx / (len(PHASES) - 1)
          for idx, phase in enumerate(PHASES)}
    values = {(g["condition"], g["phase"]): g["mean_persona_index"]
              for g in report["groups"]}

    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title desc">',
        '<title id="title">Persona index across transcript checkpoints</title>',
    ]
    # The desc is intentionally short and always distinguishes the graph from an activation plot.
    fragments.append('<desc id="desc">Behavioral lexical index, not internal model activations.</desc>')
    fragments.extend([
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="82" y="38" font-family="sans-serif" font-size="22" font-weight="bold" '
        'fill="#222">Persona carryover after scripted history</text>',
        '<text x="82" y="59" font-family="sans-serif" font-size="12" '
        'fill="#666">Behavioral lexical proxy; score between 0 and 1</text>',
    ])
    if report.get("synthetic"):
        fragments.append('<text x="818" y="38" text-anchor="end" font-size="12" '
                         'font-family="sans-serif" font-weight="bold" fill="#ae3e33">'
                         'SYNTHETIC DEMO</text>')

    for v in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - v)
        fragments.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
                         f'y2="{y:.1f}" stroke="#e6e6e6"/>')
        fragments.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" '
                         f'font-family="sans-serif" font-size="12" fill="#555">{v:.2f}</text>')
    fragments.append(f'<line x1="{left}" y1="{top}" x2="{left}" '
                     f'y2="{top+plot_h}" stroke="#888"/>')
    for phase, x in xs.items():
        label = phase.replace("_", " ")
        fragments.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" '
                         f'font-size="11" font-family="sans-serif" fill="#444">{escape(label)}</text>')
    for condition in CONDITIONS:
        points = [(xs[phase], top + plot_h * (1 - values[condition, phase]))
                  for phase in PHASES if (condition, phase) in values]
        if not points:
            continue
        color = COLORS[condition]
        path = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}"
                        for i, (x, y) in enumerate(points))
        fragments.append(f'<path d="{path}" fill="none" stroke="{color}" '
                         'stroke-width="3" stroke-linejoin="round"/>')
        for x, y in points:
            fragments.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" '
                             f'fill="{color}" stroke="white" stroke-width="1.5"/>')
    for i, condition in enumerate(CONDITIONS):
        x = left + i * 165
        fragments.append(f'<line x1="{x}" y1="{height-42}" x2="{x+25}" '
                         f'y2="{height-42}" stroke="{COLORS[condition]}" stroke-width="3"/>')
        fragments.append(f'<text x="{x+32}" y="{height-37}" font-size="12" '
                         f'font-family="sans-serif" fill="#333">{condition}</text>')
    fragments.append('</svg>')
    return "\n".join(fragments) + "\n"
