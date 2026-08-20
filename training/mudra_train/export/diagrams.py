"""Generate handshape diagrams for the dictionary.

The v1 dictionary referenced ~150 images that never existed, so every card fell
through to a 🤟 placeholder — the whole feature was non-functional (docs/AUDIT.md, A1).

The obvious fix is to source photographs, but every candidate set is either
non-redistributable, non-commercial, or requires an account. Rather than take a licence
risk on ~150 files, these diagrams are **generated from the same kinematic hand model
the pipeline already uses**. That makes them:

* **Ours.** MIT, like the rest of the repository. No attribution burden, no takedown risk.
* **Consistent.** One visual language across every sign, at any size, as vector art.
* **Honest.** Schematic line drawings clearly read as diagrams rather than pretending
  to be photographs of a signer.

They are a genuine improvement on a broken placeholder, and a weaker teaching aid than
video of a fluent signer. The dictionary links to PopSign clips for word-level signs in
Phase 4; these cover the alphabet and digits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..ingest.synthetic import _HANDSHAPES, _build_hand

#: ASL digits. Same ``(curls, thumb_tuck, rotation)`` format as the letters.
#: Curl order is index, middle, ring, pinky; 0.0 extended, 1.0 curled.
_DIGITS: dict[str, tuple[tuple[float, float, float, float], float, float]] = {
    "0": ((0.75, 0.75, 0.75, 0.75), 0.7, 0.0),
    "1": ((0.0, 1.0, 1.0, 1.0), 0.8, 0.0),
    "2": ((0.0, 0.0, 1.0, 1.0), 0.8, 0.0),
    "3": ((0.0, 0.0, 1.0, 1.0), 0.0, 0.0),
    "4": ((0.0, 0.0, 0.0, 0.0), 0.9, 0.0),
    "5": ((0.0, 0.0, 0.0, 0.0), 0.0, 0.0),
    "6": ((0.0, 0.0, 0.0, 0.9), 0.6, 0.0),
    "7": ((0.0, 0.0, 0.9, 0.0), 0.6, 0.0),
    "8": ((0.0, 0.9, 0.0, 0.0), 0.6, 0.0),
    "9": ((0.9, 0.0, 0.0, 0.0), 0.9, 0.0),
    "10": ((1.0, 1.0, 1.0, 1.0), 0.0, 0.0),
}

#: Bones, as landmark index pairs. Matches MediaPipe's hand topology.
_BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # knuckle line
)

#: Fingertips, drawn larger so the extended fingers read at a glance.
_TIPS = frozenset({4, 8, 12, 16, 20})

# Palette lifted from the app so diagrams sit naturally on the cards.
_BACKGROUND = "#111127"
_BONE = "#8b5cf6"
_JOINT = "#00d4c8"
_LABEL = "#64748b"

_SIZE = 220
_MARGIN = 26


def _to_canvas(points: np.ndarray) -> np.ndarray:
    """Fit a hand into the canvas, preserving aspect ratio."""
    xy = points[:, :2].astype(np.float64).copy()
    xy -= xy.min(axis=0)

    extent = xy.max(axis=0)
    scale = (_SIZE - 2 * _MARGIN) / max(float(extent.max()), 1e-6)
    xy *= scale

    # Centre whatever slack remains on each axis.
    xy += (_SIZE - xy.max(axis=0)) / 2
    return xy


def render_svg(name: str, curls, thumb: float, rotation: float) -> str:
    """Render one handshape as a self-contained SVG."""
    rng = np.random.default_rng(0)  # deterministic: no jitter in diagrams
    points = _build_hand(curls, thumb, rotation, rng, jitter=0.0)
    xy = _to_canvas(points)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SIZE} {_SIZE}" '
        f'width="{_SIZE}" height="{_SIZE}" role="img" '
        f'aria-label="ASL handshape for {name}">',
        f'<title>ASL handshape for {name}</title>',
        f'<rect width="{_SIZE}" height="{_SIZE}" rx="14" fill="{_BACKGROUND}"/>',
        f'<g stroke="{_BONE}" stroke-width="7" stroke-linecap="round" opacity="0.95">',
    ]

    for start, end in _BONES:
        x1, y1 = xy[start]
        x2, y2 = xy[end]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    parts.append("</g>")

    parts.append(f'<g fill="{_JOINT}">')
    for index, (x, y) in enumerate(xy):
        radius = 6.0 if index in _TIPS else 3.4
        if index == 0:
            radius = 7.0  # wrist anchors the drawing
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}"/>')
    parts.append("</g>")

    parts.append(
        f'<text x="{_SIZE - 14}" y="{_SIZE - 12}" text-anchor="end" fill="{_LABEL}" '
        f'font-family="system-ui, sans-serif" font-size="20" font-weight="600">{name}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def generate_all(output_root: Path) -> int:
    """Write every alphabet and digit diagram. Returns the file count."""
    written = 0

    alphabets = output_root / "alphabets"
    alphabets.mkdir(parents=True, exist_ok=True)
    for letter, (curls, thumb, rotation) in _HANDSHAPES.items():
        (alphabets / f"{letter.lower()}.svg").write_text(
            render_svg(letter, curls, thumb, rotation), encoding="utf-8"
        )
        written += 1

    # J and Z are motion letters with no single static shape. Drawing their starting
    # handshape and saying so beats leaving two gaps in the alphabet.
    for motion_letter, base in (("J", "I"), ("Z", "D")):
        curls, thumb, rotation = _HANDSHAPES[base]
        svg = render_svg(f"{motion_letter} (start)", curls, thumb, rotation)
        (alphabets / f"{motion_letter.lower()}.svg").write_text(svg, encoding="utf-8")
        written += 1

    numbers = output_root / "numbers"
    numbers.mkdir(parents=True, exist_ok=True)
    for digit, (curls, thumb, rotation) in _DIGITS.items():
        (numbers / f"{digit}.svg").write_text(
            render_svg(digit, curls, thumb, rotation), encoding="utf-8"
        )
        written += 1

    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Directory to write into")
    args = parser.parse_args()

    count = generate_all(args.out)
    print(f"Wrote {count} handshape diagrams to {args.out}")


if __name__ == "__main__":
    main()
