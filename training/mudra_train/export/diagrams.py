"""Generate handshape diagrams for the dictionary.

The v1 dictionary referenced ~150 images that never existed, so every card fell
through to a 🤟 placeholder — the whole feature was non-functional (docs/AUDIT.md, A1).

The obvious fix is to source photographs, but every candidate set is either
non-redistributable, non-commercial, or requires an account. Rather than take a licence
risk on ~150 files, these diagrams are **generated from the same kinematic hand model
the classifier is trained on**. That makes them:

* **Ours.** MIT, like the rest of the repository. No attribution burden, no takedown risk.
* **Consistent.** One visual language across every sign, at any size, as vector art.
* **Honest.** Schematic line drawings clearly read as diagrams rather than pretending
  to be photographs of a signer.
* **True.** They show the letter the model will actually recognise.

## Why "the same model" is load-bearing

These diagrams were previously built from ``ingest/synthetic.py`` — the crude fixture
generator that ``ingest/simulated.py`` describes, correctly, as useless as training
data. It has no thumb opposition, no finger splay, no crossing and no 3-D projection:
a letter is four curl values, one "thumb tuck" scalar and a flat 2-D rotation.

The classifier, meanwhile, learns from ``ingest/handmodel.py`` driven by
``ingest/asl_alphabet.py``. Two independent definitions of the alphabet, and they
disagreed:

* **G and H** pointed in **opposite directions** — 169° and 160° apart. A signer who
  copied the diagram produced a hand the classifier had never seen, and it does not
  degrade into a near miss: measured on the shipped pack, G held as the diagram
  illustrated it was accepted 0% of the time and read as ``none`` 100% of the time.
* **R** was drawn with two straight parallel fingers — the fixture model cannot express
  crossing at all — which is a picture of U.
* **K** was drawn with the thumb hanging down beside the fist rather than pushed up
  between the fingers, which is the whole letter.
* **E and S** differed only by a curl of 0.8 against 1.0, so they drew as near-identical
  fists.

So the dictionary was teaching four letters wrongly and two indistinguishably, and the
recogniser was being blamed for it. Generating both from one definition is what stops
that recurring; ``tests/test_diagrams.py`` asserts it stays that way.

The remaining honest limitation is unchanged: these are schematics, and video of a
fluent signer teaches better. The dictionary links to PopSign clips for word-level
signs; these cover the alphabet and digits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..ingest.asl_alphabet import (
    ALPHABET,
    MOTION_LETTERS,
    THUMB_CONTACT,
    orientation_for,
)
from ..ingest.asl_numbers import NUMBER_ALIASES, NUMBER_CONTACT, NUMBER_SHAPES
from ..ingest.handmodel import HandGeometry, HandPose, build_hand, project, resolve_contact

#: The four fingers, as chains of landmark indices. Drawn as chains rather than as a
#: flat list of bones so each finger can be depth-sorted and shaded as a unit.
_FINGERS: tuple[tuple[int, ...], ...] = (
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
#: The thumb, from its base.
_THUMB_CHAIN: tuple[int, ...] = (1, 2, 3, 4)
#: The palm, as an outline: wrist round the knuckle line and back. Drawn as one filled
#: shape rather than as four separate bones radiating from the wrist.
#:
#: This is the difference between a diagram of a fist and a diagram of a splayed hand.
#: A folded finger points at the camera, so it projects to almost nothing, and the four
#: full-length metacarpals then dominate the picture — A, S, M, N and T all drew as an
#: open fan of lines with a few stubs on top, which is to say they all drew the same.
#: An outlined palm reads as a solid mass, and whatever sits in front of it reads as a
#: finger.
_PALM: tuple[int, ...] = (0, 5, 9, 13, 17)

#: Fingertips, drawn larger so the extended fingers read at a glance.
_TIPS = frozenset({4, 8, 12, 16, 20})
#: The thumb tip, drawn larger still. In half the alphabet the thumb *is* the letter —
#: A against S, M against N against T, K against V — and it is the landmark a reader is
#: least likely to find on their own in a line drawing.
_THUMB_TIP = 4

# Palette lifted from the app so diagrams sit naturally on the cards.
_BACKGROUND = "#111127"
_BONE = "#8b5cf6"
_PALM_FILL = "#2e2461"
_PALM_EDGE = "#4c3a96"
_JOINT = "#00d4c8"
_THUMB = "#f472b6"
_MOTION = "#fbbf24"
_LABEL = "#64748b"

_SIZE = 220
_MARGIN = 26

#: The camera the diagrams are drawn through. Fixed, unlike training, because a
#: diagram is one picture rather than a distribution — but the *same* projection, so a
#: reader who copies the picture produces geometry the classifier has seen.
_DIAGRAM_DISTANCE = 7.0
_DIAGRAM_SCALE = 0.22

#: Every shape the dictionary illustrates, and where its thumb goes.
_SHAPES: dict[str, HandPose] = {**ALPHABET, **NUMBER_SHAPES}
_CONTACTS = {**THUMB_CONTACT, **NUMBER_CONTACT}

#: The motion letters: which handshape each starts from, and the path it traces.
#:
#: The path is in canvas fractions relative to the tip that draws it, and is deliberately
#: schematic — it says *hooks down and round* and *cuts across three times*, which is
#: what a still cannot otherwise say. Previously these two cards showed the starting
#: handshape relabelled, so J's card was a picture of I and Z's was a picture of D, with
#: nothing to indicate that either letter moves at all.
_MOTION_PATHS: dict[str, tuple[str, int, tuple[tuple[float, float], ...]]] = {
    # J: the pinky drops and hooks back toward the thumb side.
    "J": ("I", 20, ((0.0, 0.0), (0.05, 0.34), (-0.10, 0.52), (-0.30, 0.44))),
    # Z: the index cuts across, back down the diagonal, and across again.
    "Z": ("D", 8, ((0.0, 0.0), (0.42, 0.0), (0.0, 0.40), (0.42, 0.40))),
}


#: The viewpoint diagrams are drawn from, where each letter's band allows it: a
#: three-quarter view from slightly below, which is how hands are drawn in every
#: alphabet chart and is not an accident of taste.
#:
#: * **Yaw** turns the radial side toward the reader. Half the alphabet is separated by
#:   where the thumb sits — A against S, M against N against T, K against V — and
#:   palm-on hides the thumb behind the fingers. It is also what makes C read as a
#:   curve rather than as a spread hand.
#: * **Pitch** tips the hand so folded fingers project *down over* the palm instead of
#:   up past the knuckles. Face-on, a fist's proximal bones point straight at the
#:   camera and draw as four short upright bars, so a closed hand reads as an open one
#:   and five letters read as each other.
#:
#: Both are clamped into the letter's own :class:`~.asl_alphabet.Orientation` band, so
#: the picture is always a hand the classifier was trained on. That is the invariant
#: that failed before — the diagrams were generated from a different hand model
#: entirely, and G and H ended up pointing the opposite way — and
#: ``tests/test_diagrams.py`` now pins it.
DIAGRAM_YAW = -20.0
DIAGRAM_PITCH = 10.0


def diagram_orientation(name: str) -> tuple[float, float, float]:
    """The single viewpoint a diagram is drawn from, inside the letter's band.

    Yaw and pitch take :data:`DIAGRAM_YAW` and :data:`DIAGRAM_PITCH` where the band
    allows and the nearest edge otherwise. Roll takes the band's centre: roll is the
    axis that carries meaning — it is what makes an H an H and not a U — so a diagram
    shows it as typically signed rather than at whatever angle reads most cleanly.
    """
    band = orientation_for(name)
    return (
        min(max(DIAGRAM_YAW, band.yaw[0]), band.yaw[1]),
        min(max(DIAGRAM_PITCH, band.pitch[0]), band.pitch[1]),
        _centre(band.roll),
    )


def landmarks_for(name: str) -> np.ndarray:
    """The 21 projected landmarks a diagram draws, in raw camera coordinates.

    Exported so the parity test can compare a diagram against the training pose without
    reaching into the renderer.
    """
    pose = _SHAPES[name]
    geometry = HandGeometry()
    if name in _CONTACTS:
        pose = resolve_contact(pose, geometry, _CONTACTS[name])

    yaw, pitch, roll = diagram_orientation(name)
    return project(
        build_hand(pose, geometry),
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        distance=_DIAGRAM_DISTANCE,
        scale=_DIAGRAM_SCALE,
        hand="right",
    )


def _centre(band: tuple[float, float]) -> float:
    return (band[0] + band[1]) / 2.0


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


def render_svg(name: str, *, motion: str | None = None) -> str:
    """Render one handshape as a self-contained SVG.

    ``motion`` names a letter in :data:`_MOTION_PATHS`, which draws its stroke over the
    starting handshape.
    """
    points = landmarks_for(name)
    xy = _to_canvas(points)
    depth = _depth(points)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SIZE} {_SIZE}" '
        f'width="{_SIZE}" height="{_SIZE}" role="img" '
        f'aria-label="{_description(name, motion)}">',
        f"<title>{_description(name, motion)}</title>",
        f'<rect width="{_SIZE}" height="{_SIZE}" rx="14" fill="{_BACKGROUND}"/>',
    ]

    if motion is not None:
        parts += _motion_stroke(motion, xy)

    # The palm first and flat, so every finger reads as being in front of it.
    outline = " ".join(f"{xy[i][0]:.1f},{xy[i][1]:.1f}" for i in _PALM)
    parts.append(
        f'<polygon points="{outline}" fill="{_PALM_FILL}" stroke="{_PALM_EDGE}" '
        'stroke-width="6" stroke-linejoin="round"/>'
    )

    # Then the digits, furthest first. A finger curled toward the camera is drawn over
    # the palm and thicker than one folded away from it, which is the only cue a line
    # drawing has for a shape that is mostly pointing at the reader.
    chains = [(_THUMB_CHAIN, True), *((finger, False) for finger in _FINGERS)]
    for chain, is_thumb in sorted(chains, key=lambda item: depth[list(item[0])].mean()):
        parts.append(_chain_svg(chain, xy, depth, is_thumb))

    parts.append(f'<g fill="{_JOINT}">')
    for index, (x, y) in enumerate(xy):
        if index == _THUMB_TIP:
            continue
        radius = 5.5 if index in _TIPS else 3.0
        if index == 0:
            radius = 6.5  # wrist anchors the drawing
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}"/>')
    parts.append("</g>")

    thumb_x, thumb_y = xy[_THUMB_TIP]
    parts.append(
        f'<circle cx="{thumb_x:.1f}" cy="{thumb_y:.1f}" r="7.5" fill="{_THUMB}" '
        f'stroke="{_BACKGROUND}" stroke-width="2"/>'
    )

    label = motion or name
    parts.append(
        f'<text x="{_SIZE - 14}" y="{_SIZE - 12}" text-anchor="end" fill="{_LABEL}" '
        f'font-family="system-ui, sans-serif" font-size="20" font-weight="600">{label}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _depth(points: np.ndarray) -> np.ndarray:
    """Per-landmark depth, 0 furthest from the camera and 1 nearest."""
    z = points[:, 2].astype(np.float64)
    span = float(z.max() - z.min())
    return (z - z.min()) / span if span > 1e-9 else np.full(len(z), 0.5)


def _chain_svg(chain: tuple[int, ...], xy: np.ndarray, depth: np.ndarray, is_thumb: bool) -> str:
    """One digit, drawn bone by bone with width and opacity following depth."""
    colour = _THUMB if is_thumb else _BONE
    bones = []
    for start, end in zip(chain, chain[1:], strict=False):
        near = (depth[start] + depth[end]) / 2.0
        width = 4.4 + 3.6 * near
        opacity = 0.55 + 0.45 * near
        bones.append(
            f'<line x1="{xy[start][0]:.1f}" y1="{xy[start][1]:.1f}" '
            f'x2="{xy[end][0]:.1f}" y2="{xy[end][1]:.1f}" stroke="{colour}" '
            f'stroke-width="{width:.1f}" stroke-linecap="round" opacity="{opacity:.2f}"/>'
        )
    return "".join(bones)


def _description(name: str, motion: str | None) -> str:
    if motion is None:
        return f"ASL handshape for {name}"
    return f"ASL handshape for {motion}: the {name} hand, traced through its motion"


def _motion_stroke(letter: str, xy: np.ndarray) -> list[str]:
    """The path a motion letter's fingertip traces, drawn behind the hand."""
    _, tip, path = _MOTION_PATHS[letter]
    origin = xy[tip]
    span = float(_SIZE - 2 * _MARGIN)
    points = [(origin[0] + dx * span, origin[1] + dy * span) for dx, dy in path]

    steps = " ".join(
        ("M" if index == 0 else "L") + f"{x:.1f},{y:.1f}" for index, (x, y) in enumerate(points)
    )
    end_x, end_y = points[-1]
    before_x, before_y = points[-2]
    angle = float(np.degrees(np.arctan2(end_y - before_y, end_x - before_x)))

    return [
        f'<path d="{steps}" fill="none" stroke="{_MOTION}" stroke-width="5" '
        'stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="9 7" '
        'opacity="0.85"/>',
        f'<polygon points="0,-6 13,0 0,6" fill="{_MOTION}" opacity="0.85" '
        f'transform="translate({end_x:.1f},{end_y:.1f}) rotate({angle:.1f})"/>',
    ]


def generate_all(output_root: Path) -> int:
    """Write every alphabet and digit diagram. Returns the file count."""
    written = 0

    alphabets = output_root / "alphabets"
    alphabets.mkdir(parents=True, exist_ok=True)
    for letter in ALPHABET:
        (alphabets / f"{letter.lower()}.svg").write_text(render_svg(letter), encoding="utf-8")
        written += 1

    # J and Z have no single static shape. Drawing the starting handshape *and the path
    # the hand travels* is the most a still can say, and it is a great deal more than
    # relabelling the starting handshape and hoping.
    for motion_letter in MOTION_LETTERS:
        base, _, _ = _MOTION_PATHS[motion_letter]
        (alphabets / f"{motion_letter.lower()}.svg").write_text(
            render_svg(base, motion=motion_letter), encoding="utf-8"
        )
        written += 1

    numbers = output_root / "numbers"
    numbers.mkdir(parents=True, exist_ok=True)
    for digit, label in NUMBER_ALIASES.items():
        (numbers / f"{digit}.svg").write_text(render_svg(label), encoding="utf-8")
        written += 1
    for digit in NUMBER_SHAPES:
        (numbers / f"{digit}.svg").write_text(render_svg(digit), encoding="utf-8")
        written += 1
    # Ten is the only digit with no handshape of its own in the trained vocabulary: it
    # is the A hand shaken, and the A hand is what a reader needs to see.
    (numbers / "10.svg").write_text(render_svg("A"), encoding="utf-8")
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
