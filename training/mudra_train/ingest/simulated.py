"""Generate a fingerspelling dataset from the kinematic hand.

This is the pragmatic answer to a real constraint: the datasets that would train a good
model — FSboard above all — need a Kaggle account and a large download, and someone who
has just cloned the repo has neither. Rather than ship nothing and call the app finished,
this simulates the data.

**It is not a substitute for real data, and the model card says so.** What it is worth
turning on is *domain randomisation*: the model never sees the same hand twice, because
every axis along which a real signer differs from the idealised model is varied, and
varied more widely than reality.

Randomised per simulated signer — held constant within them, which is what makes a
signer-independent split meaningful:

* bone lengths and knuckle positions (:func:`~.handmodel.random_geometry`)
* a systematic articulation bias — some people sign crisply, others lazily
* a habitual palm orientation
* a tracking-noise level

Randomised per sample:

* viewpoint in three axes, plus how strong the perspective is
* position and apparent size in frame
* per-joint angle noise
* landmark noise, heavier at the fingertips where MediaPipe is genuinely least certain

The **sim-to-real gap** is the honest weakness. A simulated hand has no skin, no motion
blur, no self-occlusion beyond what geometry implies, and MediaPipe's own failure modes
on a real hand are not modelled — its errors are structured, and Gaussian noise is a poor
imitation of them. Expect a model trained here to be usable and clearly imperfect. The
fix is real data, and `packages/web/recorder.html` is thirty minutes away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .asl_alphabet import (
    ALPHABET,
    MOTION_LETTERS,
    THUMB_CONTACT,
    Orientation,
    Variation,
    orientation_for,
    variation_for,
)
from .asl_numbers import NUMBER_ALIASES, NUMBER_CONTACT, NUMBER_SHAPES
from .handmodel import (
    Contact,
    FingerPose,
    HandGeometry,
    HandPose,
    ThumbPose,
    build_hand,
    project,
    random_geometry,
    resolve_contact,
)
from .recorder import Dataset

#: Share of the dataset that is ``none``.
#:
#: Generous on purpose. The negative class is what stops every transition between letters
#: being force-classified as one of them, and it has to cover a far larger region of pose
#: space than any single letter — so it needs more examples, not fewer.
NONE_FRACTION = 0.22

#: Fraction of samples generated as left hands.
LEFT_HAND_FRACTION = 0.3

#: The orientation band for ``none``: everything. A hand doing nothing in particular is
#: at no particular angle, and the negative class has to cover the angles between every
#: pair of letters as well as the ones inside them.
ANY_ORIENTATION = Orientation(yaw=(-40.0, 40.0), pitch=(-45.0, 45.0), roll=(-180.0, 180.0))

#: Per-landmark tracking noise in image units, before the per-signer multiplier.
#:
#: MediaPipe's jitter on a well-lit, front-facing hand is a few thousandths of the frame.
#: Fingertips are worse than knuckles — they move fastest, occlude most, and sit at the
#: end of the kinematic chain — so the noise is scaled up for them rather than uniform.
BASE_NOISE = 0.0035
FINGERTIPS = (4, 8, 12, 16, 20)
TIP_NOISE_MULTIPLIER = 2.2

CONDITIONS = ("normal", "near", "far", "angled", "dim")


@dataclass(frozen=True, slots=True)
class Signer:
    """One simulated person: their hand, their habits, and their tracking quality."""

    name: str
    geometry: HandGeometry
    #: Multiplies every flexion angle. Below 1 is a lazy signer who under-articulates —
    #: the single most common way a real signer differs from a textbook diagram.
    articulation: float
    #: Habitual palm orientation offset, degrees (yaw, pitch, roll).
    posture: tuple[float, float, float]
    #: Multiplies :data:`BASE_NOISE`. Stands in for lighting, camera and distance.
    noise: float
    #: How much their joint angles wander between repetitions, degrees.
    variability: float


def make_signers(count: int, rng: np.random.Generator) -> list[Signer]:
    """Invent ``count`` distinct signers."""
    return [
        Signer(
            name=f"sim-{index:02d}",
            geometry=random_geometry(rng),
            articulation=float(rng.uniform(0.82, 1.10)),
            posture=(
                float(rng.normal(0.0, 9.0)),
                float(rng.normal(0.0, 7.0)),
                float(rng.normal(0.0, 6.0)),
            ),
            noise=float(rng.uniform(0.6, 2.0)),
            variability=float(rng.uniform(2.5, 7.0)),
        )
        for index in range(count)
    ]


#: Every handshape the model is trained on: the static letters plus the digits that are
#: not already one of them.
#:
#: The digits 0, 2, 6 and 9 are deliberately absent — they *are* O, V, W and F, and giving
#: them their own classes would split probability mass between identical shapes and make
#: the model worse at letters. See `asl_numbers.py`.
SHAPES: dict[str, HandPose] = {**ALPHABET, **NUMBER_SHAPES}

#: Thumb contacts for every shape that has one.
CONTACTS: dict[str, Contact] = {**THUMB_CONTACT, **NUMBER_CONTACT}


#: How many discrete levels a letter's curl range is sampled at.
#:
#: Discrete rather than continuous so the thumb-contact solve can be cached. The solve
#: is the expensive step in generating a dataset — an L-BFGS-B fit per sample put a full
#: run at ~35 minutes, nearly all of it re-deriving answers that differ by less than the
#: per-repetition wobble applied immediately afterwards. Nine levels across the widest
#: declared range is a step of 0.05, well under that wobble, and the signer's own
#: articulation multiplier stays continuous on top — so the distribution the model
#: trains on is still continuous.
CURL_LEVELS = 9


def _styled(letter: str, rng: np.random.Generator) -> tuple[HandPose, int]:
    """One stylistic reading of a letter, before any signer's anatomy touches it.

    Samples the letter's declared :class:`~.asl_alphabet.Variation`: how curled, how
    spread, how far crossed. This is the axis the previous pack had no coverage of at
    all — every sample of a letter used the single declared pose, so the model learned
    the midpoint of a range and rejected the ends of it.

    Returns the pose and the curl level it was drawn at, which keys the contact cache.
    """
    declared = SHAPES[letter]
    variation = variation_for(letter)
    level = int(rng.integers(CURL_LEVELS))
    low, high = variation.curl
    curl = low + (high - low) * level / (CURL_LEVELS - 1)

    fingers = []
    for finger in declared.fingers:
        cross = finger.cross
        if cross and variation.cross is not None:
            # Sign carries which finger crosses which way; only the depth varies.
            cross = float(np.sign(cross) * rng.uniform(*variation.cross))
        fingers.append(
            FingerPose(
                mcp=finger.mcp * curl,
                pip=finger.pip * curl,
                dip=finger.dip * curl,
                splay=finger.splay + float(rng.normal(0.0, variation.splay)),
                cross=cross,
            )
        )
    return HandPose(fingers=tuple(fingers), thumb=declared.thumb), level  # type: ignore[arg-type]


#: Solved thumb poses, keyed by (signer, letter, curl level). See :data:`CURL_LEVELS`.
ContactCache = dict[tuple[str, str, int], ThumbPose]


def _resolve_pose(
    letter: str,
    signer: Signer,
    rng: np.random.Generator,
    cache: ContactCache | None = None,
) -> HandPose:
    """The letter as *this* signer's hand makes it, in one of its stylistic readings.

    Contacts are re-solved against their bone lengths, so the thumb still meets the
    fingertip it is supposed to meet rather than merely reusing an angle that happened to
    work for the average hand. They are also re-solved per *curl level* now that the
    shape varies within a signer: a thumb solved against a loosely curled E does not meet
    anything on a tightly curled one.
    """
    styled, level = _styled(letter, rng)
    fingers = tuple(
        FingerPose(
            mcp=finger.mcp * signer.articulation,
            pip=finger.pip * signer.articulation,
            dip=finger.dip * signer.articulation,
            splay=finger.splay,
            cross=finger.cross,
        )
        for finger in styled.fingers
    )
    pose = HandPose(fingers=fingers, thumb=styled.thumb)  # type: ignore[arg-type]

    if letter not in CONTACTS:
        return pose

    key = (signer.name, letter, level)
    if cache is None or key not in cache:
        solved = resolve_contact(pose, signer.geometry, CONTACTS[letter]).thumb
        if cache is not None:
            cache[key] = solved
    else:
        solved = cache[key]
    return HandPose(fingers=pose.fingers, thumb=solved)  # type: ignore[arg-type]


def _wobble(pose: HandPose, signer: Signer, rng: np.random.Generator) -> HandPose:
    """Perturb a resolved pose the way a repetition of the same letter differs."""
    spread = signer.variability
    fingers = tuple(
        FingerPose(
            mcp=max(0.0, finger.mcp + rng.normal(0.0, spread)),
            pip=max(0.0, finger.pip + rng.normal(0.0, spread)),
            dip=max(0.0, finger.dip + rng.normal(0.0, spread)),
            splay=finger.splay + rng.normal(0.0, spread * 0.5),
            cross=finger.cross,
        )
        for finger in pose.fingers
    )
    thumb = ThumbPose(
        radial=pose.thumb.radial + rng.normal(0.0, spread),
        palmar=pose.thumb.palmar + rng.normal(0.0, spread),
        mcp=max(0.0, pose.thumb.mcp + rng.normal(0.0, spread)),
        ip=max(0.0, pose.thumb.ip + rng.normal(0.0, spread)),
    )
    return HandPose(fingers=fingers, thumb=thumb)  # type: ignore[arg-type]


def _sample_orientation(
    band: Orientation, signer: Signer, rng: np.random.Generator
) -> tuple[float, float, float]:
    """One viewpoint from a letter's declared band, biased by the signer's posture.

    Uniform within the band rather than Gaussian around its centre. That is the whole
    reason the band exists: a Gaussian is thin at the edges, so the orientations a
    less typical signer actually uses are the ones the model sees least of — and an
    orientation the model has not seen does not degrade gracefully, it lands in
    ``none``.

    The signer's habitual posture shifts them within the band and is then clamped back
    to it, so someone who always tilts piles up against an edge (which is realistic)
    without ever crossing it (which for H, P and Q would mean signing U, K or G under
    the wrong label).
    """
    return (
        _within(band.yaw, rng.uniform(*band.yaw) + signer.posture[0]),
        _within(band.pitch, rng.uniform(*band.pitch) + signer.posture[1]),
        _within(band.roll, rng.uniform(*band.roll) + signer.posture[2]),
    )


def _within(band: tuple[float, float], value: float) -> float:
    return float(min(max(value, band[0]), band[1]))


def _off_centre(band: tuple[float, float], value: float) -> float:
    """How far into a band's outer reaches a value sits: 0 at the centre, 1 at an edge."""
    half = (band[1] - band[0]) / 2.0
    if half <= 0.0:
        return 0.0
    return abs(value - (band[0] + half)) / half


def _render(
    pose: HandPose,
    signer: Signer,
    band: Orientation,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str, str]:
    """Pose → landmarks as a camera would see them. Returns points, hand, condition."""
    yaw, pitch, roll = _sample_orientation(band, signer, rng)

    # Perspective strength and apparent size, which together stand in for how close the
    # signer is sitting to the camera.
    distance = float(rng.uniform(4.0, 10.0))
    scale = float(rng.uniform(0.15, 0.32))
    condition = "near" if scale > 0.27 else "far" if scale < 0.19 else "normal"
    # "Angled" means unusual *for this letter*, not far from upright. G is signed on its
    # side by definition, so an absolute threshold would file every G under `angled` and
    # turn the worst-slice metric into a report on four letters.
    if _off_centre(band.yaw, yaw) > 0.9 or _off_centre(band.pitch, pitch) > 0.9:
        condition = "angled"

    hand = "left" if rng.random() < LEFT_HAND_FRACTION else "right"

    points = project(
        build_hand(pose, signer.geometry),
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        distance=distance,
        scale=scale,
        centre=(float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.3, 0.7))),
        hand=hand,
    )

    noise = np.full(len(points), BASE_NOISE * signer.noise)
    noise[list(FINGERTIPS)] *= TIP_NOISE_MULTIPLIER
    points[:, :2] += rng.normal(0.0, noise[:, None], size=(len(points), 2))
    points[:, 2] += rng.normal(0.0, noise * 1.5)

    return points, hand, condition


def _random_pose(rng: np.random.Generator) -> HandPose:
    """A hand doing nothing in particular — the ``none`` class.

    Uniform over joint space rather than a blend of letters. Blends alone would teach the
    model that ``none`` means *between two letters*, when most of what a camera sees is a
    hand resting, moving, or half-way to somewhere unrelated.
    """
    fingers = tuple(
        FingerPose(
            mcp=float(rng.uniform(0.0, 95.0)),
            pip=float(rng.uniform(0.0, 105.0)),
            dip=float(rng.uniform(0.0, 70.0)),
            splay=float(rng.uniform(-16.0, 16.0)),
        )
        for _ in range(4)
    )
    thumb = ThumbPose(
        radial=float(rng.uniform(-20.0, 78.0)),
        palmar=float(rng.uniform(-12.0, 65.0)),
        mcp=float(rng.uniform(0.0, 52.0)),
        ip=float(rng.uniform(0.0, 75.0)),
    )
    return HandPose(fingers=fingers, thumb=thumb)  # type: ignore[arg-type]


def _blend(first: HandPose, second: HandPose, weight: float) -> HandPose:
    """Half-way between two letters — geometrically, what a transition looks like."""

    def mix(a: float, b: float) -> float:
        return a * (1.0 - weight) + b * weight

    fingers = tuple(
        FingerPose(
            mcp=mix(x.mcp, y.mcp),
            pip=mix(x.pip, y.pip),
            dip=mix(x.dip, y.dip),
            splay=mix(x.splay, y.splay),
            cross=mix(x.cross, y.cross),
        )
        for x, y in zip(first.fingers, second.fingers, strict=True)
    )
    thumb = ThumbPose(
        radial=mix(first.thumb.radial, second.thumb.radial),
        palmar=mix(first.thumb.palmar, second.thumb.palmar),
        mcp=mix(first.thumb.mcp, second.thumb.mcp),
        ip=mix(first.thumb.ip, second.thumb.ip),
    )
    return HandPose(fingers=fingers, thumb=thumb)  # type: ignore[arg-type]


def generate(
    samples_per_class: int = 900,
    signers: int = 14,
    seed: int = 20260820,
) -> Dataset:
    """Simulate a fingerspelling dataset.

    Args:
        samples_per_class: Samples per letter, split across signers.
        signers: How many distinct simulated people. More is strictly better for the
            signer-independent split — the point of that split is that the test signers
            were never seen, and with too few there is not enough anatomy to hold out.
    """
    if signers < 4:
        raise ValueError("At least 4 signers are needed for a meaningful held-out split")

    rng = np.random.default_rng(seed)
    people = make_signers(signers, rng)
    letters = list(SHAPES)
    per_signer = max(1, samples_per_class // signers)

    landmarks: list[np.ndarray] = []
    labels: list[str] = []
    signer_ids: list[str] = []
    hands: list[str] = []
    conditions: list[str] = []

    cache: ContactCache = {}

    for signer in people:
        for letter in letters:
            band = orientation_for(letter)
            for _ in range(per_signer):
                pose = _wobble(_resolve_pose(letter, signer, rng, cache), signer, rng)
                points, hand, condition = _render(pose, signer, band, rng)
                landmarks.append(points)
                labels.append(letter)
                signer_ids.append(signer.name)
                hands.append(hand)
                conditions.append(condition)

        none_count = int(len(letters) * per_signer * NONE_FRACTION)
        for index in range(none_count):
            if index % 2 == 0:
                first = _resolve_pose(letters[int(rng.integers(len(letters)))], signer, rng, cache)
                second = _resolve_pose(letters[int(rng.integers(len(letters)))], signer, rng, cache)
                pose = _blend(first, second, float(rng.uniform(0.3, 0.7)))
                condition_hint = "transition"
            else:
                pose = _random_pose(rng)
                condition_hint = "idle"

            points, hand, _ = _render(pose, signer, ANY_ORIENTATION, rng)
            landmarks.append(points)
            labels.append("none")
            signer_ids.append(signer.name)
            hands.append(hand)
            conditions.append(condition_hint)

    return Dataset(
        landmarks=np.stack(landmarks),
        labels=np.array(labels),
        signers=np.array(signer_ids),
        hands=np.array(hands),
        conditions=np.array(conditions),
    )


def describe() -> str:
    """One-paragraph provenance line for the model card."""
    excluded = ", ".join(sorted(MOTION_LETTERS))
    aliases = ", ".join(f"{digit}={label}" for digit, label in sorted(NUMBER_ALIASES.items()))
    return (
        f"Simulated from a kinematic hand model: {len(SHAPES)} handshapes covering "
        f"{len(ALPHABET)} static letters and the digits 0-9, of which {aliases} share a "
        f"handshape with a letter and are therefore not separate classes. "
        f"{excluded} are excluded as motion letters. Joint angles, bone lengths, "
        "viewpoint, perspective and tracking noise are randomised per simulated signer "
        "and per sample. No real hands were recorded, and none were seen during "
        "training — see the limitations section."
    )
