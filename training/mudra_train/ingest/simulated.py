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

from .asl_alphabet import ALPHABET, MOTION_LETTERS, THUMB_CONTACT, orientation_for
from .handmodel import (
    FingerPose,
    HandGeometry,
    HandPose,
    ThumbPose,
    build_hand,
    project,
    random_geometry,
    solve_thumb_contact,
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


def _resolve_pose(letter: str, signer: Signer) -> HandPose:
    """The letter as *this* signer's hand makes it.

    Contacts are re-solved against their bone lengths, so the thumb still meets the
    fingertip it is supposed to meet rather than merely reusing an angle that happened to
    work for the average hand.
    """
    declared = ALPHABET[letter]
    fingers = tuple(
        FingerPose(
            mcp=finger.mcp * signer.articulation,
            pip=finger.pip * signer.articulation,
            dip=finger.dip * signer.articulation,
            splay=finger.splay,
            cross=finger.cross,
        )
        for finger in declared.fingers
    )
    pose = HandPose(fingers=fingers, thumb=declared.thumb)  # type: ignore[arg-type]

    if letter in THUMB_CONTACT:
        landmark, distance = THUMB_CONTACT[letter]
        pose = HandPose(
            fingers=pose.fingers,
            thumb=solve_thumb_contact(pose, signer.geometry, landmark, distance),
        )
    return pose


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


def _render(
    pose: HandPose,
    signer: Signer,
    orientation: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, str, str]:
    """Pose → landmarks as a camera would see them. Returns points, hand, condition."""
    yaw, pitch, roll = orientation
    yaw += signer.posture[0] + rng.normal(0.0, 14.0)
    pitch += signer.posture[1] + rng.normal(0.0, 12.0)
    roll += signer.posture[2] + rng.normal(0.0, 10.0)

    # Perspective strength and apparent size, which together stand in for how close the
    # signer is sitting to the camera.
    distance = float(rng.uniform(4.0, 10.0))
    scale = float(rng.uniform(0.15, 0.32))
    condition = "near" if scale > 0.27 else "far" if scale < 0.19 else "normal"
    if abs(yaw) > 28 or abs(pitch) > 28:
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
    letters = list(ALPHABET)
    per_signer = max(1, samples_per_class // signers)

    landmarks: list[np.ndarray] = []
    labels: list[str] = []
    signer_ids: list[str] = []
    hands: list[str] = []
    conditions: list[str] = []

    for signer in people:
        # Contacts are solved once per signer per letter, not per sample: it is an
        # optimisation, and the answer only depends on their bone lengths.
        resolved = {letter: _resolve_pose(letter, signer) for letter in letters}

        for letter in letters:
            orientation = orientation_for(letter)
            for _ in range(per_signer):
                points, hand, condition = _render(
                    _wobble(resolved[letter], signer, rng), signer, orientation, rng
                )
                landmarks.append(points)
                labels.append(letter)
                signer_ids.append(signer.name)
                hands.append(hand)
                conditions.append(condition)

        none_count = int(len(letters) * per_signer * NONE_FRACTION)
        for index in range(none_count):
            if index % 2 == 0:
                first = resolved[letters[int(rng.integers(len(letters)))]]
                second = resolved[letters[int(rng.integers(len(letters)))]]
                pose = _blend(first, second, float(rng.uniform(0.3, 0.7)))
                condition_hint = "transition"
            else:
                pose = _random_pose(rng)
                condition_hint = "idle"

            points, hand, _ = _render(
                pose,
                signer,
                (
                    float(rng.uniform(-40, 40)),
                    float(rng.uniform(-45, 45)),
                    float(rng.uniform(-180, 180)),
                ),
                rng,
            )
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
    return (
        f"Simulated from a kinematic hand model ({len(ALPHABET)} static letters; "
        f"{excluded} excluded as motion letters). Joint angles, bone lengths, viewpoint, "
        "perspective and tracking noise are randomised per simulated signer and per "
        "sample. No real hands were recorded, and no real hands were seen during "
        "training — see the limitations section."
    )
