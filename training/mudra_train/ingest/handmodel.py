"""A kinematic hand, articulated the way a real one is.

This exists because the datasets that would train a good fingerspelling model — FSboard,
PopSign — need a Kaggle account and a large download, and not everyone has either. A hand
is a well-understood mechanism, the ASL manual alphabet is defined in terms of joint
configuration and finger contact, and MediaPipe's landmark topology is public. Simulating
it is therefore a real option rather than a gesture at one.

It is not equal to real data, and the model card says so. What it is:

* **Articulated, not drawn.** Every landmark comes from forward kinematics over a bone
  skeleton with anthropometric proportions. Curling a finger moves three joints through
  their real ranges; it does not shorten a line.
* **Three-dimensional.** Letters are separated by finger *spread* (U vs V), by crossing
  (R), by thumb opposition (F, O, D) and by orientation (K/P, G/Q). A flat model cannot
  express any of those, which is exactly why the fixture generator in `synthetic.py` is
  useless as training data.
* **Projected.** The hand is rotated in space and projected through a pinhole, because
  what MediaPipe reports is a projection — foreshortening is signal, not noise.

Coordinate frame, hand-local, right hand:

* ``+x`` radial — toward the thumb side
* ``+y`` distal — from the wrist toward the middle-finger knuckle
* ``+z`` palmar — out of the palm, toward a camera the palm is facing

Image space flips ``y`` (screens grow downward); :func:`project` does that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LANDMARKS_PER_HAND = 21

#: MediaPipe landmark order: wrist, then thumb, index, middle, ring, pinky, four each.
THUMB = (1, 2, 3, 4)
FINGER_ROOTS = (5, 9, 13, 17)
FINGER_NAMES = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True, slots=True)
class HandGeometry:
    """Bone proportions, in units of palm length (wrist → middle MCP).

    Defaults are population averages; :func:`random_geometry` perturbs them per simulated
    signer, which is what makes a signer-independent split mean something here.
    """

    #: Knuckle positions (x, y, z) for index, middle, ring, pinky. The knuckle line is
    #: arched and tilted, not a straight row — that arch is why a fist is not a flat
    #: stack of fingers.
    knuckles: tuple[tuple[float, float, float], ...] = (
        (0.29, 0.98, 0.02),
        (0.09, 1.00, 0.00),
        (-0.11, 0.96, -0.01),
        (-0.30, 0.86, -0.04),
    )
    #: Proximal, middle, distal phalanx lengths per finger.
    phalanges: tuple[tuple[float, float, float], ...] = (
        (0.80, 0.45, 0.30),
        (0.88, 0.52, 0.33),
        (0.80, 0.48, 0.32),
        (0.62, 0.35, 0.28),
    )
    #: Thumb carpometacarpal joint — sits low and radial, not on the knuckle line.
    thumb_base: tuple[float, float, float] = (0.34, 0.20, 0.06)
    #: Thumb metacarpal, proximal, distal lengths.
    thumb_bones: tuple[float, float, float] = (0.62, 0.42, 0.32)
    #: Resting splay of each finger, degrees, positive = toward the thumb.
    rest_splay: tuple[float, float, float, float] = (6.0, 0.0, -5.0, -11.0)


@dataclass(frozen=True, slots=True)
class FingerPose:
    """One finger's joint angles, in degrees.

    ``mcp``/``pip``/``dip`` are flexion — curling toward the palm. ``splay`` is abduction
    within the palm plane, positive toward the thumb. Splay is the whole difference
    between U and V, so it is a first-class parameter rather than jitter.
    """

    mcp: float = 0.0
    pip: float = 0.0
    dip: float = 0.0
    splay: float = 0.0
    #: Sideways offset applied *after* kinematics, in palm units. Used only for R, where
    #: the fingers physically cross over one another — a rotation cannot express that.
    cross: float = 0.0


@dataclass(frozen=True, slots=True)
class ThumbPose:
    """Thumb joint angles, in degrees.

    ``radial`` swings the thumb away from the index finger within the palm plane: 0 is
    tucked alongside the index (A, S, M, N, T), ~70 is the right angle of L.
    ``palmar`` lifts it out of the palm plane toward the camera, which is what opposition
    — the thumb reaching across to meet a fingertip — is built from.
    """

    radial: float = 20.0
    palmar: float = 15.0
    mcp: float = 0.0
    ip: float = 0.0


@dataclass(frozen=True, slots=True)
class HandPose:
    """A complete hand configuration: four fingers and a thumb."""

    fingers: tuple[FingerPose, FingerPose, FingerPose, FingerPose]
    thumb: ThumbPose = field(default_factory=ThumbPose)


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis."""
    theta = np.deg2rad(degrees)
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    k = 1.0 - c
    return np.array(
        [
            [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
            [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
            [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
        ]
    )


_Z_AXIS = np.array([0.0, 0.0, 1.0])
_Y_AXIS = np.array([0.0, 1.0, 0.0])
_X_AXIS = np.array([1.0, 0.0, 0.0])


def _finger_chain(
    base: np.ndarray,
    lengths: tuple[float, float, float],
    pose: FingerPose,
    rest_splay: float,
) -> np.ndarray:
    """Forward kinematics down one finger. Returns the three joint positions after the MCP.

    Splay rotates the finger within the palm plane; flexion then rotates it out of that
    plane toward the palm, about an axis that has been splayed with it. Flexion
    accumulates down the chain, which is why a fully curled finger folds back on itself
    rather than merely bending once.
    """
    # Negated: rotating the distal axis toward +x — the thumb side — is a *negative*
    # rotation about +z, so without this a positive splay would fan the fingers the wrong
    # way and V, W and R would spread toward the little finger.
    splay = _rotation(_Z_AXIS, -(pose.splay + rest_splay))
    direction = splay @ _Y_AXIS
    # The flexion axis travels with the finger, otherwise a splayed finger would curl
    # sideways out of its own plane.
    axis = splay @ _X_AXIS

    points = np.empty((3, 3))
    position = base.astype(np.float64).copy()
    total = 0.0

    for index, (length, angle) in enumerate(
        zip(lengths, (pose.mcp, pose.pip, pose.dip), strict=True)
    ):
        total += angle
        position = position + (_rotation(axis, total) @ direction) * length
        points[index] = position

    if pose.cross:
        # R crosses the middle finger over the index. The offset grows along the finger,
        # because they cross at the tips and not at the knuckles.
        for index in range(3):
            points[index, 0] += pose.cross * (index + 1) / 3.0
            points[index, 2] += abs(pose.cross) * 0.25 * (index + 1) / 3.0

    return points


def _thumb_chain(geometry: HandGeometry, pose: ThumbPose) -> np.ndarray:
    """Forward kinematics down the thumb. Returns MCP, IP and tip positions."""
    # In-plane swing away from the index. Negative because +x is radial, so rotating the
    # distal axis toward +x is a negative rotation about +z.
    swing = _rotation(_Z_AXIS, -pose.radial)
    # Palmar abduction then lifts the thumb out of the palm plane, about the palm's width
    # axis carried round by the swing.
    lift = _rotation(swing @ _X_AXIS, pose.palmar)

    direction = lift @ (swing @ _Y_AXIS)
    # Flexion curls the tip across the palm, about the same carried axis.
    axis = lift @ (swing @ _X_AXIS)

    points = np.empty((3, 3))
    position = np.array(geometry.thumb_base, dtype=np.float64)
    total = 0.0

    for index, (length, angle) in enumerate(
        zip(geometry.thumb_bones, (0.0, pose.mcp, pose.ip), strict=True)
    ):
        total += angle
        position = position + (_rotation(axis, total) @ direction) * length
        points[index] = position

    return points


def build_hand(pose: HandPose, geometry: HandGeometry | None = None) -> np.ndarray:
    """Build 21 hand-local landmarks from a pose. Shape ``(21, 3)``."""
    geometry = geometry or HandGeometry()
    points = np.zeros((LANDMARKS_PER_HAND, 3), dtype=np.float64)

    points[0] = (0.0, 0.0, 0.0)  # wrist
    points[1] = geometry.thumb_base
    points[2:5] = _thumb_chain(geometry, pose.thumb)

    for finger in range(4):
        root = FINGER_ROOTS[finger]
        knuckle = np.array(geometry.knuckles[finger], dtype=np.float64)
        points[root] = knuckle
        points[root + 1 : root + 4] = _finger_chain(
            knuckle,
            geometry.phalanges[finger],
            pose.fingers[finger],
            geometry.rest_splay[finger],
        )

    return points


#: Anatomical limits for the thumb, degrees. The solver may not leave these.
THUMB_BOUNDS = ((-25.0, 80.0), (-15.0, 70.0), (0.0, 55.0), (0.0, 80.0))


def solve_thumb_contact(
    pose: HandPose,
    geometry: HandGeometry,
    landmark: int,
    distance: float,
) -> ThumbPose:
    """Adjust the thumb so its tip sits ``distance`` from ``landmark``.

    Several letters are defined by *contact*, not by angle: in F and O the thumb meets the
    index fingertip, in D it meets the middle. Hand-tuning four angles to achieve that is
    fragile — it holds for one set of bone lengths and silently breaks for the next, so
    every simulated signer would need retuning and any that were missed would be signing a
    subtly wrong letter.

    Declaring the contact and solving for it instead means the constraint holds for
    *whatever* hand the model is given, which is what makes per-signer anatomy safe to
    randomise.

    The declared angles act as both the starting point and a weak prior, so among the many
    thumb poses that satisfy a distance constraint the solver returns one close to the
    handshape as described, rather than an anatomically legal contortion.
    """
    from scipy.optimize import minimize  # imported lazily; only simulation needs it

    start = np.array([pose.thumb.radial, pose.thumb.palmar, pose.thumb.mcp, pose.thumb.ip])

    def cost(angles: np.ndarray) -> float:
        candidate = HandPose(
            fingers=pose.fingers,
            thumb=ThumbPose(*(float(a) for a in angles)),
        )
        points = build_hand(candidate, geometry)
        gap = float(np.linalg.norm(points[4] - points[landmark]))
        # The prior is deliberately weak — enough to choose between equally valid
        # contacts, not enough to prevent one being reached.
        prior = float(np.sum(((angles - start) / 60.0) ** 2))
        return (gap - distance) ** 2 + 0.02 * prior

    result = minimize(cost, start, method="L-BFGS-B", bounds=THUMB_BOUNDS)
    return ThumbPose(*(float(a) for a in result.x))


def project(
    points: np.ndarray,
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    distance: float = 6.0,
    scale: float = 0.22,
    centre: tuple[float, float] = (0.5, 0.5),
    hand: str = "right",
) -> np.ndarray:
    """Rotate the hand in space and project it the way a camera does.

    Foreshortening is the point. A hand tilted away from the camera produces shorter
    fingers in the image, and a model that has only ever seen a fronto-parallel hand
    treats that as a different letter. Perspective divide is what puts that in the data.

    Returns landmarks in MediaPipe's convention: ``x``/``y`` normalised to the image with
    ``y`` growing downward, ``z`` depth relative to the wrist on roughly the same scale.
    """
    rotation = _rotation(_Y_AXIS, yaw) @ _rotation(_X_AXIS, pitch) @ _rotation(_Z_AXIS, roll)
    rotated = points @ rotation.T

    if hand == "left":
        rotated[:, 0] = -rotated[:, 0]

    # Pinhole with the camera down +z. Depth is measured from the wrist so the divisor
    # cannot go negative for any plausible hand size.
    depth = distance - rotated[:, 2]
    projected = np.empty_like(rotated)
    # Both axes are negated, and for the same reason: this is a camera looking *at* the
    # signer, so its image axes run opposite to the signer's own.
    #
    # y, because screens grow downward. x, because the person's right — the radial, thumb
    # side of their right hand — appears on the *left* of the image. MediaPipe reports
    # coordinates in that raw camera frame: `getUserMedia` hands over an unmirrored feed,
    # and the selfie-style mirroring in the app is a CSS transform on the video element,
    # which never touches the pixels the landmarker samples.
    #
    # Getting this backwards is silent and total. Every hand the model is shown at
    # inference is then the mirror image of everything it trained on, the normaliser
    # cannot help — it canonicalises left to right, not raw to mirrored — and the
    # classifier rejects almost every frame as `none`. Measured: accuracy 0.95 → 0.24,
    # with 69% of real letters falling to the negative class.
    projected[:, 0] = centre[0] - scale * distance * rotated[:, 0] / depth
    projected[:, 1] = centre[1] - scale * distance * rotated[:, 1] / depth
    projected[:, 2] = scale * (rotated[:, 2] - rotated[0, 2])

    return projected


def random_geometry(rng: np.random.Generator) -> HandGeometry:
    """A plausible individual hand.

    Real hands differ in finger-length ratios far more than in overall size, and overall
    size is normalised away anyway. Varying the *ratios* is therefore what produces a
    genuinely held-out signer rather than a scaled copy of the same one.
    """
    base = HandGeometry()

    def jitter(value: float, spread: float) -> float:
        return float(value * (1.0 + rng.normal(0.0, spread)))

    phalanges = tuple(
        tuple(jitter(length, 0.07) for length in finger) for finger in base.phalanges
    )
    knuckles = tuple(
        (jitter(x, 0.06), jitter(y, 0.04), z + float(rng.normal(0.0, 0.01)))
        for x, y, z in base.knuckles
    )
    thumb_base = (
        jitter(base.thumb_base[0], 0.08),
        jitter(base.thumb_base[1], 0.12),
        base.thumb_base[2] + float(rng.normal(0.0, 0.02)),
    )
    thumb_bones = tuple(jitter(length, 0.08) for length in base.thumb_bones)
    rest_splay = tuple(value + float(rng.normal(0.0, 2.5)) for value in base.rest_splay)

    return HandGeometry(
        knuckles=knuckles,  # type: ignore[arg-type]
        phalanges=phalanges,  # type: ignore[arg-type]
        thumb_base=thumb_base,
        thumb_bones=thumb_bones,  # type: ignore[arg-type]
        rest_splay=rest_splay,  # type: ignore[arg-type]
    )
