"""The ASL manual alphabet, as joint configurations.

Each letter is a :class:`HandPose` — flexion at every knuckle, finger spread, thumb
opposition — plus the orientation the letter is normally signed at. This is the actual
definition of a fingerspelled letter: linguistics describes them by handshape, finger
contact and palm orientation, not by appearance.

Writing them down this way makes the hard distinctions explicit instead of accidental:

* **U vs V** differ only in *splay*. Same flexion, same orientation.
* **R** crosses middle over index; no joint angle expresses that, so it gets a lateral
  offset that grows toward the fingertips, which is where the fingers actually cross.
* **M, N, T** differ only in how many fingers lie over the tucked thumb — three, two, one.
* **K vs P** and **G vs Q** are the *same handshape* at different orientations. They are
  the reason `normalise.py` deliberately does not cancel rotation.
* **F, O, D** are defined by which fingertip the thumb touches, which is what thumb
  palmar abduction is in the model for.

**J and Z are excluded.** Both are motion letters — J traces a hook, Z draws the letter —
and a single frame cannot represent either. The continuous CTC pipeline handles them;
a static handshape classifier that claimed to would be lying.

The angles are anatomically grounded rather than measured: full flexion is roughly 90° at
the MCP, 100° at the PIP and 70° at the DIP, and the values here are read off the standard
alphabet against those ranges.
"""

from __future__ import annotations

from .handmodel import FingerPose, HandPose, ThumbPose

# Common finger poses, so a letter reads as a handshape rather than twelve numbers.

#: Straight.
EXT = FingerPose(mcp=2.0, pip=3.0, dip=2.0)
#: Folded into the palm.
FIST = FingerPose(mcp=88.0, pip=100.0, dip=68.0)
#: Barely closed — the wide, open arc of C.
CURVE = FingerPose(mcp=38.0, pip=44.0, dip=18.0)
#: Curled right round to meet a lifted thumb — the closed ring of O.
#:
#: Distinctly tighter than :data:`CURVE`. C and O are the same *gesture* at two degrees of
#: closure, so if these two are similar the classifier cannot separate them either.
TOUCH = FingerPose(mcp=58.0, pip=74.0, dip=34.0)
#: Folded down over the thumb — E, and the covering fingers of M, N and T.
CLOSED = FingerPose(mcp=70.0, pip=88.0, dip=42.0)


def _splayed(base: FingerPose, splay: float) -> FingerPose:
    return FingerPose(mcp=base.mcp, pip=base.pip, dip=base.dip, splay=splay, cross=base.cross)


#: Palm orientation per letter: (yaw, pitch, roll) degrees applied before projection.
#:
#: Most letters are signed palm-out at a slight angle. The exceptions carry meaning: G and
#: H point sideways, P and Q are K and G rotated to point downward, and those rotations
#: are the *only* thing separating those pairs.
ORIENTATION: dict[str, tuple[float, float, float]] = {
    # G and H lie the hand on its side and point across the body. Roll does that work;
    # a large pitch would turn the hand edge-on and collapse it to a line.
    "G": (34.0, -10.0, -76.0),
    "H": (34.0, -10.0, -76.0),
    # P and Q are K and G aimed at the floor.
    "P": (0.0, 20.0, 168.0),
    "Q": (0.0, 20.0, -160.0),
}
DEFAULT_ORIENTATION = (0.0, -8.0, 0.0)

#: Letters defined by where the thumb touches: ``letter -> (landmark, distance)``.
#:
#: Distance is in palm units, so 0.05 is contact and 0.45 is the deliberate opening of C.
#: These are solved per simulated signer rather than baked in as angles, because the angle
#: that produces contact depends on the bone lengths of the hand doing it — see
#: :func:`~.handmodel.solve_thumb_contact`.
THUMB_CONTACT: dict[str, tuple[int, float]] = {
    "E": (12, 0.12),  # fingertips come down onto a thumb lying across the palm centre
    "C": (8, 0.52),  # an open curve: near the index tip, deliberately not touching
    "D": (12, 0.06),  # thumb meets the middle finger, index stands alone
    "F": (8, 0.05),  # the ring of F
    "O": (8, 0.05),  # every tip meets the thumb; index is the reference
    "X": (7, 0.20),  # thumb rests beside the hooked index
    "K": (10, 0.26),  # thumb pushes up against the middle finger's proximal bone
    "P": (10, 0.26),  # K, aimed downward
    # M, N and T bury the thumb under the fingers and differ only in how many cover it —
    # three, two, one. So the thumb tip emerges one finger further round the hand each
    # time, and the target is the *knuckle* of the finger it emerges beside: in a folded
    # hand the PIP joints are the visible top of the fist, which is where the thumb tip
    # actually shows. Aiming at the fingertips instead puts it below the fist, where no
    # thumb has ever been.
    "T": (6, 0.16),  # wedged between index and middle
    "N": (10, 0.16),  # under two fingers, emerging by the middle knuckle
    "M": (14, 0.16),  # under three, emerging by the ring knuckle
}


ALPHABET: dict[str, HandPose] = {
    # Fist with the thumb alongside the index, not across it — that is what separates
    # A from S.
    "A": HandPose(
        fingers=(FIST, FIST, FIST, FIST),
        thumb=ThumbPose(radial=12.0, palmar=8.0, mcp=8.0, ip=6.0),
    ),
    # Flat hand, fingers together, thumb folded across the palm.
    "B": HandPose(
        fingers=(
            _splayed(EXT, -8.0),
            _splayed(EXT, -1.0),
            _splayed(EXT, 4.0),
            _splayed(EXT, 10.0),
        ),
        thumb=ThumbPose(radial=-6.0, palmar=-8.0, mcp=32.0, ip=12.0),
    ),
    # An open curve; the thumb mirrors the fingers to close the C.
    "C": HandPose(
        fingers=(CURVE, CURVE, CURVE, CURVE),
        thumb=ThumbPose(radial=34.0, palmar=44.0, mcp=16.0, ip=8.0),
    ),
    # Index up; the other three curl down to meet the thumb.
    "D": HandPose(
        fingers=(EXT, TOUCH, TOUCH, TOUCH),
        thumb=ThumbPose(radial=26.0, palmar=52.0, mcp=26.0, ip=22.0),
    ),
    # All four fold down over a thumb lying flat across the palm.
    #
    # The contact is with the *middle* finger, not the index. Declaring E as "thumb
    # touches index" is tempting — they do end up close — but that is O's defining
    # feature, and reusing it collapses the two letters. The thumb in E lies across the
    # centre of the palm, so the middle finger is what comes down on it.
    "E": HandPose(
        fingers=(CLOSED, CLOSED, CLOSED, CLOSED),
        thumb=ThumbPose(radial=-14.0, palmar=2.0, mcp=42.0, ip=18.0),
    ),
    # Thumb and index meet in a ring; the other three stand up.
    "F": HandPose(
        fingers=(TOUCH, _splayed(EXT, 4.0), _splayed(EXT, 2.0), _splayed(EXT, 4.0)),
        thumb=ThumbPose(radial=30.0, palmar=56.0, mcp=22.0, ip=18.0),
    ),
    # Index pointing, thumb parallel beside it. Signed sideways — see ORIENTATION.
    "G": HandPose(
        fingers=(EXT, FIST, FIST, FIST),
        thumb=ThumbPose(radial=16.0, palmar=10.0, mcp=4.0, ip=2.0),
    ),
    # Index and middle together, pointing sideways.
    "H": HandPose(
        fingers=(_splayed(EXT, -7.0), _splayed(EXT, 2.0), FIST, FIST),
        thumb=ThumbPose(radial=8.0, palmar=12.0, mcp=30.0, ip=18.0),
    ),
    # Pinky alone.
    "I": HandPose(
        fingers=(FIST, FIST, FIST, EXT),
        thumb=ThumbPose(radial=8.0, palmar=10.0, mcp=34.0, ip=20.0),
    ),
    # Index and middle in a V with the thumb pushed up between them.
    "K": HandPose(
        fingers=(_splayed(EXT, 12.0), _splayed(EXT, -14.0), FIST, FIST),
        thumb=ThumbPose(radial=22.0, palmar=48.0, mcp=10.0, ip=4.0),
    ),
    # The right angle: index up, thumb straight out.
    "L": HandPose(
        fingers=(EXT, FIST, FIST, FIST),
        thumb=ThumbPose(radial=72.0, palmar=6.0, mcp=2.0, ip=2.0),
    ),
    # Three fingers over the tucked thumb.
    "M": HandPose(
        fingers=(CLOSED, CLOSED, CLOSED, FIST),
        thumb=ThumbPose(radial=-14.0, palmar=6.0, mcp=48.0, ip=34.0),
    ),
    # Two fingers over the tucked thumb.
    "N": HandPose(
        fingers=(CLOSED, CLOSED, FIST, FIST),
        thumb=ThumbPose(radial=-8.0, palmar=8.0, mcp=44.0, ip=32.0),
    ),
    # Every fingertip meets the thumb.
    "O": HandPose(
        fingers=(TOUCH, TOUCH, TOUCH, TOUCH),
        thumb=ThumbPose(radial=30.0, palmar=54.0, mcp=24.0, ip=18.0),
    ),
    # K, pointed downward.
    "P": HandPose(
        fingers=(_splayed(EXT, 12.0), _splayed(EXT, -14.0), FIST, FIST),
        thumb=ThumbPose(radial=22.0, palmar=48.0, mcp=10.0, ip=4.0),
    ),
    # G, pointed downward.
    "Q": HandPose(
        fingers=(EXT, FIST, FIST, FIST),
        thumb=ThumbPose(radial=16.0, palmar=12.0, mcp=4.0, ip=2.0),
    ),
    # Crossed fingers. `cross` is what a rotation cannot express.
    "R": HandPose(
        fingers=(
            FingerPose(mcp=6.0, pip=14.0, dip=6.0, splay=-9.0, cross=-0.34),
            FingerPose(mcp=10.0, pip=20.0, dip=8.0, splay=9.0, cross=0.34),
            FIST,
            FIST,
        ),
        thumb=ThumbPose(radial=6.0, palmar=10.0, mcp=34.0, ip=22.0),
    ),
    # Fist with the thumb clamped across the front of the fingers.
    "S": HandPose(
        fingers=(FIST, FIST, FIST, FIST),
        thumb=ThumbPose(radial=-18.0, palmar=4.0, mcp=42.0, ip=26.0),
    ),
    # One finger over the tucked thumb.
    "T": HandPose(
        fingers=(CLOSED, FIST, FIST, FIST),
        thumb=ThumbPose(radial=-4.0, palmar=10.0, mcp=40.0, ip=30.0),
    ),
    # Index and middle up, held together.
    "U": HandPose(
        fingers=(_splayed(EXT, -7.0), _splayed(EXT, 2.0), FIST, FIST),
        thumb=ThumbPose(radial=6.0, palmar=10.0, mcp=34.0, ip=22.0),
    ),
    # Index and middle up, apart. Only the splay separates this from U.
    "V": HandPose(
        fingers=(_splayed(EXT, 15.0), _splayed(EXT, -15.0), FIST, FIST),
        thumb=ThumbPose(radial=6.0, palmar=10.0, mcp=34.0, ip=22.0),
    ),
    # Three up and spread.
    "W": HandPose(
        fingers=(_splayed(EXT, 17.0), _splayed(EXT, 0.0), _splayed(EXT, -17.0), FIST),
        thumb=ThumbPose(radial=4.0, palmar=12.0, mcp=38.0, ip=26.0),
    ),
    # The hook: the knuckle stays straight, the finger bends.
    "X": HandPose(
        fingers=(FingerPose(mcp=18.0, pip=88.0, dip=30.0), FIST, FIST, FIST),
        thumb=ThumbPose(radial=8.0, palmar=12.0, mcp=36.0, ip=24.0),
    ),
    # Thumb and pinky out, everything else down.
    "Y": HandPose(
        fingers=(FIST, FIST, FIST, EXT),
        thumb=ThumbPose(radial=76.0, palmar=10.0, mcp=2.0, ip=2.0),
    ),
}

#: Letters this model cannot represent, and why.
MOTION_LETTERS = {
    "J": "traces a hook with the pinky; a single frame cannot show the path",
    "Z": "draws the letter in the air with the index finger",
}


def orientation_for(letter: str) -> tuple[float, float, float]:
    """Palm orientation for a letter, in degrees."""
    return ORIENTATION.get(letter, DEFAULT_ORIENTATION)
