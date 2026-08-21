"""ASL numbers 0–9, and the fact that half of them are already letters.

This is the part worth reading before adding numbers to anything.

**In ASL itself, several numbers and letters are the same handshape.** Not similar —
identical:

===========  ===============
Number       Same shape as
===========  ===============
``0``        ``O``
``2``        ``V``
``6``        ``W``
``9``        ``F``
===========  ===============

A signer tells them apart from context, not from the hand. So a classifier asked to
output 26 letters *and* 10 digits as 36 flat classes is being asked to split classes that
carry no distinguishing signal. It cannot, and the cost does not fall only on the digits:
probability mass that should sit on ``V`` gets divided between ``V`` and ``2``, so
**letter accuracy gets worse** — the model is made worse at the thing it used to do well,
in exchange for a capability that cannot work.

So this module defines only the **six digits that are genuinely new handshapes** — 1, 3,
4, 5, 7 and 8 — and records the other four as aliases. The model learns 30 handshapes; a
*mode* decides whether a given shape is read as a letter or a digit. That is how a human
reads them too.

The four thumb-contact digits are a nice demonstration of why the contact solver exists:
6, 7, 8 and 9 are the same gesture — a fingertip touching the thumb — walking one finger
along the hand each time. Declaring them as contacts says exactly that, and the solver
finds the angles for whatever hand it is given.
"""

from __future__ import annotations

from .asl_alphabet import EXT, FIST, _splayed
from .handmodel import Contact, FingerPose, HandPose, ThumbPose

#: Digits that are the same handshape as a letter, so the model never learns them twice.
#:
#: The value is the label the classifier is trained on; the digit is what a numbers-mode
#: reader displays for it.
NUMBER_ALIASES: dict[str, str] = {
    "0": "O",
    "2": "V",
    "6": "W",
    "9": "F",
}

#: Digits that are genuinely distinct handshapes and get their own class.
NUMBER_SHAPES: dict[str, HandPose] = {
    # Index up, everything else in a fist, thumb resting against the middle finger.
    # Distinct from D, where the other three fingers curl up to *meet* the thumb.
    "1": HandPose(
        fingers=(EXT, FIST, FIST, FIST),
        thumb=ThumbPose(radial=6.0, palmar=10.0, mcp=36.0, ip=24.0),
    ),
    # Thumb, index and middle all extended. The thumb being out is what separates it from
    # K, where the thumb tucks up between the two fingers.
    "3": HandPose(
        fingers=(_splayed(EXT, 12.0), _splayed(EXT, -6.0), FIST, FIST),
        thumb=ThumbPose(radial=64.0, palmar=8.0, mcp=4.0, ip=2.0),
    ),
    # Four fingers up and *spread*, thumb folded across the palm. B is the same hand with
    # the fingers held together — the same distinction that separates U from V.
    "4": HandPose(
        fingers=(
            _splayed(EXT, 13.0),
            _splayed(EXT, 5.0),
            _splayed(EXT, -4.0),
            _splayed(EXT, -13.0),
        ),
        thumb=ThumbPose(radial=-6.0, palmar=-6.0, mcp=34.0, ip=14.0),
    ),
    # The open hand: all five spread. No letter has the thumb out beside four raised
    # fingers, so this one is unambiguous.
    "5": HandPose(
        fingers=(
            _splayed(EXT, 15.0),
            _splayed(EXT, 5.0),
            _splayed(EXT, -5.0),
            _splayed(EXT, -15.0),
        ),
        thumb=ThumbPose(radial=68.0, palmar=14.0, mcp=2.0, ip=2.0),
    ),
    # Ring finger down to the thumb, the rest standing.
    "7": HandPose(
        fingers=(
            _splayed(EXT, 12.0),
            _splayed(EXT, 2.0),
            FingerPose(mcp=58.0, pip=74.0, dip=34.0, splay=-6.0),
            _splayed(EXT, -14.0),
        ),
        thumb=ThumbPose(radial=18.0, palmar=48.0, mcp=22.0, ip=18.0),
    ),
    # Middle finger down to the thumb.
    "8": HandPose(
        fingers=(
            _splayed(EXT, 12.0),
            FingerPose(mcp=58.0, pip=74.0, dip=34.0, splay=2.0),
            _splayed(EXT, -6.0),
            _splayed(EXT, -14.0),
        ),
        thumb=ThumbPose(radial=20.0, palmar=50.0, mcp=22.0, ip=18.0),
    ),
}

#: Where the thumb meets, for the digits defined by contact. Same machinery as F and O.
NUMBER_CONTACT: dict[str, Contact] = {
    "7": Contact(16, 0.05),  # ring fingertip
    "8": Contact(12, 0.05),  # middle fingertip
}

#: What a numbers-mode reader displays for each trained label.
#:
#: Built from the two halves above so the two can never disagree — which they would, the
#: first time a digit moved between them.
NUMBER_VOCABULARY: dict[str, str] = {
    **{label: digit for digit, label in NUMBER_ALIASES.items()},
    **{label: label for label in NUMBER_SHAPES},
}
