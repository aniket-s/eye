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
* **K vs P**, **G vs Q** and **H vs U** are the *same handshape* at different
  orientations. They are the reason `normalise.py` deliberately does not cancel
  rotation, and the reason orientation is declared as a **band** per letter rather
  than a single angle — see :data:`ORIENTATION`.
* **F, O, D** are defined by which fingertip the thumb touches, which is what thumb
  palmar abduction is in the model for.
* **S vs A vs E** are separated by which *side* of the folded fingers the thumb is on.
  S clamps across the front of them, A rides the radial edge behind them, and in E the
  fingertips come down on top of a thumb lying flat across the palm. Dropping ``z``
  hides the side itself, but it decides where the thumb lands in the picture, so
  getting it wrong moves the thumb to somewhere no thumb has ever been. Declared with
  :func:`~.handmodel.solve_thumb_contact`'s ``in_front_of`` / ``behind`` argument and
  pinned by ``tests/test_alphabet_anatomy.py``.

**J and Z are excluded.** Both are motion letters — J traces a hook, Z draws the letter —
and a single frame cannot represent either. The continuous CTC pipeline handles them;
a static handshape classifier that claimed to would be lying.

The angles are anatomically grounded rather than measured: full flexion is roughly 90° at
the MCP, 100° at the PIP and 70° at the DIP, and the values here are read off the standard
alphabet against those ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

from .handmodel import Contact, FingerPose, HandPose, ThumbPose

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


@dataclass(frozen=True, slots=True)
class Orientation:
    """The range of palm orientations a letter is legitimately signed at, in degrees.

    A **range**, not an angle, and that distinction is the whole point. Declaring one
    angle and relying on noise plus blind rotation augmentation to cover the rest gives
    Gaussian coverage: dense at the declared angle, thin in the tails, and stopping dead
    at the augmentation bound. A signer whose wrist sits 40° off the declared angle for
    G falls outside everything the model ever saw, and an unseen orientation does not
    read as a near miss — it reads as ``none``. Measured on the previous pack: signing G
    at the orientation the dictionary illustrated produced 0% accepted, 100% ``none``.

    A declared band is sampled uniformly, so coverage is even right up to its edge, and
    the edge is a stated fact about the letter rather than a side effect of the noise
    model.

    ``roll`` is the axis that carries meaning, because it is the one that turns a letter
    into a *different letter*: H is U rolled, P is K rolled, Q is G rolled. Those three
    pairs — and only those three — must keep disjoint roll bands, which is what
    :func:`assert_bands_are_separable` checks. Every other pair differs in handshape as
    well, so their bands may overlap freely and are widened until the projection
    degenerates rather than kept artificially narrow.
    """

    yaw: tuple[float, float]
    pitch: tuple[float, float]
    roll: tuple[float, float]


#: Most letters are signed roughly upright and palm-out, with the spread here covering
#: how a real wrist actually sits rather than how a diagram draws it.
#:
#: The roll band is generous because for most letters tilt is pure nuisance — nothing
#: else is reachable by rotating them. The exceptions are U and K, which have their own
#: tighter entries below: they are what H and P *are* when rolled, so their bands have a
#: neighbour to stay clear of.
DEFAULT_ORIENTATION = Orientation(yaw=(-34.0, 34.0), pitch=(-32.0, 16.0), roll=(-30.0, 30.0))

#: Palm orientation per letter, where it differs from :data:`DEFAULT_ORIENTATION`.
#:
#: Six entries, for two different reasons. G, H, P and Q are the letters whose
#: orientation *is* their identity — G and H lay the hand on its side and point across
#: the body, P and Q are K and G aimed at the floor. U and K are ordinary upright
#: letters that happen to be what H and P become when rolled, so they are here only to
#: give up the part of the default roll band that would reach their twin. Everything
#: else inherits the default untouched.
ORIENTATION: dict[str, Orientation] = {
    # U and K are ordinary upright letters that happen to have a rolled twin — H and P.
    # Their roll band is the one part of the default they cannot have, because tilting a
    # U far enough makes it an H.
    "U": Orientation(yaw=(-34.0, 34.0), pitch=(-32.0, 16.0), roll=(-16.0, 16.0)),
    "K": Orientation(yaw=(-34.0, 34.0), pitch=(-32.0, 16.0), roll=(-16.0, 16.0)),
    # G and H lie the hand on its side. The roll band is wide because wrists differ by
    # far more than the ±25° of augmentation the previous pack relied on, and it stops
    # short of U's band because H *is* U rolled.
    #
    # The yaw band is the other half of the fix, and the larger one. G and H are signed
    # palm-in by some signers and palm-angled by others; the previous pack was trained
    # at a single yaw and rejected the other palm-facing outright (measured: 0-8%
    # accepted at yaw -60 to -80). Pointing direction is what makes a G, not which way
    # the palm happens to face, so the band covers both.
    "G": Orientation(yaw=(-48.0, 78.0), pitch=(-28.0, 12.0), roll=(-112.0, -42.0)),
    "H": Orientation(yaw=(-48.0, 78.0), pitch=(-28.0, 12.0), roll=(-112.0, -42.0)),
    # P and Q aim at the floor. Real signers get there by rotating the wrist forward as
    # well as rolling, so the pitch band runs well past level — the hand is seen edge-on
    # or from the back, not palm-first.
    "P": Orientation(yaw=(-26.0, 26.0), pitch=(0.0, 62.0), roll=(140.0, 204.0)),
    "Q": Orientation(yaw=(-26.0, 26.0), pitch=(0.0, 62.0), roll=(-206.0, -142.0)),
}

#: The pairs that are one handshape at two orientations, and must therefore stay apart.
#:
#: Listed explicitly because it is a linguistic fact about ASL, not a property anything
#: in the code can derive. Every other pair of letters differs in handshape too, so their
#: orientation bands are free to overlap.
ROTATION_ONLY_PAIRS: tuple[tuple[str, str], ...] = (("H", "U"), ("K", "P"), ("G", "Q"))

#: Roll degrees two rotation-only bands must stay apart, after augmentation widens both.
#:
#: Small on purpose. A large margin looks safe and is not: it buys separation between two
#: letters at the price of a dead zone between them, and a signer who lands in the dead
#: zone gets ``none`` — the failure this whole table exists to remove. Better to let the
#: two bands nearly touch and let the classifier work at the boundary.
MIN_BAND_SEPARATION = 8.0

#: Letters defined by where the thumb touches: ``letter -> (landmark, distance)``.
#:
#: Distance is in palm units, so 0.05 is contact and 0.45 is the deliberate opening of C.
#: These are solved per simulated signer rather than baked in as angles, because the angle
#: that produces contact depends on the bone lengths of the hand doing it — see
#: :func:`~.handmodel.solve_thumb_contact`.
THUMB_CONTACT: dict[str, Contact] = {
    # E: the fingers come down *on top of* the thumb, so the thumb stays behind the
    # fingertips. Without saying so the solver satisfied the distance from the front
    # instead, putting the thumb over the curled fingers — which is a flat O, not an E,
    # and is why the previous pack read a slightly loose E as O half the time.
    "E": Contact(12, 0.12, behind=12),
    "C": Contact(8, 0.52),  # an open curve: near the index tip, deliberately not touching
    "D": Contact(12, 0.06),  # thumb meets the middle finger, index stands alone
    "F": Contact(8, 0.05),  # the ring of F
    "O": Contact(8, 0.05),  # every tip meets the thumb; index is the reference
    "X": Contact(7, 0.20),  # thumb rests beside the hooked index
    "K": Contact(10, 0.26),  # thumb pushes up against the middle finger's proximal bone
    "P": Contact(10, 0.26),  # K, aimed downward
    # S: the thumb clamps *across the front* of the folded fingers, landing over the
    # middle finger's knuckle. The previous definition left it in the palm plane, which
    # put it behind the fist beside the index — where A's thumb goes. Two letters were
    # sharing one thumb position, and correcting it costs nothing: `in_front_of` picks
    # the other solution to the same distance constraint.
    "S": Contact(10, 0.14, in_front_of=10),
    # M, N and T bury the thumb under the fingers and differ only in how many cover it —
    # three, two, one. So the thumb tip emerges one finger further round the hand each
    # time, and the target is the *knuckle* of the finger it emerges beside: in a folded
    # hand the PIP joints are the visible top of the fist, which is where the thumb tip
    # actually shows. Aiming at the fingertips instead puts it below the fist, where no
    # thumb has ever been.
    #
    # `behind` states the other half: the covering fingers are over the thumb, which is
    # what "buried" means and what separates all three from S.
    "T": Contact(6, 0.16, behind=6),  # wedged between index and middle
    "N": Contact(10, 0.16, behind=10),  # under two fingers, emerging by the middle knuckle
    "M": Contact(14, 0.16, behind=14),  # under three, emerging by the ring knuckle
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
        thumb=ThumbPose(radial=-12.0, palmar=-6.0, mcp=44.0, ip=22.0),
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
    #
    # The V is deliberately *narrower* than the one in V itself. Textbook K and V have
    # similar spread and are told apart by the thumb — but a thumb wedged between two
    # fingers is the one landmark a camera reliably cannot see, so a model given only
    # that cue has nothing left when the thumb is occluded. The v1 pipeline, tuned
    # against a real webcam, separated them by fingertip separation alone and never
    # looked at the thumb; this encodes what it learned. Spread is also a genuine
    # difference — K's fingers really do sit closer than V's — so this sharpens a true
    # cue rather than inventing one.
    "K": HandPose(
        fingers=(_splayed(EXT, 7.0), _splayed(EXT, -7.0), FIST, FIST),
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
    # K, pointed downward. Same handshape by definition, so the spread must match K's.
    "P": HandPose(
        fingers=(_splayed(EXT, 7.0), _splayed(EXT, -7.0), FIST, FIST),
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
    #
    # `palmar` lifts the thumb out of the palm plane and over the folded fingers, which
    # is what "across the front" means and what separates S from A. At the previous
    # value of 4 it lay in the palm plane — beside the fist rather than over it — and
    # the solved contact in THUMB_CONTACT now states the side explicitly as well.
    "S": HandPose(
        fingers=(FIST, FIST, FIST, FIST),
        thumb=ThumbPose(radial=-8.0, palmar=40.0, mcp=44.0, ip=28.0),
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

#: Letters whose thumb a camera genuinely loses.
#:
#: In each of these the thumb is tucked under, behind, or between the fingers, and
#: MediaPipe does not report an occluded thumb as missing — it regresses one toward an
#: average hand. A model that has only ever seen the thumb exactly where the geometry
#: puts it therefore loses its single cue at the moment it matters most. K is the
#: clearest case: the v1 pipeline, tuned against a real webcam, gave up on K's thumb
#: entirely and separated K from V by fingertip spread instead.
#:
#: Deliberately a list rather than an augmentation applied to everything. L, Y, G and Q
#: hold the thumb out where nothing can hide it, and teaching the model that L's thumb
#: might vanish would turn L into a plain pointing finger.
OCCLUDED_THUMB = frozenset({"A", "E", "K", "M", "N", "O", "P", "S", "T", "X"})

#: Letters whose index and middle fingers a camera cannot tell apart.
#:
#: R winds them round each other, so from the front they overlap and MediaPipe returns
#: two nearly coincident tips with the depth ordering guessed. That degraded reading is
#: what a real R usually arrives as, and it is why the previous pack — trained only on
#: cleanly separated crossed fingers — read a real R as U.
MERGED_FINGERS = frozenset({"R"})

#: Letters this model cannot represent, and why.
MOTION_LETTERS = {
    "J": "traces a hook with the pinky; a single frame cannot show the path",
    "Z": "draws the letter in the air with the index finger",
}


@dataclass(frozen=True, slots=True)
class Variation:
    """How much a letter's *shape* legitimately differs between signers.

    Distinct from the per-signer ``articulation`` trait in ``simulated.py``, which says
    how crisply someone signs everything. This says how much room a particular letter
    has, and the two compound.

    It exists because the range is not the same for every letter, and treating it as if
    it were is what put E on a knife edge: a fraction looser and the previous pack read
    it as O, a fraction tighter and it read it as S, because it had only ever seen the
    midpoint. Meanwhile U and V have almost no room at all — spread *is* the letter, and
    widening it would walk one into the other.
    """

    #: Multiplier applied to every flexion angle, sampled per repetition.
    curl: tuple[float, float] = (0.95, 1.05)
    #: Extra finger splay, ± degrees.
    splay: float = 2.0
    #: Range for :attr:`~.handmodel.FingerPose.cross`, for the letters that cross.
    cross: tuple[float, float] | None = None


#: Per-letter shape variation, where it differs from the default.
VARIATION: dict[str, Variation] = {
    # E runs from a loose claw to fingertips pressed flat onto the thumb, and both are
    # E. The band is wide because measurement said it had to be: at 0.75x the previous
    # pack called it O half the time, at 1.25x it called it S half the time.
    "E": Variation(curl=(0.80, 1.20), splay=3.0),
    # O and C are the same gesture at two degrees of closure, so their curl ranges are
    # kept apart deliberately rather than widened toward each other.
    "O": Variation(curl=(0.92, 1.06)),
    "C": Variation(curl=(0.94, 1.08)),
    "S": Variation(curl=(0.92, 1.10)),
    "A": Variation(curl=(0.92, 1.10)),
    # K and P share a handshape, so they share a range. Wider than V's because a
    # narrow-V K is a common and perfectly correct way to sign it — and one the previous
    # pack rejected 14% of the time.
    "K": Variation(curl=(0.94, 1.06), splay=5.0),
    "P": Variation(curl=(0.94, 1.06), splay=5.0),
    # U and V are separated by spread alone. Neither gets room to reach the other.
    "U": Variation(curl=(0.96, 1.04), splay=1.5),
    "V": Variation(curl=(0.96, 1.04), splay=1.5),
    # R's crossing runs from barely-there to fully wound round. The shallow end matters
    # most: MediaPipe under-resolves crossed fingers, so what a real R reaches the
    # classifier as is usually a shallow crossing, and the previous pack — trained on
    # one deep value — read that as U 38% of the time.
    "R": Variation(curl=(0.94, 1.08), splay=2.0, cross=(0.14, 0.52)),
}

DEFAULT_VARIATION = Variation()


def orientation_for(letter: str) -> Orientation:
    """The band of palm orientations this letter may be signed at."""
    return ORIENTATION.get(letter, DEFAULT_ORIENTATION)


def variation_for(letter: str) -> Variation:
    """How much this letter's shape may vary between signers."""
    return VARIATION.get(letter, DEFAULT_VARIATION)


def assert_bands_are_separable(augmentation_degrees: float) -> None:
    """Check that no rotation-only pair's roll bands can meet.

    H is U rolled, P is K rolled, Q is G rolled. If augmentation widens their declared
    bands until they touch, the training data contains the same geometry under two
    labels and the classifier cannot do better than a coin flip on either.

    Called by the trainer rather than at import, because the safe band width depends on
    how much augmentation is configured — and the failure it guards against is silent:
    the report still looks fine, because both letters are still separable from every
    letter that is *not* their twin.

    :raises ValueError: naming the pair, the gap, and both bands.
    """
    for first, second in ROTATION_ONLY_PAIRS:
        one = _widened(orientation_for(first).roll, augmentation_degrees)
        two = _widened(orientation_for(second).roll, augmentation_degrees)
        gap = _circular_gap(one, two)
        if gap < MIN_BAND_SEPARATION:
            raise ValueError(
                f"{first} and {second} are one handshape at two orientations, and their "
                f"roll bands are only {gap:.1f}° apart after ±{augmentation_degrees:.0f}° "
                f"of augmentation ({first}: {one}, {second}: {two}). Widen the gap or "
                f"reduce the augmentation — below {MIN_BAND_SEPARATION}° the training set "
                "labels the same geometry both ways."
            )


def _widened(band: tuple[float, float], margin: float) -> tuple[float, float]:
    return (band[0] - margin, band[1] + margin)


def _circular_gap(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Smallest angular distance between two arcs of roll, or 0 if they overlap.

    Roll is periodic, so a linear comparison gets P and K wrong: P sits near 180° and K
    near 0°, which are far apart the short way round and further still the long way, but
    a subtraction can report either.
    """
    spans = (first[1] - first[0]) + (second[1] - second[0])
    forward = (second[0] - first[1]) % 360.0
    backward = (first[0] - second[1]) % 360.0
    # Two disjoint arcs and the two gaps between them tile the circle exactly.
    if forward + backward + spans > 360.0 + 1e-9:
        return 0.0
    return min(forward, backward)
