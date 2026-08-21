"""The ASL alphabet's defining anatomy, asserted rather than assumed.

``test_handmodel.py`` checks that the kinematics work: a curled finger shortens, a
solved contact reaches its target. This file checks something the kinematics cannot —
that the joint angles in ``asl_alphabet.py`` describe the letters they claim to.

That distinction is not academic. Every check here corresponds to a defect the previous
pack shipped with, each of which was invisible to every metric the trainer reported,
because a wrong letter is still perfectly separable from its own simulated twin:

* **S's thumb sat behind the folded fingers** rather than clamped across the front of
  them, which is where A's thumb goes. Two letters, one thumb position. Signing a
  correct S — thumb over the fingers — was read as E 41% of the time.
* **E's thumb sat in front of the fingertips**, so the fingers were curled *under* the
  thumb instead of down onto it. That is a flat O, and a slightly loose E was read as
  O half the time.
* **K's spread matched V's**, leaving the thumb as the only thing between them — and
  the thumb in K is wedged between two fingers, which is the single landmark a camera
  most reliably cannot see.

The assertions are on *relationships* — which side, which is nearer, which is wider —
never on absolute angles, so the definitions stay free to be tuned.
"""

from __future__ import annotations

import numpy as np
import pytest

from mudra_train.ingest.asl_alphabet import (
    ALPHABET,
    MIN_BAND_SEPARATION,
    ROTATION_ONLY_PAIRS,
    THUMB_CONTACT,
    _circular_gap,
    assert_bands_are_separable,
    orientation_for,
    variation_for,
)
from mudra_train.ingest.handmodel import HandGeometry, build_hand, resolve_contact

WRIST = 0
THUMB_TIP = 4
#: Per finger: MCP, PIP, TIP.
INDEX, MIDDLE, RING, PINKY = (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)


def hand(letter: str) -> np.ndarray:
    """The letter as the trainer builds it, in hand-local 3-D. ``+z`` is palmar."""
    geometry = HandGeometry()
    pose = ALPHABET[letter]
    if letter in THUMB_CONTACT:
        pose = resolve_contact(pose, geometry, THUMB_CONTACT[letter])
    return build_hand(pose, geometry)


def spread(points: np.ndarray) -> float:
    """Distance between the index and middle fingertips."""
    return float(np.linalg.norm(points[INDEX[2]] - points[MIDDLE[2]]))


class TestTheThumbIsOnTheRightSide:
    """Which side of the fingers the thumb is on, for the letters that turn on it.

    Invisible to the model directly — the normaliser drops ``z`` — but it decides where
    the thumb lands in the 2-D picture, which is all the model has.
    """

    def test_s_clamps_the_thumb_across_the_front(self) -> None:
        points = hand("S")
        assert points[THUMB_TIP, 2] > points[INDEX[1], 2], "S's thumb is behind the index"
        assert points[THUMB_TIP, 2] > points[MIDDLE[1], 2], "S's thumb is behind the middle"

    def test_a_leaves_the_thumb_behind_the_fingers(self) -> None:
        """A is the same fist as S with the thumb *not* crossed over it.

        If both put the thumb in front the two letters are one letter, and the
        classifier's only honest answer is a coin flip.
        """
        points = hand("A")
        assert points[THUMB_TIP, 2] < points[INDEX[1], 2]

    def test_a_and_s_do_not_put_the_thumb_in_the_same_place(self) -> None:
        gap = float(np.linalg.norm(hand("A")[THUMB_TIP] - hand("S")[THUMB_TIP]))
        assert gap > 0.35, f"A and S thumbs are {gap:.2f} palm units apart — too close to read"

    def test_e_curls_the_fingers_down_onto_the_thumb(self) -> None:
        """In E the fingers rest *on* the thumb. Thumb in front instead makes a flat O."""
        points = hand("E")
        assert points[INDEX[2], 2] > points[THUMB_TIP, 2]
        assert points[MIDDLE[2], 2] > points[THUMB_TIP, 2]

    @pytest.mark.parametrize("letter", ["M", "N", "T"])
    def test_the_tucked_letters_bury_the_thumb(self, letter: str) -> None:
        """M, N and T cover the thumb with fingers. That is what makes them not-S."""
        points = hand(letter)
        covering = {"M": [INDEX, MIDDLE, RING], "N": [INDEX, MIDDLE], "T": [INDEX]}[letter]
        for finger in covering:
            assert points[THUMB_TIP, 2] < points[finger[1], 2], (
                f"{letter}'s thumb is in front of a finger that should be covering it"
            )


class TestSeparationsThatSurviveACamera:
    """Cues that hold up when MediaPipe cannot see what the geometry says is there."""

    def test_k_is_narrower_than_v(self) -> None:
        """K's thumb is between two fingers, so a camera loses it. Spread has to carry.

        The v1 pipeline, hand-tuned against a real webcam, separated K from V by
        fingertip distance and never looked at the thumb at all.
        """
        assert spread(hand("K")) < spread(hand("V"))

    def test_k_is_wider_than_u(self) -> None:
        """...but still a V, not two fingers held together."""
        assert spread(hand("K")) > spread(hand("U"))

    def test_p_is_the_k_handshape(self) -> None:
        """P is K aimed at the floor, so the shape must match — only the roll differs."""
        assert spread(hand("P")) == pytest.approx(spread(hand("K")), abs=1e-9)

    def test_q_is_the_g_handshape(self) -> None:
        for landmark in (INDEX[2], MIDDLE[2], THUMB_TIP):
            assert hand("Q")[landmark] == pytest.approx(hand("G")[landmark], abs=0.05)

    def test_r_crosses_its_fingers(self) -> None:
        """The tips end up on opposite sides of where their knuckles started."""
        points = hand("R")
        at_knuckles = points[INDEX[0], 0] - points[MIDDLE[0], 0]
        at_tips = points[INDEX[2], 0] - points[MIDDLE[2], 0]
        assert at_knuckles * at_tips < 0, "R's fingers do not cross"

    @pytest.mark.parametrize("letter", ["U", "V"])
    def test_the_uncrossed_letters_do_not_cross(self, letter: str) -> None:
        points = hand(letter)
        at_knuckles = points[INDEX[0], 0] - points[MIDDLE[0], 0]
        at_tips = points[INDEX[2], 0] - points[MIDDLE[2], 0]
        assert at_knuckles * at_tips > 0

    def test_g_holds_the_thumb_parallel_to_the_index(self) -> None:
        """G is a caliper. L is a right angle. That angle is the whole difference."""
        points = hand("G")
        index = points[INDEX[2]] - points[INDEX[0]]
        thumb = points[THUMB_TIP] - points[2]
        cosine = float(index @ thumb / (np.linalg.norm(index) * np.linalg.norm(thumb)))
        assert np.degrees(np.arccos(np.clip(cosine, -1, 1))) < 30

    def test_l_does_not(self) -> None:
        points = hand("L")
        index = points[INDEX[2]] - points[INDEX[0]]
        thumb = points[THUMB_TIP] - points[2]
        cosine = float(index @ thumb / (np.linalg.norm(index) * np.linalg.norm(thumb)))
        assert np.degrees(np.arccos(np.clip(cosine, -1, 1))) > 45


class TestOrientationBands:
    def test_every_rotation_only_pair_stays_apart(self) -> None:
        for first, second in ROTATION_ONLY_PAIRS:
            gap = _circular_gap(orientation_for(first).roll, orientation_for(second).roll)
            assert gap >= MIN_BAND_SEPARATION, f"{first} and {second} overlap in roll"

    def test_the_shipped_augmentation_is_checked_against_the_bands(self) -> None:
        """The trainer calls this. Here to prove it fails when it should."""
        from mudra_train.features.augment import AugmentConfig

        assert_bands_are_separable(AugmentConfig().max_rotation_degrees)
        with pytest.raises(ValueError, match="one handshape at two orientations"):
            assert_bands_are_separable(90.0)

    @pytest.mark.parametrize("letter", ["G", "H"])
    def test_g_and_h_cover_both_palm_facings(self, letter: str) -> None:
        """The failure that started this: one palm facing trained, the other rejected.

        Measured on the previous pack, which was trained at a single yaw: signing G with
        the palm turned the other way was accepted 0-8% of the time, and read as
        ``none`` otherwise. A signer whose wrist sits differently from the simulator's
        got silence, not a near miss.
        """
        band = orientation_for(letter).yaw
        assert band[0] < -30.0 and band[1] > 60.0, f"{letter}'s yaw band is {band}"

    @pytest.mark.parametrize("letter", ["P", "Q"])
    def test_p_and_q_are_reached_by_turning_the_wrist_down(self, letter: str) -> None:
        """Not by rolling a palm-forward hand upside down, which is nobody's P."""
        assert orientation_for(letter).pitch[1] >= 45.0

    def test_bands_are_wide_enough_to_be_worth_declaring(self) -> None:
        for letter in ALPHABET:
            band = orientation_for(letter)
            for axis, values in (("yaw", band.yaw), ("pitch", band.pitch), ("roll", band.roll)):
                assert values[1] - values[0] >= 20.0, f"{letter}.{axis} spans {values}"


class TestVariation:
    def test_e_has_room_at_both_ends(self) -> None:
        """A loose E and a pressed-tight E are both E, and the previous pack knew one."""
        low, high = variation_for("E").curl
        assert low <= 0.85 and high >= 1.15

    @pytest.mark.parametrize("letter", ["U", "V"])
    def test_u_and_v_are_not_given_room_to_reach_each_other(self, letter: str) -> None:
        """Spread *is* the letter here, so it is the one thing that may not wander."""
        assert variation_for(letter).splay <= 2.0

    def test_r_covers_a_shallow_crossing(self) -> None:
        """What a real R arrives as, once MediaPipe has failed to resolve the overlap."""
        band = variation_for("R").cross
        assert band is not None and band[0] <= 0.2

    def test_k_and_p_vary_together(self) -> None:
        assert variation_for("K") == variation_for("P")
