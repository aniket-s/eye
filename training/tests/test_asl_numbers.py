"""ASL numbers, and the four that are already letters.

The point of this module is a linguistic fact with an engineering consequence, so that is
what these tests pin: 0/O, 2/V, 6/W and 9/F are *the same handshape*, and training them as
separate classes would ask the model to split classes carrying no distinguishing signal —
making it worse at letters in exchange for a capability that cannot work.
"""

from __future__ import annotations

import numpy as np
import pytest

from mudra_train.export.pack import build_vocabularies
from mudra_train.ingest.asl_alphabet import ALPHABET
from mudra_train.ingest.asl_numbers import (
    NUMBER_ALIASES,
    NUMBER_CONTACT,
    NUMBER_SHAPES,
    NUMBER_VOCABULARY,
)
from mudra_train.ingest.handmodel import (
    FINGER_ROOTS,
    HandGeometry,
    HandPose,
    build_hand,
    solve_thumb_contact,
)
from mudra_train.ingest.simulated import CONTACTS, SHAPES

TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def hand(label: str) -> np.ndarray:
    geometry = HandGeometry()
    pose = SHAPES[label]
    if label in CONTACTS:
        landmark, distance = CONTACTS[label]
        pose = HandPose(
            fingers=pose.fingers,
            thumb=solve_thumb_contact(pose, geometry, landmark, distance),
        )
    return build_hand(pose, geometry)


def extension(points: np.ndarray, finger: int) -> float:
    root = FINGER_ROOTS[finger]
    return float(np.linalg.norm(points[root + 3] - points[root]))


class TestAliases:
    def test_the_four_shared_shapes_are_declared_not_retrained(self) -> None:
        assert NUMBER_ALIASES == {"0": "O", "2": "V", "6": "W", "9": "F"}

    def test_no_alias_gets_its_own_handshape(self) -> None:
        """The whole point. A separate class for 2 would split V's probability mass."""
        assert not set(NUMBER_ALIASES) & set(NUMBER_SHAPES)

    def test_every_alias_targets_a_real_letter(self) -> None:
        for digit, letter in NUMBER_ALIASES.items():
            assert letter in ALPHABET, f"{digit} aliases {letter}, which is not a letter"

    def test_all_ten_digits_are_covered(self) -> None:
        assert set(NUMBER_ALIASES) | set(NUMBER_SHAPES) == set("0123456789")

    def test_the_vocabulary_is_derived_from_both_halves(self) -> None:
        """Built rather than written out, so the two cannot drift apart."""
        assert len(NUMBER_VOCABULARY) == 10
        assert NUMBER_VOCABULARY["V"] == "2"
        assert NUMBER_VOCABULARY["7"] == "7"


class TestShapes:
    def test_only_six_digits_need_their_own_class(self) -> None:
        assert set(NUMBER_SHAPES) == {"1", "3", "4", "5", "7", "8"}

    def test_they_join_the_alphabet_without_colliding(self) -> None:
        assert len(SHAPES) == len(ALPHABET) + len(NUMBER_SHAPES)

    @pytest.mark.parametrize(
        ("digit", "extended", "closed"),
        [
            ("1", (0,), (1, 2, 3)),
            ("4", (0, 1, 2, 3), ()),
            ("5", (0, 1, 2, 3), ()),
        ],
    )
    def test_the_right_fingers_are_up(
        self, digit: str, extended: tuple[int, ...], closed: tuple[int, ...]
    ) -> None:
        points = hand(digit)
        for finger in extended:
            assert extension(points, finger) > 1.10
        for finger in closed:
            assert extension(points, finger) < 0.80

    def test_four_is_five_with_the_thumb_folded_in(self) -> None:
        """Their only difference, and the same distinction that separates B from 4."""
        assert hand("5")[TIPS["thumb"], 0] > hand("4")[TIPS["thumb"], 0] + 0.5

    def test_four_spreads_its_fingers_and_b_does_not(self) -> None:
        def spread(label: str) -> float:
            points = hand(label)
            return float(np.linalg.norm(points[TIPS["index"]] - points[TIPS["pinky"]]))

        assert spread("4") > spread("B") * 1.3

    def test_seven_and_eight_touch_different_fingers_to_the_thumb(self) -> None:
        """6, 7, 8 and 9 are one gesture walking a finger along the hand, which is why
        they are declared as contacts rather than as angles."""
        for digit, tip in (("7", TIPS["ring"]), ("8", TIPS["middle"])):
            points = hand(digit)
            gap = float(np.linalg.norm(points[TIPS["thumb"]] - points[tip]))
            assert gap < 0.15, f"{digit}: the thumb should meet that fingertip"

    def test_the_contacts_are_registered_with_the_generator(self) -> None:
        for digit in NUMBER_CONTACT:
            assert digit in CONTACTS

    def test_three_puts_the_thumb_out_and_k_does_not(self) -> None:
        assert hand("3")[TIPS["thumb"], 0] > hand("K")[TIPS["thumb"], 0] + 0.3


class TestVocabularyExport:
    def test_builds_both_readings_for_a_pack_with_digits(self) -> None:
        vocabularies = build_vocabularies(sorted(SHAPES) + ["none"])
        assert set(vocabularies) == {"letters", "numbers"}
        assert len(vocabularies["numbers"]) == 10
        assert vocabularies["numbers"]["V"] == "2"
        assert "none" not in vocabularies["letters"]

    def test_letters_reading_covers_every_letter(self) -> None:
        vocabularies = build_vocabularies(sorted(SHAPES) + ["none"])
        assert set(vocabularies["letters"]) == set(ALPHABET)

    def test_emits_nothing_for_a_letters_only_pack(self) -> None:
        """There is no mode to offer, so the switch should not appear at all."""
        assert build_vocabularies(sorted(ALPHABET) + ["none"]) == {}
