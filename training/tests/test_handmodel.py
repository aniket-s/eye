"""The kinematic hand, and the alphabet defined on it.

These tests are the specification. A letter of the manual alphabet is defined by which
fingers are extended, how far they are spread, and where the thumb touches — so that is
what is asserted, rather than coordinates that would have to be rewritten every time a
bone length is adjusted.

The pairs that are easy to get wrong get their own tests, because getting them wrong does
not raise: it produces a model that trains cleanly and confuses two letters forever.
"""

from __future__ import annotations

import numpy as np
import pytest

from mudra_train.ingest.asl_alphabet import (
    ALPHABET,
    MOTION_LETTERS,
    THUMB_CONTACT,
    orientation_for,
)
from mudra_train.ingest.handmodel import (
    FINGER_ROOTS,
    LANDMARKS_PER_HAND,
    FingerPose,
    HandGeometry,
    HandPose,
    ThumbPose,
    build_hand,
    project,
    random_geometry,
    solve_thumb_contact,
)

TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
WRIST = 0


def hand(letter: str, geometry: HandGeometry | None = None) -> np.ndarray:
    """Build a letter with its thumb contact resolved, as the generator does."""
    geometry = geometry or HandGeometry()
    pose = ALPHABET[letter]
    if letter in THUMB_CONTACT:
        landmark, distance = THUMB_CONTACT[letter]
        pose = HandPose(
            fingers=pose.fingers,
            thumb=solve_thumb_contact(pose, geometry, landmark, distance),
        )
    return build_hand(pose, geometry)


def extension(points: np.ndarray, finger: int) -> float:
    """How far a fingertip reaches from its own knuckle. Large = extended."""
    root = FINGER_ROOTS[finger]
    return float(np.linalg.norm(points[root + 3] - points[root]))


class TestKinematics:
    def test_produces_the_mediapipe_topology(self) -> None:
        points = build_hand(HandPose(fingers=(FingerPose(),) * 4))
        assert points.shape == (LANDMARKS_PER_HAND, 3)
        assert np.allclose(points[WRIST], 0.0)

    def test_positive_splay_fans_toward_the_thumb(self) -> None:
        """The sign convention the letters are written against.

        Get it backwards and V, W and R spread toward the little finger instead — which
        looks plausible in isolation and quietly makes R indistinguishable from U.
        """
        points = build_hand(
            HandPose(fingers=(FingerPose(splay=25), FingerPose(splay=-25)) + (FingerPose(),) * 2)
        )
        neutral = build_hand(HandPose(fingers=(FingerPose(),) * 4))
        assert points[TIPS["index"], 0] > neutral[TIPS["index"], 0]
        assert points[TIPS["middle"], 0] < neutral[TIPS["middle"], 0]

    def test_flexion_shortens_a_finger_in_space(self) -> None:
        """The whole point of forward kinematics: a curled finger folds, it does not shrink."""
        straight = build_hand(HandPose(fingers=(FingerPose(),) * 4))
        curled = build_hand(
            HandPose(fingers=(FingerPose(mcp=88, pip=100, dip=68),) * 4)
        )
        for finger in range(4):
            assert extension(curled, finger) < extension(straight, finger) * 0.6

    def test_a_closed_fist_puts_the_fingertips_on_the_palm(self) -> None:
        """Not below the wrist, and not out in front — folded onto the palm itself."""
        points = build_hand(HandPose(fingers=(FingerPose(mcp=88, pip=100, dip=68),) * 4))
        for finger in range(4):
            tip = points[FINGER_ROOTS[finger] + 3]
            knuckle = points[FINGER_ROOTS[finger]]
            assert 0.0 < tip[1] < knuckle[1], "fingertip should sit between wrist and knuckles"

    def test_bone_lengths_are_preserved_under_flexion(self) -> None:
        """A joint rotates a segment; it must not stretch it."""
        geometry = HandGeometry()
        relaxed = build_hand(HandPose(fingers=(FingerPose(),) * 4), geometry)
        curled = build_hand(HandPose(fingers=(FingerPose(mcp=60, pip=70, dip=40),) * 4), geometry)
        for finger in range(4):
            root = FINGER_ROOTS[finger]
            for segment in range(3):
                a = float(np.linalg.norm(relaxed[root + segment + 1] - relaxed[root + segment]))
                b = float(np.linalg.norm(curled[root + segment + 1] - curled[root + segment]))
                assert a == pytest.approx(b, rel=1e-9)

    def test_splay_separates_fingertips_without_curling_them(self) -> None:
        together = build_hand(HandPose(fingers=(FingerPose(splay=0),) * 4))
        apart = build_hand(
            HandPose(
                fingers=(
                    FingerPose(splay=18),
                    FingerPose(splay=-18),
                    FingerPose(),
                    FingerPose(),
                )
            )
        )
        gap = lambda p: float(np.linalg.norm(p[TIPS["index"]] - p[TIPS["middle"]]))  # noqa: E731
        assert gap(apart) > gap(together) * 2


class TestProjection:
    def test_returns_image_convention_coordinates(self) -> None:
        """x/y around the frame centre with y growing downward, z relative to the wrist."""
        points = project(hand("B"), centre=(0.5, 0.5))
        assert points[TIPS["middle"], 1] < points[WRIST, 1], "fingers point up the screen"
        assert points[WRIST, 2] == pytest.approx(0.0)

    def test_turning_away_foreshortens(self) -> None:
        """Perspective is signal. A model that never sees it treats a tilt as a new letter.

        Measured across the knuckle line rather than the whole hand: the knuckles lie in
        the plane of the palm, so they compress by cos(yaw). Overall width does not,
        because the thumb stands out of that plane and swings *toward* the camera as the
        hand turns.
        """

        def palm_width(points: np.ndarray) -> float:
            return float(np.linalg.norm(points[5, :2] - points[17, :2]))

        facing = palm_width(project(hand("B"), yaw=0.0))
        turned = palm_width(project(hand("B"), yaw=60.0))
        assert turned == pytest.approx(facing * np.cos(np.deg2rad(60)), rel=0.25)

    def test_left_hands_are_mirrored(self) -> None:
        right = project(hand("L"), hand="right")
        left = project(hand("L"), hand="left")
        assert np.allclose(right[:, 0] - 0.5, -(left[:, 0] - 0.5))
        assert np.allclose(right[:, 1], left[:, 1])

    def test_scale_and_centre_move_the_hand_without_reshaping_it(self) -> None:
        small = project(hand("V"), scale=0.15, centre=(0.3, 0.3))
        large = project(hand("V"), scale=0.30, centre=(0.7, 0.7))

        def shape(points: np.ndarray) -> np.ndarray:
            flat = points[:, :2] - points[:, :2].mean(axis=0)
            return flat / np.sqrt((flat**2).sum(axis=1).mean())

        assert np.allclose(shape(small), shape(large), atol=0.02)


class TestThumbContactSolver:
    @pytest.mark.parametrize("letter", sorted(THUMB_CONTACT))
    def test_reaches_the_declared_contact(self, letter: str) -> None:
        """Tolerance is loose on purpose.

        Some contacts sit at the edge of what the joint limits allow — K asks the thumb to
        press against the middle finger's proximal bone, which it can only approach. The
        assertion that matters is that the solver gets into the right region; a broken
        solver misses by the length of a finger, not by a tenth of a palm.
        """
        landmark, distance = THUMB_CONTACT[letter]
        points = hand(letter)
        reached = float(np.linalg.norm(points[TIPS["thumb"]] - points[landmark]))
        assert reached == pytest.approx(distance, abs=0.12)

    def test_holds_for_an_unusual_hand(self) -> None:
        """The reason contact is solved rather than baked in as angles.

        A thumb angle tuned for average bone lengths silently misses on a longer hand, and
        every simulated signer has different bone lengths.
        """
        rng = np.random.default_rng(7)
        for _ in range(6):
            geometry = random_geometry(rng)
            points = hand("F", geometry)
            gap = float(np.linalg.norm(points[TIPS["thumb"]] - points[TIPS["index"]]))
            assert gap < 0.15, "the ring of F must close on any hand"

    def test_stays_within_anatomical_limits(self) -> None:
        geometry = HandGeometry()
        # An impossible request: the thumb cannot reach the pinky tip of an open hand.
        solved = solve_thumb_contact(ALPHABET["B"], geometry, TIPS["pinky"], 0.0)
        assert -25.0 <= solved.radial <= 80.0
        assert -15.0 <= solved.palmar <= 70.0
        assert 0.0 <= solved.mcp <= 55.0
        assert 0.0 <= solved.ip <= 80.0


class TestAlphabet:
    def test_covers_every_static_letter(self) -> None:
        expected = set("ABCDEFGHIKLMNOPQRSTUVWXY")
        assert set(ALPHABET) == expected

    def test_motion_letters_are_excluded_and_explained(self) -> None:
        """J and Z cannot be one frame. Silently including them would be a lie."""
        assert set(MOTION_LETTERS) == {"J", "Z"}
        assert not set(MOTION_LETTERS) & set(ALPHABET)
        for reason in MOTION_LETTERS.values():
            assert len(reason) > 20

    @pytest.mark.parametrize(
        ("letter", "extended", "closed"),
        [
            ("B", (0, 1, 2, 3), ()),
            ("D", (0,), ()),
            ("I", (3,), (0, 1, 2)),
            ("L", (0,), (1, 2, 3)),
            ("U", (0, 1), (2, 3)),
            ("V", (0, 1), (2, 3)),
            ("W", (0, 1, 2), (3,)),
            ("Y", (3,), (0, 1, 2)),
            ("A", (), (0, 1, 2, 3)),
            ("S", (), (0, 1, 2, 3)),
        ],
    )
    def test_the_right_fingers_are_up(
        self, letter: str, extended: tuple[int, ...], closed: tuple[int, ...]
    ) -> None:
        points = hand(letter)
        for finger in extended:
            assert extension(points, finger) > 1.10, f"{letter}: finger {finger} should be out"
        for finger in closed:
            assert extension(points, finger) < 0.80, f"{letter}: finger {finger} should be closed"

    def test_d_stands_one_finger_above_the_rest(self) -> None:
        """D's other three fingers curl to the thumb rather than closing into a fist, so
        they are shortened but not closed — the assertion is relative, not absolute."""
        points = hand("D")
        for finger in (1, 2, 3):
            assert extension(points, finger) < extension(points, 0) * 0.85

    def test_v_is_u_with_the_fingers_apart(self) -> None:
        """The only difference. If splay ever stops mattering, these two merge."""
        gap = lambda letter: float(  # noqa: E731
            np.linalg.norm(hand(letter)[TIPS["index"]] - hand(letter)[TIPS["middle"]])
        )
        assert gap("V") > gap("U") * 2.5
        assert gap("U") < 0.35, "U's fingers are pressed together"

    def test_r_crosses_its_fingers(self) -> None:
        """In every other two-finger letter the index stays on the thumb side of the middle."""
        crossed = hand("R")
        straight = hand("U")
        assert crossed[TIPS["index"], 0] < crossed[TIPS["middle"], 0]
        assert straight[TIPS["index"], 0] > straight[TIPS["middle"], 0]

    def test_a_and_s_differ_by_where_the_thumb_sits(self) -> None:
        """Same fist. In A the thumb rests alongside; in S it clamps across the front."""
        assert hand("A")[TIPS["thumb"], 0] > hand("S")[TIPS["thumb"], 0] + 0.3

    def test_o_closes_a_ring_and_c_does_not(self) -> None:
        def opening(letter: str) -> float:
            points = hand(letter)
            return float(np.linalg.norm(points[TIPS["thumb"]] - points[TIPS["index"]]))

        assert opening("O") < 0.12
        assert opening("C") > 0.35

    def test_e_is_not_o(self) -> None:
        """They are close in every model and identical in a careless one.

        O closes a ring against the *index*; E's thumb lies across the centre of the palm
        under the *middle* finger. So E's thumb tip sits further round toward the little
        finger. Declaring both as "thumb touches index" collapses them, which is exactly
        what an earlier version of this file did.
        """
        assert hand("E")[TIPS["thumb"], 0] < hand("O")[TIPS["thumb"], 0] - 0.15

    def test_m_n_and_t_bury_the_thumb_progressively_further_round(self) -> None:
        """Three fingers over it, then two, then one — the thumb tip moves ulnar-ward."""
        thumb_x = {letter: hand(letter)[TIPS["thumb"], 0] for letter in ("T", "N", "M")}
        assert thumb_x["T"] > thumb_x["N"] > thumb_x["M"]

    def test_k_and_p_are_one_handshape_at_two_orientations(self) -> None:
        """Which is precisely why the normaliser must not cancel rotation."""
        assert np.allclose(hand("K"), hand("P"))
        assert orientation_for("K") != orientation_for("P")

    def test_g_and_q_are_one_handshape_at_two_orientations(self) -> None:
        assert orientation_for("G") != orientation_for("Q")
        assert orientation_for("G") != orientation_for("A")

    def test_most_letters_share_a_default_orientation(self) -> None:
        rotated = {"G", "H", "P", "Q"}
        for letter in ALPHABET:
            if letter not in rotated:
                assert orientation_for(letter) == orientation_for("A")
