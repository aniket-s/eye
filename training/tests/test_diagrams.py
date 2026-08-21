"""The dictionary must illustrate the letter the classifier was trained on.

This is the guard on a failure that had nothing to do with the model and looked exactly
like a model failure.

The diagrams were generated from ``ingest/synthetic.py`` — the crude fixture generator,
which has no thumb opposition, no splay, no crossing and no 3-D projection. The
classifier was trained from ``ingest/asl_alphabet.py`` through ``ingest/handmodel.py``.
Two definitions of the alphabet, maintained independently, and they diverged:

* G and H were drawn pointing **169° and 160°** from where the classifier expected them.
* R was drawn with two straight parallel fingers, which is a picture of U.
* K was drawn with the thumb beside the fist rather than between the fingers.
* E and S were drawn as near-identical fists.

Measured on the shipped pack: a G held as the diagram illustrated it was accepted **0%**
of the time and read as ``none`` in every frame. Not a wrong letter — nothing at all.
A user following the app's own teaching aid could not sign G, and no amount of work on
the recogniser would have fixed it.

So the assertion is not "the diagrams look right". It is that a reader who copies a
diagram produces geometry the model has seen: one hand model, one pose table, one
orientation band, checked here.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mudra_train.export import diagrams
from mudra_train.ingest.asl_alphabet import ALPHABET, MOTION_LETTERS, orientation_for
from mudra_train.ingest.asl_numbers import NUMBER_ALIASES, NUMBER_SHAPES

INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
THUMB_TIP = 4


def bearing(points: np.ndarray, start: int, end: int) -> float:
    """Compass bearing of ``start -> end`` in image space: 0 up, +90 image-right."""
    dx = points[end, 0] - points[start, 0]
    dy = points[end, 1] - points[start, 1]  # image y grows downward
    return math.degrees(math.atan2(dx, -dy))


class TestEveryDiagramIsAHandTheModelHasSeen:
    """The invariant that broke. Everything else here is a consequence of it."""

    @pytest.mark.parametrize("letter", sorted(ALPHABET))
    def test_the_viewpoint_is_inside_the_letters_band(self, letter: str) -> None:
        yaw, pitch, roll = diagrams.diagram_orientation(letter)
        band = orientation_for(letter)
        assert band.yaw[0] <= yaw <= band.yaw[1]
        assert band.pitch[0] <= pitch <= band.pitch[1]
        assert band.roll[0] <= roll <= band.roll[1]

    @pytest.mark.parametrize("letter", sorted(ALPHABET))
    def test_the_pose_is_the_training_pose(self, letter: str) -> None:
        """Not merely similar: the same table, resolved by the same solver."""
        from mudra_train.ingest.asl_alphabet import THUMB_CONTACT
        from mudra_train.ingest.handmodel import (
            HandGeometry,
            build_hand,
            project,
            resolve_contact,
        )

        geometry = HandGeometry()
        pose = ALPHABET[letter]
        if letter in THUMB_CONTACT:
            pose = resolve_contact(pose, geometry, THUMB_CONTACT[letter])
        yaw, pitch, roll = diagrams.diagram_orientation(letter)
        expected = project(
            build_hand(pose, geometry),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            distance=diagrams._DIAGRAM_DISTANCE,
            scale=diagrams._DIAGRAM_SCALE,
            hand="right",
        )
        assert np.allclose(diagrams.landmarks_for(letter), expected)


class TestTheLettersThatWereDrawnWrong:
    """One case each, phrased as the reader sees it rather than as an angle."""

    @pytest.mark.parametrize("letter", ["G", "H"])
    def test_g_and_h_point_sideways(self, letter: str) -> None:
        """They were drawn pointing the opposite way, and it cost the letter entirely."""
        heading = bearing(diagrams.landmarks_for(letter), INDEX_MCP, INDEX_TIP)
        assert 45.0 < abs(heading) < 135.0, f"{letter} points at {heading:.0f}°, not sideways"

    @pytest.mark.parametrize("letter", ["G", "H", "P", "Q"])
    def test_the_direction_matches_what_training_generates(self, letter: str) -> None:
        """The direct guard on the failure. For these four, direction *is* the letter.

        Drawn against sampled training hands rather than against a hard-coded angle: if
        the diagram and the trainer ever disagree about which way a letter points, this
        is what says so, whichever of them moved.
        """
        from mudra_train.ingest.handmodel import HandGeometry, build_hand, project
        from mudra_train.ingest.simulated import _resolve_pose, make_signers

        rng = np.random.default_rng(11)
        signers = make_signers(4, rng)
        trained = []
        for signer in signers:
            band = orientation_for(letter)
            for _ in range(8):
                pose = _resolve_pose(letter, signer, rng)
                yaw = float(rng.uniform(*band.yaw))
                pitch = float(rng.uniform(*band.pitch))
                roll = float(rng.uniform(*band.roll))
                points = project(
                    build_hand(pose, signer.geometry), yaw=yaw, pitch=pitch, roll=roll
                )
                trained.append(bearing(points, INDEX_MCP, INDEX_TIP))

        drawn = bearing(diagrams.landmarks_for(letter), INDEX_MCP, INDEX_TIP)
        offsets = [abs((angle - drawn + 180.0) % 360.0 - 180.0) for angle in trained]
        assert float(np.median(offsets)) < 45.0, (
            f"{letter} is drawn pointing at {drawn:.0f}°, a median {np.median(offsets):.0f}° "
            f"from where training puts it. This is exactly how G and H ended up drawn "
            f"backwards."
        )

    @pytest.mark.parametrize("letter", ["P", "Q"])
    def test_p_and_q_point_downward(self, letter: str) -> None:
        heading = abs(bearing(diagrams.landmarks_for(letter), INDEX_MCP, INDEX_TIP))
        assert heading > 110.0, f"{letter} points at {heading:.0f}°, not down"

    def test_r_crosses_its_fingers(self) -> None:
        """It was drawn as U — the old generator could not express crossing at all."""
        points = diagrams.landmarks_for("R")
        at_knuckles = points[INDEX_MCP, 0] - points[MIDDLE_MCP, 0]
        at_tips = points[INDEX_TIP, 0] - points[MIDDLE_TIP, 0]
        assert at_knuckles * at_tips < 0

    def test_k_wedges_the_thumb_between_the_two_fingers(self) -> None:
        """It was drawn hanging down beside the fist, which is the letter thrown away."""
        points = diagrams.landmarks_for("K")
        low, high = sorted((points[INDEX_TIP, 0], points[MIDDLE_TIP, 0]))
        assert low < points[THUMB_TIP, 0] < high, "K's thumb is outside the V, not in it"

    def test_ks_thumb_reaches_the_middle_finger_and_vs_does_not(self) -> None:
        """K and V are the same two fingers. Where the thumb goes is the difference."""
        MIDDLE_PIP = 10
        reach = lambda letter: float(  # noqa: E731 — a name would be longer than the body
            np.linalg.norm(
                diagrams.landmarks_for(letter)[THUMB_TIP, :2]
                - diagrams.landmarks_for(letter)[MIDDLE_PIP, :2]
            )
        )
        assert reach("K") < reach("V") * 0.75

    def test_e_and_s_are_not_the_same_picture(self) -> None:
        """They differed by a curl of 0.8 against 1.0 and drew indistinguishably."""
        assert _picture_distance("E", "S") > 1.5

    def test_a_and_s_are_not_the_same_picture(self) -> None:
        """One fist with the thumb across the front, one with it up the side."""
        assert _picture_distance("A", "S") > 1.5

    @pytest.mark.parametrize("pair", [("M", "N"), ("N", "T"), ("U", "V"), ("K", "V")])
    def test_the_close_pairs_stay_distinguishable(self, pair: tuple[str, str]) -> None:
        """The threshold is set just under the tightest real pair, K and V at 0.67.

        Low on purpose. These letters *are* nearly the same picture — that is a fact
        about ASL, not a defect — so this catches a diagram collapsing onto its
        neighbour without pretending they should look unalike.
        """
        assert _picture_distance(*pair) > 0.5


def _picture_distance(first: str, second: str) -> float:
    """How different two diagrams are, after the size and position a reader ignores.

    Measured in the classifier's own feature space, so the number means the same thing
    to this test as it does to the model.
    """
    from mudra_train.features.normalise import normalise_hand

    one = normalise_hand(diagrams.landmarks_for(first))
    two = normalise_hand(diagrams.landmarks_for(second))
    return float(np.linalg.norm(one - two))


class TestTheMotionLetters:
    @pytest.mark.parametrize("letter", sorted(MOTION_LETTERS))
    def test_the_card_shows_a_path_and_not_just_a_handshape(self, letter: str) -> None:
        """J's card used to be a picture of I, and Z's a picture of D.

        Nothing on either said the letter moves, which is the only thing that makes it
        that letter.
        """
        base, _, _ = diagrams._MOTION_PATHS[letter]
        svg = diagrams.render_svg(base, motion=letter)
        assert "stroke-dasharray" in svg, f"{letter} has no motion path"
        assert "<polygon" in svg, f"{letter}'s path has no direction"
        assert f">{letter}</text>" in svg, f"{letter}'s card is labelled as its base shape"

    def test_j_traces_the_pinky_and_z_the_index(self) -> None:
        assert diagrams._MOTION_PATHS["J"][1] == 20
        assert diagrams._MOTION_PATHS["Z"][1] == 8


class TestGeneratedFiles:
    def test_every_letter_and_digit_gets_a_card(self, tmp_path: Path) -> None:
        written = diagrams.generate_all(tmp_path)
        letters = {path.stem.upper() for path in (tmp_path / "alphabets").glob("*.svg")}
        assert letters == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        digits = {path.stem for path in (tmp_path / "numbers").glob("*.svg")}
        assert digits == set(NUMBER_ALIASES) | set(NUMBER_SHAPES) | {"10"}
        assert written == len(letters) + len(digits)

    def test_cards_are_self_contained_svg(self, tmp_path: Path) -> None:
        diagrams.generate_all(tmp_path)
        for path in (tmp_path / "alphabets").glob("*.svg"):
            body = path.read_text(encoding="utf-8")
            assert body.startswith("<svg") and body.rstrip().endswith("</svg>")
            assert "http" not in body.replace("http://www.w3.org/2000/svg", "")

    def test_cards_name_the_letter_for_a_screen_reader(self, tmp_path: Path) -> None:
        diagrams.generate_all(tmp_path)
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            body = (tmp_path / "alphabets" / f"{letter.lower()}.svg").read_text(encoding="utf-8")
            assert f'aria-label="ASL handshape for {letter}' in body
