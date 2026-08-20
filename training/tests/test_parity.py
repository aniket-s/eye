"""Cross-language parity: Python normalisation must match TypeScript exactly.

If these two drift apart, the model is trained on one feature distribution and
served another. Offline metrics stay healthy while real-world accuracy quietly
degrades — the hardest class of ML bug to notice, because nothing errors.

The fixtures are generated from the TypeScript implementation by
``npm run fixtures:parity``. Changing either normaliser without regenerating them
fails this test, which is the point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mudra_train.features.normalise import (
    FEATURE_LENGTH,
    NORMALISATION_SCHEME,
    normalise_batch,
    normalise_hand,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "normalisation.json"

#: Tolerance. TypeScript computes in float64 and stores float32; the residual
#: difference is float32 rounding, ~1e-7.
TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def fixtures() -> dict:
    if not FIXTURES.exists():
        pytest.fail(
            f"{FIXTURES} is missing. Run `npm run fixtures:parity` from the repo root."
        )
    return json.loads(FIXTURES.read_text())


def test_scheme_matches(fixtures: dict) -> None:
    """The fixtures must describe the scheme this module implements."""
    assert fixtures["scheme"] == NORMALISATION_SCHEME


def test_every_case_matches_typescript(fixtures: dict) -> None:
    """Every fixture reproduces the TypeScript output."""
    failures = []
    for case in fixtures["cases"]:
        actual = normalise_hand(np.array(case["landmarks"]), case["handedness"])
        expected = np.array(case["expected"], dtype=np.float32)

        assert actual.shape == (FEATURE_LENGTH,), case["name"]
        delta = float(np.max(np.abs(actual - expected)))
        if delta > TOLERANCE:
            failures.append(f"{case['name']}: max delta {delta:.3e}")

    assert not failures, "Python and TypeScript normalisation disagree:\n" + "\n".join(failures)


def test_batch_matches_single(fixtures: dict) -> None:
    """The vectorised path must agree with the per-sample path.

    Training uses ``normalise_batch`` for speed while the parity fixtures exercise
    ``normalise_hand``; without this, the fast path could drift unchecked.
    """
    cases = fixtures["cases"]
    landmarks = np.array([c["landmarks"] for c in cases])
    handedness = np.array([c["handedness"] for c in cases])

    batched = normalise_batch(landmarks, handedness)
    for i, case in enumerate(cases):
        single = normalise_hand(np.array(case["landmarks"]), case["handedness"])
        assert np.max(np.abs(batched[i] - single)) < TOLERANCE, case["name"]


def test_handedness_canonicalisation() -> None:
    """A left hand and its mirror image normalise identically.

    This is the fix for the defect that made left-handed signers fail outright.
    """
    rng = np.random.default_rng(7)
    right = rng.uniform(0.2, 0.8, size=(21, 3))
    left = right.copy()
    left[:, 0] = -left[:, 0]

    assert np.max(np.abs(normalise_hand(right, "right") - normalise_hand(left, "left"))) < TOLERANCE


def test_translation_and_scale_invariance() -> None:
    rng = np.random.default_rng(11)
    hand = rng.uniform(0.2, 0.8, size=(21, 3))
    reference = normalise_hand(hand, "right")

    moved = hand + np.array([0.15, -0.2, 0.0])
    assert np.max(np.abs(normalise_hand(moved, "right") - reference)) < TOLERANCE

    centre = hand.mean(axis=0)
    zoomed = centre + (hand - centre) * 2.5
    assert np.max(np.abs(normalise_hand(zoomed, "right") - reference)) < TOLERANCE


def test_rotation_is_preserved() -> None:
    """Rotation must NOT be normalised away.

    K/P, G/Q and H/U are the same handshape at different orientations. Canonicalising
    rotation would merge each pair into a single class.
    """
    rng = np.random.default_rng(13)
    hand = rng.uniform(0.2, 0.8, size=(21, 3))
    centre = hand.mean(axis=0)

    angle = np.pi / 2
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = hand.copy()
    rotated[:, :2] = (hand[:, :2] - centre[:2]) @ rotation.T + centre[:2]

    delta = np.max(np.abs(normalise_hand(rotated, "right") - normalise_hand(hand, "right")))
    assert delta > 0.5, "rotation was collapsed; K/P and G/Q would become unreadable"


def test_z_is_ignored() -> None:
    rng = np.random.default_rng(17)
    hand = rng.uniform(0.2, 0.8, size=(21, 3))
    noisy = hand.copy()
    noisy[:, 2] += rng.normal(0, 0.5, size=21)

    assert np.array_equal(normalise_hand(hand, "right"), normalise_hand(noisy, "right"))


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        normalise_hand(np.zeros((20, 3)), "right")
    with pytest.raises(ValueError):
        normalise_hand(np.zeros((21, 3)), "sideways")
    with pytest.raises(ValueError):
        normalise_batch(np.zeros((2, 20, 3)), np.array(["right", "right"]))
