"""The simulated dataset.

The generator's job is not merely to produce arrays of the right shape — it is to produce
*variation*, since a model trained on an unvarying hand learns that hand rather than the
alphabet. So most of these tests measure whether things actually differ: between signers,
between repetitions, between viewpoints.

The one that matters most is
:meth:`TestSigners.test_signers_are_anatomically_distinct`. If simulated signers were
merely scaled copies of one another, the signer-independent split would report a number
that looked like generalisation and measured nothing at all — which is the exact failure
`docs/AUDIT.md` A3 is about.
"""

from __future__ import annotations

import numpy as np
import pytest

from mudra_train.features.normalise import normalise_hand
from mudra_train.ingest.asl_alphabet import ALPHABET, MOTION_LETTERS
from mudra_train.ingest.asl_numbers import NUMBER_ALIASES, NUMBER_SHAPES
from mudra_train.ingest.simulated import (
    LEFT_HAND_FRACTION,
    NONE_FRACTION,
    SHAPES,
    generate,
    make_signers,
)


@pytest.fixture(scope="module")
def dataset():
    """Small but structurally complete: every letter, every signer, both hands."""
    return generate(samples_per_class=48, signers=6, seed=11)


class TestDataset:
    def test_covers_every_handshape_plus_none(self, dataset) -> None:
        assert set(dataset.labels) == set(SHAPES) | {"none"}

    def test_covers_the_letters_and_the_six_distinct_digits(self, dataset) -> None:
        """0, 2, 6 and 9 are absent on purpose — they are O, V, W and F, and a separate
        class for each would split probability mass between identical shapes."""
        assert set(ALPHABET) <= set(dataset.labels)
        assert set(NUMBER_SHAPES) <= set(dataset.labels)
        assert not set(NUMBER_ALIASES) & set(dataset.labels)

    def test_excludes_the_motion_letters(self, dataset) -> None:
        """J and Z cannot be a single frame; a static pack must not claim them."""
        assert not set(dataset.labels) & set(MOTION_LETTERS)

    def test_landmarks_have_the_mediapipe_shape(self, dataset) -> None:
        assert dataset.landmarks.shape[1:] == (21, 3)
        assert np.isfinite(dataset.landmarks).all()

    def test_every_array_is_the_same_length(self, dataset) -> None:
        n = len(dataset)
        for array in (dataset.labels, dataset.signers, dataset.hands, dataset.conditions):
            assert len(array) == n

    def test_letters_are_balanced(self, dataset) -> None:
        counts = {label: int((dataset.labels == label).sum()) for label in SHAPES}
        assert len(set(counts.values())) == 1, "an imbalanced letter would train weaker"

    def test_none_is_well_represented(self, dataset) -> None:
        """`none` has to cover far more of pose space than any one letter, so it needs
        more examples rather than an equal share."""
        share = float((dataset.labels == "none").mean())
        expected = NONE_FRACTION / (1 + NONE_FRACTION)
        assert share == pytest.approx(expected, rel=0.15)

    def test_generates_both_hands(self, dataset) -> None:
        left = float((dataset.hands == "left").mean())
        assert left == pytest.approx(LEFT_HAND_FRACTION, abs=0.06)

    def test_is_deterministic_for_a_given_seed(self) -> None:
        """Reproducibility is what makes a metric comparable between two runs."""
        a = generate(samples_per_class=24, signers=5, seed=3)
        b = generate(samples_per_class=24, signers=5, seed=3)
        assert np.array_equal(a.landmarks, b.landmarks)
        assert np.array_equal(a.labels, b.labels)

    def test_a_different_seed_gives_different_data(self) -> None:
        a = generate(samples_per_class=24, signers=5, seed=3)
        b = generate(samples_per_class=24, signers=5, seed=4)
        assert not np.array_equal(a.landmarks, b.landmarks)

    def test_refuses_too_few_signers(self) -> None:
        """Below a handful there is not enough anatomy to hold any of it out."""
        with pytest.raises(ValueError, match="signers"):
            generate(samples_per_class=24, signers=2)


class TestSigners:
    def test_signers_are_anatomically_distinct(self) -> None:
        """The load-bearing property.

        A signer-independent split only means something if the held-out signers really are
        different people. Bone *ratios* are what must differ — overall size is normalised
        away, so scaling one hand up would produce a "new signer" the model finds
        identical, and an evaluation score that measures nothing.
        """
        rng = np.random.default_rng(5)
        people = make_signers(8, rng)
        ratios = [
            np.array(person.geometry.phalanges).ravel() / person.geometry.phalanges[1][0]
            for person in people
        ]
        distances = [
            float(np.abs(a - b).mean()) for i, a in enumerate(ratios) for b in ratios[i + 1 :]
        ]
        assert min(distances) > 0.01, "two simulated signers have near-identical proportions"

    def test_signers_have_distinct_names(self) -> None:
        people = make_signers(10, np.random.default_rng(1))
        assert len({person.name for person in people}) == 10

    def test_signer_traits_stay_in_plausible_ranges(self) -> None:
        for person in make_signers(30, np.random.default_rng(2)):
            assert 0.7 < person.articulation < 1.2, "nobody under- or over-articulates that far"
            assert person.noise > 0
            assert person.variability > 0

    def test_each_signer_appears_in_the_data(self, dataset) -> None:
        assert len(set(dataset.signers)) == 6


class TestVariation:
    def test_repetitions_of_a_letter_differ(self, dataset) -> None:
        """Identical repetitions would let the model memorise instead of generalise."""
        mask = (dataset.labels == "B") & (dataset.signers == dataset.signers[0])
        samples = dataset.landmarks[mask]
        assert len(samples) > 3
        assert not np.allclose(samples[0], samples[1])

    def test_the_same_letter_still_normalises_to_a_similar_shape(self, dataset) -> None:
        """Variation must not be so wild that a letter stops being one thing.

        Within-letter spread has to stay below between-letter spread, or the class is
        noise. This is the check that the randomisation ranges have not been set to
        something that merely destroys the signal.
        """

        def features(label: str) -> np.ndarray:
            mask = dataset.labels == label
            points = dataset.landmarks[mask]
            hands = dataset.hands[mask]
            return np.stack(
                [normalise_hand(p, h) for p, h in zip(points[:40], hands[:40], strict=False)]
            )

        same = features("L")
        other = features("B")
        within = float(np.linalg.norm(same - same.mean(axis=0), axis=1).mean())
        between = float(np.linalg.norm(same.mean(axis=0) - other.mean(axis=0)))
        assert within < between, "L varies more within itself than it differs from B"

    def test_viewpoint_is_randomised(self, dataset) -> None:
        """Every sample from the same viewpoint teaches the model that letters are 2-D."""
        mask = dataset.labels == "B"
        points = dataset.landmarks[mask]
        # Palm width relative to hand height varies only if the hand is being turned.
        width = np.linalg.norm(points[:, 5, :2] - points[:, 17, :2], axis=1)
        height = np.linalg.norm(points[:, 12, :2] - points[:, 0, :2], axis=1)
        assert float((width / height).std()) > 0.03

    def test_apparent_size_and_position_vary(self, dataset) -> None:
        centres = dataset.landmarks[:, :, :2].mean(axis=1)
        assert centres[:, 0].std() > 0.05
        assert centres[:, 1].std() > 0.05

    def test_conditions_are_labelled_for_slice_reporting(self, dataset) -> None:
        """Per-slice metrics are how a subgroup failure is caught, so the slices have to
        exist in the data."""
        assert {"normal", "angled", "transition", "idle"} <= set(dataset.conditions)


class TestNegatives:
    def test_none_covers_both_transitions_and_idle_poses(self, dataset) -> None:
        """Blends alone would teach the model that `none` means *between two letters*,
        when most of what a camera sees is a hand doing something else entirely."""
        none = dataset.conditions[dataset.labels == "none"]
        assert (none == "transition").sum() > 0
        assert (none == "idle").sum() > 0

    def test_none_is_spread_more_widely_than_a_letter(self, dataset) -> None:
        def spread(label: str) -> float:
            mask = dataset.labels == label
            points, hands = dataset.landmarks[mask], dataset.hands[mask]
            features = np.stack(
                [normalise_hand(p, h) for p, h in zip(points[:120], hands[:120], strict=False)]
            )
            return float(np.linalg.norm(features - features.mean(axis=0), axis=1).mean())

        assert spread("none") > spread("L")
