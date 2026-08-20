"""The model pack committed to the repository.

Every other test exercises code. This one exercises the *artefact* — the file a user
actually gets — because the two can disagree. A pack can be exported from a healthy
pipeline and still be wrong, and the way it goes wrong is silent.

The orientation check is the reason this file exists. A pack trained on mirrored data
scores beautifully against its own mirrored evaluation set and then rejects almost every
real hand, which is exactly what shipped once. Asserting the *asymmetry* — that the pack
does far better on correctly-oriented landmarks than on mirrored ones — pins the
convention to the weights rather than to a comment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mudra_train.features.normalise import FEATURE_LENGTH, normalise_hand
from mudra_train.ingest.simulated import generate

PACK = Path(__file__).resolve().parents[2] / "packages/web/public/models/asl-fingerspell"

pytestmark = pytest.mark.skipif(
    not (PACK / "model.onnx").is_file(),
    reason="no pack committed; run `npm run train:simulated`",
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((PACK / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def session():
    import onnxruntime as ort

    return ort.InferenceSession(str(PACK / "model.onnx"))


@pytest.fixture(scope="module")
def held_out():
    """Data the pack never saw: a different seed, so different anatomy and different noise."""
    return generate(samples_per_class=60, signers=6, seed=99)


def predict(session, manifest, dataset, transform=None) -> np.ndarray:
    labels = np.array(manifest["labels"])
    features = np.stack(
        [
            normalise_hand(transform(points.copy()) if transform else points, hand)
            for points, hand in zip(dataset.landmarks, dataset.hands, strict=True)
        ]
    ).astype(np.float32)
    logits = session.run(["logits"], {"features": features})[0]
    return labels[logits.argmax(axis=1)]


def mirror(points: np.ndarray) -> np.ndarray:
    """Flip about the image's vertical axis, as a mirrored preview would."""
    return np.stack([1.0 - points[:, 0], points[:, 1], points[:, 2]], axis=1)


class TestManifest:
    def test_declares_the_scheme_the_runtime_implements(self, manifest) -> None:
        assert manifest["input"]["normalisation"] == "centroid-rms-v2"
        assert manifest["input"]["featureLength"] == FEATURE_LENGTH

    def test_excludes_the_motion_letters(self, manifest) -> None:
        assert "J" not in manifest["labels"]
        assert "Z" not in manifest["labels"]

    def test_reads_its_handshapes_as_letters_or_numbers(self, manifest) -> None:
        """And does *not* train the four digits that are already letters.

        0/O, 2/V, 6/W and 9/F are one handshape each. Separate classes would split their
        probability mass and make the model worse at letters.
        """
        vocabularies = manifest["vocabularies"]
        assert set(vocabularies) == {"letters", "numbers"}
        assert sorted(vocabularies["numbers"].values()) == list("0123456789")
        for digit in ("0", "2", "6", "9"):
            assert digit not in manifest["labels"]
        assert vocabularies["numbers"]["V"] == "2"

    def test_keeps_a_negative_class(self, manifest) -> None:
        """Without it every transition between letters is force-classified as a letter."""
        assert manifest["labels"][-1] == "none"

    def test_was_evaluated_on_held_out_signers(self, manifest) -> None:
        assert manifest["metrics"]["testSigners"] >= 2

    def test_is_labelled_as_simulated(self, manifest) -> None:
        """Provenance travels with the weights, not only in the README."""
        assert "simulated" in manifest["name"].lower()
        assert "simulated" in manifest["licence"].lower()

    def test_weights_match_the_declared_checksum(self, manifest) -> None:
        import hashlib

        digest = hashlib.sha256((PACK / manifest["modelFile"]).read_bytes()).hexdigest()
        assert digest == manifest["sha256"]


class TestBehaviour:
    def test_generalises_to_signers_it_never_saw(self, session, manifest, held_out) -> None:
        accuracy = float((predict(session, manifest, held_out) == held_out.labels).mean())
        assert accuracy > 0.85, f"the shipped pack only scores {accuracy:.3f}"

    def test_is_oriented_for_a_raw_camera_frame(self, session, manifest, held_out) -> None:
        """The bug this file exists for.

        MediaPipe reports raw camera coordinates, in which a right hand's thumb sits left
        of its pinky. A pack trained on the mirror image is internally consistent, scores
        0.97 on its own evaluation, and rejects most real hands — so the only way to catch
        it is to check the pack is *asymmetric in the right direction*.
        """
        upright = float((predict(session, manifest, held_out) == held_out.labels).mean())
        flipped = float((predict(session, manifest, held_out, mirror) == held_out.labels).mean())
        assert upright > flipped * 2, (
            f"the pack scores {upright:.3f} upright and {flipped:.3f} mirrored — if those "
            "are close the orientation convention has been lost; if flipped is higher the "
            "pack is trained backwards and will reject nearly every real hand"
        )

    def test_rejects_almost_nothing_it_should_recognise(self, session, manifest, held_out) -> None:
        """A pack that quietly answers `none` looks identical to a broken pipeline."""
        predicted = predict(session, manifest, held_out)
        letters = held_out.labels != "none"
        rejected = float((predicted[letters] == "none").mean())
        assert rejected < 0.05, f"{rejected:.1%} of letters rejected as `none`"

    def test_exports_a_unit_norm_embedding_for_custom_signs(self, session) -> None:
        outputs = [output.name for output in session.get_outputs()]
        assert "embedding" in outputs, "custom signs need the ArcFace embedding head"

        features = np.zeros((1, FEATURE_LENGTH), dtype=np.float32)
        embedding = session.run(["embedding"], {"features": features})[0]
        assert float(np.linalg.norm(embedding)) == pytest.approx(1.0, abs=1e-3)
