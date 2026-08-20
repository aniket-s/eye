"""Load **FSboard** — 3M+ characters of ASL fingerspelling from 147 Deaf signers.

FSboard (Tanzer et al., CVPR 2025, CC BY 4.0) is the best openly-licensed fingerspelling
corpus available, and training on it is the intended upgrade path away from the simulated
pack this repository ships. It lives on Kaggle, so it has to be downloaded by hand:

.. code-block:: bash

    pip install kaggle                       # then put your API token in ~/.kaggle/
    kaggle datasets download -d google/fsboard
    unzip fsboard.zip -d ./fsboard

    # Look before you leap — this prints the schema it found and checks it can read it.
    python -m mudra_train.ingest.fsboard --root ./fsboard --inspect

    python -m mudra_train.train_ctc --fsboard ./fsboard --out ../packages/web/public/models/asl-fingerspell

**It trains a CTC pack, not a static one.** FSboard is continuous spelling with
sequence-level labels: each clip carries the phrase that was signed, with no alignment
saying which frame is which letter. There is no honest way to cut per-letter training
examples out of that, and a static classifier needs exactly that. The temporal pipeline
learns the alignment itself, which is the whole point of CTC.

It also means **J and Z come for free** — as motion letters they are ordinary labels to a
model that sees sequences, and impossible for one that sees frames.

Schema handling
---------------

This adapter **discovers** the schema rather than assuming it. FSboard follows Google's
ASL Fingerspelling competition layout — landmark parquet keyed by ``sequence_id``, columns
named ``x_right_hand_0`` and so on, with a metadata CSV giving ``participant_id`` and
``phrase`` — but the release could differ in details, and this code cannot be tested
against the real files from where it was written. So it looks for what is there, names
what it found, and fails with the actual column list rather than a ``KeyError`` twenty
frames in. Run ``--inspect`` first; if it disagrees with your download, the output says
exactly how.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .recorder import LANDMARKS_PER_HAND
from .sequences import SequenceDataset

#: Letters a fingerspelling pack can represent. FSboard phrases are addresses, URLs and
#: phone numbers, so they carry digits and punctuation that no handshape label covers.
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: Metadata column names, in the order they are tried.
PARTICIPANT_COLUMNS = ("participant_id", "signer_id", "signer", "participant")
PHRASE_COLUMNS = ("phrase", "text", "label", "sentence")
PATH_COLUMNS = ("path", "file", "filepath")
SEQUENCE_COLUMNS = ("sequence_id", "sequence", "id")

_LANDMARK = re.compile(r"^(?P<axis>[xyz])_(?P<group>[a-z_]+?)_(?P<index>\d+)$")


@dataclass(frozen=True, slots=True)
class Schema:
    """What a particular FSboard download actually contains."""

    metadata_file: Path
    landmark_root: Path
    participant_column: str
    phrase_column: str
    path_column: str | None
    sequence_column: str
    #: Landmark group name → how many landmarks it has, e.g. ``{"right_hand": 21}``.
    groups: dict[str, int]

    @property
    def has_hands(self) -> bool:
        return "right_hand" in self.groups and "left_hand" in self.groups

    def describe(self) -> str:
        groups = ", ".join(f"{name} ({count})" for name, count in sorted(self.groups.items()))
        return (
            f"metadata:    {self.metadata_file.name}\n"
            f"  signer:    {self.participant_column}\n"
            f"  phrase:    {self.phrase_column}\n"
            f"  sequence:  {self.sequence_column}\n"
            f"  path:      {self.path_column or '(derived from file_id)'}\n"
            f"landmarks:   {groups}"
        )


def _pick(available: list[str], candidates: tuple[str, ...], what: str) -> str:
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(
        f"Could not find the {what} column. Tried {', '.join(candidates)}.\n"
        f"The file has: {', '.join(sorted(available))}\n"
        "If FSboard names it something else, pass the right name explicitly or open an "
        "issue with this list."
    )


def discover_schema(root: Path) -> Schema:
    """Work out how this download is laid out.

    Deliberately verbose on failure. A dataset adapter that dies with ``KeyError: 'phrase'``
    after ten minutes of loading is worse than useless; this fails immediately and prints
    what it actually saw.
    """
    import pandas as pd

    root = root.expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory. Did the download finish?")

    candidates = sorted(root.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV metadata found in {root}. Expected something like `train.csv` beside "
            "the landmark parquet directories."
        )
    # Prefer the training split when several are present.
    metadata_file = next((c for c in candidates if "train" in c.name.lower()), candidates[0])

    head = pd.read_csv(metadata_file, nrows=5)
    columns = list(head.columns)

    parquets = sorted(root.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No .parquet landmark files under {root}.")

    # Read one row group's schema only — these files are large.
    import pyarrow.parquet as pq

    landmark_columns = list(pq.ParquetFile(parquets[0]).schema.names)

    groups: dict[str, int] = {}
    for name in landmark_columns:
        match = _LANDMARK.match(name)
        if match is None:
            continue
        group = match.group("group")
        groups[group] = max(groups.get(group, 0), int(match.group("index")) + 1)

    if not groups:
        raise ValueError(
            f"No landmark columns in {parquets[0].name}. Expected names like "
            f"`x_right_hand_0`. Found: {', '.join(landmark_columns[:12])}…"
        )

    return Schema(
        metadata_file=metadata_file,
        landmark_root=parquets[0].parent,
        participant_column=_pick(columns, PARTICIPANT_COLUMNS, "signer"),
        phrase_column=_pick(columns, PHRASE_COLUMNS, "phrase"),
        path_column=next((c for c in PATH_COLUMNS if c in columns), None),
        sequence_column=_pick(columns, SEQUENCE_COLUMNS, "sequence id"),
        groups=groups,
    )


def _hand_array(frame, group: str) -> np.ndarray:
    """Pull one hand out of a landmark frame as ``(frames, 21, 3)``. NaN where absent."""
    points = np.full((len(frame), LANDMARKS_PER_HAND, 3), np.nan)
    for index in range(LANDMARKS_PER_HAND):
        for axis, column in enumerate("xyz"):
            name = f"{column}_{group}_{index}"
            if name in frame.columns:
                points[:, index, axis] = frame[name].to_numpy(dtype=np.float64)
    return points


def spellable(phrase: str) -> list[str]:
    """The letters of a phrase that a fingerspelling pack can label.

    FSboard prompts are addresses, phone numbers and URLs, so most phrases contain digits
    and punctuation. Those characters are dropped rather than mapped to a nearest letter:
    inventing a label is how a dataset quietly teaches a model something false.
    """
    return [character for character in phrase.upper() if character in ALPHABET]


def iter_sequences(
    root: Path,
    schema: Schema | None = None,
    *,
    limit: int | None = None,
    min_frames: int = 8,
    min_detection: float = 0.6,
) -> Iterator[tuple[np.ndarray, list[str], str, str]]:
    """Yield ``(landmarks, letters, signer, hand)`` for each usable clip.

    Args:
        limit: Stop after this many clips. FSboard is large; this is how you try
            something on a subset before committing an afternoon to it.
        min_frames: Skip clips too short to contain their phrase.
        min_detection: Skip clips where the dominant hand was tracked in less than this
            fraction of frames. A clip that is mostly missing data teaches the model
            mostly about missing data.
    """
    import pandas as pd

    schema = schema or discover_schema(root)
    metadata = pd.read_csv(schema.metadata_file)

    yielded = 0
    for row in metadata.itertuples(index=False):
        if limit is not None and yielded >= limit:
            return

        letters = spellable(str(getattr(row, schema.phrase_column)))
        if not letters:
            continue

        if schema.path_column is not None:
            path = root / str(getattr(row, schema.path_column))
        else:
            path = schema.landmark_root / f"{getattr(row, 'file_id')}.parquet"
        if not path.is_file():
            continue

        sequence_id = getattr(row, schema.sequence_column)
        frame = pd.read_parquet(path, filters=[(schema.sequence_column, "==", sequence_id)])
        if len(frame) < min_frames:
            continue

        right = _hand_array(frame, "right_hand")
        left = _hand_array(frame, "left_hand")

        right_seen = int((~np.isnan(right).all(axis=(1, 2))).sum())
        left_seen = int((~np.isnan(left).all(axis=(1, 2))).sum())
        if max(right_seen, left_seen) / len(frame) < min_detection:
            continue

        # Whichever hand the signer actually used. Assuming right-dominance would mirror
        # every left-handed signer's clips into the wrong slot.
        dominant, hand = (right, "right") if right_seen >= left_seen else (left, "left")

        # Drop frames where the hand vanished. The temporal model reads a continuous
        # trajectory; a gap of NaN is not one, and interpolating across it would invent
        # motion that never happened.
        present = ~np.isnan(dominant).all(axis=(1, 2))
        dominant = dominant[present]
        if len(dominant) < min_frames:
            continue

        # A stray missing landmark inside an otherwise tracked frame is filled from its
        # neighbours in the same frame, which is far less wrong than leaving NaN.
        centre = np.nanmean(dominant, axis=1, keepdims=True)
        dominant = np.where(np.isnan(dominant), centre, dominant)

        yield dominant, letters, str(getattr(row, schema.participant_column)), hand
        yielded += 1


def load_fsboard(
    root: Path,
    *,
    limit: int | None = None,
    min_frames: int = 8,
    min_detection: float = 0.6,
) -> SequenceDataset:
    """Load FSboard into the CTC pipeline's dataset.

    :raises ValueError: if nothing usable was found, rather than returning an empty
        dataset that would train happily and produce nonsense.
    """
    sequences: list[np.ndarray] = []
    labels: list[list[str]] = []
    signers: list[str] = []
    hands: list[str] = []

    for points, letters, signer, hand in iter_sequences(
        root, limit=limit, min_frames=min_frames, min_detection=min_detection
    ):
        sequences.append(points.astype(np.float32))
        labels.append(letters)
        signers.append(signer)
        hands.append(hand)

    if not sequences:
        raise ValueError(
            f"No usable clips found under {root}. Run with --inspect to see the schema "
            "that was detected, and check the detection-rate threshold is not too strict."
        )

    return SequenceDataset(
        sequences=sequences,
        labels=labels,
        signers=np.array(signers),
        hands=np.array(hands),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Extracted FSboard directory")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the detected schema and read a few clips, then stop",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    schema = discover_schema(args.root)
    print("Detected schema:\n")
    print(schema.describe())
    print()

    if not schema.has_hands:
        print("⚠  No left/right hand landmark groups found. Fingerspelling needs them.")
        return

    if args.inspect:
        print("Reading a few clips…\n")
        for index, (points, letters, signer, hand) in enumerate(
            iter_sequences(args.root, schema, limit=5)
        ):
            print(
                f"  {index + 1}. {len(points):>4} frames  {hand:<5}  signer {signer:<10} "
                f"{''.join(letters)[:40]}"
            )
        print("\nLooks readable. Train with `python -m mudra_train.train_ctc --fsboard ...`")
        return

    dataset = load_fsboard(args.root, limit=args.limit)
    unique = sorted(set(dataset.signers.tolist()))
    total = sum(len(label) for label in dataset.labels)
    print(f"{len(dataset)} clips, {total:,} characters, {len(unique)} signers")


if __name__ == "__main__":
    main()
