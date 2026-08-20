# Model card — synthetic-words

Built 2026-08-20.

> ## ⚠ NOT A USABLE MODEL
>
> Trained on procedurally generated clips, not recordings of signers.
> It exists to prove the word-level pipeline runs end to end.

## Intended use

Recognising isolated ASL word signs from two-handed MediaPipe landmarks. The app
segments on motion and classifies each completed sign once.

## Architecture

- Conv1D + self-attention encoder, 1,663,887 parameters.
- Input: 96-frame clip × 100 features,
  `shoulder-frame-v1` — both hands and upper-body pose in one
  shoulder-anchored reference frame, so hand position relative to the body is
  preserved. That matters: signs can share a handshape and differ only in location.
- Motion features (first and second temporal differences) are computed inside the
  graph, so the browser cannot compute them differently from training.

## Evaluation

Signer-independent.

- Held-out signers: synthetic-00, synthetic-04
- top-1 accuracy: **1.0000**
- top-5 accuracy: 1.0000
- macro-F1: 1.0000
- Worst slice: dominant=left at 1.0000

## Limitations

- Isolated signs only. Continuous signing is not segmented linguistically; the
  segmenter splits on motion, which is a proxy.
- Facial expression is not modelled. ASL non-manual markers carry grammar, so
  signs distinguished only by expression are not separable here.
- Vocabulary is fixed at training time. Users can add their own signs through the
  few-shot path instead.
- Requires visible shoulders for the body frame; a tight crop falls back to a
  hand-derived frame and loses location information.
