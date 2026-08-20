# Model card — synthetic-fingerspell-ctc

Built 2026-08-20.

> ## ⚠ NOT A USABLE MODEL
>
> Trained on procedurally generated sequences, not real fingerspelling.
> It exists to prove the CTC pipeline runs end to end.

## Intended use

Continuous ASL fingerspelling from MediaPipe hand landmarks, decoded with CTC.
Unlike the static pack, this handles J and Z natively — they are motion-defined,
and a temporal model sees motion — and can spell doubled letters.

## Architecture

- Causal depthwise-separable Conv1D encoder, 433,461 parameters.
- Input: 64-frame window × 42 features, `centroid-rms-v2`.
- Output: 23 classes (blank at index 0).
- Causal throughout, so the same weights serve offline and streaming decoding.

## Evaluation

Signer-independent. Scored with character error rate, which penalises
misrecognition, double-firing and misses together — per-frame accuracy
penalises none of them.

- Held-out signers: synthetic-00, synthetic-04
- Character error rate: **0.0792**
- Exact-match rate: 82.5%

## Limitations

- Fingerspelling only. Word-level signs are Phase 4.
- Latency is bounded by the decode stride and the LocalAgreement window; text
  settles a few hundred milliseconds behind the hand.
- A very long pause mid-word flushes the buffer and starts a new word.
