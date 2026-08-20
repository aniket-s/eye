# Model card — synthetic-fingerspell

Built 2026-08-20.

> ## ⚠ NOT A USABLE MODEL
>
> Trained on procedurally generated handshapes, not real hands. It exists to
> prove the pipeline runs end to end. It will fail on a real webcam.
> Train on FSboard or your own recordings before shipping anything.

## Intended use

Recognising static ASL fingerspelling handshapes from MediaPipe hand landmarks,
in a browser, on the signer's own device. Not a medical, legal, or safety-critical
tool, and not a substitute for a qualified interpreter.

## Architecture

- Residual MLP encoder with an ArcFace head, 481,792 parameters.
- Input: 42 features (21 landmarks × x, y), `centroid-rms-v2`.
- Output: 25 classes — A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, none.

## Evaluation

Signer-independent: no signer appears in both training and test.

- Held-out signers: synthetic-00, synthetic-04
- macro-F1: **0.8407**
- accuracy: 0.8632
- Expected calibration error: 0.0932
- Rejection AUROC (known vs `none`): 0.9838

### Per slice

| Slice                | Support | macro-F1 |
| -------------------- | ------- | -------- |
| condition=near       | 181     | 0.8302   |
| condition=angled     | 205     | 0.8387   |
| hand=left            | 339     | 0.8390   |
| condition=far        | 202     | 0.8406   |
| hand=right           | 765     | 0.8413   |
| condition=dim        | 200     | 0.8514   |
| condition=normal     | 172     | 0.8732   |
| condition=transition | 144     | 0.9008   |

**Worst slice: condition=near at 0.8302.**

### Most frequent confusions

- S → N (×25)
- V → U (×24)
- N → M (×16)
- S → T (×15)
- V → R (×11)
- U → R (×10)
- M → N (×7)
- R → U (×7)

## Limitations

- Static handshapes only. J and Z require motion and are not covered here.
- Trained on the conditions present in the data. Lighting, camera angles, or
  skin tones outside that range are untested.
- Rotation is preserved rather than normalised, because K/P and G/Q differ only
  by orientation. Tilt beyond the ±25° augmentation range will degrade.
- The `none` class rejects transitions, but no rejector is perfect; spurious
  letters remain possible during fast fingerspelling.
