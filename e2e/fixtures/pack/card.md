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
- macro-F1: **0.8649**
- accuracy: 0.8777
- Expected calibration error: 0.0386
- Rejection AUROC (known vs `none`): 0.9820

### Per slice

| Slice                | Support | macro-F1 |
| -------------------- | ------- | -------- |
| condition=angled     | 205     | 0.8567   |
| condition=near       | 181     | 0.8573   |
| hand=right           | 765     | 0.8611   |
| condition=dim        | 200     | 0.8619   |
| condition=far        | 202     | 0.8650   |
| hand=left            | 339     | 0.8730   |
| condition=normal     | 172     | 0.8938   |
| condition=transition | 144     | 0.9294   |

**Worst slice: condition=angled at 0.8567.**

### Most frequent confusions

- V → U (×21)
- N → M (×17)
- S → N (×14)
- S → T (×13)
- T → S (×12)
- V → R (×12)
- R → U (×8)
- U → R (×8)

## Limitations

- Static handshapes only. J and Z require motion and are not covered here.
- Trained on the conditions present in the data. Lighting, camera angles, or
  skin tones outside that range are untested.
- Rotation is preserved rather than normalised, because K/P and G/Q differ only
  by orientation. Tilt beyond the ±25° augmentation range will degrade.
- The `none` class rejects transitions, but no rejector is perfect; spurious
  letters remain possible during fast fingerspelling.
