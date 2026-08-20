# Model card — asl-fingerspell

Built 2026-08-20.

> ## ⚠ TRAINED ON SIMULATED HANDS
>
> No real hands were recorded, and none were seen during training. Every
> sample comes from a kinematic hand model — a bone skeleton with
> anthropometric proportions, articulated to the joint configurations that
> define the manual alphabet, then projected through a camera model with
> viewpoint, anatomy and tracking noise randomised.
>
> **What this means in practice.** It works on real hands, and less well than
> a model trained on real ones. The scores below are measured on held-out
> *simulated* signers, so they describe how well the model generalises across
> simulated anatomy — **not** how well it will do on your camera. Treat them
> as an upper bound and expect the real figure to be lower.
>
> **Known gaps.** MediaPipe's errors on a real hand are structured — it
> confuses occluded fingers in specific, repeatable ways — and Gaussian noise
> is a poor imitation of that. There is no skin, no motion blur, and no
> self-occlusion beyond what the geometry implies. M, N, T and the
> thumb-tucked letters are the least faithful, because how far the thumb
> disappears under the fingers is exactly what a geometric model guesses at.
>
> **J and Z are absent.** Both are motion letters and cannot be represented by
> a single frame. Use a `temporal-ctc` pack for those.
>
> Recording thirty minutes of your own data with `packages/web/recorder.html`
> and retraining will beat this. That is the intended upgrade path.

## Intended use

Recognising static ASL fingerspelling handshapes from MediaPipe hand landmarks,
in a browser, on the signer's own device. Not a medical, legal, or safety-critical
tool, and not a substitute for a qualified interpreter.

## Architecture

- Residual MLP encoder with an ArcFace head, 482,560 parameters.
- Input: 42 features (21 landmarks × x, y), `centroid-rms-v2`.
- Output: 31 classes — 1, 3, 4, 5, 7, 8, A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, none.

## Evaluation

Signer-independent: no signer appears in both training and test.

- Held-out signers: sim-03, sim-04, sim-08, sim-12, sim-16
- macro-F1: **0.9730**
- accuracy: 0.9703
- Expected calibration error: 0.0093
- Rejection AUROC (known vs `none`): 0.9938

### Per slice

| Slice | Support | macro-F1 |
| --- | --- | --- |
| condition=transition | 1240 | 0.9324 |
| condition=angled | 2534 | 0.9609 |
| hand=left | 4091 | 0.9714 |
| hand=right | 9634 | 0.9737 |
| condition=far | 2040 | 0.9801 |
| condition=normal | 4071 | 0.9847 |
| condition=near | 2605 | 0.9909 |
| condition=idle | 1235 | 0.9918 |

**Worst slice: condition=transition at 0.9324.**

### Most frequent confusions

- T → N (×29)
- none → H (×16)
- none → C (×15)
- E → M (×13)
- K → V (×12)
- S → M (×11)
- none → P (×11)
- none → U (×11)

## Limitations

- Static handshapes only. J and Z require motion and are not covered here.
- Trained on the conditions present in the data. Lighting, camera angles, or
  skin tones outside that range are untested.
- Rotation is preserved rather than normalised, because K/P and G/Q differ only
  by orientation. Tilt beyond the ±25° augmentation range will degrade.
- The `none` class rejects transitions, but no rejector is perfect; spurious
  letters remain possible during fast fingerspelling.
