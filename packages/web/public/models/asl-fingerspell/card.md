# Model card — asl-fingerspell

Built 2026-08-21.

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
> **J and Z are absent.** Both are handshapes *plus a path*, and a single
> frame carries no path, so no static pack can contain them. The app reads
> them with `MotionLetterRecognizer`, which runs after this model; a
> `temporal-ctc` pack makes them ordinary labels instead.
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
- macro-F1: **0.9682**
- accuracy: 0.9663
- Expected calibration error: 0.0085
- Rejection AUROC (known vs `none`): 0.9894

### Per slice

| Slice | Support | macro-F1 |
| --- | --- | --- |
| condition=transition | 1240 | 0.9287 |
| hand=right | 9520 | 0.9675 |
| condition=far | 1659 | 0.9680 |
| hand=left | 4205 | 0.9696 |
| condition=angled | 4077 | 0.9732 |
| condition=normal | 3406 | 0.9811 |
| condition=near | 2108 | 0.9831 |
| condition=idle | 1235 | 0.9914 |

**Worst slice: condition=transition at 0.9287.**

### What reaches the screen

macro-F1 above scores the `argmax`. The app never shows the `argmax` — it
shows a letter only once the probability and the margin over the runner-up
both clear this pack's thresholds. So a letter can score well above and
still, to the person signing, be a letter the app cannot do. This is the
share of each letter that is **accepted and correct**, which is what they
actually get.

| Letter | Accepted and correct |
| --- | --- |
| E | 0.877 |
| T | 0.885 |
| S | 0.901 |
| O | 0.936 |
| N | 0.947 |
| A | 0.949 |
| K | 0.952 |
| M | 0.963 |
| V | 0.973 |
| 1 | 0.979 |
| 4 | 0.979 |
| U | 0.981 |
| X | 0.981 |
| B | 0.984 |
| P | 0.984 |
| C | 0.989 |
| D | 0.989 |
| Q | 0.992 |
| 7 | 0.992 |
| F | 0.995 |
| G | 0.997 |
| W | 0.997 |
| H | 1.000 |
| I | 1.000 |
| L | 1.000 |
| R | 1.000 |
| Y | 1.000 |
| 3 | 1.000 |
| 5 | 1.000 |
| 8 | 1.000 |

### Most frequent confusions

- E → O (×41)
- S → A (×20)
- A → S (×16)
- none → Q (×15)
- none → O (×14)
- O → E (×13)
- T → S (×13)
- none → 1 (×13)

## Limitations

- Static handshapes only. J and Z are paths, and are read separately.
- Trained on the conditions present in the data. Lighting, camera angles, or
  skin tones outside that range are untested.
- Rotation is preserved rather than normalised, because H/U, K/P and G/Q differ
  only by orientation. Each letter declares the band of orientations it is
  signed at (`ingest/asl_alphabet.py`); outside its band, expect `none` rather
  than a near miss.
- The `none` class rejects transitions, but no rejector is perfect; spurious
  letters remain possible during fast fingerspelling.
