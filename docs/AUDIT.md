# Audit of the v1 implementation

Reviewed 19 August 2026 against `index.html` (1,325 lines) and `model_weights.json` (2.5 MB),
as recovered from the deployed site. No source repository, dataset, or training code existed.

Every finding has a stable id (M1, V2, A3…) referenced from code comments and tests, so a
reader who encounters a `docs/AUDIT.md, M4` marker can find the reasoning. Findings are
retired here as their phase lands.

| Status       | Meaning                                    |
| ------------ | ------------------------------------------ |
| 🔴 Open      | Not yet addressed                          |
| 🟡 Mitigated | Partially addressed or documented in place |
| 🟢 Fixed     | Resolved, with a test asserting the fix    |

---

## 1. Model (`M`)

Architecture: scikit-learn `MLPClassifier`, `63 → 256 → 256 → 128 → 27`, ReLU, ~115k parameters.
Input: 21 landmarks × (x, y, z), wrist-origin, divided by the 3D wrist→middle-MCP distance.
Labels: 26 letters plus `space`.

| id     | Finding                                                                                                                                                                                                     | Status                                              |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **M1** | **No handedness invariance.** A left-handed signer produces a mirrored feature vector the model has never seen. Asserted by `landmarkVector.test.ts`.                                                       | 🔴 Phase 2                                          |
| **M2** | **No rotation invariance.** Only translation and scale are normalised, so a 30° tilt moves the vector off-manifold. Asserted by `landmarkVector.test.ts`.                                                   | 🔴 Phase 2                                          |
| **M3** | **`z` is included.** MediaPipe's normalised `z` is weakly calibrated and depth-ambiguous from a single view. Every top competition solution drops it.                                                       | 🔴 Phase 2                                          |
| **M4** | **No negative class.** Only `space` exists, and it is suppressed before display. Every transitional pose between two letters is force-classified as a letter. The largest single source of spurious output. | 🔴 Phase 2                                          |
| **M5** | **Confidence gate is ineffective.** `0.45` on an uncalibrated softmax. An overfit MLP with no negative class is _confidently_ wrong on off-manifold input, so this filters far less than it appears to.     | 🔴 Phase 2                                          |
| **M6** | **2.5 MB of JSON for 115k parameters** (~22 bytes/param), parsed on the main thread. An int8 ONNX export is ~250 KB.                                                                                        | 🟡 Parse moved to the worker; size fixed in Phase 2 |

---

## 2. The geometric override layer

`geometricFix()` — 85 lines rewriting the model's output for 20 of 26 letters, with 15+ magic
constants. **This is the finding that matters most**, because it has trapped the project rather
than merely degraded it.

### It is self-inconsistent

The function is meant to correct the model _using hand geometry_. If it did that, a fixed hand
would produce one answer regardless of what the model guessed. It does not. Measured, and
pinned in `geometricFix.test.ts`:

| Hand        | Distinct outputs across 24 seed letters |
| ----------- | --------------------------------------- |
| Open hand   | **8** — B, E, H, K, L, R, U, Y          |
| Closed fist | **10** — A, C, E, F, G, L, M, N, S, Y   |

Concretely, on one unchanging closed fist: `I → S`, `U → N`, `M → M`. Three answers, one
physical handshape. On an open hand, `I`, `M`, `S` and `V` all collapse to `K`.

### It makes the model un-retrainable

Improving the model changes its error distribution, which invalidates every branch. The model
and the overrides can only be changed together, by hand. This is why v1 is a dead end rather
than a starting point.

### It is overfit and untestable

The constants were tuned against one hand, one camera, one distance, one lighting setup, across
20+ interacting branches with no tests.

### Its units are mixed

`tipAbovePip` **multiplies** by palm scale; `dist` **divides** by it. Thresholds are therefore
not comparable across branches. `geometricFix` also computes palm scale in 2D while
`landmarksToVector` uses 3D for the same conceptual quantity.

**Resolution:** deleted in Phase 2, not extended. Correct normalisation plus a `none` class
makes all of it unnecessary. Status: 🟡 isolated in `packages/core/src/legacy/`, fully
characterised by tests.

---

## 3. J/Z motion state machine

The instinct is right — J and Z are the only motion-defined letters — but:

| id     | Finding                                                                                                                                                                                  | Status     |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| **J1** | `isZShape` accepts **both** stroke directions (`fwd \|\| rev`), roughly doubling the false-positive rate. A horizontal wiggle reads as Z. Asserted by `jzStateMachine.test.ts`.          | 🔴 Phase 3 |
| **J2** | Frames are counted, not timed. Every threshold is frame-rate dependent and drifts under CPU load. The same physical motion at half the sample rate fails to register — asserted by test. | 🔴 Phase 3 |
| **J3** | A confirmed detection **latches the output for 32 frames** regardless of what the hand does next.                                                                                        | 🔴 Phase 3 |

**Resolution:** CTC decoding in Phase 3 makes J and Z ordinary labels. The state machine is
deleted entirely.

**Partial mitigation (Phase 1):** the commit timer is now millisecond-based
(`TimedHoldCommitDetector`), so _that_ part of the pipeline behaves identically at 15, 30 and
60 fps. The J/Z state machine itself still counts frames — J2 remains open until Phase 3.

---

## 4. Vision layer (`V`)

| id     | Finding                                                                                                                                                                                  | Status                                                      |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **V1** | Uses `@mediapipe/hands` + `camera_utils`, **end-of-life since 1 March 2023**, maintained "on an as-is basis" with no fixes.                                                              | 🟢 Replaced by `@mediapipe/tasks-vision@1.0.1`, self-hosted |
| **V2** | Loaded **unpinned** from jsDelivr. An upstream publish could break production with no rollback.                                                                                          | 🟢 Self-hosted and checksum-pinned via `models.lock.json`   |
| **V3** | `maxNumHands: 1`. **Blocks the entire word-level roadmap** — most ASL words are two-handed.                                                                                              | 🟢 `numHands: 2`                                            |
| **V4** | No pose or face landmarks. ASL non-manual markers (eyebrows, mouth morphemes) are grammatically load-bearing; the winning GISLR solution ranked **lips** among its top-2 feature groups. | 🟢 Upper-body pose captured                                 |
| **V5** | 1280×720 capture, detection every frame, no throttle. Burns the frame budget before the classifier runs.                                                                                 | 🟢 640×480 + frame-drop backpressure                        |

---

## 5. Temporal logic

Prediction is per-frame `argmax`, then `holdFrames` counts consecutive identical labels
(`HOLD_NEEDED = 20`). Because `argmax` flickers, the counter resets constantly and the user must
hold unnaturally still — **one stray frame discards all accumulated progress**, asserted in
`holdCommit.test.ts`.

There is no probability-level smoothing, which is the correct place to filter. Commit is a dwell
timer rather than an evidence test. A v1 off-by-one also means `framesToCommit + 1` frames are
actually required.

**Resolution:** Phase 3 — median filtering over a probability queue, energy-score rejection, and
CTC blank-transition commit wrapped in LocalAgreement-2.

---

## 6. Application and process (`A`)

| id      | Finding                                                                                                                                                                                                                    | Status                                                           |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **A1**  | **The Dictionary is 100% broken.** Images are referenced at `alphabets/a.jpg` etc.; none of those directories exist. All ~150 cards fall through to the 🤟 placeholder.                                                    | 🟡 Pinned by e2e test; fixed Phase 2 with CC0 imagery            |
| **A2**  | **No training pipeline in the repo.** The error string reads _"run Script 2 first"_. No dataset, training code, eval, or model card. The model is an unversioned blob nobody can regenerate.                               | 🔴 Phase 2                                                       |
| **A3**  | **"Test Model Accuracy" is not evaluation.** Same user, camera and session; no held-out signers, no confusion matrix. **The most dangerous item** — it produces a number that looks like validation and hides regressions. | 🟡 On-screen caveat added; real eval Phase 2                     |
| **A4**  | Single 1,325-line HTML file. No modules, build, tests, types, linting, or CI.                                                                                                                                              | 🟢 Phase 0                                                       |
| **A5**  | Inference ran on the main thread inside the MediaPipe callback, allocating 4 arrays per frame.                                                                                                                             | 🔴 Phase 1 (Web Worker)                                          |
| **A6**  | No accessibility work — no ARIA, no keyboard path, no visible focus, no reduced-motion. This is an **accessibility product**.                                                                                              | 🟡 Keyboard nav, focus ring, ARIA state added; full pass Phase 2 |
| **A7**  | Camera failure handled by a bare `alert()`. No secure-context guard, no permission-recovery guidance.                                                                                                                      | 🟢 Typed error mapping in `legacyHands.ts`                       |
| **A8**  | No privacy statement despite requesting camera access.                                                                                                                                                                     | 🟢 `docs/PRIVACY.md`                                             |
| **A9**  | The camera kept running after navigating away from the Translator, leaving the webcam light on.                                                                                                                            | 🟢 Released on navigation                                        |
| **A10** | Third-party webfont from `fonts.googleapis.com` — an external request in an app whose premise is that nothing leaves the device, and a hard failure on restricted networks.                                                | 🔴 Phase 5 (self-host with offline/PWA)                          |

---

## 7. What was worth keeping

- The **landmark-based approach** is correct and matches every winning competition solution.
- The **product concept** — translator, dictionary, sentence builder, speech — is well-judged.
- Treating **J and Z as motion-dependent** shows real domain understanding.
- The **visual design** is good and carried over unchanged.
