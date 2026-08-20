# MudraPragyan.AI

**Sign language recognition that runs entirely in your browser.** No server, no upload, no cost.

> **Status: Phases 0 and 1 complete.** The application has been restructured from a single 1,325-line
> HTML file into a tested monorepo, and the vision layer has been rebuilt on MediaPipe Tasks
> Vision with two-handed tracking and worker-based inference. **The classifier itself is still
> the v1 model**, with all its known defects — those are Phase 2. See
> [`docs/AUDIT.md`](docs/AUDIT.md) for what is wrong and the roadmap below for the plan.

---

## Quick start

```bash
npm install
npm run setup:assets   # downloads the MediaPipe models (~10 MB, once)
npm run dev            # http://localhost:5173
```

The `.task` model files are not committed — `models.lock.json` pins them by checksum instead.
Without them the app loads and the Dictionary works, but the Translator will tell you to run
setup.

| Command          | What it does                                                 |
| ---------------- | ------------------------------------------------------------ |
| `npm run dev`    | Development server with hot reload                           |
| `npm run build`  | Production build into `packages/web/dist`                    |
| `npm test`       | 127 unit tests, no browser needed (~2 s)                     |
| `npm run e2e`    | 16 Playwright smoke tests against the built app              |
| `npm run verify` | Format check → lint → typecheck → tests. Run before pushing. |

Camera access needs a secure context, so use `localhost` or HTTPS.

**Deploying?** See [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Layout

```
packages/core/     Pure TypeScript. No DOM. All recognition logic, fully unit-tested.
packages/web/      The application. UI, camera, wiring. No recognition decisions.
docs/adr/          Architecture decision records — why things are the way they are.
e2e/               Playwright smoke tests.
scripts/           setup-assets.mjs — self-hosts the MediaPipe WASM and models.
```

**Recording training data:** `npm run dev` then open `/recorder.html`. It captures hand
landmarks only — never video — tagged with signer id and lighting/distance condition, and
exports JSONL. Phase 2's signer-independent evaluation depends on those tags.

The `core` / `web` split is load-bearing, not cosmetic: because `core` cannot touch the DOM,
every recognition decision is testable in Node. See [ADR 0004](docs/adr/0004-monorepo-and-pure-core.md).

## How it works

```
camera → MediaPipe hand landmarks → normalise → classifier → smoothing → letter → sentence
```

Everything after the camera happens on your device. Nothing is uploaded. See
[`docs/PRIVACY.md`](docs/PRIVACY.md) — and verify it yourself with the Network tab.

## Known limitations

These are **real and documented**, not hidden. Full detail in [`docs/AUDIT.md`](docs/AUDIT.md).

- **Left-handed signers are not supported.** The model has no handedness invariance (M1).
- **The hand must be held roughly upright.** No rotation invariance (M2).
- **Transitional poses produce spurious letters.** There is no "no sign" class (M4).
- **The Dictionary shows no images.** None of the referenced files exist (A1).
- **The in-app accuracy test is not a benchmark.** It measures one person in one session and
  cannot predict performance for anyone else (A3).

## Roadmap

| Phase | Delivers                                                                                                                                                           | Status  |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| **0** | Monorepo, tests, CI, audit, privacy statement — behaviour unchanged                                                                                                | ✅ Done |
| **1** | MediaPipe Tasks Vision, two hands, pose, Web Worker, landmark recorder                                                                                             | ✅ Done |
| **2** | Retrained model with handedness/rotation invariance and a `none` class. **`geometricFix` deleted.** Signer-independent evaluation as a CI gate. Dictionary images. |         |
| **3** | Continuous fingerspelling via CTC. J/Z state machine and dwell timer removed.                                                                                      |         |
| **4** | 250 word-level signs, two-handed                                                                                                                                   |         |
| **5** | Indian Sign Language, custom user signs, offline PWA                                                                                                               |         |

## Data and licensing

All training data is openly licensed and redistributable:

| Purpose           | Dataset                                                                                                               | Licence   |
| ----------------- | --------------------------------------------------------------------------------------------------------------------- | --------- |
| Fingerspelling    | [FSboard](https://arxiv.org/abs/2407.15806) — 3M+ characters, 147 Deaf signers                                        | CC BY 4.0 |
| Word-level        | [PopSign ASL v1.0](https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/index.html) — 250 signs, 47 Deaf signers | CC BY 4.0 |
| Indian SL         | [INCLUDE](https://zenodo.org/records/4010759) — 263 words                                                             | CC BY 4.0 |
| Dictionary images | [Roboflow ASL Letters](https://public.roboflow.com/object-detection/american-sign-language-letters)                   | CC0       |

Datasets that are non-commercial, research-only, or unredistributable — WLASL, MS-ASL,
ASL Citizen, AUTSL, and the Kaggle competition sets — are deliberately excluded.

## Contributing

`npm run verify` must pass. New recognition logic belongs in `packages/core` with tests.
If a file in `core` needs `window` or `document`, the design is wrong.

## Licence

MIT for the code. Model packs carry their own licence and attribution in their model card.
