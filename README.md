# MudraPragyan.AI

**Sign language recognition that runs entirely in your browser.** No server, no upload, no cost.

> **Status: Phases 0–4 complete.** The application has been restructured from a single 1,325-line
> HTML file into a tested monorepo, the vision layer rebuilt on MediaPipe Tasks Vision, and a
> complete training pipeline built with signer-independent evaluation and ONNX model packs.
>
> **You need to train a model.** The pipeline is tested end to end, but no real dataset can be
> shipped with it. Until you train one, the app falls back to the v1 model and says so.
> See [Training a model](#training-a-model) — it takes about half an hour.

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

## Training a model

The app ships with no trained model, because no dataset can legally or practically be
bundled. Two routes:

**Fastest — record your own (~30 minutes).**

```bash
npm run dev                    # open http://localhost:5173/recorder.html
# Record 24 letters + `none`, twice: once per hand, across a few lighting conditions.
# Get a second person to record too — a single-signer dataset is refused, on purpose.

pip install -e "training[dev]"
cd training
python -m mudra_train.train --data ./recordings --out ../packages/web/public/models/asl-fingerspell
```

**Best quality — FSboard.** 3M+ characters from 147 Deaf signers, CC BY 4.0. Needs a free
Kaggle account. Point `--data` at the extracted landmarks.

Three model types are available, and the app switches mode automatically based on which
pack is installed:

| Pack                | Trainer       | What it does                                         |
| ------------------- | ------------- | ---------------------------------------------------- |
| `static-handshape`  | `train`       | One letter at a time, hold to commit                 |
| `temporal-ctc`      | `train_ctc`   | Continuous fingerspelling — no pausing, J and Z work |
| `temporal-isolated` | `train_words` | Word signs, two-handed, segmented on motion          |

```bash
# Continuous fingerspelling
python -m mudra_train.train_ctc --data ./recordings --out ../packages/web/public/models/asl-fingerspell

# Word signs, from PopSign or GISLR landmark parquet
python -m mudra_train.train_words --index /data/popsign/train.csv --data-root /data/popsign \
    --out ../packages/web/public/models/asl-fingerspell
```

**Just checking the pipeline runs:**

```bash
npm run train:synthetic          # static model, procedural fixtures — NOT usable
npm run train:synthetic:ctc      # continuous fingerspelling, same caveat
npm run train:synthetic:words    # word signs, same caveat
```

Training always evaluates on **held-out signers** and reports per-slice scores, so a model
that works for right-handed signers in good light and fails otherwise cannot hide behind a
healthy average.

## Layout

```
packages/core/     Pure TypeScript. No DOM. All recognition logic, fully unit-tested.
packages/web/      The application. UI, camera, wiring. No recognition decisions.
docs/adr/          Architecture decision records — why things are the way they are.
e2e/               Playwright smoke tests.
scripts/           setup-assets.mjs — self-hosts the MediaPipe WASM and models.
training/          Python: ingest → augment → train → evaluate → export ONNX packs.
```

**Ingesting video corpora:** datasets that ship clips rather than coordinates — INCLUDE
for Indian Sign Language, PopSign in raw form — go through
`training/mudra_train/ingest/video.py`, which runs the _same_ MediaPipe Tasks bundles the
browser does. The derived landmarks are far smaller than the video and, for a CC BY 4.0
source, redistributable.

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

The app is a PWA: install it, then use it with the network off. On a second visit the
~25 MB of WASM and model weights come from the local cache.

## Teaching it your own signs

Open the Translator, type a name, and hold a handshape for a couple of seconds. Six
examples are enough.

This works with no training and no server because the classifier is trained with an
**ArcFace** head, which places every handshape on a unit hypersphere. Averaging a few
embeddings gives a class centroid, and classification is then a cosine similarity — a
single dot product. It is the only practical route to name signs, regional variants and
personal shortcuts, none of which will ever appear in a public dataset.

A custom sign is only consulted when the trained vocabulary _declines_ a frame: a trained
class rests on thousands of examples from many signers, yours on six. The app also warns
when your examples were inconsistent, or when a new sign is too close to an existing one
to tell apart. Everything is stored in IndexedDB on your device, namespaced by pack.

Requires a pack exported with an embedding output; the panel says so when one is not. The
design, and its limits, are in [ADR 0005](docs/adr/0005-few-shot-custom-signs.md).

## Known limitations

These are **real and documented**, not hidden. Full detail in [`docs/AUDIT.md`](docs/AUDIT.md).

- **No trained model ships with the repo.** Until you train one the app runs the v1 fallback,
  which has all the defects below. It says so on screen.
- **The in-app accuracy test is not a benchmark.** It measures one person in one session and
  cannot predict performance for anyone else (A3). Real evaluation lives in `training/`.
- **Word categories in the Dictionary have no artwork.** The alphabet and numbers do.
- **Facial expression is not modelled.** ASL non-manual markers carry grammar, so signs
  that differ only by expression are not separable.

Fixed once a pack is installed: left-handed signers (M1), depth noise (M3), spurious
letters during transitions (M4, M5), the 2.5 MB model download (M6 — now ~520 KB int8),
and, with a CTC pack, the dwell timer and the J/Z state machine (J1–J3).

## Roadmap

| Phase | Delivers                                                                                          | Status  |
| ----- | ------------------------------------------------------------------------------------------------- | ------- |
| **0** | Monorepo, tests, CI, audit, privacy statement — behaviour unchanged                               | ✅ Done |
| **1** | MediaPipe Tasks Vision, two hands, pose, Web Worker, landmark recorder                            | ✅ Done |
| **2** | Training pipeline, normalisation, `none` class, ONNX packs, signer-independent evaluation         | ✅ Done |
| **3** | Continuous fingerspelling via CTC. J/Z state machine and dwell timer removed. Dictionary artwork. | ✅ Done |
| **4** | 250 word-level signs, two-handed                                                                  | ✅ Done |
| **5** | Custom user signs, offline PWA, multi-pack registry, video ingestion for ISL                      | ✅ Done |

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
