# MudraPragyan.AI

**Sign language recognition that runs entirely in your browser.** No server, no upload, no cost.

> **Status: Phases 0–5 complete.** The application has been restructured from a single 1,325-line
> HTML file into a tested monorepo, the vision layer rebuilt on MediaPipe Tasks Vision, and a
> complete training pipeline built with signer-independent evaluation and ONNX model packs.
>
> **A trained pack is committed at `packages/web/public/models/asl-fingerspell`**, so the
> Translator runs the v2 pipeline out of the box. Without a pack the app falls back to the
> v1 legacy model, whose hand-tuned overrides systematically misread E, K, M, N, T and X
> (`docs/AUDIT.md` §2) — if the status bar ever says “Legacy model”, the pack is missing
> from the deployment. J and Z are motion letters: a static pack cannot represent them, a
> `temporal-ctc` pack can — see [Training on real data](#training-on-real-data).

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

**A trained model pack ships with the repository**, so the Translator works out of the box.
It was trained on simulated hands rather than recordings of real people — see
[Where the model comes from](#where-the-model-comes-from) for what that costs you, and the
[model card](packages/web/public/models/asl-fingerspell/card.md) for the numbers.

| Command          | What it does                                                 |
| ---------------- | ------------------------------------------------------------ |
| `npm run dev`    | Development server with hot reload                           |
| `npm run build`  | Production build into `packages/web/dist`                    |
| `npm test`       | 294 unit tests, no browser needed (~6 s)                     |
| `npm run e2e`    | 42 Playwright tests against the built app                    |
| `npm run verify` | Format check → lint → typecheck → tests. Run before pushing. |

Camera access needs a secure context, so use `localhost` or HTTPS.

**Deploying?** See [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Building words, not just letters

Per-letter accuracy compounds, and that is the real obstacle. At 93% — around the best
anyone achieves on real fingerspelling — a five-letter word arrives intact 70% of the
time and a seven-letter word 60%. Pushing per-letter accuracy is a losing fight against
an exponent.

So the app constrains output to real words. As you spell, it offers candidates from a
30,000-word frequency-ranked list; press <kbd>1</kbd>–<kbd>4</kbd> or click to accept.
Two things make it better than a spell-checker:

- **It uses the model's own confusion profile.** Every pack ships the letters it actually
  mistakes for which, measured on its held-out signers, in its manifest. So a displayed
  V is cheap to reinterpret as a K if _this_ model confuses those, and expensive if it
  does not. A retrain replaces the profile automatically.
- **Corrections are marked.** A suggestion that changes a letter is shown in a different
  colour from one that merely completes what you spelled. Silently swapping what someone
  signed for something else is the one behaviour that would make the feature untrustworthy.

Everything runs on the device. The word list is fetched lazily, ~80 KB compressed.

## Where the model comes from

The bundled pack is **simulated**. No dataset of real signers can be legally or practically
bundled with a repository, so rather than ship nothing, the training data is generated from a
kinematic hand model: a bone skeleton with anthropometric proportions, articulated to the joint
configurations and finger contacts that define the manual alphabet, then projected through a
camera with viewpoint, anatomy and tracking noise randomised per simulated signer.

|                   |                                             |
| ----------------- | ------------------------------------------- |
| macro-F1          | **0.968** on 5 held-out _simulated_ signers |
| Worst slice       | 0.922 (transitions between letters)         |
| Calibration error | 0.009                                       |
| Vocabulary        | 24 static letters + `none`                  |
| Size              | 525 KB, int8                                |

**Read that first number carefully.** It measures generalisation across simulated anatomy, not
performance on your camera. It is an upper bound; the real figure will be lower. A simulated
hand has no skin, no motion blur, and MediaPipe's errors on a real hand are structured in ways
Gaussian noise does not imitate. M, N and T — where the thumb hides under the fingers — are the
least faithful.

**J and Z are absent.** Both are motion letters and a single frame cannot represent either.
Train a `temporal-ctc` pack for those.

### Numbers, and why there is a mode switch

In ASL, several digits **are** letters. Not similar — the same handshape:

| Digit | Same shape as |
| ----- | ------------- |
| `0`   | `O`           |
| `2`   | `V`           |
| `6`   | `W`           |
| `9`   | `F`           |

A signer separates them by context. So the model is trained on 30 _handshapes_, and a
letters/numbers switch decides what a handshape means — which is how a human reads them
too.

The alternative, one flat classifier over 26 letters and 10 digits, does not merely fail
at the digits: it makes the **letters worse**, because probability mass belonging to `V`
gets divided between `V` and `2` and neither clears the margin threshold. Adding numbers
this way cost nothing — macro-F1 went from 0.968 across 24 classes to 0.973 across 30.

The whole thing is reproducible in about ten minutes:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e "training[dev]"
npm run train:simulated
```

(A virtual environment because recent macOS and Linux Pythons refuse a plain
`pip install` into the system interpreter. None of this is needed to run the app.)

## Training on real data

Simulation is a floor, not a ceiling. Both routes below will beat it.

**Fastest — record your own (~30 minutes).**

```bash
npm run dev                    # open http://localhost:5173/recorder.html
# Record 24 letters + `none`, twice: once per hand, across a few lighting conditions.
# Get a second person to record too — a single-signer dataset is refused, on purpose.

python3 -m venv .venv && source .venv/bin/activate
pip install -e "training[dev]"
cd training
python -m mudra_train.train --data ./recordings --out ../packages/web/public/models/asl-fingerspell
```

**Best quality — FSboard.** 3M+ characters from 147 Deaf signers, CC BY 4.0. Needs a free
Kaggle account.

```bash
pip install kaggle                       # API token goes in ~/.kaggle/kaggle.json
kaggle datasets download -d google/fsboard
unzip fsboard.zip -d ./fsboard

cd training
# Check the adapter agrees with your download before committing to a long run.
python -m mudra_train.ingest.fsboard --root ../fsboard --inspect
python -m mudra_train.train_ctc --fsboard ../fsboard \
  --out ../packages/web/public/models/asl-fingerspell
```

FSboard trains a **CTC pack, not a static one**, and that is a feature. Its labels are
whole phrases with no per-frame alignment, so there is no honest way to cut per-letter
examples out of it — the temporal model learns the alignment itself. Which also means
**J and Z come for free**: motion letters are ordinary labels to a model that sees
sequences, and impossible for one that sees frames. The dwell timer disappears too, so
spelling becomes continuous instead of pausing on each letter.

The adapter discovers the schema rather than assuming it, and `--inspect` prints what it
found. That matters because it was written without access to the download; if the release
differs, the output says exactly how.

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
exports JSONL. Phase 2's signer-independent evaluation depends on those tags. The page
shows what it is doing while you record: a skeleton overlay and state badge make dropped
tracking visible, each held sign is checked against the current model in real time
(advisory — capture never waits for it), a wrong-hand warning guards the handedness tag,
and every sample autosaves locally so a refresh or crash loses nothing.

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

- **The shipped pack is trained on simulated hands.** It replaces the v1 fallback (which
  systematically misread E, K, M, N, T and X), but M, N, T and E remain its weakest letters
  and J/Z need a `temporal-ctc` pack. Retraining on real recordings beats it.
- **The in-app accuracy test is not a benchmark.** It measures one person in one session and
  cannot predict performance for anyone else (A3). Real evaluation lives in `training/`.
- **The Dictionary lists only the alphabet and numbers.** Word categories were removed
  until a word-level pack and artwork exist to back them.
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
