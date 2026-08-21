/**
 * Model pack manifests.
 *
 * A pack is a directory carrying everything needed to run a model:
 *
 * ```
 * models/asl-fingerspell/v1.0.0/
 * ├── manifest.json
 * ├── model.onnx
 * └── card.md
 * ```
 *
 * The point is that **thresholds and normalisation travel with the weights**. v1
 * hardcoded its confidence threshold and 85 lines of correction heuristics into the
 * application, so retraining silently invalidated both — the coupling that made the
 * model un-retrainable (docs/AUDIT.md §2, ADR 0003).
 *
 * Adding Indian Sign Language, a word-level model, or a user's own signs is then a
 * new pack rather than a code change.
 */
import type { RejectionThresholds } from '../decode/rejection.js';

/** What a pack's model consumes and emits. */
export type PackTask =
  /** One frame in, one class out. Fingerspelling and numbers. */
  | 'static-handshape'
  /** A sequence of frames in, a sequence of labels out. Phase 3. */
  | 'temporal-ctc'
  /** A sequence in, a single class out. Word-level signs, Phase 4. */
  | 'temporal-isolated';

/** Feature extraction the model was trained against. */
export interface PackInput {
  /** Feature vector length. Must match what the normaliser produces. */
  readonly featureLength: number;
  /** Number of hands the model consumes. */
  readonly hands: 1 | 2;
  /**
   * Normalisation scheme identifier, e.g. `centroid-rms-v2`.
   *
   * Checked at load time. A pack trained under a different scheme than the runtime
   * implements is refused rather than silently mispredicting — this is the guard
   * against train/inference skew.
   */
  readonly normalisation: string;
  /** Frames per window. Absent for single-frame models. */
  readonly windowFrames?: number;
}

/**
 * `true letter -> { predicted letter -> probability }`.
 *
 * Rows need not sum to 1: correct predictions are excluded, and pairs too rare to
 * estimate are dropped at export.
 */
export type ConfusionProfile = Readonly<Record<string, Readonly<Record<string, number>>>>;

/** `vocabulary -> { trained label -> displayed text }`. */
export type Vocabularies = Readonly<Record<string, Readonly<Record<string, string>>>>;

/** Headline metrics, copied from the evaluation report. */
export interface PackMetrics {
  /** Macro-averaged F1 on the signer-independent test split. */
  readonly macroF1: number;
  /** Worst per-slice macro-F1. The number that actually matters. */
  readonly worstSliceF1: number;
  /** Expected calibration error. */
  readonly ece?: number;
  /** Signers held out of training. */
  readonly testSigners: number;
}

export interface PackManifest {
  /** Stable identifier, e.g. `asl-fingerspell`. */
  readonly id: string;
  /** Semantic version of this pack's weights. */
  readonly version: string;
  /** Human-readable name for the model picker. */
  readonly name: string;
  /** Sign language this pack recognises, as a BCP-47-style tag, e.g. `ase`, `ins`. */
  readonly language: string;
  readonly task: PackTask;
  /** Class labels, in model output order. Includes `none` for rejection. */
  readonly labels: readonly string[];
  readonly input: PackInput;
  /** Operating point chosen during evaluation, not guessed by the app. */
  readonly thresholds: RejectionThresholds;
  readonly metrics: PackMetrics;
  /**
   * Which letters this model mistakes for which, as `P(predicted | true)` measured on the
   * held-out signers.
   *
   * Travels with the weights for the same reason the thresholds do: it describes *this*
   * model's behaviour and is invalidated by a retrain. Word correction uses it to weight
   * substitutions — a K read by a given pack may be overwhelmingly likely to be a V and
   * vanishingly likely to be a W, and a generic spell-checker cannot know that.
   *
   * Optional: packs exported before this existed have none, and correction falls back to
   * treating every substitution as equally likely.
   */
  readonly confusions?: ConfusionProfile;
  /**
   * How often each label was accepted *and* correct on the held-out signers, at this
   * pack's own thresholds: `label -> 0..1`.
   *
   * Distinct from `metrics.macroF1`, and the distinction is the point. F1 scores the
   * `argmax`; the app never shows the `argmax`. It shows a letter only once the
   * smoothed probability and the margin over the runner-up both clear the thresholds
   * above, and shows nothing at all otherwise. A letter can therefore carry an
   * excellent F1 and still almost never appear, because its runner-up sits just behind
   * it — which is invisible in every aggregate the pack reports.
   *
   * Surfaced in the Translator's debug panel so "this letter never works" has an
   * answer that does not require a retrain to discover.
   *
   * Optional: packs exported before this existed have none.
   */
  readonly acceptance?: Readonly<Record<string, number>>;
  /**
   * Named readings of the trained handshapes: `vocabulary -> { label -> displayed text }`.
   *
   * The model predicts handshapes; a vocabulary decides what one *means*. ASL numbers
   * make this necessary rather than decorative — 2 and V are the same handshape, as are
   * 6/W, 9/F and 0/O, and a signer tells them apart from context. Training them as
   * separate classes would ask the network to split classes carrying no distinguishing
   * signal, and would cost letter accuracy to do it.
   *
   * Absent for a pack with only one reading, which is what a letters-only pack is.
   */
  readonly vocabularies?: Vocabularies;
  /** Model file name within the pack directory. */
  readonly modelFile: string;
  /** SHA-256 of the model file, so CI can verify what shipped. */
  readonly sha256: string;
  /** Licence of the data the model was trained on. */
  readonly licence: string;
  /** Attribution required by that licence. */
  readonly attribution?: string;
}

/** Single-hand scheme, for fingerspelling packs. */
export const NORMALISATION_SCHEME = 'centroid-rms-v2';

/** Two-handed body-relative scheme, for word-level packs. */
export const BODY_SCHEME = 'shoulder-frame-v1';

/**
 * Every normalisation scheme this build implements.
 *
 * A pack naming anything else was trained against feature extraction the runtime does
 * not have, and must be refused rather than left to mispredict silently.
 */
export const SUPPORTED_SCHEMES: readonly string[] = [NORMALISATION_SCHEME, BODY_SCHEME];

/** Which scheme each task expects, so a mismatched pair is caught too. */
const SCHEME_FOR_HANDS: Record<number, string> = {
  1: NORMALISATION_SCHEME,
  2: BODY_SCHEME,
};

/**
 * Validate an untrusted manifest.
 *
 * @throws TypeError naming the specific field that is wrong.
 */
export function parseManifest(value: unknown): PackManifest {
  if (typeof value !== 'object' || value === null) {
    throw new TypeError('Manifest must be an object');
  }
  const raw = value as Record<string, unknown>;

  const requireString = (key: string): string => {
    const found = raw[key];
    if (typeof found !== 'string' || found.length === 0) {
      throw new TypeError(`Manifest: \`${key}\` must be a non-empty string`);
    }
    return found;
  };

  const labels = raw['labels'];
  if (
    !Array.isArray(labels) ||
    labels.length === 0 ||
    !labels.every((l) => typeof l === 'string')
  ) {
    throw new TypeError('Manifest: `labels` must be a non-empty array of strings');
  }
  if (new Set(labels).size !== labels.length) {
    throw new TypeError('Manifest: `labels` contains duplicates');
  }

  const task = requireString('task');
  if (task !== 'static-handshape' && task !== 'temporal-ctc' && task !== 'temporal-isolated') {
    throw new TypeError(`Manifest: unknown task "${task}"`);
  }

  const input = raw['input'];
  if (typeof input !== 'object' || input === null) {
    throw new TypeError('Manifest: `input` must be an object');
  }
  const inputRaw = input as Record<string, unknown>;
  const featureLength = inputRaw['featureLength'];
  if (typeof featureLength !== 'number' || !Number.isInteger(featureLength) || featureLength <= 0) {
    throw new TypeError('Manifest: `input.featureLength` must be a positive integer');
  }
  const hands = inputRaw['hands'];
  if (hands !== 1 && hands !== 2) {
    throw new TypeError('Manifest: `input.hands` must be 1 or 2');
  }
  const normalisation = inputRaw['normalisation'];
  if (typeof normalisation !== 'string') {
    throw new TypeError('Manifest: `input.normalisation` must be a string');
  }
  if (!SUPPORTED_SCHEMES.includes(normalisation)) {
    throw new TypeError(
      `Manifest: normalisation "${normalisation}" is not implemented by this build ` +
        `(supported: ${SUPPORTED_SCHEMES.join(', ')}). ` +
        'Refusing to load rather than mispredict silently.',
    );
  }
  const expectedScheme = SCHEME_FOR_HANDS[hands];
  if (expectedScheme !== undefined && normalisation !== expectedScheme) {
    throw new TypeError(
      `Manifest: a ${hands}-handed pack must use "${expectedScheme}", not "${normalisation}". ` +
        'Refusing to load rather than mispredict silently.',
    );
  }

  const thresholds = raw['thresholds'];
  if (typeof thresholds !== 'object' || thresholds === null) {
    throw new TypeError('Manifest: `thresholds` must be an object');
  }
  const thresholdsRaw = thresholds as Record<string, unknown>;
  const minProbability = thresholdsRaw['minProbability'];
  const minMargin = thresholdsRaw['minMargin'];
  const maxEnergy = thresholdsRaw['maxEnergy'] ?? null;
  if (typeof minProbability !== 'number' || typeof minMargin !== 'number') {
    throw new TypeError('Manifest: `thresholds.minProbability` and `.minMargin` must be numbers');
  }
  if (maxEnergy !== null && typeof maxEnergy !== 'number') {
    throw new TypeError('Manifest: `thresholds.maxEnergy` must be a number or null');
  }

  const metrics = raw['metrics'];
  if (typeof metrics !== 'object' || metrics === null) {
    throw new TypeError('Manifest: `metrics` must be an object');
  }
  const metricsRaw = metrics as Record<string, unknown>;
  const macroF1 = metricsRaw['macroF1'];
  const worstSliceF1 = metricsRaw['worstSliceF1'];
  const testSigners = metricsRaw['testSigners'];
  if (typeof macroF1 !== 'number' || typeof worstSliceF1 !== 'number') {
    throw new TypeError('Manifest: `metrics.macroF1` and `.worstSliceF1` must be numbers');
  }
  if (typeof testSigners !== 'number' || testSigners < 1) {
    throw new TypeError(
      'Manifest: `metrics.testSigners` must be at least 1 — a model with no held-out ' +
        'signers has not been evaluated (docs/AUDIT.md, A3)',
    );
  }

  const confusions = parseConfusions(raw['confusions']);
  const vocabularies = parseVocabularies(raw['vocabularies'], labels);
  const acceptance = parseAcceptance(raw['acceptance'], labels);

  const manifest: PackManifest = {
    id: requireString('id'),
    version: requireString('version'),
    name: requireString('name'),
    language: requireString('language'),
    task,
    labels,
    input: {
      featureLength,
      hands,
      normalisation,
      ...(typeof inputRaw['windowFrames'] === 'number'
        ? { windowFrames: inputRaw['windowFrames'] }
        : {}),
    },
    thresholds: { minProbability, minMargin, maxEnergy },
    metrics: {
      macroF1,
      worstSliceF1,
      testSigners,
      ...(typeof metricsRaw['ece'] === 'number' ? { ece: metricsRaw['ece'] } : {}),
    },
    ...(confusions === null ? {} : { confusions }),
    ...(acceptance === null ? {} : { acceptance }),
    ...(vocabularies === null ? {} : { vocabularies }),
    modelFile: requireString('modelFile'),
    sha256: requireString('sha256'),
    licence: requireString('licence'),
    ...(typeof raw['attribution'] === 'string' ? { attribution: raw['attribution'] } : {}),
  };

  return manifest;
}

/**
 * Read the acceptance profile, if a pack carries one.
 *
 * Lenient by design: this is diagnostic data, not an operating parameter. A malformed
 * or partial profile costs the debug panel a line, so entries that do not name a
 * trained label or do not carry a share in 0..1 are dropped rather than thrown over.
 * Refusing to load a working model because a reporting field is odd would be the wrong
 * trade.
 */
function parseAcceptance(raw: unknown, labels: readonly string[]): Record<string, number> | null {
  if (raw === undefined || raw === null) return null;
  if (typeof raw !== 'object' || Array.isArray(raw)) return null;

  const known = new Set(labels);
  const parsed: Record<string, number> = {};
  for (const [label, share] of Object.entries(raw as Record<string, unknown>)) {
    if (!known.has(label)) continue;
    if (typeof share !== 'number' || !Number.isFinite(share) || share < 0 || share > 1) {
      continue;
    }
    parsed[label] = share;
  }
  return Object.keys(parsed).length > 0 ? parsed : null;
}

/**
 * Validate the vocabularies, if a pack declares any.
 *
 * Every entry must name a label the model was actually trained on. A vocabulary
 * referencing a class that does not exist would silently mask that reading out of
 * existence — the mode would appear in the UI and recognise nothing.
 */
function parseVocabularies(raw: unknown, labels: readonly string[]): Vocabularies | null {
  if (raw === undefined || raw === null) return null;
  if (typeof raw !== 'object' || Array.isArray(raw)) {
    throw new TypeError('Manifest: `vocabularies` must be an object');
  }

  const known = new Set(labels);
  const parsed: Record<string, Record<string, string>> = {};

  for (const [name, entries] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof entries !== 'object' || entries === null || Array.isArray(entries)) {
      throw new TypeError(`Manifest: \`vocabularies.${name}\` must be an object`);
    }
    const mapping: Record<string, string> = {};
    for (const [label, display] of Object.entries(entries as Record<string, unknown>)) {
      if (typeof display !== 'string' || display === '') {
        throw new TypeError(`Manifest: \`vocabularies.${name}.${label}\` must be a string`);
      }
      if (!known.has(label)) {
        throw new TypeError(
          `Manifest: \`vocabularies.${name}\` refers to "${label}", which is not one of ` +
            "this pack's labels",
        );
      }
      mapping[label] = display;
    }
    if (Object.keys(mapping).length === 0) {
      throw new TypeError(`Manifest: \`vocabularies.${name}\` is empty`);
    }
    parsed[name] = mapping;
  }

  return parsed;
}

/**
 * Validate the confusion profile, if a pack carries one.
 *
 * Absent is normal — older packs have none. Malformed is not, and is rejected rather than
 * ignored: a profile with a stray string or a probability of 4 would silently skew every
 * suggestion the app makes, which is far harder to notice than a refusal to load.
 */
function parseConfusions(raw: unknown): ConfusionProfile | null {
  if (raw === undefined || raw === null) return null;
  if (typeof raw !== 'object' || Array.isArray(raw)) {
    throw new TypeError('Manifest: `confusions` must be an object');
  }

  const profile: Record<string, Record<string, number>> = {};
  for (const [trueLabel, row] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof row !== 'object' || row === null || Array.isArray(row)) {
      throw new TypeError(`Manifest: \`confusions.${trueLabel}\` must be an object`);
    }
    const parsed: Record<string, number> = {};
    for (const [predicted, probability] of Object.entries(row as Record<string, unknown>)) {
      if (typeof probability !== 'number' || !Number.isFinite(probability)) {
        throw new TypeError(`Manifest: \`confusions.${trueLabel}.${predicted}\` must be a number`);
      }
      if (probability <= 0 || probability > 1) {
        throw new TypeError(
          `Manifest: \`confusions.${trueLabel}.${predicted}\` must be a probability in (0, 1]`,
        );
      }
      parsed[predicted] = probability;
    }
    profile[trueLabel] = parsed;
  }
  return profile;
}

/** `id@version`, for logs and the model picker. */
export function packRef(manifest: PackManifest): string {
  return `${manifest.id}@${manifest.version}`;
}
