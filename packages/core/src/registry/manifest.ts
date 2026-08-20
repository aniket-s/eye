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
  /** Model file name within the pack directory. */
  readonly modelFile: string;
  /** SHA-256 of the model file, so CI can verify what shipped. */
  readonly sha256: string;
  /** Licence of the data the model was trained on. */
  readonly licence: string;
  /** Attribution required by that licence. */
  readonly attribution?: string;
}

/** The normalisation scheme this build of `core` implements. */
export const NORMALISATION_SCHEME = 'centroid-rms-v2';

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
  if (normalisation !== NORMALISATION_SCHEME) {
    throw new TypeError(
      `Manifest: normalisation "${normalisation}" does not match this build's ` +
        `"${NORMALISATION_SCHEME}". Refusing to load rather than mispredict silently.`,
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
    modelFile: requireString('modelFile'),
    sha256: requireString('sha256'),
    licence: requireString('licence'),
    ...(typeof raw['attribution'] === 'string' ? { attribution: raw['attribution'] } : {}),
  };

  return manifest;
}

/** `id@version`, for logs and the model picker. */
export function packRef(manifest: PackManifest): string {
  return `${manifest.id}@${manifest.version}`;
}
