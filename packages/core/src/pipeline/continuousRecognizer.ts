/**
 * Continuous fingerspelling.
 *
 * Landmarks arrive at 30–60 Hz; a temporal model over a 60-frame window costs ~10 ms.
 * Running it per frame would consume the entire budget for no benefit, because sign
 * boundaries do not move at 60 Hz. So frames accumulate in a ring buffer and the model
 * runs on a **stride** — typically every 5 frames, or about 6 Hz — which is ample
 * resolution for letter boundaries and leaves the frame budget intact.
 *
 * Each run re-decodes the whole window, so the hypothesis for recent frames can change
 * as more context arrives. {@link LocalAgreementCommitter} absorbs that: stable text is
 * committed, the unstable tail stays provisional, and the user watches it settle rather
 * than watching letters flicker.
 *
 * What this replaces:
 * - The dwell timer, which forced a two-thirds-of-a-second pause per letter.
 * - The J/Z state machine, since a temporal model reads motion directly and J and Z
 *   are just labels in the CTC alphabet.
 */
import { ctcGreedyDecode, LocalAgreementCommitter, type CtcDecodeResult } from '../decode/ctc.js';
import { normaliseHand, FEATURE_LENGTH } from '../features/normalise.js';
import type { HandLandmarks } from '../types.js';
import type { Handedness } from '../vision/frame.js';

/** A model that maps a window of frames to per-frame class probabilities. */
export interface SequenceClassifier {
  /** Labels including the blank symbol. */
  readonly labels: readonly string[];
  readonly blankIndex: number;
  /**
   * @param window Flattened features, `frames × featureLength`.
   * @param frames How many frames the window holds.
   * @returns One probability vector per frame.
   */
  classifySequence(window: Float32Array, frames: number): Promise<number[][]>;
}

export interface ContinuousRecognizerOptions {
  /** Frames held in the window. */
  readonly windowFrames: number;
  /** Run the model every this many frames. */
  readonly strideFrames?: number;
  /** Hypotheses that must agree before text is committed. */
  readonly agreement?: number;
  /** Feature length per frame. */
  readonly featureLength?: number;
  /**
   * Frames of absent hand before the window is flushed and reset.
   *
   * Holding stale frames across a pause would let the decoder splice two unrelated
   * spellings into one word.
   */
  readonly idleResetFrames?: number;
}

export interface ContinuousResult {
  /** Text the decoder is confident about. Only ever grows. */
  readonly committed: readonly string[];
  /** Current best guess for the tail, shown greyed. */
  readonly provisional: readonly string[];
  /** True when the model ran on this frame rather than only buffering. */
  readonly decoded: boolean;
  readonly confidence: number;
}

const DEFAULT_STRIDE = 5;
const DEFAULT_AGREEMENT = 2;
const DEFAULT_IDLE_RESET = 20;

export class ContinuousRecognizer {
  readonly #classifier: SequenceClassifier;
  readonly #windowFrames: number;
  readonly #strideFrames: number;
  readonly #featureLength: number;
  readonly #idleResetFrames: number;
  readonly #committer: LocalAgreementCommitter;

  /** Ring buffer of normalised frames, `windowFrames × featureLength`. */
  readonly #buffer: Float32Array;
  /** Scratch used to present the ring buffer in chronological order. */
  readonly #ordered: Float32Array;

  #filled = 0;
  #writeIndex = 0;
  #sinceDecode = 0;
  #idleFrames = 0;
  #lastResult: ContinuousResult = {
    committed: [],
    provisional: [],
    decoded: false,
    confidence: 0,
  };

  constructor(classifier: SequenceClassifier, options: ContinuousRecognizerOptions) {
    if (options.windowFrames < 1) throw new RangeError('windowFrames must be at least 1');

    this.#classifier = classifier;
    this.#windowFrames = options.windowFrames;
    this.#strideFrames = options.strideFrames ?? DEFAULT_STRIDE;
    this.#featureLength = options.featureLength ?? FEATURE_LENGTH;
    this.#idleResetFrames = options.idleResetFrames ?? DEFAULT_IDLE_RESET;
    this.#committer = new LocalAgreementCommitter(options.agreement ?? DEFAULT_AGREEMENT);

    const capacity = this.#windowFrames * this.#featureLength;
    this.#buffer = new Float32Array(capacity);
    this.#ordered = new Float32Array(capacity);
  }

  /** Everything committed so far, as a string. */
  get text(): string {
    return this.#committer.committed.join('');
  }

  /** Clear the window and all committed text. */
  reset(): void {
    this.#filled = 0;
    this.#writeIndex = 0;
    this.#sinceDecode = 0;
    this.#idleFrames = 0;
    this.#committer.reset();
    this.#lastResult = { committed: [], provisional: [], decoded: false, confidence: 0 };
  }

  /** Drop buffered frames but keep committed text, e.g. after a pause. */
  flushWindow(): void {
    this.#filled = 0;
    this.#writeIndex = 0;
    this.#sinceDecode = 0;
  }

  /**
   * Feed one frame.
   *
   * @param landmarks The primary hand, or `null` when none is visible.
   * @param handedness Which hand it is; left hands are canonicalised.
   */
  async push(landmarks: HandLandmarks | null, handedness: Handedness): Promise<ContinuousResult> {
    if (landmarks === null || landmarks.length === 0) {
      this.#idleFrames++;
      // A sustained pause ends the current word. Without this, the letters before and
      // after a pause would be decoded as one continuous spelling.
      if (this.#idleFrames >= this.#idleResetFrames && this.#filled > 0) {
        this.flushWindow();
      }
      return { ...this.#lastResult, decoded: false };
    }
    this.#idleFrames = 0;

    const offset = this.#writeIndex * this.#featureLength;
    normaliseHand(
      landmarks,
      handedness,
      this.#buffer.subarray(offset, offset + this.#featureLength),
    );
    this.#writeIndex = (this.#writeIndex + 1) % this.#windowFrames;
    this.#filled = Math.min(this.#filled + 1, this.#windowFrames);
    this.#sinceDecode++;

    // Wait for enough context before the first decode, then run on the stride.
    const minimumFrames = Math.min(this.#windowFrames, this.#strideFrames * 2);
    if (this.#filled < minimumFrames || this.#sinceDecode < this.#strideFrames) {
      return { ...this.#lastResult, decoded: false };
    }
    this.#sinceDecode = 0;

    const window = this.#chronological();
    const frameProbabilities = await this.#classifier.classifySequence(window, this.#filled);
    const decoded: CtcDecodeResult = ctcGreedyDecode(
      frameProbabilities,
      this.#classifier.labels,
      this.#classifier.blankIndex,
    );

    const { committed, provisional } = this.#committer.update(decoded.labels);
    this.#lastResult = {
      committed,
      provisional,
      decoded: true,
      confidence: decoded.confidence,
    };
    return this.#lastResult;
  }

  /**
   * Present the ring buffer oldest-frame-first.
   *
   * The model needs chronological order; the ring stores frames wherever the write
   * head happened to be. Copying into a reused scratch array keeps this allocation-free.
   */
  #chronological(): Float32Array {
    const view = this.#ordered.subarray(0, this.#filled * this.#featureLength);

    if (this.#filled < this.#windowFrames) {
      // Not yet wrapped: frames sit in order from index 0.
      view.set(this.#buffer.subarray(0, this.#filled * this.#featureLength));
      return view;
    }

    // Wrapped: the oldest frame is at the write head.
    const splitFrame = this.#writeIndex;
    const splitOffset = splitFrame * this.#featureLength;
    const tailLength = this.#buffer.length - splitOffset;
    view.set(this.#buffer.subarray(splitOffset), 0);
    view.set(this.#buffer.subarray(0, splitOffset), tailLength);
    return view;
  }
}
