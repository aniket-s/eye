/**
 * Word-level sign recognition.
 *
 * Structurally different from both earlier pipelines. Fingerspelling classifies every
 * frame (static) or decodes a continuous stream (CTC); a word sign is a discrete
 * gesture with a beginning and an end, and produces one label for the whole thing.
 *
 * So the flow is: **segment, then classify**. {@link SignSegmenter} watches motion and
 * decides when a sign has finished; only then does the classifier run, once, on the
 * captured clip. That is far cheaper than classifying continuously — the model runs
 * perhaps once a second instead of six times — and it is also more accurate, because
 * the classifier sees the complete gesture rather than a fragment of one.
 *
 * Long clips are downsampled rather than truncated. Truncating would cut the *end* off
 * a sign, and in many signs the ending carries the meaning.
 */
import { BODY_FEATURE_LENGTH, normaliseBody } from '../features/normaliseBody.js';
import { SignSegmenter, type SignSegmenterOptions } from '../temporal/signSegmenter.js';
import type { Handedness, VisionFrame } from '../vision/frame.js';

/** A model that maps a clip of frames to class scores. */
export interface IsolatedClassifier {
  readonly labels: readonly string[];
  /**
   * @param clip Flattened features, `frames × featureLength`.
   * @param frames Frame count.
   * @returns One probability per label.
   */
  classifyClip(clip: Float32Array, frames: number): Promise<number[]>;
}

export interface IsolatedSignRecognizerOptions {
  /** Maximum frames retained per clip. Longer clips are downsampled to fit. */
  readonly maxFrames?: number;
  /** Minimum probability for a prediction to be reported. */
  readonly minProbability?: number;
  readonly featureLength?: number;
  readonly segmenter?: SignSegmenterOptions;
}

export interface SignPrediction {
  readonly label: string;
  readonly probability: number;
}

export interface IsolatedSignResult {
  /** True while a sign is being captured. */
  readonly capturing: boolean;
  /** Set on the frame a sign is recognised. */
  readonly recognised: SignPrediction | null;
  /** Runners-up, for a "did you mean" affordance. */
  readonly alternatives: readonly SignPrediction[];
  /** Smoothed motion, for the debug overlay. */
  readonly motion: number;
}

const DEFAULT_MAX_FRAMES = 96;
const DEFAULT_MIN_PROBABILITY = 0.35;
const ALTERNATIVE_COUNT = 3;

const IDLE: IsolatedSignResult = {
  capturing: false,
  recognised: null,
  alternatives: [],
  motion: 0,
};

export class IsolatedSignRecognizer {
  readonly #classifier: IsolatedClassifier;
  readonly #segmenter: SignSegmenter;
  readonly #maxFrames: number;
  readonly #minProbability: number;
  readonly #featureLength: number;

  /** Captured frames for the sign in progress. */
  readonly #buffer: Float32Array;
  #frames = 0;

  constructor(classifier: IsolatedClassifier, options: IsolatedSignRecognizerOptions = {}) {
    this.#classifier = classifier;
    this.#segmenter = new SignSegmenter(options.segmenter);
    this.#maxFrames = options.maxFrames ?? DEFAULT_MAX_FRAMES;
    this.#minProbability = options.minProbability ?? DEFAULT_MIN_PROBABILITY;
    this.#featureLength = options.featureLength ?? BODY_FEATURE_LENGTH;
    // Twice the cap, so a long sign can be captured in full and then downsampled
    // rather than being cut short as it is recorded.
    this.#buffer = new Float32Array(this.#maxFrames * 2 * this.#featureLength);
  }

  reset(): void {
    this.#segmenter.reset();
    this.#frames = 0;
  }

  /**
   * Feed one frame.
   *
   * @param frame Hands and pose from the vision layer.
   * @param dominant The signer's dominant hand.
   */
  async push(frame: VisionFrame, dominant: Handedness = 'right'): Promise<IsolatedSignResult> {
    const segment = this.#segmenter.update(frame, dominant);

    if (segment.state === 'active') {
      this.#capture(frame, dominant);
    }

    if (!segment.segmentComplete || this.#frames === 0) {
      return {
        capturing: segment.state === 'active',
        recognised: null,
        alternatives: [],
        motion: segment.motion,
      };
    }

    const { clip, frames } = this.#prepareClip();
    this.#frames = 0;

    const probabilities = await this.#classifier.classifyClip(clip, frames);
    const ranked = this.#rank(probabilities);
    const best = ranked[0];

    if (best === undefined || best.probability < this.#minProbability) {
      // Below threshold, report nothing rather than guessing. A word-level model with
      // 250 classes is confidently wrong on gestures that are not signs at all.
      return { ...IDLE, motion: segment.motion };
    }

    return {
      capturing: false,
      recognised: best,
      alternatives: ranked.slice(1, 1 + ALTERNATIVE_COUNT),
      motion: segment.motion,
    };
  }

  /** Append one frame's features, if there is room. */
  #capture(frame: VisionFrame, dominant: Handedness): void {
    const capacity = this.#buffer.length / this.#featureLength;
    if (this.#frames >= capacity) return;

    const offset = this.#frames * this.#featureLength;
    normaliseBody(frame, dominant, this.#buffer.subarray(offset, offset + this.#featureLength));
    this.#frames++;
  }

  /**
   * Produce the clip to classify, downsampling if it is longer than the cap.
   *
   * Uniform downsampling keeps the beginning *and* the end. Truncation would discard
   * the end, which in many signs is where the meaning lives — the difference between
   * a sign and its opposite is often the final movement.
   */
  #prepareClip(): { clip: Float32Array; frames: number } {
    if (this.#frames <= this.#maxFrames) {
      return {
        clip: this.#buffer.subarray(0, this.#frames * this.#featureLength),
        frames: this.#frames,
      };
    }

    const clip = new Float32Array(this.#maxFrames * this.#featureLength);
    for (let target = 0; target < this.#maxFrames; target++) {
      const source = Math.round((target * (this.#frames - 1)) / (this.#maxFrames - 1));
      clip.set(
        this.#buffer.subarray(source * this.#featureLength, (source + 1) * this.#featureLength),
        target * this.#featureLength,
      );
    }
    return { clip, frames: this.#maxFrames };
  }

  /** Sort labels by probability, descending. */
  #rank(probabilities: readonly number[]): SignPrediction[] {
    return probabilities
      .map((probability, index) => ({
        label: this.#classifier.labels[index] ?? '?',
        probability,
      }))
      .sort((a, b) => b.probability - a.probability);
  }
}
