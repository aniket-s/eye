/**
 * The v2 recognition pipeline.
 *
 * Frame → normalise → classify → smooth → judge → letter.
 *
 * Contrast with {@link LegacyRecognizer}, which it replaces:
 *
 * | Stage | v1 | v2 |
 * | --- | --- | --- |
 * | Features | wrist-relative, includes `z`, right hands only | centroid/RMS, 2-D, both hands |
 * | Correction | 85 lines of hand-tuned overrides | none — the model is simply right |
 * | Rejection | one threshold on an uncalibrated softmax | `none` class + margin + energy |
 * | Smoothing | compares consecutive argmax labels | median filter over probabilities |
 *
 * **There is no `geometricFix` here, and there never will be.** Those overrides
 * existed to patch a classifier that could not generalise; a correctly normalised
 * model trained with a negative class does not need patching, and reintroducing
 * per-letter special cases would recreate the coupling that made v1 un-retrainable
 * (docs/AUDIT.md §2).
 */
import { normaliseHand } from '../features/normalise.js';
import {
  judge,
  ProbabilitySmoother,
  type RejectionThresholds,
  type Verdict,
} from '../decode/rejection.js';
import { NO_PREDICTION, type HandLandmarks } from '../types.js';
import type { Handedness } from '../vision/frame.js';

/** Anything that can turn features into class scores. */
export interface Classifier {
  readonly labels: readonly string[];
  classify(features: Float32Array): Promise<{
    readonly probabilities: number[];
    readonly logits: number[];
  }>;
}

export interface HandshapeRecognizerOptions {
  readonly thresholds: RejectionThresholds;
  /** Frames to smooth over. */
  readonly smoothingWindow?: number;
  readonly featureLength: number;
}

export interface HandshapeResult {
  /** The letter to display, or {@link NO_PREDICTION}. */
  readonly letter: string;
  /** Full verdict, for the debug overlay. */
  readonly verdict: Verdict | null;
}

const EMPTY: HandshapeResult = { letter: NO_PREDICTION, verdict: null };

export class HandshapeRecognizer {
  readonly #classifier: Classifier;
  readonly #thresholds: RejectionThresholds;
  readonly #smoother: ProbabilitySmoother;
  readonly #features: Float32Array;

  constructor(classifier: Classifier, options: HandshapeRecognizerOptions) {
    this.#classifier = classifier;
    this.#thresholds = options.thresholds;
    this.#smoother = new ProbabilitySmoother(
      classifier.labels.length,
      options.smoothingWindow ?? 8,
    );
    this.#features = new Float32Array(options.featureLength);
  }

  /** Clear smoothing history, e.g. when the hand leaves frame. */
  reset(): void {
    this.#smoother.reset();
  }

  /**
   * Recognise one frame.
   *
   * @param landmarks The primary hand, or `null` if none was detected.
   * @param handedness Which of the signer's hands it is. Left hands are canonicalised
   *   during normalisation, which is what makes left-handed signers work at all.
   */
  async recognise(
    landmarks: HandLandmarks | null,
    handedness: Handedness,
  ): Promise<HandshapeResult> {
    if (landmarks === null || landmarks.length === 0) {
      this.reset();
      return EMPTY;
    }

    normaliseHand(landmarks, handedness, this.#features);
    const { probabilities, logits } = await this.#classifier.classify(this.#features);

    // Smooth probabilities, not labels. One flickered frame is outvoted by the
    // median instead of resetting everything downstream.
    const smoothed = this.#smoother.push(probabilities);
    const verdict = judge(this.#classifier.labels, smoothed, logits, this.#thresholds);

    return { letter: verdict.accepted ? verdict.label : NO_PREDICTION, verdict };
  }
}
