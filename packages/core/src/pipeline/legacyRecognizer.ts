/**
 * The v1 recognition pipeline, assembled from its extracted parts.
 *
 * Frame → feature vector → MLP → geometric overrides → J/Z motion → displayed letter.
 *
 * Phase 0 preserves this exactly. Phases 2 and 3 dismantle it: the geometric overrides
 * disappear when the classifier is retrained with proper normalisation and a `none`
 * class, and the J/Z stage disappears when CTC makes J and Z ordinary labels.
 */
import { landmarksToVector } from '../features/landmarkVector.js';
import { geometricFix } from '../legacy/geometricFix.js';
import { JzStateMachine } from '../legacy/jzStateMachine.js';
import type { MlpClassifier } from '../model/mlp.js';
import { NO_PREDICTION, type HandLandmarks } from '../types.js';

/**
 * Minimum softmax confidence for a prediction to be considered at all.
 *
 * Ineffective in v1: an overfit MLP with no negative class is *confidently* wrong on
 * transitional poses, so this filters far less than it appears to (docs/AUDIT.md, M5).
 */
export const MIN_CONFIDENCE = 0.45;

/**
 * Label the v1 model emits for "no sign". Suppressed rather than surfaced, because
 * it was never trained against real negatives (docs/AUDIT.md, M4).
 */
export const SPACE_LABEL = 'space';

export interface RecognitionResult {
  /** The letter to display, or {@link NO_PREDICTION}. */
  readonly letter: string;
  /** The model's raw `argmax` label before any override, for the debug overlay. */
  readonly rawLabel: string | null;
  /** The model's confidence in `rawLabel`. */
  readonly confidence: number | null;
}

const EMPTY_RESULT: RecognitionResult = {
  letter: NO_PREDICTION,
  rawLabel: null,
  confidence: null,
};

export class LegacyRecognizer {
  readonly #classifier: MlpClassifier;
  readonly #jz = new JzStateMachine();

  constructor(classifier: MlpClassifier) {
    this.#classifier = classifier;
  }

  /** Clear motion history, e.g. when the hand leaves the frame. */
  reset(): void {
    this.#jz.reset();
  }

  /**
   * Recognise one frame.
   *
   * @param landmarks The 21 landmarks of the primary hand, or `null` if none was detected.
   */
  recognise(landmarks: HandLandmarks | null): RecognitionResult {
    if (landmarks === null || landmarks.length === 0) {
      this.reset();
      return EMPTY_RESULT;
    }

    const prediction = this.#classifier.predict(landmarksToVector(landmarks));

    if (prediction.label === SPACE_LABEL) {
      return {
        letter: NO_PREDICTION,
        rawLabel: prediction.label,
        confidence: prediction.confidence,
      };
    }
    if (prediction.confidence < MIN_CONFIDENCE) {
      return {
        letter: NO_PREDICTION,
        rawLabel: prediction.label,
        confidence: prediction.confidence,
      };
    }

    const corrected = geometricFix(prediction.label, landmarks);
    const withMotion = this.#jz.update(corrected, prediction.confidence, landmarks);

    return {
      letter: withMotion,
      rawLabel: prediction.label,
      confidence: prediction.confidence,
    };
  }
}
