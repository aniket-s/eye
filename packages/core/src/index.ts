/**
 * `@mudrapragyan/core` — DOM-free recognition logic.
 *
 * Every export here is pure TypeScript with no browser dependencies, so the whole
 * recognition path can be unit-tested in Node against recorded landmark fixtures.
 */

export * from './types.js';

export { relu, softmax, argmax } from './math/activations.js';
export { MIN_SCALE, distance2d, distance3d, palmScale2d, palmScale3d } from './math/geometry.js';

export { landmarksToVector, LANDMARK_VECTOR_LENGTH } from './features/landmarkVector.js';
export { normaliseHand, mirrorHand, FEATURE_LENGTH } from './features/normalise.js';
export {
  normaliseBody,
  BODY_FEATURE_LENGTH,
  BODY_LAYOUT,
  BODY_NORMALISATION_SCHEME,
  POSE_POINTS,
} from './features/normaliseBody.js';

export {
  SignSegmenter,
  type SignSegmenterOptions,
  type SegmenterState,
  type SegmenterUpdate,
} from './temporal/signSegmenter.js';

export {
  IsolatedSignRecognizer,
  type IsolatedClassifier,
  type IsolatedSignRecognizerOptions,
  type IsolatedSignResult,
  type SignPrediction,
} from './pipeline/isolatedSignRecognizer.js';

export {
  energyScore,
  topMargin,
  judge,
  ProbabilitySmoother,
  NONE_LABEL,
  DEFAULT_THRESHOLDS,
  type RejectionThresholds,
  type Verdict,
} from './decode/rejection.js';

export {
  parseManifest,
  packRef,
  NORMALISATION_SCHEME,
  BODY_SCHEME,
  SUPPORTED_SCHEMES,
  type PackManifest,
  type PackTask,
  type PackInput,
  type PackMetrics,
} from './registry/manifest.js';
export {
  buildCustomSign,
  cosineSimilarity,
  normaliseVector,
  CustomSignBook,
  DEFAULT_MATCH_THRESHOLD,
  MIN_SAMPLES,
  type CustomSign,
  type CustomSignMatch,
} from './registry/customSigns.js';
export { MlpClassifier } from './model/mlp.js';

export { SentenceBuffer, type SentenceListener } from './sentence/sentenceBuffer.js';
export {
  emptyFrame,
  selectPrimaryHand,
  type Handedness,
  type HandObservation,
  type VisionFrame,
} from './vision/frame.js';
export { LandmarkView, packLandmarks, HAND_BUFFER_LENGTH } from './vision/landmarkBuffer.js';
export { packVisionFrame, VisionFrameView, FRAME_BUFFER_LENGTH } from './vision/framePacking.js';

export {
  TimedHoldCommitDetector,
  type TimedHoldCommitOptions,
  type TimedHoldCommitResult,
} from './temporal/timedHoldCommit.js';
export {
  HoldCommitDetector,
  type HoldCommitOptions,
  type HoldCommitResult,
} from './temporal/holdCommit.js';

export {
  BLANK_INDEX,
  ctcGreedyDecode,
  commonPrefix,
  levenshtein,
  characterErrorRate,
  LocalAgreementCommitter,
  type CtcDecodeResult,
} from './decode/ctc.js';

export {
  ContinuousRecognizer,
  type SequenceClassifier,
  type ContinuousRecognizerOptions,
  type ContinuousResult,
} from './pipeline/continuousRecognizer.js';

export {
  HandshapeRecognizer,
  type Classifier,
  type HandshapeRecognizerOptions,
  type HandshapeResult,
} from './pipeline/handshapeRecognizer.js';

export {
  LegacyRecognizer,
  MIN_CONFIDENCE,
  SPACE_LABEL,
  type RecognitionResult,
} from './pipeline/legacyRecognizer.js';

// Legacy — removed in Phases 2 and 3. See src/legacy/README.md.
export { geometricFix } from './legacy/geometricFix.js';
export { JzStateMachine } from './legacy/jzStateMachine.js';
