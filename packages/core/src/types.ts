/**
 * Shared types for the recognition pipeline.
 *
 * Everything in `@mudrapragyan/core` is DOM-free and side-effect-free so it can be
 * unit-tested in Node against recorded landmark fixtures. Nothing here may import
 * from `window`, `document`, or any browser global.
 */

/** A single MediaPipe hand landmark in normalised image coordinates. */
export interface Landmark {
  /** Horizontal position, 0 (image left) to 1 (image right). */
  readonly x: number;
  /** Vertical position, 0 (image top) to 1 (image bottom). Y grows downward. */
  readonly y: number;
  /**
   * Depth relative to the wrist, in roughly the same scale as `x`.
   *
   * Weakly calibrated and depth-ambiguous from a single view. Retained here because
   * the v1 model consumes it; scheduled for removal in Phase 2 (see docs/AUDIT.md, M3).
   */
  readonly z: number;
}

/**
 * One hand's landmarks. MediaPipe's hand topology is always exactly 21 points,
 * indexed per {@link HandLandmarkIndex}.
 */
export type HandLandmarks = readonly Landmark[];

/** Named indices into a 21-point MediaPipe hand, replacing bare magic numbers. */
export const HandLandmarkIndex = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20,
} as const;

export type HandLandmarkIndex = (typeof HandLandmarkIndex)[keyof typeof HandLandmarkIndex];

/** Number of landmarks MediaPipe reports per hand. */
export const LANDMARKS_PER_HAND = 21;

/** A classifier's output for one frame. */
export interface Prediction {
  /** The winning class label. */
  readonly label: string;
  /** Softmax probability of the winning class, in [0, 1]. */
  readonly confidence: number;
}

/**
 * Weights for the v1 multi-layer perceptron, as exported from scikit-learn.
 *
 * Superseded in Phase 2 by versioned ONNX model packs; see docs/adr/0003-model-packs.md.
 */
export interface MlpWeights {
  /** Class labels, ordered to match the final layer's output units. */
  readonly labels: readonly string[];
  /** Per-feature mean used to standardise the input vector. */
  readonly scaler_mean: readonly number[];
  /** Per-feature standard deviation used to standardise the input vector. */
  readonly scaler_std: readonly number[];
  /** Weight matrices, one per layer, each shaped `[fanIn][fanOut]`. */
  readonly coefs: readonly (readonly (readonly number[])[])[];
  /** Bias vectors, one per layer, each of length `fanOut`. */
  readonly intercepts: readonly (readonly number[])[];
  /** Hidden-layer activation. Only `"relu"` is supported. */
  readonly activation?: string;
}

/** Sentinel emitted when no hand is present or the prediction was rejected. */
export const NO_PREDICTION = '?' as const;
export type NoPrediction = typeof NO_PREDICTION;
