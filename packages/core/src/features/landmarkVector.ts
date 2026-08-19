import { palmScale3d } from '../math/geometry.js';
import { HandLandmarkIndex, LANDMARKS_PER_HAND, type HandLandmarks } from '../types.js';

/** Length of the feature vector consumed by the v1 MLP: 21 landmarks × (x, y, z). */
export const LANDMARK_VECTOR_LENGTH = LANDMARKS_PER_HAND * 3;

/**
 * Flatten one hand into the 63-element feature vector the v1 model expects.
 *
 * The hand is made translation-invariant (wrist moved to the origin) and
 * scale-invariant (divided by the 3D wrist-to-middle-MCP distance).
 *
 * **Known limitations**, carried forward unchanged from v1 and addressed in Phase 2:
 * - Not invariant to handedness — a left hand yields a mirrored vector the model
 *   has never seen (docs/AUDIT.md, M1).
 * - Not invariant to in-plane rotation (M2).
 * - Includes `z`, which is weakly calibrated from a single view (M3).
 *
 * @param landmarks Exactly 21 MediaPipe hand landmarks.
 * @returns 63 numbers, ordered `[x0, y0, z0, x1, y1, z1, ...]`.
 * @throws RangeError if `landmarks` is not exactly 21 points.
 */
export function landmarksToVector(landmarks: HandLandmarks): number[] {
  if (landmarks.length !== LANDMARKS_PER_HAND) {
    throw new RangeError(`Expected ${LANDMARKS_PER_HAND} landmarks, received ${landmarks.length}`);
  }

  const wrist = landmarks[HandLandmarkIndex.WRIST]!;
  const middleMcp = landmarks[HandLandmarkIndex.MIDDLE_MCP]!;
  const scale = palmScale3d(wrist, middleMcp);

  const vector = new Array<number>(LANDMARK_VECTOR_LENGTH);
  for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
    const point = landmarks[i]!;
    const base = i * 3;
    vector[base] = (point.x - wrist.x) / scale;
    vector[base + 1] = (point.y - wrist.y) / scale;
    vector[base + 2] = (point.z - wrist.z) / scale;
  }
  return vector;
}
