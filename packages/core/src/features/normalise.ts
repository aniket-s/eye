/**
 * Feature normalisation — v2.
 *
 * Replaces {@link landmarksToVector}, which was translation- and scale-invariant but
 * nothing else (docs/AUDIT.md, M1–M3).
 *
 * ## What is normalised, and what deliberately is not
 *
 * | Nuisance | Handled by |
 * | --- | --- |
 * | Where the hand sits in frame | Centering on the landmark centroid |
 * | Distance from the camera | Dividing by RMS spread |
 * | Left vs right hand | Mirroring left hands (M1) |
 * | Depth noise | Dropping `z` (M3) |
 * | Small hand tilt | **Augmentation at training time, not here** |
 *
 * ## Why rotation is NOT normalised
 *
 * This is the important design decision, and it is the opposite of what the audit's
 * M2 wording implies. Rotating every hand to a canonical orientation would make the
 * model *unable to read ASL*, because several letters differ **only** by orientation:
 *
 * - **K** and **P** are the same handshape; P points downward.
 * - **G** and **Q** are the same handshape; Q points downward.
 * - **H** and **U** are the same handshape at different angles.
 *
 * Canonicalising rotation would merge each of those pairs into one class. So gravity-
 * relative orientation is preserved as signal, and robustness to *incidental* tilt is
 * bought with rotation augmentation during training instead. Two independent top
 * solutions in Google's sign-recognition competitions report the same finding: explicit
 * rotation canonicalisation did not help.
 *
 * ## Why the centroid rather than the wrist
 *
 * v1 placed the wrist at the origin. The wrist sits at the edge of the hand, so
 * MediaPipe's jitter there moves every other point. The centroid of all 21 landmarks
 * is far more stable frame to frame.
 *
 * ## Why RMS spread rather than a bone length
 *
 * v1 divided by the wrist→middle-MCP distance. That single bone foreshortens sharply
 * when the palm tilts toward the camera, so the scale factor changed with pose rather
 * than distance. RMS spread over all landmarks is a pose-stable measure of apparent
 * hand size.
 */
import { LANDMARKS_PER_HAND, type HandLandmarks } from '../types.js';
import type { Handedness } from '../vision/frame.js';

/** Output length: 21 landmarks × (x, y). */
export const FEATURE_LENGTH = LANDMARKS_PER_HAND * 2;

/** Smallest permitted scale divisor, guarding a degenerate hand. */
const MIN_SPREAD = 1e-6;

/**
 * Turn one hand into the model's input features.
 *
 * @param landmarks Exactly 21 MediaPipe hand landmarks.
 * @param handedness Which of the signer's hands this is. Left hands are mirrored so
 *   the model only ever sees right-handed geometry — this is the fix for M1, and it
 *   also halves the input distribution the model has to cover.
 * @param target Optional buffer to write into, to avoid a per-frame allocation.
 * @returns 42 numbers, ordered `[x0, y0, x1, y1, ...]`.
 * @throws RangeError if the landmark count or buffer length is wrong.
 */
export function normaliseHand(
  landmarks: HandLandmarks,
  handedness: Handedness,
  target: Float32Array = new Float32Array(FEATURE_LENGTH),
): Float32Array {
  if (landmarks.length !== LANDMARKS_PER_HAND) {
    throw new RangeError(`Expected ${LANDMARKS_PER_HAND} landmarks, received ${landmarks.length}`);
  }
  if (target.length !== FEATURE_LENGTH) {
    throw new RangeError(`Expected a buffer of ${FEATURE_LENGTH}, received ${target.length}`);
  }

  // 1. Handedness. Mirroring x is sufficient for a single hand: the landmark indices
  //    are anatomically defined, so a mirrored left hand *is* a right hand. (A
  //    two-handed model must additionally swap the left and right index groups.)
  const flip = handedness === 'left' ? -1 : 1;

  // 2. Centre on the centroid.
  let centroidX = 0;
  let centroidY = 0;
  for (const point of landmarks) {
    centroidX += point.x * flip;
    centroidY += point.y;
  }
  centroidX /= LANDMARKS_PER_HAND;
  centroidY /= LANDMARKS_PER_HAND;

  // 3. Scale by RMS distance from the centroid.
  let sumSquares = 0;
  for (const point of landmarks) {
    const dx = point.x * flip - centroidX;
    const dy = point.y - centroidY;
    sumSquares += dx * dx + dy * dy;
  }
  const spread = Math.max(Math.sqrt(sumSquares / LANDMARKS_PER_HAND), MIN_SPREAD);

  // 4. Write out. `z` is dropped (M3).
  for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
    const point = landmarks[i]!;
    target[i * 2] = (point.x * flip - centroidX) / spread;
    target[i * 2 + 1] = (point.y - centroidY) / spread;
  }
  return target;
}

/**
 * Mirror a hand horizontally, as the training-time augmentation does.
 *
 * Exported so tests and the recorder can produce the same geometry the trainer sees.
 */
export function mirrorHand(landmarks: HandLandmarks): HandLandmarks {
  return landmarks.map((point) => ({ x: -point.x, y: point.y, z: point.z }));
}
