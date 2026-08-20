/**
 * Two-handed, body-relative normalisation — for word-level signs.
 *
 * Fingerspelling needs one hand's shape. Word signs need considerably more: most are
 * two-handed, and **where** the hands are relative to the body is part of the sign's
 * meaning. The sign for "hat" and the sign for "shoes" can share a handshape and differ
 * only in whether the hand is at the head or the feet.
 *
 * So this cannot simply run {@link normaliseHand} twice. Normalising each hand
 * independently would discard exactly the information that distinguishes the signs —
 * their positions relative to each other and to the torso. Both hands share **one**
 * reference frame instead:
 *
 * - **Origin:** the midpoint between the shoulders. Stable, and the anatomical centre
 *   that signing space is organised around.
 * - **Scale:** shoulder width. Independent of camera distance, and roughly constant
 *   for a given signer regardless of pose.
 *
 * ## Handedness
 *
 * Canonicalising is more involved than for one hand. Mirroring the geometry is not
 * enough: the left and right landmark *groups* must be swapped too, or a left-dominant
 * signer's dominant hand would land in the non-dominant slot. Getting this wrong is
 * subtle — the model still trains, just badly, because half the examples have their
 * hands the wrong way round.
 *
 * ## Missing hands
 *
 * One-handed signs are common, and MediaPipe drops a hand whenever it leaves frame or
 * is occluded. Absent hands are zero-filled with an explicit presence flag, so the
 * model can distinguish "hand at the origin" from "no hand" — which zeros alone cannot
 * express.
 */
import { LANDMARKS_PER_HAND, type Landmark } from '../types.js';
import type { Handedness, VisionFrame } from '../vision/frame.js';

/**
 * Pose landmarks retained, in the order {@link VisionLandmarker} emits them:
 * nose, shoulders, elbows, wrists.
 */
export const POSE_POINTS = 7;

/** Index pairs swapped when mirroring: shoulders, elbows, wrists. Nose stays put. */
const POSE_MIRROR_PAIRS: readonly (readonly [number, number])[] = [
  [1, 2],
  [3, 4],
  [5, 6],
];

const HAND_VALUES = LANDMARKS_PER_HAND * 2;

/** Offsets into the feature vector. */
export const BODY_LAYOUT = {
  /** Dominant hand, 21 × (x, y). */
  dominantHand: 0,
  /** Non-dominant hand, 21 × (x, y). */
  otherHand: HAND_VALUES,
  /** Upper-body pose, 7 × (x, y). */
  pose: HAND_VALUES * 2,
  /** 1 when the dominant hand was detected, else 0. */
  dominantPresent: HAND_VALUES * 2 + POSE_POINTS * 2,
  /** 1 when the non-dominant hand was detected, else 0. */
  otherPresent: HAND_VALUES * 2 + POSE_POINTS * 2 + 1,
} as const;

/** Total feature length: 42 + 42 + 14 + 2. */
export const BODY_FEATURE_LENGTH = BODY_LAYOUT.otherPresent + 1;

/** Identifier recorded in a pack manifest for this scheme. */
export const BODY_NORMALISATION_SCHEME = 'shoulder-frame-v1';

const MIN_SCALE = 1e-6;

/**
 * Build the per-frame feature vector for a word-level model.
 *
 * @param frame Hands and pose from the vision layer.
 * @param dominant The signer's dominant hand. Left-dominant signers are mirrored so
 *   the model only ever sees right-dominant geometry.
 * @param target Optional buffer to fill, avoiding a per-frame allocation.
 * @throws RangeError if `target` is the wrong length.
 */
export function normaliseBody(
  frame: VisionFrame,
  dominant: Handedness,
  target: Float32Array = new Float32Array(BODY_FEATURE_LENGTH),
): Float32Array {
  if (target.length !== BODY_FEATURE_LENGTH) {
    throw new RangeError(`Expected a buffer of ${BODY_FEATURE_LENGTH}, received ${target.length}`);
  }
  target.fill(0);

  const flip = dominant === 'left' ? -1 : 1;
  const pose = frame.pose ?? null;

  // ── Reference frame ──
  let originX: number;
  let originY: number;
  let scale: number;

  if (pose !== null && pose.length >= 3) {
    const leftShoulder = pose[1]!;
    const rightShoulder = pose[2]!;
    originX = ((leftShoulder.x + rightShoulder.x) / 2) * flip;
    originY = (leftShoulder.y + rightShoulder.y) / 2;
    scale = Math.max(
      Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y),
      MIN_SCALE,
    );
  } else {
    // No pose: fall back to the hands themselves. Loses absolute body position, which
    // costs accuracy on location-dependent signs, but keeps the app working when the
    // signer is framed too tightly for shoulders to be visible.
    const fallback = handsReference(frame, flip);
    originX = fallback.x;
    originY = fallback.y;
    scale = fallback.scale;
  }

  const write = (offset: number, point: Landmark, index: number): void => {
    target[offset + index * 2] = (point.x * flip - originX) / scale;
    target[offset + index * 2 + 1] = (point.y - originY) / scale;
  };

  // ── Hands ──
  // After mirroring, a left hand *is* a right hand, so handedness is read against the
  // flip: for a left-dominant signer the mirrored 'left' observation is the dominant one.
  for (const hand of frame.hands) {
    const isDominant = hand.handedness === dominant;
    const offset = isDominant ? BODY_LAYOUT.dominantHand : BODY_LAYOUT.otherHand;
    const presence = isDominant ? BODY_LAYOUT.dominantPresent : BODY_LAYOUT.otherPresent;

    if (hand.landmarks.length !== LANDMARKS_PER_HAND) continue;
    for (let i = 0; i < LANDMARKS_PER_HAND; i++) write(offset, hand.landmarks[i]!, i);
    target[presence] = 1;
  }

  // ── Pose ──
  if (pose !== null) {
    for (let i = 0; i < Math.min(POSE_POINTS, pose.length); i++) {
      // Mirroring swaps the body's own left and right, so the landmark groups swap too.
      const source = flip === -1 ? mirroredPoseIndex(i) : i;
      const point = pose[source];
      if (point !== undefined) write(BODY_LAYOUT.pose, point, i);
    }
  }

  return target;
}

/** Which pose index supplies slot `i` after mirroring. */
function mirroredPoseIndex(index: number): number {
  for (const [a, b] of POSE_MIRROR_PAIRS) {
    if (index === a) return b;
    if (index === b) return a;
  }
  return index;
}

/** Origin and scale derived from the hands, when pose is unavailable. */
function handsReference(frame: VisionFrame, flip: number): { x: number; y: number; scale: number } {
  let sumX = 0;
  let sumY = 0;
  let count = 0;

  for (const hand of frame.hands) {
    for (const point of hand.landmarks) {
      sumX += point.x * flip;
      sumY += point.y;
      count++;
    }
  }
  if (count === 0) return { x: 0, y: 0, scale: 1 };

  const x = sumX / count;
  const y = sumY / count;

  let sumSquares = 0;
  for (const hand of frame.hands) {
    for (const point of hand.landmarks) {
      const dx = point.x * flip - x;
      const dy = point.y - y;
      sumSquares += dx * dx + dy * dy;
    }
  }

  // ×2.5 so the resulting scale is roughly comparable to shoulder width, keeping the
  // feature magnitudes in the same range whichever path produced them.
  return { x, y, scale: Math.max(Math.sqrt(sumSquares / count) * 2.5, MIN_SCALE) };
}
