import type { Landmark } from '../types.js';

/** Smallest permitted scale divisor, guarding against a degenerate (zero-size) hand. */
export const MIN_SCALE = 1e-4;

/** Euclidean distance between two landmarks in the image plane, ignoring `z`. */
export function distance2d(a: Landmark, b: Landmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Euclidean distance between two landmarks including `z`. */
export function distance3d(a: Landmark, b: Landmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * Palm size measured in the image plane, used to make pixel thresholds
 * independent of how close the hand is to the camera.
 *
 * Note this is the **2D** convention. {@link palmScale3d} is a separate, larger
 * value; the two are not interchangeable. See docs/AUDIT.md (§1.2) — the v1 code
 * mixed both conventions, which is why they are named explicitly here.
 */
export function palmScale2d(wrist: Landmark, middleMcp: Landmark): number {
  return Math.max(distance2d(wrist, middleMcp), MIN_SCALE);
}

/** Palm size including `z`, used to normalise the model's input feature vector. */
export function palmScale3d(wrist: Landmark, middleMcp: Landmark): number {
  return Math.max(distance3d(wrist, middleMcp), MIN_SCALE);
}
