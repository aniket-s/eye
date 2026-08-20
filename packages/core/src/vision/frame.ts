/**
 * The frame types that flow from the vision layer into recognition.
 *
 * These are deliberately independent of MediaPipe. `packages/web` adapts whatever
 * the landmarker emits into these shapes, so swapping HandLandmarker for
 * HolisticLandmarker — or replaying a recorded session in a test — changes nothing
 * downstream.
 */
import { LANDMARKS_PER_HAND, type HandLandmarks, type Landmark } from '../types.js';

/** Which of the signer's hands a set of landmarks belongs to. */
export type Handedness = 'left' | 'right';

/** One detected hand. */
export interface HandObservation {
  readonly handedness: Handedness;
  /** Confidence in the handedness call, 0–1. */
  readonly handednessScore: number;
  readonly landmarks: HandLandmarks;
}

/**
 * Everything the vision layer produced for a single video frame.
 *
 * `timestampMs` is the media time of the frame, not `Date.now()`. Deriving all
 * timing from it is what makes behaviour identical at 30 fps and 60 fps
 * (docs/AUDIT.md, J2).
 */
export interface VisionFrame {
  readonly timestampMs: number;
  readonly hands: readonly HandObservation[];
  /**
   * Upper-body pose landmarks, when a pose model is running.
   *
   * Unused by the v1 classifier. Captured from Phase 1 so that the word-level model
   * in Phase 4 has body-relative context available without another vision change.
   */
  readonly pose: readonly Landmark[] | null;
}

/** An empty frame, for when no hand is visible. */
export function emptyFrame(timestampMs: number): VisionFrame {
  return { timestampMs, hands: [], pose: null };
}

/**
 * Choose the hand to classify when more than one is visible.
 *
 * The v1 model is single-handed, so something must pick. Preferring the hand with
 * the most landmarks inside the frame beats trusting detection order, which is
 * unstable between frames and makes the output flicker when both hands are up.
 *
 * Phase 4's word-level model consumes both hands and will not call this.
 */
export function selectPrimaryHand(hands: readonly HandObservation[]): HandObservation | null {
  if (hands.length === 0) return null;
  if (hands.length === 1) return hands[0]!;

  let best = hands[0]!;
  let bestScore = visibilityScore(best.landmarks);
  for (let i = 1; i < hands.length; i++) {
    const candidate = hands[i]!;
    const score = visibilityScore(candidate.landmarks);
    if (score > bestScore) {
      best = candidate;
      bestScore = score;
    }
  }
  return best;
}

/** Fraction of a hand's landmarks that lie inside the normalised image bounds. */
function visibilityScore(landmarks: HandLandmarks): number {
  let inside = 0;
  for (const point of landmarks) {
    if (point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1) inside++;
  }
  return inside / LANDMARKS_PER_HAND;
}
