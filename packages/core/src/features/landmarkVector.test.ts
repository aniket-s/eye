import { describe, expect, it } from 'vitest';
import { LANDMARK_VECTOR_LENGTH, landmarksToVector } from './landmarkVector.js';
import { mirror, openHand, rotate, scale, translate } from '../test/landmarkFactory.js';

/** Largest absolute difference between two equal-length vectors. */
function maxDelta(a: readonly number[], b: readonly number[]): number {
  return a.reduce((worst, value, i) => Math.max(worst, Math.abs(value - (b[i] as number))), 0);
}

describe('landmarksToVector', () => {
  it('emits 63 values — 21 landmarks × (x, y, z)', () => {
    expect(landmarksToVector(openHand())).toHaveLength(LANDMARK_VECTOR_LENGTH);
  });

  it('places the wrist at the origin', () => {
    const [x, y, z] = landmarksToVector(openHand());
    expect(x).toBe(0);
    expect(y).toBe(0);
    expect(z).toBe(0);
  });

  it('rejects a hand with the wrong number of landmarks', () => {
    expect(() => landmarksToVector(openHand().slice(0, 20))).toThrow(RangeError);
  });

  it('is invariant to translation', () => {
    const reference = landmarksToVector(openHand());
    const moved = landmarksToVector(translate(openHand(), 0.2, -0.15));
    expect(maxDelta(reference, moved)).toBeLessThan(1e-12);
  });

  it('is invariant to scale', () => {
    const reference = landmarksToVector(openHand());
    const nearer = landmarksToVector(scale(openHand(), 1.8));
    expect(maxDelta(reference, nearer)).toBeLessThan(1e-12);
  });

  // ── The following two tests document known defects, not desired behaviour. ──
  // They exist so that Phase 2 can assert the opposite and prove the fix landed.
  // See docs/AUDIT.md M1 and M2.

  it('DEFECT M1: is NOT invariant to handedness — a left hand looks unfamiliar', () => {
    const right = landmarksToVector(openHand());
    const left = landmarksToVector(mirror(openHand()));
    // A large delta here is exactly why left-handed signers fail: the mirrored
    // vector lies far outside the distribution the model was trained on.
    expect(maxDelta(right, left)).toBeGreaterThan(0.5);
  });

  it('DEFECT M2: is NOT invariant to in-plane rotation', () => {
    const upright = landmarksToVector(openHand());
    const tilted = landmarksToVector(rotate(openHand(), Math.PI / 6)); // 30°
    expect(maxDelta(upright, tilted)).toBeGreaterThan(0.5);
  });
});
