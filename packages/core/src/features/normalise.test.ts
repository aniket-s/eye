import { describe, expect, it } from 'vitest';
import { FEATURE_LENGTH, normaliseHand } from './normalise.js';
import { mirror, openHand, rotate, scale, translate } from '../test/landmarkFactory.js';
import type { HandLandmarks } from '../types.js';

function features(hand: HandLandmarks, handedness: 'left' | 'right' = 'right'): number[] {
  return Array.from(normaliseHand(hand, handedness));
}

function maxDelta(a: readonly number[], b: readonly number[]): number {
  return a.reduce((worst, value, i) => Math.max(worst, Math.abs(value - (b[i] as number))), 0);
}

describe('normaliseHand — shape and guards', () => {
  it('emits 42 values — 21 landmarks × (x, y), with z dropped', () => {
    expect(normaliseHand(openHand(), 'right')).toHaveLength(FEATURE_LENGTH);
  });

  it('writes into a caller-supplied buffer', () => {
    const target = new Float32Array(FEATURE_LENGTH);
    expect(normaliseHand(openHand(), 'right', target)).toBe(target);
  });

  it('rejects the wrong number of landmarks', () => {
    expect(() => normaliseHand(openHand().slice(0, 20), 'right')).toThrow(RangeError);
  });

  it('rejects a mis-sized buffer', () => {
    expect(() => normaliseHand(openHand(), 'right', new Float32Array(10))).toThrow(RangeError);
  });

  it('ignores z entirely, so depth noise cannot move the features', () => {
    const noisy = openHand().map((p, i) => ({ ...p, z: p.z + i * 0.13 }));
    expect(maxDelta(features(openHand()), features(noisy))).toBe(0);
  });

  it('survives a degenerate hand where every landmark coincides', () => {
    const collapsed = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
    expect(features(collapsed).every(Number.isFinite)).toBe(true);
  });
});

// ── These three invert the defect assertions in landmarkVector.test.ts. ──
// v1 failed all of them; that failure is what M1 and M3 describe.

describe('normaliseHand — the invariances v1 lacked', () => {
  it('FIXES M1: a left hand normalises identically to the mirrored right hand', () => {
    const right = features(openHand(), 'right');
    const left = features(mirror(openHand()), 'left');
    // v1's equivalent delta was > 0.5, which is why left-handed signers failed.
    expect(maxDelta(right, left)).toBeLessThan(1e-5);
  });

  it('is invariant to where the hand sits in frame', () => {
    expect(
      maxDelta(features(openHand()), features(translate(openHand(), 0.25, -0.2))),
    ).toBeLessThan(1e-5);
  });

  it('is invariant to distance from the camera', () => {
    expect(maxDelta(features(openHand()), features(scale(openHand(), 2.3)))).toBeLessThan(1e-5);
  });

  it('confines wrist jitter to the wrist instead of moving the whole hand', () => {
    const jittered = openHand().map((p, i) => (i === 0 ? { ...p, x: p.x + 0.02 } : p));

    // v1 placed the wrist at the origin, so jitter there displaced all 21 landmarks
    // by the full amount. Centring on the centroid spreads it across the hand, so
    // the other 20 landmarks barely move — only the landmark that actually moved does.
    const baseline = features(openHand()).slice(2); // drop the wrist's own x, y
    const shifted = features(jittered).slice(2);
    expect(maxDelta(baseline, shifted)).toBeLessThan(0.03);

    // The wrist itself does move, which is correct — it genuinely changed position.
    const wristDelta = Math.abs(
      (features(jittered)[0] as number) - (features(openHand())[0] as number),
    );
    expect(wristDelta).toBeGreaterThan(0.1);
  });
});

describe('normaliseHand — rotation is preserved on purpose', () => {
  /**
   * Not a defect. Several ASL letters differ *only* by orientation — K/P, G/Q, H/U —
   * so canonicalising rotation would merge those classes and make the letters
   * unreadable. Robustness to incidental tilt comes from training augmentation.
   * See the module docblock in normalise.ts.
   */
  it('does NOT collapse rotated hands, because orientation carries meaning', () => {
    const upright = features(openHand());
    const rotated = features(rotate(openHand(), Math.PI / 2));
    expect(maxDelta(upright, rotated)).toBeGreaterThan(0.5);
  });

  it('changes smoothly with rotation, so augmentation can cover the range', () => {
    const upright = features(openHand());
    const small = maxDelta(upright, features(rotate(openHand(), Math.PI / 36))); // 5°
    const large = maxDelta(upright, features(rotate(openHand(), Math.PI / 6))); // 30°
    expect(small).toBeLessThan(large);
    // A 5° tilt must not move features anywhere near as far as a class change would.
    expect(small).toBeLessThan(0.35);
  });
});

describe('normaliseHand — output statistics', () => {
  it('centres the features on the origin', () => {
    const values = features(openHand());
    let sumX = 0;
    let sumY = 0;
    for (let i = 0; i < values.length; i += 2) {
      sumX += values[i] as number;
      sumY += values[i + 1] as number;
    }
    expect(sumX / 21).toBeCloseTo(0, 5);
    expect(sumY / 21).toBeCloseTo(0, 5);
  });

  it('scales the features to unit RMS spread', () => {
    const values = features(openHand());
    let sumSquares = 0;
    for (const value of values) sumSquares += value * value;
    expect(Math.sqrt(sumSquares / 21)).toBeCloseTo(1, 5);
  });
});
