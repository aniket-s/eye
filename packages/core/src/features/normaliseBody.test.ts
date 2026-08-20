import { describe, expect, it } from 'vitest';
import { BODY_FEATURE_LENGTH, BODY_LAYOUT, normaliseBody, POSE_POINTS } from './normaliseBody.js';
import { openHand, translate } from '../test/landmarkFactory.js';
import type { Handedness, HandObservation, VisionFrame } from '../vision/frame.js';
import type { Landmark } from '../types.js';

/** Upper-body pose: nose, shoulders, elbows, wrists. */
function pose(centreX = 0.5): Landmark[] {
  return [
    { x: centreX, y: 0.2, z: 0 }, // nose
    { x: centreX - 0.12, y: 0.35, z: 0 }, // left shoulder
    { x: centreX + 0.12, y: 0.35, z: 0 }, // right shoulder
    { x: centreX - 0.18, y: 0.5, z: 0 }, // left elbow
    { x: centreX + 0.18, y: 0.5, z: 0 }, // right elbow
    { x: centreX - 0.2, y: 0.62, z: 0 }, // left wrist
    { x: centreX + 0.2, y: 0.62, z: 0 }, // right wrist
  ];
}

function hand(handedness: Handedness, dx = 0, dy = 0): HandObservation {
  return {
    handedness,
    handednessScore: 0.95,
    landmarks: translate(openHand(), dx, dy),
  };
}

function frame(hands: HandObservation[], withPose = true): VisionFrame {
  return { timestampMs: 0, hands, pose: withPose ? pose() : null };
}

function features(f: VisionFrame, dominant: Handedness = 'right'): number[] {
  return Array.from(normaliseBody(f, dominant));
}

function slice(values: number[], offset: number, count: number): number[] {
  return values.slice(offset, offset + count);
}

function maxDelta(a: readonly number[], b: readonly number[]): number {
  return a.reduce((worst, value, i) => Math.max(worst, Math.abs(value - (b[i] as number))), 0);
}

describe('normaliseBody — shape and layout', () => {
  it('emits the declared feature length', () => {
    expect(normaliseBody(frame([hand('right')]), 'right')).toHaveLength(BODY_FEATURE_LENGTH);
    // 21×2 per hand, ×2 hands, plus 7×2 pose, plus 2 presence flags.
    expect(BODY_FEATURE_LENGTH).toBe(42 * 2 + POSE_POINTS * 2 + 2);
  });

  it('writes into a caller-supplied buffer', () => {
    const target = new Float32Array(BODY_FEATURE_LENGTH);
    expect(normaliseBody(frame([hand('right')]), 'right', target)).toBe(target);
  });

  it('rejects a mis-sized buffer', () => {
    expect(() => normaliseBody(frame([hand('right')]), 'right', new Float32Array(4))).toThrow(
      RangeError,
    );
  });

  it('clears the buffer between calls so stale hands cannot linger', () => {
    const target = new Float32Array(BODY_FEATURE_LENGTH);
    normaliseBody(frame([hand('right'), hand('left')]), 'right', target);
    normaliseBody(frame([hand('right')]), 'right', target);

    expect(target[BODY_LAYOUT.otherPresent]).toBe(0);
    expect(slice(Array.from(target), BODY_LAYOUT.otherHand, 42).every((v) => v === 0)).toBe(true);
  });
});

describe('normaliseBody — presence flags', () => {
  /**
   * Zeros alone cannot distinguish "hand resting at the origin" from "no hand". One-
   * handed signs are common and MediaPipe drops a hand whenever it leaves frame, so
   * the model needs an explicit signal.
   */
  it('marks which hands were detected', () => {
    const both = features(frame([hand('right'), hand('left')]));
    expect(both[BODY_LAYOUT.dominantPresent]).toBe(1);
    expect(both[BODY_LAYOUT.otherPresent]).toBe(1);

    const one = features(frame([hand('right')]));
    expect(one[BODY_LAYOUT.dominantPresent]).toBe(1);
    expect(one[BODY_LAYOUT.otherPresent]).toBe(0);

    const none = features(frame([]));
    expect(none[BODY_LAYOUT.dominantPresent]).toBe(0);
    expect(none[BODY_LAYOUT.otherPresent]).toBe(0);
  });

  it('routes the dominant hand to the dominant slot', () => {
    const values = features(frame([hand('left'), hand('right')]), 'right');
    expect(values[BODY_LAYOUT.dominantPresent]).toBe(1);
    expect(slice(values, BODY_LAYOUT.dominantHand, 42).some((v) => v !== 0)).toBe(true);
  });
});

describe('normaliseBody — the reference frame', () => {
  it('is invariant to where the signer stands in frame', () => {
    const centred: VisionFrame = { timestampMs: 0, hands: [hand('right')], pose: pose(0.5) };
    const offset: VisionFrame = {
      timestampMs: 0,
      hands: [hand('right', 0.2, 0)],
      pose: pose(0.7),
    };
    expect(maxDelta(features(centred), features(offset))).toBeLessThan(1e-5);
  });

  it('places the shoulder midpoint at the origin', () => {
    const values = features(frame([hand('right')]));
    // Pose slots 1 and 2 are the shoulders; their midpoint must be (0, 0).
    const leftShoulderX = values[BODY_LAYOUT.pose + 2] as number;
    const rightShoulderX = values[BODY_LAYOUT.pose + 4] as number;
    expect((leftShoulderX + rightShoulderX) / 2).toBeCloseTo(0, 5);
  });

  it('scales by shoulder width, so shoulders are one unit apart', () => {
    const values = features(frame([hand('right')]));
    const leftShoulderX = values[BODY_LAYOUT.pose + 2] as number;
    const rightShoulderX = values[BODY_LAYOUT.pose + 4] as number;
    expect(Math.abs(leftShoulderX - rightShoulderX)).toBeCloseTo(1, 5);
  });

  /**
   * The property that separates this from running the single-hand normaliser twice:
   * *where* the hands are relative to the body is part of a word sign's meaning, so
   * moving one hand must change the features.
   */
  it('preserves hand position relative to the body', () => {
    const atChest = features(frame([hand('right', 0, 0)]));
    const atHead = features(frame([hand('right', 0, -0.25)]));
    expect(maxDelta(atChest, atHead)).toBeGreaterThan(0.3);
  });

  it('preserves the spatial relationship between the two hands', () => {
    const together = features(frame([hand('right', 0, 0), hand('left', 0.05, 0)]));
    const apart = features(frame([hand('right', 0, 0), hand('left', 0.35, 0)]));
    expect(maxDelta(together, apart)).toBeGreaterThan(0.3);
  });

  it('falls back to a hand-derived frame when pose is unavailable', () => {
    const values = features(frame([hand('right')], false));
    expect(values.every(Number.isFinite)).toBe(true);
    expect(values[BODY_LAYOUT.dominantPresent]).toBe(1);
  });

  it('survives a frame with neither hands nor pose', () => {
    const values = features({ timestampMs: 0, hands: [], pose: null });
    expect(values.every((v) => v === 0)).toBe(true);
  });
});

describe('normaliseBody — handedness canonicalisation', () => {
  /**
   * Mirroring the geometry is not sufficient. The left and right landmark *groups*
   * must swap too, or a left-dominant signer's dominant hand lands in the non-dominant
   * slot. The model would still train — just badly, on half its examples reversed.
   */
  it('presents a left-dominant signer identically to a mirrored right-dominant one', () => {
    const rightDominant: VisionFrame = {
      timestampMs: 0,
      hands: [hand('right', 0.1, -0.05), hand('left', -0.1, 0.02)],
      pose: pose(0.5),
    };

    // The same scene mirrored: every x negated, and both hands relabelled.
    const mirrorLandmarks = (points: readonly Landmark[]): Landmark[] =>
      points.map((p) => ({ x: -p.x, y: p.y, z: p.z }));
    const mirrored: VisionFrame = {
      timestampMs: 0,
      hands: rightDominant.hands.map((h) => ({
        handedness: h.handedness === 'right' ? ('left' as const) : ('right' as const),
        handednessScore: h.handednessScore,
        landmarks: mirrorLandmarks(h.landmarks),
      })),
      pose: (() => {
        const flipped = mirrorLandmarks(rightDominant.pose!);
        // Swap the body's own left/right groups.
        return [
          flipped[0]!,
          flipped[2]!,
          flipped[1]!,
          flipped[4]!,
          flipped[3]!,
          flipped[6]!,
          flipped[5]!,
        ];
      })(),
    };

    expect(maxDelta(features(rightDominant, 'right'), features(mirrored, 'left'))).toBeLessThan(
      1e-5,
    );
  });

  it('keeps the nose in place when mirroring, since it has no counterpart', () => {
    const values = features(frame([hand('right')]), 'left');
    expect(Number.isFinite(values[BODY_LAYOUT.pose] as number)).toBe(true);
  });
});
