import { describe, expect, it } from 'vitest';
import { emptyFrame, selectPrimaryHand, type HandObservation } from './frame.js';
import { LandmarkView, HAND_BUFFER_LENGTH, packLandmarks } from './landmarkBuffer.js';
import { openHand, translate } from '../test/landmarkFactory.js';
import type { HandLandmarks } from '../types.js';

function observation(handedness: 'left' | 'right', landmarks: HandLandmarks): HandObservation {
  return { handedness, handednessScore: 0.95, landmarks };
}

describe('emptyFrame', () => {
  it('carries the timestamp and no hands', () => {
    const frame = emptyFrame(1234);
    expect(frame.timestampMs).toBe(1234);
    expect(frame.hands).toEqual([]);
    expect(frame.pose).toBeNull();
  });
});

describe('selectPrimaryHand', () => {
  it('returns null when nothing was detected', () => {
    expect(selectPrimaryHand([])).toBeNull();
  });

  it('returns the only hand without inspecting it', () => {
    const only = observation('right', openHand());
    expect(selectPrimaryHand([only])).toBe(only);
  });

  it('prefers the hand more fully inside the frame', () => {
    const inside = observation('right', openHand());
    // Shift the other hand mostly out of view.
    const outside = observation('left', translate(openHand(), 0.85, 0));
    expect(selectPrimaryHand([outside, inside])).toBe(inside);
    // Order must not matter — detection order is unstable between frames.
    expect(selectPrimaryHand([inside, outside])).toBe(inside);
  });

  it('keeps the first hand when both are equally visible, so output does not flicker', () => {
    const first = observation('right', openHand());
    const second = observation('left', openHand());
    expect(selectPrimaryHand([first, second])).toBe(first);
  });
});

describe('packLandmarks / LandmarkView', () => {
  it('round-trips landmarks exactly', () => {
    const original = openHand();
    const view = new LandmarkView();
    const restored = view.read(packLandmarks(original));

    expect(restored).toHaveLength(original.length);
    original.forEach((point, i) => {
      expect(restored[i]!.x).toBeCloseTo(point.x, 6);
      expect(restored[i]!.y).toBeCloseTo(point.y, 6);
      expect(restored[i]!.z).toBeCloseTo(point.z, 6);
    });
  });

  it('produces a buffer of the expected length', () => {
    expect(packLandmarks(openHand())).toHaveLength(HAND_BUFFER_LENGTH);
  });

  it('writes into a caller-supplied buffer, allocating nothing', () => {
    const target = new Float32Array(HAND_BUFFER_LENGTH);
    expect(packLandmarks(openHand(), target)).toBe(target);
  });

  it('reuses the same landmark objects across reads', () => {
    const view = new LandmarkView();
    const first = view.read(packLandmarks(openHand()));
    const second = view.read(packLandmarks(translate(openHand(), 0.1, 0.1)));
    // Identity is intentional: the hot path must not allocate 21 objects per frame.
    expect(second).toBe(first);
    expect(second[0]!.x).toBeCloseTo(openHand()[0]!.x + 0.1, 6);
  });

  it('rejects the wrong number of landmarks', () => {
    expect(() => packLandmarks(openHand().slice(0, 5))).toThrow(RangeError);
  });

  it('rejects a mis-sized target buffer', () => {
    expect(() => packLandmarks(openHand(), new Float32Array(10))).toThrow(RangeError);
  });

  it('rejects a mis-sized buffer on read', () => {
    expect(() => new LandmarkView().read(new Float32Array(10))).toThrow(RangeError);
  });
});
