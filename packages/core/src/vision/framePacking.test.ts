import { describe, expect, it } from 'vitest';
import { FRAME_BUFFER_LENGTH, packVisionFrame, VisionFrameView } from './framePacking.js';
import { openHand, translate } from '../test/landmarkFactory.js';
import type { Landmark } from '../types.js';
import type { VisionFrame } from './frame.js';

function pose(): Landmark[] {
  return Array.from({ length: 7 }, (_, i) => ({ x: 0.4 + i * 0.02, y: 0.3 + i * 0.03, z: 0 }));
}

function frame(options: { dominant?: boolean; other?: boolean; pose?: boolean } = {}): VisionFrame {
  const hands = [];
  if (options.dominant !== false) {
    hands.push({
      handedness: 'right' as const,
      handednessScore: 0.9,
      landmarks: openHand(),
    });
  }
  if (options.other === true) {
    hands.push({
      handedness: 'left' as const,
      handednessScore: 0.9,
      landmarks: translate(openHand(), 0.2, 0.05),
    });
  }
  return { timestampMs: 42, hands, pose: options.pose === false ? null : pose() };
}

describe('packVisionFrame / VisionFrameView', () => {
  it('round-trips both hands and pose', () => {
    const original = frame({ other: true });
    const restored = new VisionFrameView().read(
      packVisionFrame(original, 'right'),
      original.timestampMs,
      'right',
    );

    expect(restored.timestampMs).toBe(42);
    expect(restored.hands).toHaveLength(2);
    expect(restored.pose).not.toBeNull();

    const originalRight = original.hands.find((h) => h.handedness === 'right')!;
    const restoredRight = restored.hands.find((h) => h.handedness === 'right')!;
    originalRight.landmarks.forEach((point, i) => {
      expect(restoredRight.landmarks[i]!.x).toBeCloseTo(point.x, 6);
      expect(restoredRight.landmarks[i]!.y).toBeCloseTo(point.y, 6);
    });
  });

  it('produces a buffer of the declared length', () => {
    expect(packVisionFrame(frame(), 'right')).toHaveLength(FRAME_BUFFER_LENGTH);
  });

  /**
   * Presence flags rather than NaN sentinels. NaN propagates silently through
   * arithmetic, so a missing hand would poison the features rather than being handled.
   */
  it('round-trips a one-handed frame without inventing a second hand', () => {
    const restored = new VisionFrameView().read(
      packVisionFrame(frame({ other: false }), 'right'),
      0,
      'right',
    );
    expect(restored.hands).toHaveLength(1);
    expect(restored.hands[0]!.handedness).toBe('right');
  });

  it('round-trips a frame with no pose', () => {
    const restored = new VisionFrameView().read(
      packVisionFrame(frame({ pose: false }), 'right'),
      0,
      'right',
    );
    expect(restored.pose).toBeNull();
  });

  it('round-trips an entirely empty frame', () => {
    const empty: VisionFrame = { timestampMs: 7, hands: [], pose: null };
    const restored = new VisionFrameView().read(packVisionFrame(empty, 'right'), 7, 'right');
    expect(restored.hands).toEqual([]);
    expect(restored.pose).toBeNull();
  });

  it('labels hands against the dominance the buffer was packed with', () => {
    const original: VisionFrame = {
      timestampMs: 0,
      hands: [{ handedness: 'left', handednessScore: 0.9, landmarks: openHand() }],
      pose: null,
    };
    // Packed as left-dominant, so the hand lands in the dominant slot and reads back
    // as 'left' — not silently relabelled.
    const restored = new VisionFrameView().read(packVisionFrame(original, 'left'), 0, 'left');
    expect(restored.hands[0]!.handedness).toBe('left');
  });

  it('reuses its landmark objects between reads', () => {
    const view = new VisionFrameView();
    const first = view.read(packVisionFrame(frame(), 'right'), 0, 'right');
    const second = view.read(packVisionFrame(frame(), 'right'), 1, 'right');
    expect(second.hands[0]!.landmarks).toBe(first.hands[0]!.landmarks);
  });

  it('rejects mis-sized buffers', () => {
    expect(() => packVisionFrame(frame(), 'right', new Float32Array(4))).toThrow(RangeError);
    expect(() => new VisionFrameView().read(new Float32Array(4), 0, 'right')).toThrow(RangeError);
  });
});
