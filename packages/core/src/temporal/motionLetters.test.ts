import { describe, expect, it } from 'vitest';
import { MotionLetterRecognizer, MOTION_LETTERS } from './motionLetters.js';
import { HandLandmarkIndex, LANDMARKS_PER_HAND, type HandLandmarks } from '../types.js';
import type { Handedness } from '../vision/frame.js';

/**
 * A hand of a given apparent size with one fingertip placed where we want it.
 *
 * The other landmarks form a ring around the centre so the RMS spread — which every
 * threshold is measured in — is a stable, known quantity rather than an accident of
 * where the traced tip happens to be.
 */
function handWith(tip: number, x: number, y: number, scale = 0.1): HandLandmarks {
  const points = Array.from({ length: LANDMARKS_PER_HAND }, (_, index) => {
    const angle = (index / LANDMARKS_PER_HAND) * Math.PI * 2;
    return { x: 0.5 + Math.cos(angle) * scale, y: 0.5 + Math.sin(angle) * scale, z: 0 };
  });
  points[tip] = { x, y, z: 0 };
  return points;
}

/** Feed a path of `[x, y]` image positions, 20 ms apart, and collect what is emitted. */
function trace(
  recognizer: MotionLetterRecognizer,
  shape: string,
  tip: number,
  path: readonly (readonly [number, number])[],
  {
    handedness = 'right',
    startMs = 1000,
    stepMs = 20,
  }: { handedness?: Handedness; startMs?: number; stepMs?: number } = {},
): string[] {
  const emitted: string[] = [];
  path.forEach(([x, y], index) => {
    const letter = recognizer.update(
      shape,
      handWith(tip, x, y),
      handedness,
      startMs + index * stepMs,
    );
    if (letter !== null) emitted.push(letter);
  });
  return emitted;
}

const PINKY = HandLandmarkIndex.PINKY_TIP;
const INDEX = HandLandmarkIndex.INDEX_TIP;

/** A J: the pinky falls, then hooks toward the thumb side (image-left, right hand). */
const HOOK: (readonly [number, number])[] = [
  [0.5, 0.3],
  [0.5, 0.36],
  [0.5, 0.42],
  [0.5, 0.47],
  [0.49, 0.5],
  [0.46, 0.51],
  [0.42, 0.51],
  [0.38, 0.5],
];

/** A Z: across, back down the diagonal, across again, descending overall. */
const ZIGZAG: (readonly [number, number])[] = [
  [0.35, 0.3],
  [0.42, 0.3],
  [0.5, 0.31],
  [0.44, 0.36],
  [0.38, 0.4],
  [0.36, 0.42],
  [0.44, 0.43],
  [0.52, 0.44],
];

describe('MotionLetterRecognizer', () => {
  describe('J', () => {
    it('reads a pinky that drops and hooks', () => {
      const recognizer = new MotionLetterRecognizer();
      expect(trace(recognizer, 'I', PINKY, HOOK)).toEqual(['J']);
    });

    it('ignores a hand simply being lowered', () => {
      // The whole reason the shape test has two phases. v1 compared only start to end,
      // which a straight drop satisfies.
      const straight = HOOK.map((_, index) => [0.5, 0.3 + index * 0.03] as const);
      expect(trace(new MotionLetterRecognizer(), 'I', PINKY, straight)).toEqual([]);
    });

    it('ignores a hook made while the model is reading another letter', () => {
      expect(trace(new MotionLetterRecognizer(), 'B', PINKY, HOOK)).toEqual([]);
    });

    it('reads a left-handed J, which mirrors', () => {
      const mirrored = HOOK.map(([x, y]) => [1 - x, y] as const);
      const emitted = trace(new MotionLetterRecognizer(), 'I', PINKY, mirrored, {
        handedness: 'left',
      });
      expect(emitted).toEqual(['J']);
    });

    it('does not read a left hand tracing the right-handed path', () => {
      expect(trace(new MotionLetterRecognizer(), 'I', PINKY, HOOK, { handedness: 'left' })).toEqual(
        [],
      );
    });

    it('keeps tracing after the handshape stops being readable', () => {
      // The hand genuinely turns partway through a J, and the classifier stops calling
      // it I. Abandoning there would make the letter unreachable.
      const recognizer = new MotionLetterRecognizer();
      const emitted: string[] = [];
      HOOK.forEach(([x, y], index) => {
        const letter = recognizer.update(
          index < 2 ? 'I' : null,
          handWith(PINKY, x, y),
          'right',
          1000 + index * 20,
        );
        if (letter !== null) emitted.push(letter);
      });
      expect(emitted).toEqual(['J']);
    });
  });

  describe('Z', () => {
    it('reads three alternating strokes that descend', () => {
      expect(trace(new MotionLetterRecognizer(), 'D', INDEX, ZIGZAG)).toEqual(['Z']);
    });

    it('ignores a horizontal wiggle', () => {
      // v1's own docblock admitted this one: it had no descent requirement at all.
      const flat = ZIGZAG.map(([x], index) => [x, 0.3 + (index % 2) * 0.004] as const);
      expect(trace(new MotionLetterRecognizer(), 'D', INDEX, flat)).toEqual([]);
    });

    it('ignores a single sweep', () => {
      const sweep = ZIGZAG.map((_, index) => [0.3 + index * 0.03, 0.3 + index * 0.01] as const);
      expect(trace(new MotionLetterRecognizer(), 'D', INDEX, sweep)).toEqual([]);
    });
  });

  describe('measuring in hand widths rather than frame units', () => {
    it('reads the same gesture at half the apparent size', () => {
      // The v1 detector used raw frame distances, so the thresholds only held at one
      // distance from the camera. Same gesture, hand and path scaled together.
      const recognizer = new MotionLetterRecognizer();
      const emitted: string[] = [];
      HOOK.forEach(([x, y], index) => {
        const shrunk = handWith(PINKY, 0.5 + (x - 0.5) / 2, 0.5 + (y - 0.5) / 2, 0.05);
        const letter = recognizer.update('I', shrunk, 'right', 1000 + index * 20);
        if (letter !== null) emitted.push(letter);
      });
      expect(emitted).toEqual(['J']);
    });

    it('reads the same gesture anywhere in the frame', () => {
      // The path is measured as displacement from where the trace began, not as
      // absolute coordinates, so where the signer stands cannot change the answer.
      const corner = HOOK.map(([x, y]) => [x - 0.28, y + 0.18] as const);
      expect(trace(new MotionLetterRecognizer(), 'I', PINKY, corner)).toEqual(['J']);
    });

    it('survives the hand appearing to change size mid-gesture', () => {
      // The signer leans toward the camera through the J. The gesture is the same one
      // in hand widths; only the apparent size moves. The reference scale is frozen at
      // the start of the trace so that wobble stays out of the measured path.
      const recognizer = new MotionLetterRecognizer();
      const emitted: string[] = [];
      HOOK.forEach(([x, y], index) => {
        const zoom = 1 + index * 0.05;
        const leaning = handWith(PINKY, 0.5 + (x - 0.5) * zoom, 0.5 + (y - 0.5) * zoom, 0.1 * zoom);
        const letter = recognizer.update('I', leaning, 'right', 1000 + index * 20);
        if (letter !== null) emitted.push(letter);
      });
      expect(emitted).toEqual(['J']);
    });
  });

  describe('timing', () => {
    it('abandons a trace that takes too long', () => {
      const recognizer = new MotionLetterRecognizer({ windowMs: 60 });
      expect(trace(recognizer, 'I', PINKY, HOOK, { stepMs: 30 })).toEqual([]);
    });

    it('is not frame-rate dependent', () => {
      // Identical gesture over identical wall-clock time, sampled twice as often.
      const dense = HOOK.flatMap(([x, y], index) =>
        index === HOOK.length - 1
          ? [[x, y] as const]
          : [
              [x, y] as const,
              [(x + HOOK[index + 1]![0]) / 2, (y + HOOK[index + 1]![1]) / 2] as const,
            ],
      );
      expect(trace(new MotionLetterRecognizer(), 'I', PINKY, dense, { stepMs: 10 })).toEqual(['J']);
    });

    it('drops a trace with a gap in it', () => {
      const recognizer = new MotionLetterRecognizer({ staleMs: 50 });
      const emitted: string[] = [];
      HOOK.forEach(([x, y], index) => {
        // A stalled tab in the middle of the gesture.
        const at = 1000 + index * 20 + (index > 3 ? 500 : 0);
        const letter = recognizer.update('I', handWith(PINKY, x, y), 'right', at);
        if (letter !== null) emitted.push(letter);
      });
      expect(emitted).toEqual([]);
    });

    it('emits once per gesture, not once per frame', () => {
      const recognizer = new MotionLetterRecognizer();
      const repeated = [...HOOK, ...HOOK.slice(-1), ...HOOK.slice(-1)];
      expect(trace(recognizer, 'I', PINKY, repeated)).toEqual(['J']);
    });

    it('can read a second J after the cooldown', () => {
      const recognizer = new MotionLetterRecognizer({ cooldownMs: 10 });
      expect(trace(recognizer, 'I', PINKY, HOOK)).toEqual(['J']);
      expect(trace(recognizer, 'I', PINKY, HOOK, { startMs: 5000 })).toEqual(['J']);
    });
  });

  describe('lifecycle', () => {
    it('reports what it is tracing', () => {
      const recognizer = new MotionLetterRecognizer();
      expect(recognizer.tracking).toBeNull();
      recognizer.update('I', handWith(PINKY, 0.5, 0.3), 'right', 1000);
      expect(recognizer.tracking).toBe('J');
    });

    it('forgets everything when the hand leaves', () => {
      const recognizer = new MotionLetterRecognizer();
      HOOK.slice(0, 5).forEach(([x, y], index) => {
        recognizer.update('I', handWith(PINKY, x, y), 'right', 1000 + index * 20);
      });
      recognizer.update(null, null, 'right', 1120);
      expect(recognizer.tracking).toBeNull();
      // The remainder of the path alone is not a J.
      expect(trace(recognizer, null as unknown as string, PINKY, HOOK.slice(5))).toEqual([]);
    });

    it('knows J and Z and nothing else', () => {
      expect(MOTION_LETTERS.map((motion) => motion.letter)).toEqual(['J', 'Z']);
      expect(MOTION_LETTERS.map((motion) => motion.from)).toEqual(['I', 'D']);
    });
  });
});
