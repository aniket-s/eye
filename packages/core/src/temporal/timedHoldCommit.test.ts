import { describe, expect, it } from 'vitest';
import { TimedHoldCommitDetector } from './timedHoldCommit.js';
import { NO_PREDICTION } from '../types.js';

/** Feed `letter` for `durationMs` at a given frame rate, collecting commits. */
function hold(
  detector: TimedHoldCommitDetector,
  letter: string,
  durationMs: number,
  fps: number,
  startMs = 0,
): { committed: string[]; endMs: number } {
  const step = 1000 / fps;
  const committed: string[] = [];
  let t = startMs;
  for (; t <= startMs + durationMs; t += step) {
    const result = detector.update(letter, t);
    if (result.committed !== null) committed.push(result.committed);
  }
  return { committed, endMs: t };
}

describe('TimedHoldCommitDetector', () => {
  it('commits once the hold duration has elapsed', () => {
    const detector = new TimedHoldCommitDetector({ holdMs: 500, cooldownMs: 0 });
    expect(hold(detector, 'A', 600, 30).committed).toEqual(['A']);
  });

  it('does not commit before the hold duration', () => {
    const detector = new TimedHoldCommitDetector({ holdMs: 500, cooldownMs: 0 });
    expect(hold(detector, 'A', 400, 30).committed).toEqual([]);
  });

  /**
   * The point of this class. v1 counted frames, so the same physical hold committed
   * at 30 fps and failed at 15 fps (docs/AUDIT.md, J2 and §5).
   */
  it('behaves identically at 15, 30 and 60 fps', () => {
    for (const fps of [15, 30, 60]) {
      const detector = new TimedHoldCommitDetector({ holdMs: 500, cooldownMs: 0 });
      expect(hold(detector, 'A', 600, fps).committed, `at ${fps} fps`).toEqual(['A']);
    }
  });

  it('reports progress proportional to elapsed time, not frames', () => {
    const detector = new TimedHoldCommitDetector({ holdMs: 1000, cooldownMs: 0 });
    detector.update('A', 0);
    expect(detector.update('A', 250).progressPercent).toBe(25);
    expect(detector.update('A', 500).progressPercent).toBe(50);
    expect(detector.update('A', 999).progressPercent).toBe(100);
  });

  it('enforces the cooldown between commits', () => {
    const detector = new TimedHoldCommitDetector({ holdMs: 200, cooldownMs: 1000 });
    // 2100 ms would allow 10 commits at a 200 ms hold with no cooldown.
    const { committed } = hold(detector, 'A', 2100, 30);
    expect(committed).toEqual(['A', 'A']);
  });

  it('keeps counting through a low frame rate instead of restarting the hold', () => {
    // At 8 fps the frame interval (125 ms) exceeds toleranceMs. That must not be
    // mistaken for a stalled stream.
    const detector = new TimedHoldCommitDetector({ holdMs: 500, cooldownMs: 0 });
    expect(hold(detector, 'A', 700, 8).committed).toEqual(['A']);
  });

  it('restarts the hold when the letter changes', () => {
    const detector = new TimedHoldCommitDetector({ holdMs: 500, cooldownMs: 0 });
    hold(detector, 'A', 400, 30);
    const { committed } = hold(detector, 'B', 600, 30, 400);
    expect(committed).toEqual(['B']);
  });

  it('bridges a single dropped frame instead of discarding progress', () => {
    const detector = new TimedHoldCommitDetector({
      holdMs: 500,
      cooldownMs: 0,
      toleranceMs: 120,
    });
    detector.update('A', 0);
    detector.update('A', 100);
    detector.update(NO_PREDICTION, 133); // one dropout, within tolerance
    detector.update('A', 166);
    expect(detector.update('A', 520).committed).toBe('A');
  });

  it('discards progress when the hand is gone beyond the tolerance', () => {
    const detector = new TimedHoldCommitDetector({
      holdMs: 500,
      cooldownMs: 0,
      toleranceMs: 120,
    });
    detector.update('A', 0);
    detector.update('A', 400);
    detector.update(NO_PREDICTION, 600); // 200 ms gap, beyond tolerance
    expect(detector.update('A', 700).progressPercent).toBe(0);
    expect(detector.update('A', 900).committed).toBeNull();
  });

  it('starts a fresh hold after a long gap in the same letter', () => {
    const detector = new TimedHoldCommitDetector({ holdMs: 500, cooldownMs: 0 });
    detector.update('A', 0);
    // The tab was backgrounded; the next frame is seconds later. That is not a hold.
    expect(detector.update('A', 5000).committed).toBeNull();
  });

  it('can be reset', () => {
    const detector = new TimedHoldCommitDetector({ holdMs: 500, cooldownMs: 0 });
    hold(detector, 'A', 400, 30);
    detector.reset();
    expect(detector.update('A', 450).progressPercent).toBe(0);
  });

  it('rejects a non-positive hold duration', () => {
    expect(() => new TimedHoldCommitDetector({ holdMs: 0 })).toThrow(RangeError);
  });
});
