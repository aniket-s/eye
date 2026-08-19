import { describe, expect, it } from 'vitest';
import { HoldCommitDetector } from './holdCommit.js';
import { NO_PREDICTION } from '../types.js';

/** Feed the same letter N times and collect every commit. */
function feed(detector: HoldCommitDetector, letter: string, frames: number): string[] {
  const committed: string[] = [];
  for (let i = 0; i < frames; i++) {
    const result = detector.update(letter);
    if (result.committed !== null) committed.push(result.committed);
  }
  return committed;
}

describe('HoldCommitDetector', () => {
  /**
   * Characterisation of a v1 off-by-one: the first frame carrying a new letter sets
   * the counter to 0 rather than 1, so `framesToCommit + 1` frames are needed. The
   * behaviour is preserved here deliberately; Phase 3 replaces the whole mechanism,
   * so correcting it now would only make the A/B comparison noisier.
   */
  it('commits one frame later than the threshold implies (v1 off-by-one)', () => {
    const detector = new HoldCommitDetector({ framesToCommit: 5, cooldownFrames: 0 });
    expect(feed(detector, 'A', 5)).toEqual([]);
    expect(feed(detector, 'A', 1)).toEqual(['A']);
  });

  it('does not commit before the threshold', () => {
    const detector = new HoldCommitDetector({ framesToCommit: 5, cooldownFrames: 0 });
    expect(feed(detector, 'A', 4)).toEqual([]);
  });

  it('reports progress from 0 to 100 as the letter is held', () => {
    const detector = new HoldCommitDetector({ framesToCommit: 4, cooldownFrames: 0 });
    expect(detector.update('A').progressPercent).toBe(0); // first sighting
    expect(detector.update('A').progressPercent).toBe(25);
    expect(detector.update('A').progressPercent).toBe(50);
    expect(detector.update('A').progressPercent).toBe(75);
    expect(detector.update('A').progressPercent).toBe(100);
  });

  it('enforces a cooldown so one long hold cannot commit repeatedly', () => {
    const detector = new HoldCommitDetector({ framesToCommit: 3, cooldownFrames: 10 });
    // 30 frames would allow 10 commits without a cooldown.
    expect(feed(detector, 'A', 30).length).toBeLessThan(4);
  });

  it('resets progress when no hand is present', () => {
    const detector = new HoldCommitDetector({ framesToCommit: 5, cooldownFrames: 0 });
    feed(detector, 'A', 4);
    expect(detector.update(NO_PREDICTION).progressPercent).toBe(0);
    expect(feed(detector, 'A', 4)).toEqual([]); // had to start over
  });

  // This is the defect the audit describes (§1.5): because the v1 model's argmax
  // flickers, a single stray frame throws away all accumulated progress.
  it('DEFECT: one flickered frame discards all progress', () => {
    const detector = new HoldCommitDetector({ framesToCommit: 5, cooldownFrames: 0 });
    feed(detector, 'A', 4);
    detector.update('S'); // a single misprediction
    expect(feed(detector, 'A', 4)).toEqual([]); // 8 of 9 frames said A, nothing committed
  });

  it('starts a fresh count when the letter changes', () => {
    const detector = new HoldCommitDetector({ framesToCommit: 3, cooldownFrames: 0 });
    feed(detector, 'A', 2);
    expect(feed(detector, 'B', 4)).toEqual(['B']);
  });

  it('can be reset explicitly', () => {
    const detector = new HoldCommitDetector({ framesToCommit: 3, cooldownFrames: 0 });
    feed(detector, 'A', 2);
    detector.reset();
    expect(feed(detector, 'A', 2)).toEqual([]);
  });

  it('rejects a non-positive threshold', () => {
    expect(() => new HoldCommitDetector({ framesToCommit: 0 })).toThrow(RangeError);
  });
});
