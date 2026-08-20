import { describe, expect, it } from 'vitest';
import {
  DEFAULT_THRESHOLDS,
  energyScore,
  judge,
  NONE_LABEL,
  ProbabilitySmoother,
  topMargin,
} from './rejection.js';
import { softmax } from '../math/activations.js';

describe('energyScore', () => {
  it('is lower for confident logits than for flat ones', () => {
    // Energy tracks logit magnitude, which softmax discards.
    expect(energyScore([8, 0, 0])).toBeLessThan(energyScore([0.2, 0.1, 0.05]));
  });

  /**
   * The property that makes energy useful for rejection. These two vectors have
   * *identical* softmax distributions — softmax is shift-invariant — yet one is a
   * strong activation and the other is barely responding at all.
   */
  it('separates inputs that max-softmax cannot tell apart', () => {
    const strong = [10, 6, 4];
    const weak = [4, 0, -2]; // same differences, so identical softmax
    const strongProbs = softmax(strong);
    const weakProbs = softmax(weak);
    strongProbs.forEach((p, i) => expect(p).toBeCloseTo(weakProbs[i] as number, 10));

    expect(energyScore(strong)).toBeLessThan(energyScore(weak));
  });

  it('honours the temperature parameter', () => {
    expect(energyScore([3, 1, 0], 2)).not.toBeCloseTo(energyScore([3, 1, 0], 1), 6);
  });

  it('rejects empty logits and non-positive temperature', () => {
    expect(() => energyScore([])).toThrow(RangeError);
    expect(() => energyScore([1, 2], 0)).toThrow(RangeError);
  });
});

describe('topMargin', () => {
  it('measures the gap between the top two classes', () => {
    expect(topMargin([0.7, 0.2, 0.1])).toBeCloseTo(0.5, 10);
  });

  it('is near zero when the model is torn, even at high confidence', () => {
    // Max probability is 0.49 — "confident" by many thresholds — but useless.
    expect(topMargin([0.49, 0.48, 0.03])).toBeCloseTo(0.01, 10);
  });

  it('handles a single class', () => {
    expect(topMargin([1])).toBe(1);
  });
});

describe('ProbabilitySmoother', () => {
  it('returns the input unchanged for the first frame', () => {
    const smoother = new ProbabilitySmoother(3, 5);
    const out = smoother.push([0.6, 0.3, 0.1]);
    expect(out[0]).toBeCloseTo(0.6, 6);
  });

  it('keeps the window bounded', () => {
    const smoother = new ProbabilitySmoother(2, 3);
    for (let i = 0; i < 10; i++) smoother.push([0.5, 0.5]);
    expect(smoother.size).toBe(3);
  });

  it('always returns a valid distribution', () => {
    const smoother = new ProbabilitySmoother(3, 5);
    for (const frame of [
      [0.7, 0.2, 0.1],
      [0.1, 0.8, 0.1],
      [0.6, 0.3, 0.1],
    ]) {
      const out = smoother.push(frame);
      expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
      expect(out.every((p) => p >= 0)).toBe(true);
    }
  });

  /**
   * The behaviour v1 lacked. v1 compared consecutive argmax labels, so a single bad
   * frame reset the entire hold (docs/AUDIT.md, §5). A per-class median outvotes it.
   */
  it('suppresses a single spurious spike', () => {
    const smoother = new ProbabilitySmoother(3, 5);
    for (let i = 0; i < 4; i++) smoother.push([0.8, 0.1, 0.1]);

    const spiked = smoother.push([0.05, 0.9, 0.05]); // one confidently wrong frame
    expect(spiked.indexOf(Math.max(...spiked))).toBe(0); // class 0 still wins
  });

  it('does follow a sustained change, so real transitions still register', () => {
    const smoother = new ProbabilitySmoother(3, 5);
    for (let i = 0; i < 5; i++) smoother.push([0.8, 0.1, 0.1]);
    let out: number[] = [];
    for (let i = 0; i < 5; i++) out = smoother.push([0.05, 0.9, 0.05]);
    expect(out.indexOf(Math.max(...out))).toBe(1);
  });

  it('can be reset', () => {
    const smoother = new ProbabilitySmoother(2, 4);
    smoother.push([0.9, 0.1]);
    smoother.reset();
    expect(smoother.size).toBe(0);
  });

  it('rejects a mis-sized vector and invalid construction', () => {
    expect(() => new ProbabilitySmoother(0)).toThrow(RangeError);
    expect(() => new ProbabilitySmoother(3, 0)).toThrow(RangeError);
    expect(() => new ProbabilitySmoother(3).push([0.5, 0.5])).toThrow(RangeError);
  });
});

describe('judge', () => {
  const labels = ['A', 'B', NONE_LABEL];

  it('accepts a clear, confident prediction', () => {
    const verdict = judge(labels, [0.9, 0.05, 0.05], null);
    expect(verdict.accepted).toBe(true);
    expect(verdict.label).toBe('A');
    expect(verdict.reason).toBe('accepted');
  });

  it('reports the none class as its own outcome, not a threshold failure', () => {
    const verdict = judge(labels, [0.1, 0.1, 0.8], null);
    expect(verdict.accepted).toBe(false);
    expect(verdict.reason).toBe('none-class');
  });

  it('rejects a low-probability winner', () => {
    expect(judge(labels, [0.4, 0.35, 0.25], null).reason).toBe('low-probability');
  });

  it('rejects when the top two classes are too close', () => {
    // 0.52 clears the probability bar but the margin is only 0.06.
    const verdict = judge(labels, [0.52, 0.46, 0.02], null, {
      ...DEFAULT_THRESHOLDS,
      minProbability: 0.5,
    });
    expect(verdict.reason).toBe('low-margin');
  });

  it('rejects on energy when a ceiling is configured', () => {
    const verdict = judge(labels, [0.95, 0.03, 0.02], [0.2, 0.1, 0.05], {
      minProbability: 0.5,
      minMargin: 0.1,
      maxEnergy: -5,
    });
    expect(verdict.reason).toBe('high-energy');
  });

  it('skips the energy check when no logits are supplied', () => {
    const verdict = judge(labels, [0.95, 0.03, 0.02], null, {
      minProbability: 0.5,
      minMargin: 0.1,
      maxEnergy: -1000,
    });
    expect(verdict.accepted).toBe(true);
  });

  it('reports the diagnostic values whatever the outcome', () => {
    const verdict = judge(labels, [0.4, 0.35, 0.25], [2, 1.8, 1]);
    expect(verdict.probability).toBeCloseTo(0.4, 10);
    expect(verdict.margin).toBeCloseTo(0.05, 10);
    expect(Number.isFinite(verdict.energy)).toBe(true);
  });
});

/**
 * ASL's own ambiguity, handled where it belongs.
 *
 * 2 and V are the same handshape, as are 6/W, 9/F and 0/O. A signer separates them by
 * context. Restricting the argmax to the active vocabulary *is* that context — and
 * without it the two readings compete for one pile of probability and neither wins.
 */
describe('restricting judgement to a vocabulary', () => {
  const LABELS = ['A', 'V', '5', NONE_LABEL];

  it('ignores a label outside the active set, however confident', () => {
    const verdict = judge(
      LABELS,
      [0.9, 0.05, 0.03, 0.02],
      null,
      DEFAULT_THRESHOLDS,
      new Set(['V']),
    );
    expect(verdict.label).toBe('V');
  });

  it('behaves exactly as before when no set is given', () => {
    const probabilities = [0.9, 0.05, 0.03, 0.02];
    expect(judge(LABELS, probabilities, null).label).toBe('A');
    expect(judge(LABELS, probabilities, null, DEFAULT_THRESHOLDS, null).label).toBe('A');
  });

  it('always keeps `none` eligible', () => {
    // Rejecting a frame must never depend on which vocabulary happens to be selected.
    const verdict = judge(LABELS, [0.1, 0.1, 0.1, 0.7], null, DEFAULT_THRESHOLDS, new Set(['V']));
    expect(verdict.label).toBe(NONE_LABEL);
    expect(verdict.accepted).toBe(false);
  });

  /**
   * The margin has to be measured inside the vocabulary too, or selecting "numbers" would
   * still see V crowding 2 in the runner-up slot and reject exactly the frames the mode
   * exists to accept.
   */
  it('measures the margin within the active set', () => {
    // A and V nearly tied, which is exactly the 2-vs-V situation. Unrestricted, the
    // margin is too thin to accept anything. With V out of the vocabulary, A is the
    // clear winner and the frame is usable.
    const probabilities = [0.64, 0.28, 0.05, 0.03];
    const thresholds = { minProbability: 0.6, minMargin: 0.45, maxEnergy: null };

    expect(judge(LABELS, probabilities, null, thresholds).reason).toBe('low-margin');

    const scoped = judge(LABELS, probabilities, null, thresholds, new Set(['A', '5']));
    expect(scoped.label).toBe('A');
    expect(scoped.accepted).toBe(true);
  });

  it('rejects rather than throwing when the set matches nothing', () => {
    const verdict = judge(['A', 'B'], [0.6, 0.4], null, DEFAULT_THRESHOLDS, new Set(['Z']));
    expect(verdict.accepted).toBe(false);
    expect(verdict.label).toBe(NONE_LABEL);
  });
});
