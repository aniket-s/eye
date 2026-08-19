import { describe, expect, it } from 'vitest';
import { argmax, relu, softmax } from './activations.js';

describe('relu', () => {
  it('passes positives through and clamps negatives to zero', () => {
    expect(relu(2.5)).toBe(2.5);
    expect(relu(-2.5)).toBe(0);
    expect(relu(0)).toBe(0);
  });
});

describe('softmax', () => {
  it('produces a distribution summing to one', () => {
    const probabilities = softmax([1, 2, 3]);
    expect(probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(probabilities.every((p) => p > 0 && p < 1)).toBe(true);
  });

  it('is monotonic in the logits', () => {
    const [low, mid, high] = softmax([0, 1, 5]) as [number, number, number];
    expect(low).toBeLessThan(mid);
    expect(mid).toBeLessThan(high);
  });

  it('stays finite for large logits, where a naive implementation overflows', () => {
    // Math.exp(1000) is Infinity; without max-subtraction this yields NaN.
    const probabilities = softmax([1000, 1001, 1002]);
    expect(probabilities.every(Number.isFinite)).toBe(true);
    expect(probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it('is invariant to a constant shift of all logits', () => {
    const a = softmax([1, 2, 3]);
    const b = softmax([101, 102, 103]);
    a.forEach((value, i) => expect(value).toBeCloseTo(b[i] as number, 12));
  });

  it('is uniform when every logit is equal', () => {
    for (const p of softmax([7, 7, 7, 7])) expect(p).toBeCloseTo(0.25, 12);
  });

  it('rejects an empty input rather than returning NaN', () => {
    expect(() => softmax([])).toThrow(RangeError);
  });
});

describe('argmax', () => {
  it('finds the largest value', () => {
    expect(argmax([1, 9, 3])).toBe(1);
  });

  it('resolves ties to the lowest index', () => {
    expect(argmax([5, 5, 2])).toBe(0);
  });

  it('handles all-negative inputs', () => {
    expect(argmax([-9, -2, -7])).toBe(1);
  });

  it('returns -1 for an empty input', () => {
    expect(argmax([])).toBe(-1);
  });
});
