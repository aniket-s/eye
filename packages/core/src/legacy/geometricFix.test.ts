/**
 * Characterisation tests for the v1 geometric overrides.
 *
 * These pin the *current* behaviour so Phase 2 can delete the module and prove the
 * replacement classifier is better rather than merely different. They are not a
 * specification — several assert outcomes that are plainly wrong.
 *
 * When Phase 2 lands, this file is deleted along with `geometricFix.ts`, and the
 * golden-fixture suite becomes the contract instead. See ./README.md.
 */
import { describe, expect, it } from 'vitest';
import { geometricFix } from './geometricFix.js';
import { closedFist, openHand, scale, translate } from '../test/landmarkFactory.js';

/** The 24 static letters the v1 accuracy test exercised (J and Z are motion-based). */
const STATIC_LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'.split('');

describe('geometricFix — invariants that do hold', () => {
  it('is deterministic', () => {
    const hand = openHand();
    expect(geometricFix('S', hand)).toBe(geometricFix('S', hand));
  });

  it('is invariant to where the hand sits in frame', () => {
    for (const seed of STATIC_LETTERS) {
      expect(geometricFix(seed, translate(openHand(), 0.15, -0.2))).toBe(
        geometricFix(seed, openHand()),
      );
    }
  });

  it('is invariant to how close the hand is to the camera', () => {
    for (const seed of STATIC_LETTERS) {
      expect(geometricFix(seed, scale(openHand(), 1.6))).toBe(geometricFix(seed, openHand()));
    }
  });

  it('passes through letters it has no rule for', () => {
    // Only the branches listed in the source are handled; anything else falls through.
    expect(geometricFix('Z', closedFist())).not.toBe('');
    expect(geometricFix('unknown-label', openHand())).toBe('unknown-label');
  });
});

describe('geometricFix — DEFECT: the output depends on the model, not the hand', () => {
  /**
   * This is the central finding of docs/AUDIT.md §1.2.
   *
   * `geometricFix` is meant to correct the model using hand geometry. If it did that,
   * a fixed hand would yield one answer regardless of what the model guessed. It does
   * not: the same geometry produces 8–10 different letters depending purely on the
   * model's `argmax`. The decision surface has no single definition, which is why the
   * model cannot be retrained without re-tuning all 85 lines by hand.
   */
  it('returns 8 different letters for one unchanging open hand', () => {
    const hand = openHand();
    const outputs = new Set(STATIC_LETTERS.map((seed) => geometricFix(seed, hand)));
    expect(outputs.size).toBe(8);
    expect([...outputs].sort()).toEqual(['B', 'E', 'H', 'K', 'L', 'R', 'U', 'Y']);
  });

  it('returns 10 different letters for one unchanging closed fist', () => {
    const hand = closedFist();
    const outputs = new Set(STATIC_LETTERS.map((seed) => geometricFix(seed, hand)));
    expect(outputs.size).toBe(10);
    expect([...outputs].sort()).toEqual(['A', 'C', 'E', 'F', 'G', 'L', 'M', 'N', 'S', 'Y']);
  });

  it('disagrees with itself on the same fist: I→S but U→N', () => {
    const hand = closedFist();
    expect(geometricFix('I', hand)).toBe('S');
    expect(geometricFix('U', hand)).toBe('N');
    expect(geometricFix('M', hand)).toBe('M');
    // Three different answers for one physical handshape.
  });

  it('collapses unrelated seeds onto the same letter', () => {
    const hand = openHand();
    // I, M, S and V all become K on an open hand — the branches overlap.
    for (const seed of ['I', 'M', 'S', 'V']) {
      expect(geometricFix(seed, hand)).toBe('K');
    }
  });

  it('rewrites a correct prediction into a wrong one', () => {
    // The model says A for a closed fist and is right; the T/A branch keeps it.
    expect(geometricFix('A', closedFist())).toBe('A');
    // But the model saying B for the same fist yields F, not A.
    expect(geometricFix('B', closedFist())).toBe('F');
  });
});

describe('geometricFix — characterisation of each branch', () => {
  const cases: ReadonlyArray<readonly [seed: string, hand: 'open' | 'fist', expected: string]> = [
    ['F', 'open', 'B'],
    ['B', 'open', 'B'],
    ['W', 'open', 'B'],
    ['F', 'fist', 'F'],
    ['E', 'open', 'E'],
    ['O', 'open', 'E'],
    ['E', 'fist', 'E'],
    ['D', 'open', 'L'],
    ['X', 'open', 'L'],
    ['C', 'open', 'L'],
    ['D', 'fist', 'C'],
    ['G', 'open', 'H'],
    ['H', 'open', 'H'],
    ['G', 'fist', 'G'],
    ['S', 'open', 'K'],
    ['S', 'fist', 'S'],
    ['M', 'open', 'K'],
    ['M', 'fist', 'M'],
    ['N', 'open', 'U'],
    ['U', 'open', 'U'],
    ['N', 'fist', 'N'],
    ['T', 'open', 'B'],
    ['A', 'open', 'B'],
    ['T', 'fist', 'A'],
    ['A', 'fist', 'A'],
    ['P', 'open', 'R'],
    ['L', 'open', 'R'],
    ['Q', 'open', 'R'],
    ['R', 'open', 'R'],
    ['L', 'fist', 'L'],
    ['Y', 'open', 'Y'],
    ['Y', 'fist', 'Y'],
  ];

  it.each(cases)('%s on a %s hand becomes %s', (seed, hand, expected) => {
    expect(geometricFix(seed, hand === 'open' ? openHand() : closedFist())).toBe(expected);
  });
});
