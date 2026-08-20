import { describe, expect, it } from 'vitest';
import {
  BLANK_INDEX,
  characterErrorRate,
  commonPrefix,
  ctcGreedyDecode,
  levenshtein,
  LocalAgreementCommitter,
} from './ctc.js';

/** Labels with blank at index 0, matching the training convention. */
const LABELS = ['', 'A', 'B', 'L'];

/** Build a one-hot probability vector for a class index. */
function frame(index: number, confidence = 1): number[] {
  const vector = new Array<number>(LABELS.length).fill((1 - confidence) / (LABELS.length - 1));
  vector[index] = confidence;
  return vector;
}

describe('ctcGreedyDecode', () => {
  it('strips blanks', () => {
    const decoded = ctcGreedyDecode([frame(0), frame(1), frame(0)], LABELS);
    expect(decoded.labels).toEqual(['A']);
  });

  it('collapses a letter held across many frames into one emission', () => {
    // The user holding A for 10 frames means one A, not ten.
    const frames = Array.from({ length: 10 }, () => frame(1));
    expect(ctcGreedyDecode(frames, LABELS).labels).toEqual(['A']);
  });

  /**
   * The case a dwell timer cannot handle. v1 counted consecutive identical labels, so
   * it could not distinguish one long L from two consecutive Ls — "HELLO" was
   * unspellable. The blank symbol exists precisely to separate them.
   */
  it('emits a doubled letter when a blank separates the two', () => {
    const frames = [frame(3), frame(3), frame(0), frame(3), frame(3)];
    expect(ctcGreedyDecode(frames, LABELS).labels).toEqual(['L', 'L']);
  });

  it('does NOT double a letter with no blank between', () => {
    expect(ctcGreedyDecode([frame(3), frame(3), frame(3)], LABELS).labels).toEqual(['L']);
  });

  it('decodes a multi-letter sequence in order', () => {
    const frames = [frame(1), frame(0), frame(2), frame(0), frame(3)];
    expect(ctcGreedyDecode(frames, LABELS).labels).toEqual(['A', 'B', 'L']);
  });

  it('returns nothing for an all-blank sequence', () => {
    const result = ctcGreedyDecode([frame(0), frame(0)], LABELS);
    expect(result.labels).toEqual([]);
    expect(result.confidence).toBe(0);
  });

  it('handles an empty input', () => {
    expect(ctcGreedyDecode([], LABELS).labels).toEqual([]);
  });

  it('reports mean confidence over emitted frames only', () => {
    const frames = [frame(1, 0.8), frame(0, 0.99), frame(2, 0.6)];
    // Mean of 0.8 and 0.6; the blank frame's high confidence is irrelevant.
    expect(ctcGreedyDecode(frames, LABELS).confidence).toBeCloseTo(0.7, 6);
  });

  it('honours a non-zero blank index', () => {
    const labels = ['A', 'B', ''];
    const frames = [
      [0.9, 0.05, 0.05],
      [0.05, 0.05, 0.9],
      [0.05, 0.9, 0.05],
    ];
    expect(ctcGreedyDecode(frames, labels, 2).labels).toEqual(['A', 'B']);
  });
});

describe('commonPrefix', () => {
  it('finds the shared start', () => {
    expect(commonPrefix(['H', 'E', 'L'], ['H', 'E', 'X'])).toEqual(['H', 'E']);
  });

  it('returns empty when the first element differs', () => {
    expect(commonPrefix(['A'], ['B'])).toEqual([]);
  });

  it('handles one sequence being a prefix of the other', () => {
    expect(commonPrefix(['H', 'E'], ['H', 'E', 'L'])).toEqual(['H', 'E']);
  });
});

describe('levenshtein and characterErrorRate', () => {
  it('is zero for identical sequences', () => {
    expect(levenshtein(['A', 'B'], ['A', 'B'])).toBe(0);
  });

  it('counts substitutions, insertions and deletions', () => {
    expect(levenshtein(['A', 'B', 'C'], ['A', 'X', 'C'])).toBe(1);
    expect(levenshtein(['A', 'C'], ['A', 'B', 'C'])).toBe(1);
    expect(levenshtein(['A', 'B', 'C'], ['A', 'C'])).toBe(1);
  });

  it('handles empty sequences', () => {
    expect(levenshtein([], ['A', 'B'])).toBe(2);
    expect(levenshtein(['A'], [])).toBe(1);
    expect(levenshtein([], [])).toBe(0);
  });

  /**
   * Why Levenshtein is the metric for commit behaviour: a decoder that emits every
   * letter twice looks fine on per-frame accuracy and terrible here, which is the
   * correct verdict.
   */
  it('penalises double-firing', () => {
    expect(characterErrorRate(['H', 'I'], ['H', 'H', 'I', 'I'])).toBeCloseTo(1.0, 6);
  });

  it('normalises by reference length', () => {
    expect(characterErrorRate(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'X'])).toBeCloseTo(0.25, 6);
  });

  it('treats an empty reference sensibly', () => {
    expect(characterErrorRate([], [])).toBe(0);
    expect(characterErrorRate([], ['A'])).toBe(1);
  });
});

describe('LocalAgreementCommitter', () => {
  it('commits nothing from a single hypothesis', () => {
    const committer = new LocalAgreementCommitter(2);
    const result = committer.update(['H', 'E']);
    expect(result.committed).toEqual([]);
    expect(result.provisional).toEqual(['H', 'E']);
  });

  it('commits the prefix two consecutive hypotheses agree on', () => {
    const committer = new LocalAgreementCommitter(2);
    committer.update(['H', 'E', 'L']);
    const result = committer.update(['H', 'E', 'X']);
    expect(result.committed).toEqual(['H', 'E']);
    expect(result.provisional).toEqual(['X']);
  });

  /**
   * The behaviour that makes streaming usable. An unstable early guess is never shown
   * as committed text, so the user does not watch letters appear and then change.
   */
  it('withholds an unstable tail until it settles', () => {
    const committer = new LocalAgreementCommitter(2);
    committer.update(['H', 'S']); // model's first guess at a half-formed handshape
    expect(committer.update(['H', 'A']).committed).toEqual(['H']);
    expect(committer.update(['H', 'A']).committed).toEqual(['H', 'A']);
  });

  it('never retracts text the user has already read', () => {
    const committer = new LocalAgreementCommitter(2);
    committer.update(['H', 'E', 'L']);
    committer.update(['H', 'E', 'L']);
    expect(committer.committed).toEqual(['H', 'E', 'L']);

    // A later hypothesis disagrees about the second letter. Committed text stands.
    committer.update(['H', 'X']);
    committer.update(['H', 'X']);
    expect(committer.committed).toEqual(['H', 'E', 'L']);
  });

  it('grows the committed prefix as agreement extends', () => {
    const committer = new LocalAgreementCommitter(2);
    committer.update(['H', 'E']);
    committer.update(['H', 'E']);
    expect(committer.committed).toEqual(['H', 'E']);

    committer.update(['H', 'E', 'L', 'L', 'O']);
    committer.update(['H', 'E', 'L', 'L', 'O']);
    expect(committer.committed).toEqual(['H', 'E', 'L', 'L', 'O']);
  });

  it('is more conservative at a higher agreement setting', () => {
    const strict = new LocalAgreementCommitter(3);
    strict.update(['A']);
    strict.update(['A']);
    expect(strict.committed).toEqual([]); // two is not enough
    strict.update(['A']);
    expect(strict.committed).toEqual(['A']);
  });

  it('can be reset', () => {
    const committer = new LocalAgreementCommitter(2);
    committer.update(['A']);
    committer.update(['A']);
    committer.reset();
    expect(committer.committed).toEqual([]);
  });

  it('rejects an invalid agreement setting', () => {
    expect(() => new LocalAgreementCommitter(0)).toThrow(RangeError);
  });
});

describe('CTC end-to-end', () => {
  /**
   * The whole point of Phase 3, exercised together: a user fingerspelling "HELLO"
   * with a doubled L, decoded and committed without a dwell timer anywhere.
   */
  it('spells HELLO with a doubled L, streaming', () => {
    const labels = ['', 'H', 'E', 'L', 'O'];
    const oneHot = (index: number): number[] => {
      const v = new Array<number>(labels.length).fill(0.01);
      v[index] = 0.96;
      return v;
    };

    // Frames as a real signer would produce: each letter held briefly, blanks between.
    const frames = [
      ...Array.from({ length: 4 }, () => oneHot(1)), // H
      oneHot(BLANK_INDEX),
      ...Array.from({ length: 4 }, () => oneHot(2)), // E
      oneHot(BLANK_INDEX),
      ...Array.from({ length: 3 }, () => oneHot(3)), // L
      oneHot(BLANK_INDEX),
      ...Array.from({ length: 3 }, () => oneHot(3)), // L again
      oneHot(BLANK_INDEX),
      ...Array.from({ length: 4 }, () => oneHot(4)), // O
    ];

    const decoded = ctcGreedyDecode(frames, labels);
    expect(decoded.labels).toEqual(['H', 'E', 'L', 'L', 'O']);

    const committer = new LocalAgreementCommitter(2);
    committer.update(decoded.labels);
    const result = committer.update(decoded.labels);
    expect(result.committed).toEqual(['H', 'E', 'L', 'L', 'O']);
    expect(characterErrorRate(['H', 'E', 'L', 'L', 'O'], [...result.committed])).toBe(0);
  });
});
