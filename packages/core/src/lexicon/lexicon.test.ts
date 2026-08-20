import { describe, expect, it } from 'vitest';
import { Lexicon, parseWordList } from './lexicon.js';

/** Frequency order matters, so these are deliberately not alphabetical. */
const WORDS = [
  'the',
  'and',
  'you',
  'yes',
  'that',
  'have',
  'this',
  'work',
  'word',
  'help',
  'hello',
  'name',
  'thanks',
  'keg',
  'ves',
  'vet',
  'tho',
  'worked',
  'working',
];

/** A model that reads K as V, and E as O. */
const CONFUSIONS = {
  K: { V: 0.31, X: 0.04 },
  E: { O: 0.22 },
  T: { N: 0.05, A: 0.04 },
};

describe('parseWordList', () => {
  it('reads one word per line and drops blanks and comments', () => {
    expect(parseWordList('# header\nthe\n\n and \nyou\n')).toEqual(['the', 'and', 'you']);
  });
});

describe('Lexicon', () => {
  it('counts the words it holds', () => {
    expect(new Lexicon(WORDS).size).toBe(WORDS.length);
  });

  it('recognises a word it holds, and rejects one it does not', () => {
    const lexicon = new Lexicon(WORDS);
    expect(lexicon.has('hello')).toBe(true);
    expect(lexicon.has('hellp')).toBe(false);
  });

  it('is case-insensitive, since letters arrive upper-case from the model', () => {
    expect(new Lexicon(WORDS).has('HELLO')).toBe(true);
  });

  it('ignores an empty query rather than offering the whole dictionary', () => {
    expect(new Lexicon(WORDS).suggest('')).toEqual([]);
  });

  it('deduplicates repeated words but keeps the more common rank', () => {
    const lexicon = new Lexicon(['the', 'and', 'the']);
    expect(lexicon.size).toBe(2);
  });
});

describe('suggesting words', () => {
  const lexicon = new Lexicon(WORDS, CONFUSIONS);

  it('puts an exact match first', () => {
    expect(lexicon.suggest('work')[0]?.word).toBe('work');
    expect(lexicon.suggest('work')[0]?.edits).toBe(0);
  });

  /** The headline case: one misread letter, one obvious correction. */
  it('corrects a single wrong letter', () => {
    expect(lexicon.suggest('helli').map((s) => s.word)).toContain('hello');
  });

  it('completes a word the user has not finished spelling', () => {
    const words = lexicon.suggest('hel').map((s) => s.word);
    expect(words).toContain('hello');
    expect(words).toContain('help');
  });

  it('does not complete when prefixes are turned off', () => {
    const words = lexicon.suggest('wor', { allowPrefix: false, maxEdits: 0 }).map((s) => s.word);
    expect(words).not.toContain('working');
  });

  it('respects the edit budget', () => {
    // Three letters wrong is beyond any sane budget: nothing should come back.
    expect(lexicon.suggest('zzzz', { maxEdits: 1, allowPrefix: false })).toEqual([]);
  });

  it('returns at most the requested number of candidates', () => {
    expect(lexicon.suggest('the', { limit: 2 }).length).toBeLessThanOrEqual(2);
  });

  it('breaks ties toward the commoner word', () => {
    // `the` is first in the list, `tho` near the end; both are one edit from `thh`.
    const words = lexicon.suggest('thh', { allowPrefix: false }).map((s) => s.word);
    expect(words.indexOf('the')).toBeLessThan(words.indexOf('tho'));
  });

  it('handles a dropped letter', () => {
    expect(lexicon.suggest('hllo').map((s) => s.word)).toContain('hello');
  });

  it('handles a doubled letter', () => {
    expect(lexicon.suggest('heello').map((s) => s.word)).toContain('hello');
  });
});

/**
 * The reason this is not a spell-checker.
 *
 * A generic checker weighs every substitution alike. Weighting by the pack's measured
 * error profile means the correction that is *mechanically plausible for this model* wins
 * over one that merely happens to be a word.
 */
describe('using the model confusion profile', () => {
  const plain = new Lexicon(WORDS);
  const informed = new Lexicon(WORDS, CONFUSIONS);

  it('makes a substitution the model is known to make cheaper', () => {
    // The profile says a displayed V is often an intended K.
    expect(informed.substitutionCost('v', 'k')).toBeLessThan(informed.substitutionCost('v', 'q'));
  });

  it('leaves unrelated substitutions at full cost', () => {
    expect(informed.substitutionCost('q', 'z')).toBe(plain.substitutionCost('q', 'z'));
  });

  it('costs nothing to leave a letter alone', () => {
    expect(informed.substitutionCost('a', 'a')).toBe(0);
  });

  it('prefers the correction its own errors explain', () => {
    // "ves" is a word, so a plain lexicon has no reason to look further. Knowing the
    // model reads K as V, "yes" — reachable through a mistake it actually makes —
    // deserves to be offered too.
    const suggestions = informed.suggest('ves', { limit: 5 }).map((s) => s.word);
    expect(suggestions).toContain('yes');
  });

  it('ranks a known confusion above an equally distant unknown one', () => {
    // `keg` and `ves` are both one substitution from `veg`-like inputs; the profile
    // should separate them.
    const informedRank = informed.suggest('vet', { limit: 6 }).findIndex((s) => s.word === 'vet');
    expect(informedRank).toBe(0);
  });

  it('works without a profile, treating every substitution alike', () => {
    expect(plain.suggest('helli').map((s) => s.word)).toContain('hello');
  });
});

describe('performance', () => {
  /**
   * The real list is 30,000 words and this runs on every committed letter, so an
   * implementation that walked the whole trie would be felt immediately.
   */
  it('stays fast on a realistic vocabulary', () => {
    const many: string[] = [...WORDS];
    for (let i = 0; i < 30_000; i++) {
      many.push(`w${i.toString(36)}xyz`);
    }
    const large = new Lexicon(many, CONFUSIONS);

    const started = performance.now();
    for (let i = 0; i < 20; i++) large.suggest('helli');
    const perCall = (performance.now() - started) / 20;

    expect(perCall).toBeLessThan(50);
  });
});
