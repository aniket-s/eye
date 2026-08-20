import type { ConfusionProfile } from '../registry/manifest.js';

/**
 * Turning a stream of recognised letters into words.
 *
 * This is the single highest-leverage thing in the pipeline for spelling words, and the
 * arithmetic says why. Per-letter accuracy compounds: at 93% — around the best anyone
 * achieves on real fingerspelling data — a five-letter word comes out right 70% of the
 * time, and a seven-letter word 60%. Chasing per-letter accuracy to fix that is a losing
 * game, because the exponent is doing the damage.
 *
 * Constraining the output to real words changes the shape of the problem. Most single
 * substitutions land on a string that is not a word at all, so the error is not merely
 * detectable but usually correctable — there is often exactly one real word within one
 * edit.
 *
 * Two things make this better than a spell-checker:
 *
 * 1. **The model's own confusion profile.** A generic checker treats every substitution
 *    as equally likely. This one knows, from the pack's held-out evaluation, that this
 *    model reads K as V far more often than as W, so `ves` suggests `yes` well behind
 *    `yes`… and `kes` puts `yes` ahead of `keg`. The profile ships in the manifest and is
 *    replaced on every retrain (ADR 0003).
 * 2. **Word frequency.** The list arrives in descending frequency order, so its index is a
 *    prior. `the` outranks `tho` without storing a single number.
 *
 * No network, no language model, no cost — a trie walk over 30,000 words, bounded by the
 * edit budget.
 */

/** A candidate word, with why it was offered. */
export interface WordSuggestion {
  readonly word: string;
  /** Higher is better. Combines edit cost, the confusion profile and word frequency. */
  readonly score: number;
  /** Letters that had to be changed for this candidate to fit. Zero means an exact match. */
  readonly edits: number;
}

export interface SuggestOptions {
  /** How many candidates to return. */
  readonly limit?: number;
  /**
   * Most letters that may be changed.
   *
   * Two is the useful ceiling. At three almost any short word reaches almost any other,
   * and the suggestions stop being suggestions.
   */
  readonly maxEdits?: number;
  /** Also offer longer words that the input is a prefix of. */
  readonly allowPrefix?: boolean;
}

/**
 * Cost of a substitution the confusion profile says nothing about.
 *
 * Well below the cost of a pair the model is known to confuse, so an unexplained
 * substitution loses to an explained one — but not so high that a plausible word is
 * unreachable when the profile happens to be silent.
 */
const UNRELATED_SUBSTITUTION = 1.0;

/** Cost floor for a substitution the model *does* make, however often. */
const MIN_SUBSTITUTION = 0.15;

/** Cost of an inserted or dropped letter. */
const INDEL = 1.15;

/**
 * Cost per letter still to be spelled, when offering to complete a word.
 *
 * Completing is not a correction: the user has simply not finished. So it must not be
 * charged as an edit — `hello` is two letters past `hel`, which alone would exceed the
 * whole budget and lose a perfectly good suggestion. It is not free either, or a long
 * word would tie with a short one; this is small enough to order completions by length
 * and never outweigh a real edit.
 */
const COMPLETION = 0.03;

/** Letters a completion may add. Beyond this the suggestion is a guess, not a completion. */
const MAX_COMPLETION = 8;

interface TrieNode {
  readonly children: Map<string, TrieNode>;
  /** Frequency rank if a word ends here, else -1. Lower is more common. */
  rank: number;
}

function node(): TrieNode {
  return { children: new Map(), rank: -1 };
}

export class Lexicon {
  readonly #root = node();
  #size = 0;
  /** substitution cost, keyed `observed|intended`. */
  readonly #cost = new Map<string, number>();

  /**
   * @param words Words in descending frequency order. The order is load-bearing: the
   *   index becomes the frequency prior, so an alphabetical list would rank `aardvark`
   *   above `and`.
   * @param confusions The pack's confusion profile, if it has one.
   */
  constructor(words: Iterable<string>, confusions?: ConfusionProfile) {
    let rank = 0;
    for (const raw of words) {
      const word = raw.trim().toLowerCase();
      if (word === '') continue;

      let current = this.#root;
      for (const letter of word) {
        let next = current.children.get(letter);
        if (next === undefined) {
          next = node();
          current.children.set(letter, next);
        }
        current = next;
      }
      // Keep the first (most common) rank if a word repeats.
      if (current.rank === -1) {
        current.rank = rank;
        this.#size++;
      }
      rank++;
    }

    if (confusions !== undefined) this.#loadConfusions(confusions);
  }

  get size(): number {
    return this.#size;
  }

  /**
   * Build the substitution cost table.
   *
   * The profile reads `P(predicted | true)`. Correction runs the other way: given the
   * letter that was *displayed*, how cheap is it to suppose the signer meant something
   * else? So the table is indexed by the observed letter, and the cost falls as the
   * model's tendency to make that particular mistake rises.
   */
  #loadConfusions(confusions: ConfusionProfile): void {
    for (const [intended, row] of Object.entries(confusions)) {
      for (const [observed, probability] of Object.entries(row)) {
        const cost = Math.max(MIN_SUBSTITUTION, UNRELATED_SUBSTITUTION * (1 - probability));
        const key = `${observed.toLowerCase()}|${intended.toLowerCase()}`;
        // A letter reachable from two directions keeps the cheaper explanation.
        const existing = this.#cost.get(key);
        if (existing === undefined || cost < existing) this.#cost.set(key, cost);
      }
    }
  }

  /** Cost of supposing `observed` should have been `intended`. */
  substitutionCost(observed: string, intended: string): number {
    if (observed === intended) return 0;
    return this.#cost.get(`${observed}|${intended}`) ?? UNRELATED_SUBSTITUTION;
  }

  /** Is this exactly a word? */
  has(word: string): boolean {
    let current: TrieNode | undefined = this.#root;
    for (const letter of word.toLowerCase()) {
      current = current.children.get(letter);
      if (current === undefined) return false;
    }
    return current.rank !== -1;
  }

  /**
   * Rank words that the input might have been.
   *
   * Walks the trie once, carrying a row of the edit-distance matrix down each branch and
   * abandoning a branch as soon as its cheapest possible completion exceeds the budget.
   * That bound is what keeps this fast enough to run on every committed letter.
   */
  suggest(input: string, options: SuggestOptions = {}): WordSuggestion[] {
    const target = input.trim().toLowerCase();
    if (target === '') return [];

    const limit = options.limit ?? 5;
    const budget = options.maxEdits ?? 2;
    const allowPrefix = options.allowPrefix ?? true;

    const found: { word: string; cost: number; rank: number; edits: number }[] = [];

    // Row 0 of the Levenshtein matrix: deleting each prefix of the input.
    const first = new Float64Array(target.length + 1);
    for (let i = 1; i <= target.length; i++) first[i] = (first[i - 1] as number) + INDEL;

    const walk = (current: TrieNode, previous: Float64Array, spelled: string): void => {
      // Is this branch still spelling out the input itself, rather than deviating from
      // it? Those continue past the edit budget, because extending a prefix is not an
      // error — pruning them would discard exactly the completions the user wants.
      const extending =
        allowPrefix &&
        spelled.startsWith(target.slice(0, spelled.length)) &&
        spelled.length <= target.length + MAX_COMPLETION;

      if (!extending && Math.min(...previous) > budget) return;

      if (current.rank !== -1) {
        const cost = previous[target.length] as number;
        const completing =
          allowPrefix && spelled.length >= target.length && spelled.startsWith(target);
        const effective = completing ? (spelled.length - target.length) * COMPLETION : cost;

        if (effective <= budget) {
          found.push({
            word: spelled,
            cost: effective,
            rank: current.rank,
            edits: completing ? 0 : Math.round(cost * 10) / 10,
          });
        }
      }

      for (const [letter, child] of current.children) {
        const next = new Float64Array(target.length + 1);
        next[0] = (previous[0] as number) + INDEL;

        for (let i = 1; i <= target.length; i++) {
          const observed = target[i - 1] as string;
          const substitute = (previous[i - 1] as number) + this.substitutionCost(observed, letter);
          const insert = (next[i - 1] as number) + INDEL;
          const remove = (previous[i] as number) + INDEL;
          next[i] = Math.min(substitute, insert, remove);
        }

        walk(child, next, spelled + letter);
      }
    };

    walk(this.#root, first, '');

    // Cheapest first; among equally plausible candidates, the commoner word. The rank
    // divisor keeps frequency as a tie-breaker rather than something that can outvote a
    // whole edit.
    found.sort((a, b) => a.cost - b.cost || a.rank - b.rank);

    return found.slice(0, limit).map((entry) => ({
      word: entry.word,
      score: 1 / (1 + entry.cost + entry.rank / 100_000),
      edits: entry.edits,
    }));
  }
}

/** Parse the shipped word list: one word per line, most common first. */
export function parseWordList(text: string): string[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line !== '' && !line.startsWith('#'));
}
