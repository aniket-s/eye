/**
 * CTC decoding.
 *
 * Connectionist Temporal Classification removes the segmentation problem that v1
 * tried to solve with a dwell timer and a bespoke motion state machine.
 *
 * The model emits a distribution per frame over `blank + letters`. Decoding collapses
 * repeated labels and strips blanks, so a letter is emitted at the frame where the
 * argmax transitions **into** it — a peak detector, not a stopwatch. Three consequences
 * fall out of that, each replacing a v1 mechanism:
 *
 * - **No hold timer.** The user fingerspells at their own pace instead of pausing
 *   two thirds of a second on every letter (docs/AUDIT.md §5).
 * - **J and Z are ordinary labels.** They are defined by motion, and a temporal model
 *   sees motion, so the 100-line state machine with its frame-counted thresholds and
 *   both-directions Z test simply disappears (J1, J2, J3).
 * - **Doubled letters work.** `LL` in "HELLO" is a genuine problem for a dwell timer —
 *   it cannot tell one long L from two — and is exactly what CTC's blank symbol
 *   exists to disambiguate.
 *
 * Reference point from Google's fingerspelling competition: greedy CTC scored 0.807 at
 * 11.4 MB and 49.7 ms, against 0.812 at 12.5 MB and 190 ms for a joint CTC+attention
 * model. Half a point for a 4× speed-up makes greedy the clear choice in a browser.
 */

/** Index of the CTC blank symbol. Convention: 0, matching the training code. */
export const BLANK_INDEX = 0;

export interface CtcDecodeResult {
  /** Decoded labels, blanks stripped and repeats collapsed. */
  readonly labels: string[];
  /** Mean probability of the emitted (non-blank) frames, as a rough confidence. */
  readonly confidence: number;
}

/**
 * Greedy CTC decode: argmax per frame, collapse repeats, strip blanks.
 *
 * @param frameProbabilities One probability vector per frame, over `blank + labels`.
 * @param labels Class labels **including** the blank at {@link BLANK_INDEX}.
 * @param blankIndex Index of the blank symbol.
 */
export function ctcGreedyDecode(
  frameProbabilities: readonly (readonly number[])[],
  labels: readonly string[],
  blankIndex: number = BLANK_INDEX,
): CtcDecodeResult {
  const decoded: string[] = [];
  let previousIndex = -1;
  let emittedProbability = 0;
  let emittedCount = 0;

  for (const frame of frameProbabilities) {
    let best = 0;
    for (let i = 1; i < frame.length; i++) {
      if ((frame[i] as number) > (frame[best] as number)) best = i;
    }

    // A repeat only counts as a new label if a blank separated the two — that is the
    // entire reason blank exists, and what lets "LL" survive decoding.
    if (best !== previousIndex && best !== blankIndex) {
      const label = labels[best];
      if (label !== undefined) {
        decoded.push(label);
        emittedProbability += frame[best] as number;
        emittedCount++;
      }
    }
    previousIndex = best;
  }

  return {
    labels: decoded,
    confidence: emittedCount === 0 ? 0 : emittedProbability / emittedCount,
  };
}

/**
 * Longest common prefix of two label sequences.
 *
 * The primitive behind {@link LocalAgreementCommitter}.
 */
export function commonPrefix(a: readonly string[], b: readonly string[]): string[] {
  const limit = Math.min(a.length, b.length);
  let index = 0;
  while (index < limit && a[index] === b[index]) index++;
  return a.slice(0, index);
}

/**
 * Levenshtein edit distance.
 *
 * The correct metric for commit behaviour, because it penalises misrecognition,
 * double-firing and misses in one number. Per-frame accuracy penalises none of them —
 * a decoder that emits every letter twice can still score well per frame.
 */
export function levenshtein(a: readonly string[], b: readonly string[]): number {
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  let previous = Array.from({ length: b.length + 1 }, (_, i) => i);
  let current = new Array<number>(b.length + 1);

  for (let i = 1; i <= a.length; i++) {
    current[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const substitution = (previous[j - 1] as number) + (a[i - 1] === b[j - 1] ? 0 : 1);
      const deletion = (previous[j] as number) + 1;
      const insertion = (current[j - 1] as number) + 1;
      current[j] = Math.min(substitution, deletion, insertion);
    }
    [previous, current] = [current, previous];
  }
  return previous[b.length] as number;
}

/** Character error rate: edit distance normalised by reference length. */
export function characterErrorRate(
  reference: readonly string[],
  hypothesis: readonly string[],
): number {
  if (reference.length === 0) return hypothesis.length === 0 ? 0 : 1;
  return levenshtein(reference, hypothesis) / reference.length;
}

/**
 * Commit text once the decoder stops changing its mind.
 *
 * A streaming decoder re-decodes a growing window every few frames, and early
 * hypotheses are unstable: the model may read a half-formed handshape as `S` and
 * revise it to `A` two frames later. Emitting immediately shows the user that churn;
 * waiting a fixed time is the dwell timer again.
 *
 * LocalAgreement-*n* commits the longest prefix that the last *n* hypotheses agree on,
 * and leaves the rest provisional. It gates on **hypothesis stability** rather than
 * elapsed time, so a confident decoder commits fast and an uncertain one waits — which
 * is the behaviour you actually want. Reported cost in streaming ASR is roughly 0.2%
 * WER against offline decoding.
 *
 * The user sees committed text settle while provisional text updates live, which also
 * makes the system legible: you can watch it think.
 */
export class LocalAgreementCommitter {
  readonly #agreement: number;
  readonly #history: string[][] = [];
  #committed: string[] = [];

  /**
   * @param agreement How many consecutive hypotheses must agree. 2 is the practical
   *   default; higher is more conservative and adds latency.
   */
  constructor(agreement = 2) {
    if (agreement < 1) throw new RangeError('agreement must be at least 1');
    this.#agreement = agreement;
  }

  /** Text committed so far. */
  get committed(): readonly string[] {
    return this.#committed;
  }

  reset(): void {
    this.#history.length = 0;
    this.#committed = [];
  }

  /**
   * Offer a new hypothesis for the whole window.
   *
   * @returns The committed text and the still-provisional tail.
   */
  update(hypothesis: readonly string[]): {
    committed: readonly string[];
    provisional: readonly string[];
  } {
    this.#history.push([...hypothesis]);
    if (this.#history.length > this.#agreement) this.#history.shift();

    if (this.#history.length === this.#agreement) {
      let agreed = this.#history[0] as string[];
      for (let i = 1; i < this.#history.length; i++) {
        agreed = commonPrefix(agreed, this.#history[i] as string[]);
      }
      // Only ever grow. A later hypothesis that shortens the agreed prefix must not
      // retract text the user has already read.
      if (agreed.length > this.#committed.length) {
        this.#committed = agreed;
      }
    }

    return {
      committed: this.#committed,
      provisional: hypothesis.slice(this.#committed.length),
    };
  }
}
