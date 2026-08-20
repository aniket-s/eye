/**
 * Deciding when *not* to emit a letter.
 *
 * v1 had a single confidence threshold (`0.45`) on an uncalibrated softmax and no
 * negative class, so every transitional pose between two letters was force-classified
 * as a letter (docs/AUDIT.md, M4 and M5).
 *
 * Threshold-only rejection cannot fix that on its own. A model that never saw
 * negatives is *confidently* wrong on transitions, because a transition is
 * geometrically an interpolation between two trained classes — the network happily
 * assigns it high probability. Three layers are needed, and this module provides the
 * two that live at inference time:
 *
 * 1. **A trained `none` class.** The foundation, and it belongs to training, not here.
 * 2. **Energy score** (Liu et al., NeurIPS 2020) — aligns with density far better
 *    than max-softmax and needs no retraining.
 * 3. **Temporal smoothing** — a median filter over recent probability vectors.
 *
 * In published real-time gesture pipelines the smoothing, not the threshold value, is
 * what actually removes flicker.
 */

/** Label reserved for "no sign is being made". */
export const NONE_LABEL = 'none';

/**
 * Free energy of a logit vector: `E(x) = −T · logsumexp(logits / T)`.
 *
 * Lower energy means the input looks more like the training distribution. Unlike max
 * softmax probability, energy keeps the *magnitude* of the logits, which is precisely
 * the information softmax normalises away — and precisely what distinguishes a
 * confident in-distribution prediction from a confident out-of-distribution one.
 *
 * @param temperature Softens the score. 1 is the usual choice.
 */
export function energyScore(logits: readonly number[], temperature = 1): number {
  if (logits.length === 0) throw new RangeError('energyScore requires at least one logit');
  if (temperature <= 0) throw new RangeError('temperature must be positive');

  let max = -Infinity;
  for (const value of logits) {
    const scaled = value / temperature;
    if (scaled > max) max = scaled;
  }
  let sum = 0;
  for (const value of logits) sum += Math.exp(value / temperature - max);

  return -temperature * (max + Math.log(sum));
}

/**
 * Difference between the top two probabilities.
 *
 * More robust than max-probability alone: a model torn between two letters reports
 * high confidence in the winner while the margin stays near zero.
 */
export function topMargin(probabilities: readonly number[]): number {
  let best = -Infinity;
  let second = -Infinity;
  for (const value of probabilities) {
    if (value > best) {
      second = best;
      best = value;
    } else if (value > second) {
      second = value;
    }
  }
  return second === -Infinity ? best : best - second;
}

/**
 * Median filter over a sliding window of probability vectors.
 *
 * Smoothing *probabilities* rather than the argmax label is the key difference from
 * v1. v1 compared consecutive labels, so a single flickered frame reset the whole
 * hold. Taking a per-class median means one bad frame is outvoted and never surfaces.
 *
 * The median is used rather than the mean because it is insensitive to outliers, and
 * a spurious near-1.0 spike on the wrong class is exactly the outlier to suppress.
 */
export class ProbabilitySmoother {
  readonly #window: number;
  readonly #classes: number;
  readonly #history: number[][] = [];

  /**
   * @param classes Number of output classes.
   * @param window Frames to smooth over. 8 is roughly a quarter-second at 30 fps —
   *   long enough to outvote flicker, short enough not to blur real transitions.
   */
  constructor(classes: number, window = 8) {
    if (classes <= 0) throw new RangeError('classes must be positive');
    if (window <= 0) throw new RangeError('window must be positive');
    this.#classes = classes;
    this.#window = window;
  }

  /** Frames currently in the window. */
  get size(): number {
    return this.#history.length;
  }

  reset(): void {
    this.#history.length = 0;
  }

  /**
   * Add a frame and return the smoothed distribution.
   *
   * @returns Per-class medians, renormalised to sum to 1.
   * @throws RangeError if the vector length does not match `classes`.
   */
  push(probabilities: readonly number[]): number[] {
    if (probabilities.length !== this.#classes) {
      throw new RangeError(
        `Expected ${this.#classes} probabilities, received ${probabilities.length}`,
      );
    }

    this.#history.push([...probabilities]);
    if (this.#history.length > this.#window) this.#history.shift();

    const smoothed = new Array<number>(this.#classes);
    const column = new Array<number>(this.#history.length);
    let total = 0;

    for (let c = 0; c < this.#classes; c++) {
      for (let f = 0; f < this.#history.length; f++) column[f] = this.#history[f]![c]!;
      const median = medianOf(column);
      smoothed[c] = median;
      total += median;
    }

    // Per-class medians do not sum to 1, so renormalise to keep a valid distribution.
    if (total > 0) {
      for (let c = 0; c < this.#classes; c++) smoothed[c] = (smoothed[c] as number) / total;
    }
    return smoothed;
  }
}

/** Median of `values`. Mutates a copy, not the input. */
function medianOf(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 0
    ? ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2
    : (sorted[middle] as number);
}

export interface RejectionThresholds {
  /** Minimum smoothed probability for the winning class. */
  readonly minProbability: number;
  /** Minimum gap between the top two classes. */
  readonly minMargin: number;
  /** Reject when energy exceeds this. `null` disables the check. */
  readonly maxEnergy: number | null;
}

/**
 * Conservative defaults.
 *
 * Shipped only as a fallback: a trained model carries its own operating point in its
 * pack manifest, so a retrain can never leave the app running on thresholds tuned for
 * different weights — the coupling that trapped v1 (ADR 0003).
 */
export const DEFAULT_THRESHOLDS: RejectionThresholds = {
  minProbability: 0.6,
  minMargin: 0.15,
  maxEnergy: null,
};

export interface Verdict {
  readonly accepted: boolean;
  readonly label: string;
  readonly probability: number;
  readonly margin: number;
  readonly energy: number;
  /** Which check rejected it, for the debug overlay. */
  readonly reason: 'accepted' | 'none-class' | 'low-probability' | 'low-margin' | 'high-energy';
}

/**
 * Decide whether a prediction should be shown.
 *
 * @param labels Class labels, in model output order.
 * @param probabilities Smoothed probabilities, same order.
 * @param logits Raw logits, for the energy score. Pass `null` to skip it.
 */
export function judge(
  labels: readonly string[],
  probabilities: readonly number[],
  logits: readonly number[] | null,
  thresholds: RejectionThresholds = DEFAULT_THRESHOLDS,
  allowed: ReadonlySet<string> | null = null,
): Verdict {
  // Restricting the argmax to the active vocabulary is what makes ASL's genuinely
  // ambiguous shapes usable. 2 and V are the *same* handshape, as are 6/W, 9/F and 0/O —
  // a signer separates them by context, and this is that context. Without it the two
  // readings would compete for the same probability mass and neither would clear the
  // margin threshold.
  //
  // `none` is always eligible: rejecting a frame must never depend on which vocabulary
  // happens to be selected.
  const eligible = (index: number): boolean =>
    allowed === null || labels[index] === NONE_LABEL || allowed.has(labels[index] ?? '');

  let bestIndex = -1;
  for (let i = 0; i < probabilities.length; i++) {
    if (!eligible(i)) continue;
    if (bestIndex === -1 || (probabilities[i] as number) > (probabilities[bestIndex] as number)) {
      bestIndex = i;
    }
  }
  if (bestIndex === -1) {
    return {
      label: NONE_LABEL,
      probability: 0,
      margin: 0,
      energy: 0,
      accepted: false,
      reason: 'none-class',
    };
  }

  const label = labels[bestIndex] ?? NONE_LABEL;
  const probability = probabilities[bestIndex] ?? 0;
  // The margin is measured within the active vocabulary too. Otherwise selecting
  // "numbers" would still see V crowding 2 in the runner-up slot, and the margin check
  // would reject exactly the frames the mode exists to accept.
  const margin =
    allowed === null
      ? topMargin(probabilities)
      : topMargin(probabilities.filter((_, index) => eligible(index)));
  const energy = logits === null ? 0 : energyScore(logits);

  const base = { label, probability, margin, energy };

  // The model saying "no sign" is a decision, not a failure — check it first so the
  // debug overlay can distinguish it from a threshold rejection.
  if (label === NONE_LABEL) return { ...base, accepted: false, reason: 'none-class' };
  if (probability < thresholds.minProbability) {
    return { ...base, accepted: false, reason: 'low-probability' };
  }
  if (margin < thresholds.minMargin) return { ...base, accepted: false, reason: 'low-margin' };
  // Skip entirely when logits were not supplied. Comparing a fabricated energy of 0
  // against a configured ceiling would reject every prediction on a caller that only
  // has probabilities available.
  if (logits !== null && thresholds.maxEnergy !== null && energy > thresholds.maxEnergy) {
    return { ...base, accepted: false, reason: 'high-energy' };
  }
  return { ...base, accepted: true, reason: 'accepted' };
}
