/** Rectified linear unit. */
export function relu(x: number): number {
  return x > 0 ? x : 0;
}

/**
 * Numerically stable softmax.
 *
 * Subtracts the maximum logit before exponentiating so that large logits cannot
 * overflow to `Infinity` (which would yield `NaN` after the division).
 *
 * @param logits Unnormalised scores. Must be non-empty.
 * @returns Probabilities summing to 1, same length as `logits`.
 */
export function softmax(logits: readonly number[]): number[] {
  if (logits.length === 0) {
    throw new RangeError('softmax requires at least one logit');
  }
  let max = -Infinity;
  for (const v of logits) {
    if (v > max) max = v;
  }
  const exponentials = new Array<number>(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp((logits[i] as number) - max);
    exponentials[i] = e;
    total += e;
  }
  for (let i = 0; i < exponentials.length; i++) {
    exponentials[i] = (exponentials[i] as number) / total;
  }
  return exponentials;
}

/**
 * Index of the largest value.
 *
 * Ties resolve to the lowest index, matching `Array.prototype.indexOf(Math.max(...))`.
 *
 * @returns The winning index, or -1 for an empty input.
 */
export function argmax(values: readonly number[]): number {
  let best = -1;
  let bestValue = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (v > bestValue) {
      bestValue = v;
      best = i;
    }
  }
  return best;
}
