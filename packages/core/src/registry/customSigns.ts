/**
 * User-defined signs, recognised without training.
 *
 * This is the payoff for training the classifier with an **ArcFace** head rather than
 * plain cross-entropy (see `training/mudra_train/models/classifier.py`). ArcFace puts
 * every embedding on the unit hypersphere and pulls each class into a tight cluster, so
 * a class is well described by the *average* of its examples — its centroid.
 *
 * Which means a user can add a sign the model has never seen:
 *
 * 1. Record five or so examples.
 * 2. Average their embeddings into a centroid.
 * 3. Classify anything by cosine similarity to that centroid.
 *
 * No gradient descent, no server, no cost. It runs in the browser in microseconds, and
 * it is the only practical route to regional variants, name signs, and personal
 * shortcuts — none of which will ever appear in a public dataset.
 *
 * The base model still runs; custom signs are consulted alongside it, and only win when
 * they match more confidently than the built-in vocabulary.
 */

/** A user-defined sign. */
export interface CustomSign {
  readonly id: string;
  /** What to display when it matches. */
  readonly label: string;
  /** Unit-norm centroid: the mean of the recorded embeddings, renormalised. */
  readonly centroid: readonly number[];
  /** How many examples went into it. */
  readonly sampleCount: number;
  /** Spread of the samples around the centroid, 0–1. Lower is tighter. */
  readonly cohesion: number;
  readonly createdAt: number;
}

/**
 * Minimum cosine similarity for a custom sign to match.
 *
 * 0.7 on a unit hypersphere is a meaningful angular distance — roughly 45°. Lower and
 * unrelated handshapes start matching; higher and the user has to reproduce their
 * recording almost exactly.
 */
export const DEFAULT_MATCH_THRESHOLD = 0.7;

/** Fewest examples that give a usable centroid. */
export const MIN_SAMPLES = 3;

/** Cosine similarity of two unit vectors. Equivalently, their dot product. */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) {
    throw new RangeError(`Cannot compare embeddings of length ${a.length} and ${b.length}`);
  }
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += (a[i] as number) * (b[i] as number);
  return dot;
}

/** Scale a vector to unit length. Returns zeros unchanged. */
export function normaliseVector(values: readonly number[]): number[] {
  let sumSquares = 0;
  for (const value of values) sumSquares += value * value;
  const magnitude = Math.sqrt(sumSquares);
  if (magnitude < 1e-9) return [...values];
  return values.map((value) => value / magnitude);
}

export interface CustomSignMatch {
  readonly sign: CustomSign;
  readonly similarity: number;
}

/**
 * Build a custom sign from recorded embeddings.
 *
 * @throws RangeError if there are too few samples, or their lengths disagree.
 */
export function buildCustomSign(
  id: string,
  label: string,
  embeddings: readonly (readonly number[])[],
  createdAt: number,
): CustomSign {
  if (embeddings.length < MIN_SAMPLES) {
    throw new RangeError(
      `A custom sign needs at least ${MIN_SAMPLES} examples, received ${embeddings.length}`,
    );
  }
  const width = embeddings[0]!.length;
  if (embeddings.some((embedding) => embedding.length !== width)) {
    throw new RangeError('All embeddings must have the same length');
  }

  const sum = new Array<number>(width).fill(0);
  for (const embedding of embeddings) {
    for (let i = 0; i < width; i++) sum[i] = (sum[i] as number) + (embedding[i] as number);
  }
  const centroid = normaliseVector(sum.map((value) => value / embeddings.length));

  // Mean similarity of the samples to their own centroid. A low value means the user
  // was inconsistent — worth telling them, because the sign will match poorly.
  let total = 0;
  for (const embedding of embeddings) total += cosineSimilarity(embedding, centroid);
  const cohesion = total / embeddings.length;

  return {
    id,
    label,
    centroid,
    sampleCount: embeddings.length,
    cohesion,
    createdAt,
  };
}

/**
 * The user's collection of custom signs.
 *
 * Deliberately storage-agnostic: persistence lives in `packages/web`, so this stays
 * DOM-free and testable in Node like everything else in `core`.
 */
export class CustomSignBook {
  readonly #signs = new Map<string, CustomSign>();
  #threshold: number;

  constructor(threshold: number = DEFAULT_MATCH_THRESHOLD) {
    this.#threshold = threshold;
  }

  get size(): number {
    return this.#signs.size;
  }

  /** Every sign, newest first. */
  list(): CustomSign[] {
    return [...this.#signs.values()].sort((a, b) => b.createdAt - a.createdAt);
  }

  get threshold(): number {
    return this.#threshold;
  }

  set threshold(value: number) {
    if (value <= 0 || value > 1) {
      throw new RangeError('threshold must be within (0, 1]');
    }
    this.#threshold = value;
  }

  add(sign: CustomSign): void {
    this.#signs.set(sign.id, sign);
  }

  remove(id: string): boolean {
    return this.#signs.delete(id);
  }

  clear(): void {
    this.#signs.clear();
  }

  /** Replace the whole collection, e.g. after loading from storage. */
  load(signs: readonly CustomSign[]): void {
    this.#signs.clear();
    for (const sign of signs) this.#signs.set(sign.id, sign);
  }

  /**
   * Find the best matching custom sign.
   *
   * @returns The match, or `null` if nothing cleared the threshold.
   */
  match(embedding: readonly number[]): CustomSignMatch | null {
    let best: CustomSignMatch | null = null;

    for (const sign of this.#signs.values()) {
      if (sign.centroid.length !== embedding.length) continue; // from another model
      const similarity = cosineSimilarity(embedding, sign.centroid);
      if (similarity >= this.#threshold && (best === null || similarity > best.similarity)) {
        best = { sign, similarity };
      }
    }

    return best;
  }

  /**
   * Match, but only when the built-in vocabulary declined the frame.
   *
   * This is the arbitration rule between the two vocabularies, and it is deliberately
   * one-sided. A trained class is backed by thousands of examples from many signers; a
   * custom sign by five from one. Letting the custom sign override an accepted letter
   * would let a hastily recorded gesture shadow a trained one — and since a user records
   * custom signs precisely *because* they are not in the vocabulary, the case where a
   * custom sign should beat a confident letter does not really arise.
   *
   * @param baseAccepted Whether the classifier accepted its own prediction.
   * @returns The match, or `null` when the base model already decided or nothing cleared
   *   the threshold.
   */
  matchIfUnrecognised(
    embedding: readonly number[] | null,
    baseAccepted: boolean,
  ): CustomSignMatch | null {
    if (baseAccepted || embedding === null) return null;
    return this.match(embedding);
  }

  /**
   * Warn about signs too close together to tell apart.
   *
   * Users naturally record variants of the same handshape and are then puzzled when
   * recognition flips between them. Surfacing the collision at creation time is far
   * kinder than letting them discover it in use.
   *
   * @returns Pairs whose similarity exceeds `threshold`, most similar first.
   */
  findCollisions(threshold = 0.9): { a: CustomSign; b: CustomSign; similarity: number }[] {
    const signs = [...this.#signs.values()];
    const collisions: { a: CustomSign; b: CustomSign; similarity: number }[] = [];

    for (let i = 0; i < signs.length; i++) {
      for (let j = i + 1; j < signs.length; j++) {
        const a = signs[i]!;
        const b = signs[j]!;
        if (a.centroid.length !== b.centroid.length) continue;
        const similarity = cosineSimilarity(a.centroid, b.centroid);
        if (similarity >= threshold) collisions.push({ a, b, similarity });
      }
    }

    return collisions.sort((x, y) => y.similarity - x.similarity);
  }
}
