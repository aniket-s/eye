import { describe, expect, it } from 'vitest';
import {
  buildCustomSign,
  cosineSimilarity,
  CustomSignBook,
  DEFAULT_MATCH_THRESHOLD,
  MIN_SAMPLES,
  normaliseVector,
} from './customSigns.js';

/** A unit vector pointing mostly along `axis`, with a little noise. */
function embedding(axis: number, noise = 0, width = 8): number[] {
  const values = new Array<number>(width).fill(0);
  values[axis] = 1;
  for (let i = 0; i < width; i++) {
    values[i] = (values[i] as number) + ((i * 37) % 11) * noise * 0.01;
  }
  return normaliseVector(values);
}

function sign(id: string, axis: number, noise = 0.2): ReturnType<typeof buildCustomSign> {
  return buildCustomSign(
    id,
    id,
    [embedding(axis, noise * 0.5), embedding(axis, noise), embedding(axis, noise * 1.5)],
    Date.now(),
  );
}

describe('cosineSimilarity and normaliseVector', () => {
  it('gives 1 for identical unit vectors', () => {
    expect(cosineSimilarity(embedding(0), embedding(0))).toBeCloseTo(1, 10);
  });

  it('gives 0 for orthogonal vectors', () => {
    expect(cosineSimilarity(embedding(0), embedding(1))).toBeCloseTo(0, 10);
  });

  it('rejects mismatched lengths', () => {
    expect(() => cosineSimilarity([1, 0], [1, 0, 0])).toThrow(RangeError);
  });

  it('scales to unit length', () => {
    const unit = normaliseVector([3, 4]);
    expect(Math.hypot(unit[0] as number, unit[1] as number)).toBeCloseTo(1, 10);
  });

  it('leaves a zero vector alone rather than dividing by zero', () => {
    expect(normaliseVector([0, 0, 0]).every(Number.isFinite)).toBe(true);
  });
});

describe('buildCustomSign', () => {
  it('produces a unit-norm centroid', () => {
    const built = sign('wave', 2);
    const magnitude = Math.sqrt(built.centroid.reduce((sum, v) => sum + v * v, 0));
    expect(magnitude).toBeCloseTo(1, 6);
  });

  it('records how many samples went in', () => {
    expect(sign('wave', 2).sampleCount).toBe(3);
  });

  /**
   * Cohesion tells the user whether they signed consistently. A low value means the
   * sign will match poorly, and saying so at creation time is far kinder than letting
   * them discover it in use.
   */
  it('reports high cohesion for consistent samples and low for scattered ones', () => {
    const consistent = buildCustomSign('a', 'a', [embedding(0), embedding(0), embedding(0)], 0);
    const scattered = buildCustomSign('b', 'b', [embedding(0), embedding(1), embedding(2)], 0);
    expect(consistent.cohesion).toBeCloseTo(1, 4);
    expect(scattered.cohesion).toBeLessThan(0.7);
  });

  it('refuses too few examples', () => {
    expect(() => buildCustomSign('x', 'x', [embedding(0)], 0)).toThrow(
      new RegExp(`at least ${MIN_SAMPLES}`),
    );
  });

  it('refuses embeddings of differing lengths', () => {
    expect(() =>
      buildCustomSign('x', 'x', [embedding(0, 0, 8), embedding(0, 0, 4), embedding(0, 0, 8)], 0),
    ).toThrow(/same length/);
  });
});

describe('CustomSignBook', () => {
  it('starts empty', () => {
    expect(new CustomSignBook().size).toBe(0);
  });

  it('adds, lists and removes signs', () => {
    const book = new CustomSignBook();
    book.add(sign('wave', 0));
    book.add(sign('point', 1));

    expect(book.size).toBe(2);
    expect(
      book
        .list()
        .map((s) => s.id)
        .sort(),
    ).toEqual(['point', 'wave']);
    expect(book.remove('wave')).toBe(true);
    expect(book.remove('missing')).toBe(false);
    expect(book.size).toBe(1);
  });

  it('lists newest first', () => {
    const book = new CustomSignBook();
    book.add(buildCustomSign('old', 'old', [embedding(0), embedding(0), embedding(0)], 100));
    book.add(buildCustomSign('new', 'new', [embedding(1), embedding(1), embedding(1)], 200));
    expect(book.list()[0]!.id).toBe('new');
  });

  it('replaces the collection on load', () => {
    const book = new CustomSignBook();
    book.add(sign('old', 0));
    book.load([sign('fresh', 1)]);
    expect(book.list().map((s) => s.id)).toEqual(['fresh']);
  });

  /** The headline behaviour: recognising a sign the model was never trained on. */
  it('matches a recorded sign', () => {
    const book = new CustomSignBook();
    book.add(sign('wave', 3, 0.2));

    const match = book.match(embedding(3, 0.25));
    expect(match?.sign.id).toBe('wave');
    expect(match?.similarity).toBeGreaterThan(DEFAULT_MATCH_THRESHOLD);
  });

  it('returns null for an unrelated handshape', () => {
    const book = new CustomSignBook();
    book.add(sign('wave', 3));
    expect(book.match(embedding(6))).toBeNull();
  });

  it('picks the closest when several could match', () => {
    const book = new CustomSignBook();
    book.add(sign('wave', 0, 0.1));
    book.add(sign('point', 4, 0.1));
    expect(book.match(embedding(4, 0.12))?.sign.id).toBe('point');
  });

  /**
   * Custom signs recorded against one model are meaningless to another — the embedding
   * spaces are unrelated. Skipping on width mismatch avoids comparing nonsense.
   */
  it('ignores signs recorded against a different model', () => {
    const book = new CustomSignBook();
    book.add(
      buildCustomSign(
        'other',
        'other',
        [embedding(0, 0, 4), embedding(0, 0, 4), embedding(0, 0, 4)],
        0,
      ),
    );
    expect(book.match(embedding(0, 0, 8))).toBeNull();
  });

  it('honours a configurable threshold', () => {
    const book = new CustomSignBook(0.999);
    book.add(sign('wave', 3, 0.2));
    expect(book.match(embedding(3, 0.6))).toBeNull();

    book.threshold = 0.5;
    expect(book.match(embedding(3, 0.6))).not.toBeNull();
  });

  it('rejects a nonsensical threshold', () => {
    const book = new CustomSignBook();
    expect(() => {
      book.threshold = 0;
    }).toThrow(RangeError);
    expect(() => {
      book.threshold = 1.5;
    }).toThrow(RangeError);
  });

  describe('matchIfUnrecognised', () => {
    /**
     * The arbitration rule. A trained class rests on thousands of examples from many
     * signers; a custom sign on five from one. The trained one wins by default.
     */
    it('stays silent when the base model accepted the frame', () => {
      const book = new CustomSignBook();
      book.add(sign('wave', 3, 0.2));
      expect(book.matchIfUnrecognised(embedding(3, 0.25), true)).toBeNull();
    });

    it('matches when the base model declined', () => {
      const book = new CustomSignBook();
      book.add(sign('wave', 3, 0.2));
      expect(book.matchIfUnrecognised(embedding(3, 0.25), false)?.sign.id).toBe('wave');
    });

    /** Packs without an embedding head produce no vector; the feature simply does not run. */
    it('tolerates a pack that exposes no embedding', () => {
      const book = new CustomSignBook();
      book.add(sign('wave', 3));
      expect(book.matchIfUnrecognised(null, false)).toBeNull();
    });

    it('still returns nothing when no custom sign is close enough', () => {
      const book = new CustomSignBook();
      book.add(sign('wave', 3));
      expect(book.matchIfUnrecognised(embedding(6), false)).toBeNull();
    });
  });

  /**
   * Users naturally record variants of the same handshape and are then puzzled when
   * recognition flips between them. Detecting the collision lets the UI say so.
   */
  it('detects signs too similar to tell apart', () => {
    const book = new CustomSignBook();
    book.add(sign('wave-a', 5, 0.05));
    book.add(sign('wave-b', 5, 0.06));
    book.add(sign('distinct', 1, 0.05));

    const collisions = book.findCollisions(0.9);
    expect(collisions).toHaveLength(1);
    expect([collisions[0]!.a.id, collisions[0]!.b.id].sort()).toEqual(['wave-a', 'wave-b']);
  });

  it('reports no collisions between distinct signs', () => {
    const book = new CustomSignBook();
    book.add(sign('one', 0));
    book.add(sign('two', 4));
    expect(book.findCollisions(0.9)).toEqual([]);
  });
});
