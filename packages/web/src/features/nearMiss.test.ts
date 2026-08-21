import { describe, expect, it } from 'vitest';
import { explain } from './nearMiss.js';

const rejected = (rawLabel: string | null, confidence: number | null, reason: string) => ({
  rawLabel,
  confidence,
  reason,
});

describe('explain', () => {
  it('names the letter the signer nearly made', () => {
    const near = explain(rejected('R', 0.48, 'low-probability'));
    expect(near.letter).toBe('R');
    expect(near.message).toContain('Nearly R');
  });

  it('tells the orientation letters what to turn', () => {
    // G, H, P and Q are one handshape at two angles. Telling a signer to fix their
    // *shape* when the shape is already right is worse than saying nothing.
    for (const letter of ['G', 'H', 'P', 'Q']) {
      const near = explain(rejected(letter, 0.55, 'none-class'));
      expect(near.message).toMatch(/side|down|across|floor/);
    }
  });

  it('tells everything else to hold steadier', () => {
    expect(explain(rejected('B', 0.55, 'low-margin')).message).toContain('steadier');
  });

  it('says nothing when there is no hand', () => {
    const near = explain(rejected(null, null, 'none-class'));
    expect(near.letter).toBeNull();
    expect(near.message).toBe('Show your hand');
  });

  it('says nothing when the model itself says no sign', () => {
    expect(explain(rejected('none', 0.9, 'none-class')).letter).toBeNull();
  });

  it('stays quiet on an idle hand rather than flickering a letter per frame', () => {
    // A hand resting between letters still produces a winner every frame. Announcing
    // each one would put a different letter on screen thirty times a second.
    expect(explain(rejected('X', 0.12, 'low-probability')).letter).toBeNull();
  });

  describe('detail for the debug panel', () => {
    it('translates the decoder’s reason into the signer’s terms', () => {
      expect(explain(rejected('K', 0.6, 'low-margin')).detail).toContain('torn between');
      expect(explain(rejected('K', 0.6, 'none-class')).detail).toContain('not reading as a letter');
    });

    it('reports what this pack manages on that letter, when the pack says', () => {
      // Answers "is it me or the model?" without needing a retrain to find out.
      const detail = explain(rejected('R', 0.45, 'low-probability'), { R: 0.61 }).detail;
      expect(detail).toContain('accepts R 61%');
    });

    it('omits the figure for a pack that carries no profile', () => {
      expect(explain(rejected('R', 0.45, 'low-probability')).detail).not.toContain('accepts');
      expect(explain(rejected('R', 0.45, 'low-probability'), {}).detail).not.toContain('accepts');
    });

    it('survives a reason it has never seen', () => {
      expect(explain(rejected('R', 0.45, 'something-new')).detail).toContain('not accepted');
    });
  });
});
