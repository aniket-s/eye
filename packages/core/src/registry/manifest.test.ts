import { describe, expect, it } from 'vitest';
import { NORMALISATION_SCHEME, packRef, parseManifest } from './manifest.js';

function validManifest(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: 'asl-fingerspell',
    version: '1.0.0',
    name: 'ASL Fingerspelling',
    language: 'ase',
    task: 'static-handshape',
    labels: ['A', 'B', 'none'],
    input: { featureLength: 42, hands: 1, normalisation: NORMALISATION_SCHEME },
    thresholds: { minProbability: 0.6, minMargin: 0.15, maxEnergy: null },
    metrics: { macroF1: 0.95, worstSliceF1: 0.91, testSigners: 8 },
    modelFile: 'model.onnx',
    sha256: 'abc123',
    licence: 'CC BY 4.0',
    ...overrides,
  };
}

describe('parseManifest', () => {
  it('accepts a well-formed manifest', () => {
    const manifest = parseManifest(validManifest());
    expect(manifest.id).toBe('asl-fingerspell');
    expect(manifest.labels).toEqual(['A', 'B', 'none']);
    expect(manifest.thresholds.minProbability).toBeCloseTo(0.6, 10);
  });

  it('builds a readable reference', () => {
    expect(packRef(parseManifest(validManifest()))).toBe('asl-fingerspell@1.0.0');
  });

  it('keeps optional fields when present and omits them when absent', () => {
    const withExtras = parseManifest(
      validManifest({
        attribution: 'FSboard (CC BY 4.0)',
        metrics: { macroF1: 0.9, worstSliceF1: 0.85, testSigners: 4, ece: 0.03 },
      }),
    );
    expect(withExtras.attribution).toBe('FSboard (CC BY 4.0)');
    expect(withExtras.metrics.ece).toBeCloseTo(0.03, 10);
    expect(parseManifest(validManifest()).attribution).toBeUndefined();
  });

  it('rejects a non-object', () => {
    expect(() => parseManifest(null)).toThrow(/must be an object/);
    expect(() => parseManifest('nope')).toThrow(/must be an object/);
  });

  it('rejects missing or empty required strings', () => {
    expect(() => parseManifest(validManifest({ id: '' }))).toThrow(/`id`/);
    expect(() => parseManifest(validManifest({ licence: undefined }))).toThrow(/`licence`/);
  });

  it('rejects an unknown task', () => {
    expect(() => parseManifest(validManifest({ task: 'magic' }))).toThrow(/unknown task "magic"/);
  });

  it('rejects empty or duplicated labels', () => {
    expect(() => parseManifest(validManifest({ labels: [] }))).toThrow(/non-empty array/);
    expect(() => parseManifest(validManifest({ labels: ['A', 'A'] }))).toThrow(/duplicates/);
  });

  /**
   * The guard that makes packs safe. A model trained under different feature
   * extraction than the runtime implements would not error — it would quietly
   * mispredict, which is the hardest class of ML bug to notice.
   */
  it('REFUSES a pack whose normalisation scheme does not match this build', () => {
    expect(() =>
      parseManifest(
        validManifest({ input: { featureLength: 42, hands: 1, normalisation: 'wrist-scaled-v1' } }),
      ),
    ).toThrow(/Refusing to load rather than mispredict silently/);
  });

  it('rejects a nonsensical feature length', () => {
    expect(() =>
      parseManifest(
        validManifest({
          input: { featureLength: 0, hands: 1, normalisation: NORMALISATION_SCHEME },
        }),
      ),
    ).toThrow(/positive integer/);
  });

  it('rejects a hand count other than 1 or 2', () => {
    expect(() =>
      parseManifest(
        validManifest({
          input: { featureLength: 42, hands: 3, normalisation: NORMALISATION_SCHEME },
        }),
      ),
    ).toThrow(/must be 1 or 2/);
  });

  it('rejects malformed thresholds', () => {
    expect(() =>
      parseManifest(validManifest({ thresholds: { minProbability: 'high', minMargin: 0.1 } })),
    ).toThrow(/must be numbers/);
  });

  /**
   * A model evaluated on its own training signers reports an inflated score. Refusing
   * the manifest makes that impossible to ship by accident (docs/AUDIT.md, A3).
   */
  it('REFUSES a model with no held-out signers', () => {
    expect(() =>
      parseManifest(
        validManifest({ metrics: { macroF1: 0.99, worstSliceF1: 0.99, testSigners: 0 } }),
      ),
    ).toThrow(/has not been evaluated/);
  });
});
