import { describe, expect, it } from 'vitest';
import { MlpClassifier } from './mlp.js';
import type { MlpWeights } from '../types.js';

/**
 * A hand-checkable two-layer network: 2 inputs → 2 hidden (ReLU) → 2 outputs.
 * The identity scaler keeps the arithmetic verifiable by hand.
 */
function tinyWeights(overrides: Partial<MlpWeights> = {}): MlpWeights {
  return {
    labels: ['A', 'B'],
    scaler_mean: [0, 0],
    scaler_std: [1, 1],
    coefs: [
      [
        [1, 0],
        [0, 1],
      ],
      [
        [1, 0],
        [0, 1],
      ],
    ],
    intercepts: [
      [0, 0],
      [0, 0],
    ],
    activation: 'relu',
    ...overrides,
  };
}

describe('MlpClassifier.fromWeights', () => {
  it('accepts a well-formed blob', () => {
    expect(() => MlpClassifier.fromWeights(tinyWeights())).not.toThrow();
  });

  it('rejects an unsupported activation', () => {
    expect(() => MlpClassifier.fromWeights(tinyWeights({ activation: 'tanh' }))).toThrow(
      /unsupported activation "tanh"/,
    );
  });

  it('rejects empty labels', () => {
    expect(() => MlpClassifier.fromWeights(tinyWeights({ labels: [] }))).toThrow(/non-empty/);
  });

  it('rejects a zero in scaler_std, which would divide by zero', () => {
    expect(() => MlpClassifier.fromWeights(tinyWeights({ scaler_std: [1, 0] }))).toThrow(
      /divide by zero/,
    );
  });

  it('rejects a layer whose matrix does not match the incoming width', () => {
    expect(() =>
      MlpClassifier.fromWeights(
        tinyWeights({
          coefs: [
            [[1, 0]],
            [
              [1, 0],
              [0, 1],
            ],
          ],
        }),
      ),
    ).toThrow(/expects 2 inputs, matrix has 1 rows/);
  });

  it('rejects a final layer whose width does not match the label count', () => {
    expect(() => MlpClassifier.fromWeights(tinyWeights({ labels: ['A', 'B', 'C'] }))).toThrow(
      /emits 2 units but there are 3 labels/,
    );
  });

  it('rejects a mismatch between coefs and intercepts', () => {
    expect(() => MlpClassifier.fromWeights(tinyWeights({ intercepts: [[0, 0]] }))).toThrow(
      /expected 2 intercept vectors/,
    );
  });
});

describe('MlpClassifier.predict', () => {
  it('reports the input size', () => {
    expect(MlpClassifier.fromWeights(tinyWeights()).inputSize).toBe(2);
  });

  it('selects the class with the larger logit', () => {
    const model = MlpClassifier.fromWeights(tinyWeights());
    expect(model.predict([3, 1]).label).toBe('A');
    expect(model.predict([1, 3]).label).toBe('B');
  });

  it('returns a calibrated-looking probability that matches softmax by hand', () => {
    // Identity weights: logits are [3, 1]. softmax([3,1])[0] = 1/(1+e^-2) ≈ 0.8808
    const model = MlpClassifier.fromWeights(tinyWeights());
    expect(model.predict([3, 1]).confidence).toBeCloseTo(1 / (1 + Math.exp(-2)), 10);
  });

  it('applies ReLU on the hidden layer, clamping negative paths', () => {
    // Input [-5, 1]: hidden = relu([-5, 1]) = [0, 1]; logits = [0, 1].
    const model = MlpClassifier.fromWeights(tinyWeights());
    const probabilities = model.predictProbabilities([-5, 1]);
    expect(probabilities[0]).toBeCloseTo(Math.exp(-1) / (Math.exp(-1) + 1), 10);
  });

  it('applies the scaler before the first layer', () => {
    const model = MlpClassifier.fromWeights(
      tinyWeights({ scaler_mean: [10, 0], scaler_std: [2, 1] }),
    );
    // x0 = (14 - 10) / 2 = 2, x1 = 1 → logits [2, 1] → class A.
    expect(model.predict([14, 1]).label).toBe('A');
  });

  it('emits a full distribution summing to one', () => {
    const probabilities = MlpClassifier.fromWeights(tinyWeights()).predictProbabilities([2, 5]);
    expect(probabilities).toHaveLength(2);
    expect(probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it('rejects a feature vector of the wrong length rather than silently mispredicting', () => {
    const model = MlpClassifier.fromWeights(tinyWeights());
    expect(() => model.predict([1, 2, 3])).toThrow(/Expected 2 features, received 3/);
  });
});
