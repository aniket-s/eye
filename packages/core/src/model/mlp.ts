import { argmax, relu, softmax } from '../math/activations.js';
import type { MlpWeights, Prediction } from '../types.js';

/**
 * Feed-forward classifier matching a scikit-learn `MLPClassifier` export.
 *
 * ReLU on every hidden layer, identity on the output layer, then softmax.
 * Inputs are standardised with the exported scaler statistics.
 *
 * This is the **v1** model. It is retained so Phase 0 can refactor without changing
 * behaviour; Phase 2 replaces it with an int8 ONNX model pack loaded through
 * ONNX Runtime Web (docs/adr/0002-onnx-runtime-web-over-tfjs.md).
 */
export class MlpClassifier {
  readonly labels: readonly string[];

  readonly #scalerMean: readonly number[];
  readonly #scalerStd: readonly number[];
  readonly #coefs: readonly (readonly (readonly number[])[])[];
  readonly #intercepts: readonly (readonly number[])[];

  private constructor(weights: MlpWeights) {
    this.labels = weights.labels;
    this.#scalerMean = weights.scaler_mean;
    this.#scalerStd = weights.scaler_std;
    this.#coefs = weights.coefs;
    this.#intercepts = weights.intercepts;
  }

  /**
   * Validate an untrusted weights blob and construct a classifier.
   *
   * The input arrives from `response.json()`, so nothing about its shape is
   * guaranteed. Every field is checked here, once, which keeps `predict` free of
   * per-frame defensive checks and turns a malformed model into a clear message
   * instead of `NaN` predictions.
   *
   * @throws TypeError naming the specific field that is wrong.
   */
  static fromWeights(weights: MlpWeights): MlpClassifier {
    const { labels, scaler_mean, scaler_std, coefs, intercepts, activation } = weights;

    if (!isArrayOf(labels, isString) || labels.length === 0) {
      throw new TypeError('Model weights: `labels` must be a non-empty array of strings');
    }
    if (activation !== undefined && activation !== 'relu') {
      throw new TypeError(`Model weights: unsupported activation "${activation}"`);
    }
    if (!isArrayOf(scaler_mean, isFiniteNumber) || !isArrayOf(scaler_std, isFiniteNumber)) {
      throw new TypeError('Model weights: `scaler_mean` and `scaler_std` must be numeric arrays');
    }
    if (scaler_mean.length !== scaler_std.length) {
      throw new TypeError(
        `Model weights: scaler length mismatch (mean ${scaler_mean.length}, std ${scaler_std.length})`,
      );
    }
    if (scaler_std.some((s) => s === 0)) {
      throw new TypeError(
        'Model weights: `scaler_std` contains a zero, which would divide by zero',
      );
    }
    if (!isArrayOf(coefs, isMatrix) || coefs.length === 0) {
      throw new TypeError('Model weights: `coefs` must be a non-empty array of matrices');
    }
    if (!isArrayOf(intercepts, isNumberArray) || intercepts.length !== coefs.length) {
      throw new TypeError(
        `Model weights: expected ${coefs.length} intercept vectors, received ${
          Array.isArray(intercepts) ? intercepts.length : 'none'
        }`,
      );
    }

    // Each layer's weight matrix must be [fanIn][fanOut], and fanOut must match its bias.
    let fanIn = scaler_mean.length;
    for (let layer = 0; layer < coefs.length; layer++) {
      const matrix = coefs[layer]!;
      const bias = intercepts[layer]!;
      if (matrix.length !== fanIn) {
        throw new TypeError(
          `Model weights: layer ${layer} expects ${fanIn} inputs, matrix has ${matrix.length} rows`,
        );
      }
      const fanOut = bias.length;
      const firstRowWidth = matrix[0]?.length;
      if (firstRowWidth !== fanOut) {
        throw new TypeError(
          `Model weights: layer ${layer} matrix width ${firstRowWidth ?? 'unknown'} does not match bias length ${fanOut}`,
        );
      }
      fanIn = fanOut;
    }
    if (fanIn !== labels.length) {
      throw new TypeError(
        `Model weights: final layer emits ${fanIn} units but there are ${labels.length} labels`,
      );
    }

    return new MlpClassifier(weights);
  }

  /** Number of input features the model expects. */
  get inputSize(): number {
    return this.#scalerMean.length;
  }

  /**
   * Classify one feature vector.
   *
   * @param features Raw (unstandardised) input, of length {@link inputSize}.
   * @returns The winning label and its softmax probability.
   * @throws RangeError if `features` is the wrong length — a silent length mismatch
   *   would otherwise produce plausible-looking but meaningless predictions.
   */
  predict(features: readonly number[]): Prediction {
    const probabilities = this.predictProbabilities(features);
    const best = argmax(probabilities);
    return {
      label: this.labels[best] as string,
      confidence: probabilities[best] as number,
    };
  }

  /** Full probability distribution over {@link labels}, in label order. */
  predictProbabilities(features: readonly number[]): number[] {
    if (features.length !== this.#scalerMean.length) {
      throw new RangeError(
        `Expected ${this.#scalerMean.length} features, received ${features.length}`,
      );
    }

    // Standardise.
    let activations = new Array<number>(features.length);
    for (let i = 0; i < features.length; i++) {
      activations[i] =
        ((features[i] as number) - (this.#scalerMean[i] as number)) /
        (this.#scalerStd[i] as number);
    }

    // Dense layers. ReLU on hidden layers, identity on the output layer.
    const lastLayer = this.#coefs.length - 1;
    for (let layer = 0; layer <= lastLayer; layer++) {
      const matrix = this.#coefs[layer]!;
      const bias = this.#intercepts[layer]!;
      const output = new Array<number>(bias.length);

      for (let unit = 0; unit < bias.length; unit++) {
        let sum = bias[unit] as number;
        for (let i = 0; i < activations.length; i++) {
          sum += (activations[i] as number) * (matrix[i]![unit] as number);
        }
        output[unit] = layer < lastLayer ? relu(sum) : sum;
      }
      activations = output;
    }

    return softmax(activations);
  }
}

/** Narrow an unknown value to a finite number, rejecting NaN and Infinity. */
function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/** Narrow an unknown value to a string. */
function isString(value: unknown): value is string {
  return typeof value === 'string';
}

/**
 * Narrow an unknown value to an array whose elements all satisfy `predicate`.
 *
 * Only the first element is probed. A full scan of the 2.5 MB weights blob would
 * cost more than it protects against, and a layer-shape mismatch is caught anyway
 * by the dimension checks above.
 */
function isArrayOf<T>(value: unknown, predicate: (item: unknown) => item is T): value is T[] {
  return Array.isArray(value) && (value.length === 0 || predicate(value[0]));
}

/** Narrow to an array of finite numbers. */
function isNumberArray(value: unknown): value is readonly number[] {
  return isArrayOf(value, isFiniteNumber);
}

/** Narrow to a 2-D array of finite numbers. */
function isMatrix(value: unknown): value is readonly (readonly number[])[] {
  return isArrayOf(value, isNumberArray);
}
