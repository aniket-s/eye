import * as ort from 'onnxruntime-web/wasm';
import type { PackManifest, SequenceClassifier } from '@mudrapragyan/core';
import { BLANK_INDEX } from '@mudrapragyan/core';
import { configureOrtRuntime } from './onnxClassifier.js';

/**
 * Sequence classifier for `temporal-ctc` packs, backed by ONNX Runtime Web.
 *
 * Differs from the static classifier in two ways that matter:
 *
 * - **The frame axis is dynamic.** The browser's window is not full for the first
 *   couple of seconds, so the session is fed whatever it has. A graph with a baked-in
 *   window length would work once the buffer filled and fail before that.
 * - **The output is log probabilities per frame**, not one distribution. The model is
 *   trained with a log-softmax head for CTC, so scores are exponentiated here to give
 *   the decoder ordinary probabilities.
 */
export class OnnxSequenceClassifier implements SequenceClassifier {
  readonly #session: ort.InferenceSession;
  readonly #manifest: PackManifest;
  readonly #inputName: string;
  readonly #outputName: string;

  private constructor(session: ort.InferenceSession, manifest: PackManifest) {
    this.#session = session;
    this.#manifest = manifest;
    this.#inputName = session.inputNames[0] ?? 'features';
    this.#outputName = session.outputNames[0] ?? 'log_probabilities';
  }

  static async create(
    modelBytes: ArrayBuffer,
    manifest: PackManifest,
  ): Promise<OnnxSequenceClassifier> {
    configureOrtRuntime();
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    const classifier = new OnnxSequenceClassifier(session, manifest);
    // Warm up at a partial window, which is what the first real calls look like.
    await classifier.classifySequence(new Float32Array(8 * manifest.input.featureLength), 8);
    return classifier;
  }

  get labels(): readonly string[] {
    return this.#manifest.labels;
  }

  get blankIndex(): number {
    return BLANK_INDEX;
  }

  get manifest(): PackManifest {
    return this.#manifest;
  }

  /**
   * Classify a window of frames.
   *
   * @param window Flattened features, `frames × featureLength`.
   * @param frames Number of frames the window holds.
   * @returns One probability vector per frame.
   */
  async classifySequence(window: Float32Array, frames: number): Promise<number[][]> {
    const featureLength = this.#manifest.input.featureLength;
    if (window.length !== frames * featureLength) {
      throw new RangeError(
        `Expected ${frames * featureLength} values for ${frames} frames, received ${window.length}`,
      );
    }

    const input = new ort.Tensor('float32', window, [1, frames, featureLength]);
    const output = await this.#session.run({ [this.#inputName]: input });
    const raw = output[this.#outputName]?.data;
    if (!(raw instanceof Float32Array)) {
      throw new Error('Sequence model returned an unexpected output type');
    }

    const classes = this.#manifest.labels.length;
    const result: number[][] = new Array<number[]>(frames);
    for (let frame = 0; frame < frames; frame++) {
      const probabilities = new Array<number>(classes);
      const offset = frame * classes;
      // The head emits log-softmax for CTC training; exponentiate for the decoder.
      for (let c = 0; c < classes; c++) probabilities[c] = Math.exp(raw[offset + c] as number);
      result[frame] = probabilities;
    }
    return result;
  }

  async close(): Promise<void> {
    await this.#session.release();
  }
}
