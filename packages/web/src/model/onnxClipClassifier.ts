import * as ort from 'onnxruntime-web/wasm';
import { softmax, type IsolatedClassifier, type PackManifest } from '@mudrapragyan/core';
import { configureOrtRuntime } from './onnxClassifier.js';

/**
 * Clip classifier for `temporal-isolated` packs.
 *
 * Runs once per completed sign rather than per frame, so the latency budget is far
 * looser than the other two classifiers — a word model can afford to be several times
 * larger because it fires perhaps once a second instead of six times.
 */
export class OnnxClipClassifier implements IsolatedClassifier {
  readonly #session: ort.InferenceSession;
  readonly #manifest: PackManifest;
  readonly #inputName: string;
  readonly #outputName: string;

  private constructor(session: ort.InferenceSession, manifest: PackManifest) {
    this.#session = session;
    this.#manifest = manifest;
    this.#inputName = session.inputNames[0] ?? 'clip';
    this.#outputName = session.outputNames[0] ?? 'logits';
  }

  static async create(
    modelBytes: ArrayBuffer,
    manifest: PackManifest,
  ): Promise<OnnxClipClassifier> {
    configureOrtRuntime();
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    const classifier = new OnnxClipClassifier(session, manifest);
    const frames = manifest.input.windowFrames ?? 96;
    await classifier.classifyClip(new Float32Array(frames * manifest.input.featureLength), frames);
    return classifier;
  }

  get labels(): readonly string[] {
    return this.#manifest.labels;
  }

  get manifest(): PackManifest {
    return this.#manifest;
  }

  /**
   * Classify one completed sign.
   *
   * @param clip Flattened features, `frames × featureLength`.
   */
  async classifyClip(clip: Float32Array, frames: number): Promise<number[]> {
    const featureLength = this.#manifest.input.featureLength;
    if (clip.length !== frames * featureLength) {
      throw new RangeError(
        `Expected ${frames * featureLength} values for ${frames} frames, received ${clip.length}`,
      );
    }

    const input = new ort.Tensor('float32', clip, [1, frames, featureLength]);
    const output = await this.#session.run({ [this.#inputName]: input });
    const raw = output[this.#outputName]?.data;
    if (!(raw instanceof Float32Array)) {
      throw new Error('Clip model returned an unexpected output type');
    }

    return softmax(Array.from(raw));
  }

  async close(): Promise<void> {
    await this.#session.release();
  }
}
