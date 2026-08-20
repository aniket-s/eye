import * as ort from 'onnxruntime-web/wasm';
import { softmax, type PackManifest } from '@mudrapragyan/core';

/**
 * Classifier backed by ONNX Runtime Web.
 *
 * Runs inside the recognition worker. Configuration notes, each deliberate:
 *
 * - **`onnxruntime-web/wasm` entry point.** 52 KB of JS and one 13 MB WASM binary
 *   (~3.3 MB gzipped). The WebGPU build nearly doubles that for no benefit at this
 *   model size — dispatch overhead alone would exceed the whole compute budget.
 * - **`numThreads = 1`.** Multi-threaded WASM needs `SharedArrayBuffer`, which needs
 *   cross-origin isolation, which GitHub Pages cannot provide. At ~200k parameters a
 *   thread pool costs more to spin up than the inference it would parallelise, so
 *   this is free — and it keeps the COOP/COEP question closed. See ADR 0002.
 * - **WASM resolved by the bundler.** The runtime references its binary with
 *   `new URL(..., import.meta.url)`, which Vite rewrites to a hashed asset under the
 *   configured base path. Setting `wasmPaths` to a hand-managed copy instead would
 *   ship the binary twice — ~13 MB of dead weight — because the bundled one is emitted
 *   either way. One copy, correct under any base path, cached immutably by hash.
 *
 * The JS and WASM come from the same install by construction here, which matters: a
 * version mismatch fails at session creation with an opaque missing-symbol error.
 */

let configured = false;

function configureRuntime(): void {
  if (configured) return;
  ort.env.wasm.numThreads = 1;
  // The runtime logs a warning per session about threading; we have chosen 1 thread
  // deliberately, so this is noise.
  ort.env.logLevel = 'error';
  configured = true;
}

export interface ClassificationResult {
  readonly label: string;
  readonly probabilities: number[];
  readonly logits: number[];
}

export class OnnxClassifier {
  readonly #session: ort.InferenceSession;
  readonly #manifest: PackManifest;
  readonly #inputName: string;
  readonly #outputName: string;

  private constructor(session: ort.InferenceSession, manifest: PackManifest) {
    this.#session = session;
    this.#manifest = manifest;
    this.#inputName = session.inputNames[0] ?? 'features';
    this.#outputName = session.outputNames[0] ?? 'logits';
  }

  /**
   * Create a session from a pack's weights.
   *
   * @throws Error if the graph's input width disagrees with the manifest. That
   *   mismatch would otherwise produce confident nonsense.
   */
  static async create(modelBytes: ArrayBuffer, manifest: PackManifest): Promise<OnnxClassifier> {
    configureRuntime();

    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    const classifier = new OnnxClassifier(session, manifest);
    await classifier.#warmUp();
    return classifier;
  }

  get labels(): readonly string[] {
    return this.#manifest.labels;
  }

  get manifest(): PackManifest {
    return this.#manifest;
  }

  /**
   * Classify one feature vector.
   *
   * @param features Normalised features, of the manifest's declared length.
   */
  async classify(features: Float32Array): Promise<ClassificationResult> {
    const expected = this.#manifest.input.featureLength;
    if (features.length !== expected) {
      throw new RangeError(`Expected ${expected} features, received ${features.length}`);
    }

    const input = new ort.Tensor('float32', features, [1, expected]);
    const output = await this.#session.run({ [this.#inputName]: input });
    const raw = output[this.#outputName]?.data;
    if (!(raw instanceof Float32Array)) {
      throw new Error('Model returned an unexpected output type');
    }

    const logits = Array.from(raw);
    const probabilities = softmax(logits);
    let best = 0;
    for (let i = 1; i < probabilities.length; i++) {
      if ((probabilities[i] as number) > (probabilities[best] as number)) best = i;
    }

    return { label: this.#manifest.labels[best] ?? 'none', probabilities, logits };
  }

  /**
   * Run one throwaway inference.
   *
   * The first call to a fresh session pays for lazy kernel initialisation — tens of
   * milliseconds. Paying it here rather than on the user's first frame keeps the
   * visible latency honest.
   */
  async #warmUp(): Promise<void> {
    await this.classify(new Float32Array(this.#manifest.input.featureLength));
  }

  async close(): Promise<void> {
    await this.#session.release();
  }
}
