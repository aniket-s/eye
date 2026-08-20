/// <reference lib="webworker" />
/**
 * Recognition worker.
 *
 * Runs the classifier off the main thread so a slow inference can never drop a camera
 * frame or stall the UI (docs/AUDIT.md, A5), and downloads the model itself so the
 * parse never blocks first paint (M6).
 *
 * Two pipelines live here, and which one runs depends on whether a **v2 model pack**
 * is installed:
 *
 * - **v2 (preferred).** Proper normalisation, a trained `none` class, probability
 *   smoothing and threshold judging. No correction heuristics of any kind.
 * - **v1 (fallback).** The original MLP plus 85 lines of `geometricFix`. Used only
 *   when no pack is present, which is what a fresh clone looks like.
 *
 * Installing a trained pack retires the fallback. See `packages/core/src/legacy/README.md`.
 */
import {
  ContinuousRecognizer,
  HandshapeRecognizer,
  LandmarkView,
  LegacyRecognizer,
  MlpClassifier,
  NO_PREDICTION,
} from '@mudrapragyan/core';
import type { Handedness, MlpWeights } from '@mudrapragyan/core';
import { OnnxClassifier } from '../model/onnxClassifier.js';
import { OnnxSequenceClassifier } from '../model/onnxSequenceClassifier.js';
import { loadPack } from '../model/pack.js';
import type { WorkerRequest, WorkerResponse } from './protocol.js';

const scope = self as unknown as DedicatedWorkerGlobalScope;

/** Reused across frames so reading landmarks allocates nothing. */
const view = new LandmarkView();

let modern: HandshapeRecognizer | null = null;
let continuous: ContinuousRecognizer | null = null;
let legacy: LegacyRecognizer | null = null;

function post(message: WorkerResponse): void {
  scope.postMessage(message);
}

/**
 * Try the v2 pack, then fall back to the v1 weights.
 *
 * A pack that exists but is broken throws rather than silently degrading — a
 * corrupted model is a real failure and hiding it behind the legacy path would make
 * it invisible.
 */
async function init(fallbackModelUrl: string): Promise<void> {
  const pack = await loadPack();

  if (pack !== null && pack.manifest.task === 'temporal-ctc') {
    const classifier = await OnnxSequenceClassifier.create(pack.modelBytes, pack.manifest);
    continuous = new ContinuousRecognizer(classifier, {
      // The window length is the model's, not the app's — a pack trained on 64 frames
      // must be fed 64.
      windowFrames: pack.manifest.input.windowFrames ?? 64,
      featureLength: pack.manifest.input.featureLength,
    });
    post({
      type: 'ready',
      labelCount: pack.manifest.labels.filter((label) => label !== '').length,
      pipeline: 'ctc',
      packName: `${pack.manifest.name} v${pack.manifest.version}`,
    });
    return;
  }

  if (pack !== null) {
    const classifier = await OnnxClassifier.create(pack.modelBytes, pack.manifest);
    modern = new HandshapeRecognizer(classifier, {
      // The operating point comes from the pack, not from the app. A retrain ships
      // its own thresholds, so they can never fall out of step with the weights.
      thresholds: pack.manifest.thresholds,
      featureLength: pack.manifest.input.featureLength,
    });
    post({
      type: 'ready',
      labelCount: pack.manifest.labels.filter((label) => label !== 'none').length,
      pipeline: 'v2',
      packName: `${pack.manifest.name} v${pack.manifest.version}`,
    });
    return;
  }

  const response = await fetch(fallbackModelUrl);
  if (!response.ok) {
    throw new Error(`Model request failed with HTTP ${response.status}.`);
  }
  const weights = (await response.json()) as MlpWeights;
  const classifier = MlpClassifier.fromWeights(weights);
  legacy = new LegacyRecognizer(classifier);
  post({ type: 'ready', labelCount: classifier.labels.length, pipeline: 'v1' });
}

scope.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;

  switch (message.type) {
    case 'init':
      init(message.modelUrl).catch((error: unknown) => {
        post({
          type: 'error',
          message: error instanceof Error ? error.message : 'Model could not be loaded.',
        });
      });
      return;

    case 'reset':
      modern?.reset();
      continuous?.reset();
      legacy?.reset();
      return;

    case 'frame': {
      const landmarks = message.landmarks === null ? null : view.read(message.landmarks);

      if (continuous !== null) {
        void continuous
          .push(landmarks, message.handedness)
          .then((result) => {
            post({
              type: 'continuous',
              seq: message.seq,
              timestampMs: message.timestampMs,
              committed: result.committed.join(''),
              provisional: result.provisional.join(''),
              confidence: result.confidence,
            });
          })
          .catch((error: unknown) => {
            post({
              type: 'error',
              message: error instanceof Error ? error.message : 'Inference failed.',
            });
          });
        return;
      }

      if (modern !== null) {
        void modern
          .recognise(landmarks, message.handedness)
          .then((result) => {
            post({
              type: 'result',
              seq: message.seq,
              timestampMs: message.timestampMs,
              letter: result.letter,
              rawLabel: result.verdict?.label ?? null,
              confidence: result.verdict?.probability ?? null,
              reason: result.verdict?.reason ?? null,
            });
          })
          .catch((error: unknown) => {
            post({
              type: 'error',
              message: error instanceof Error ? error.message : 'Inference failed.',
            });
          });
        return;
      }

      if (legacy === null) return;
      const result = legacy.recognise(landmarks);
      post({
        type: 'result',
        seq: message.seq,
        timestampMs: message.timestampMs,
        letter: result.letter,
        rawLabel: result.rawLabel,
        confidence: result.confidence,
        reason: result.letter === NO_PREDICTION ? 'low-probability' : 'accepted',
      });
      return;
    }

    default: {
      // Exhaustiveness: adding a request type without handling it fails to compile.
      const unexpected: never = message;
      post({ type: 'error', message: `Unknown request: ${JSON.stringify(unexpected)}` });
    }
  }
});

export type { Handedness };
