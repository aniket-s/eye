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
  CustomSignBook,
  IsolatedSignRecognizer,
  VisionFrameView,
  HandshapeRecognizer,
  LandmarkView,
  LegacyRecognizer,
  MlpClassifier,
  NO_PREDICTION,
} from '@mudrapragyan/core';
import type { Handedness, MlpWeights, Vocabularies } from '@mudrapragyan/core';
import { OnnxClassifier } from '../model/onnxClassifier.js';
import { OnnxSequenceClassifier } from '../model/onnxSequenceClassifier.js';
import { OnnxClipClassifier } from '../model/onnxClipClassifier.js';
import { loadPack, resolvePackId } from '../model/pack.js';
import type { WorkerRequest, WorkerResponse } from './protocol.js';

const scope = self as unknown as DedicatedWorkerGlobalScope;

/** Reused across frames so reading landmarks allocates nothing. */
const view = new LandmarkView();
/** Reused for word-mode frames, which carry both hands and pose. */
const frameView = new VisionFrameView();

let modern: HandshapeRecognizer | null = null;
let continuous: ContinuousRecognizer | null = null;
let words: IsolatedSignRecognizer | null = null;
let legacy: LegacyRecognizer | null = null;

/**
 * The user's own signs, matched against each frame's embedding.
 *
 * Lives here rather than on the main thread so a match costs one dot product on the
 * thread that already holds the embedding, instead of posting 128 floats per frame.
 */
const customSigns = new CustomSignBook();
/** Whether to send embeddings back. Only true while the recorder is open. */
let capturing = false;

/**
 * The active reading of the trained handshapes, if the pack offers more than one.
 *
 * Held here rather than applied on the main thread because it changes what the model is
 * allowed to predict — masking after the fact would let an ineligible class win the
 * argmax and then be rewritten, which is not the same answer.
 */
let vocabularies: Vocabularies | undefined;
let display: Readonly<Record<string, string>> | null = null;

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
  // Which pack, if several are installed. A one-pack install skips the registry
  // entirely and behaves exactly as before.
  const pack = await loadPack(await resolvePackId());

  if (pack !== null && pack.manifest.task === 'temporal-isolated') {
    const classifier = await OnnxClipClassifier.create(pack.modelBytes, pack.manifest);
    words = new IsolatedSignRecognizer(classifier, {
      // Clip length is the model's, not the app's.
      maxFrames: pack.manifest.input.windowFrames ?? 96,
      minProbability: pack.manifest.thresholds.minProbability,
      featureLength: pack.manifest.input.featureLength,
    });
    post({
      type: 'ready',
      labelCount: pack.manifest.labels.length,
      pipeline: 'words',
      packName: `${pack.manifest.name} v${pack.manifest.version}`,
    });
    return;
  }

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
    vocabularies = pack.manifest.vocabularies;
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
      packVersion: pack.manifest.version,
      supportsCustomSigns: classifier.supportsCustomSigns,
      ...(pack.manifest.confusions === undefined ? {} : { confusions: pack.manifest.confusions }),
      ...(pack.manifest.vocabularies === undefined
        ? {}
        : { vocabularies: pack.manifest.vocabularies }),
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
      words?.reset();
      legacy?.reset();
      return;

    case 'customSigns':
      customSigns.load(message.signs);
      return;

    case 'capture':
      capturing = message.enabled;
      return;

    case 'vocabulary': {
      const chosen = message.name === null ? undefined : vocabularies?.[message.name];
      display = chosen ?? null;
      modern?.setAllowed(chosen === undefined ? null : new Set(Object.keys(chosen)));
      return;
    }

    case 'frame': {
      if (words !== null) {
        if (message.frame === undefined) return;
        const frame = frameView.read(message.frame, message.timestampMs, message.handedness);
        void words
          .push(frame, message.handedness)
          .then((result) => {
            post({
              type: 'word',
              seq: message.seq,
              timestampMs: message.timestampMs,
              capturing: result.capturing,
              sign: result.recognised?.label ?? null,
              probability: result.recognised?.probability ?? 0,
              alternatives: result.alternatives.map((a) => a.label),
              motion: result.motion,
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
            // Arbitration between the trained vocabulary and the user's own lives in
            // `CustomSignBook`, where it is unit-tested — see `matchIfUnrecognised`.
            const custom = customSigns.matchIfUnrecognised(
              result.embedding,
              result.verdict?.accepted ?? false,
            );

            post({
              type: 'result',
              seq: message.seq,
              timestampMs: message.timestampMs,
              letter: custom?.sign.label ?? result.letter,
              // What to show for it. A custom sign is already its own label; a trained
              // handshape is renamed by the active vocabulary, so `V` shows as `2`.
              ...(custom === null && display !== null && display[result.letter] !== undefined
                ? { display: display[result.letter] }
                : {}),
              rawLabel: result.verdict?.label ?? null,
              confidence: result.verdict?.probability ?? null,
              reason: custom !== null ? 'custom-sign' : (result.verdict?.reason ?? null),
              ...(custom !== null
                ? { custom: { label: custom.sign.label, similarity: custom.similarity } }
                : {}),
              ...(capturing && result.embedding !== null ? { embedding: result.embedding } : {}),
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
