/// <reference lib="webworker" />
/**
 * Recognition worker.
 *
 * Runs the classifier off the main thread so a slow inference can never drop a
 * camera frame or stall the UI (docs/AUDIT.md, A5). The worker also *fetches and
 * parses* the model itself, which moves the 2.5 MB JSON parse off the main thread
 * entirely — that parse was previously blocking first paint (M6).
 *
 * Everything the worker does comes from `@mudrapragyan/core`, so the same logic is
 * unit-tested in Node without a worker or a browser.
 */
import { LandmarkView, LegacyRecognizer, MlpClassifier } from '@mudrapragyan/core';
import type { MlpWeights } from '@mudrapragyan/core';
import type { WorkerRequest, WorkerResponse } from './protocol.js';

const scope = self as unknown as DedicatedWorkerGlobalScope;

/** Reused across frames so reading landmarks allocates nothing. */
const view = new LandmarkView();
let recognizer: LegacyRecognizer | null = null;

function post(message: WorkerResponse): void {
  scope.postMessage(message);
}

async function init(modelUrl: string): Promise<void> {
  const response = await fetch(modelUrl);
  if (!response.ok) {
    throw new Error(`Model request failed with HTTP ${response.status}.`);
  }
  const weights = (await response.json()) as MlpWeights;
  const classifier = MlpClassifier.fromWeights(weights);
  recognizer = new LegacyRecognizer(classifier);
  post({ type: 'ready', labelCount: classifier.labels.length });
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
      recognizer?.reset();
      return;

    case 'frame': {
      if (recognizer === null) return;
      const landmarks = message.landmarks === null ? null : view.read(message.landmarks);
      const result = recognizer.recognise(landmarks);
      post({
        type: 'result',
        seq: message.seq,
        timestampMs: message.timestampMs,
        letter: result.letter,
        rawLabel: result.rawLabel,
        confidence: result.confidence,
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
