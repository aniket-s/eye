import { MlpClassifier, type MlpWeights } from '@mudrapragyan/core';

/**
 * Fetch and validate the v1 model weights.
 *
 * The blob is ~2.5 MB of JSON for a 115k-parameter network — roughly 22 bytes per
 * parameter, versus ~1 byte for an int8 ONNX export (docs/AUDIT.md, M6). Phase 2
 * replaces this with a versioned model pack loaded through ONNX Runtime Web.
 */
export const MODEL_URL = `${import.meta.env.BASE_URL}model_weights.json`;

export async function loadClassifier(url: string = MODEL_URL): Promise<MlpClassifier> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (cause) {
    throw new Error(`Could not reach the model at ${url}.`, { cause });
  }

  if (!response.ok) {
    throw new Error(`Model request failed with HTTP ${response.status}.`);
  }

  const weights = (await response.json()) as MlpWeights;
  return MlpClassifier.fromWeights(weights);
}
