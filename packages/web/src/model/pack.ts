import { parseManifest, type PackManifest } from '@mudrapragyan/core';

/**
 * Loading versioned model packs.
 *
 * A pack lives at `models/<id>/manifest.json` alongside its weights. The manifest is
 * validated before anything is downloaded, so a pack trained under a different
 * normalisation scheme than this build implements is refused rather than left to
 * mispredict silently (ADR 0003).
 */

/** Where packs are served from. */
export const PACK_ROOT = `${import.meta.env.BASE_URL}models`;

/**
 * Pack the app loads when one is available.
 *
 * When absent, the app falls back to the v1 JSON model and its legacy correction
 * heuristics. Dropping a trained pack here is what retires that path — see
 * `packages/core/src/legacy/README.md`.
 */
export const DEFAULT_PACK_ID = 'asl-fingerspell';

export interface LoadedPack {
  readonly manifest: PackManifest;
  readonly modelBytes: ArrayBuffer;
}

/**
 * Fetch and validate a pack.
 *
 * @returns The pack, or `null` if no pack is installed at that id. A missing pack is
 *   an expected state, not an error — it is what a fresh clone looks like.
 * @throws Error if a pack exists but is malformed or its weights fail to download.
 *   That is a real problem and must not be swallowed.
 */
export async function loadPack(id: string = DEFAULT_PACK_ID): Promise<LoadedPack | null> {
  const manifestUrl = `${PACK_ROOT}/${id}/manifest.json`;

  let response: Response;
  try {
    response = await fetch(manifestUrl);
  } catch {
    return null; // offline or blocked — fall back rather than break the app
  }
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error(`Model pack "${id}" manifest returned HTTP ${response.status}.`);
  }

  const contentType = response.headers.get('content-type') ?? '';
  if (!contentType.includes('json')) {
    // A dev server that answers every path with index.html would otherwise produce a
    // confusing JSON parse error here.
    return null;
  }

  const manifest = parseManifest(await response.json());

  const modelUrl = `${PACK_ROOT}/${id}/${manifest.modelFile}`;
  const modelResponse = await fetch(modelUrl);
  if (!modelResponse.ok) {
    throw new Error(
      `Model pack "${id}" declares ${manifest.modelFile} but it returned HTTP ${modelResponse.status}.`,
    );
  }

  return { manifest, modelBytes: await modelResponse.arrayBuffer() };
}

/** One-line description for the status area and logs. */
export function describePack(manifest: PackManifest): string {
  const letters = manifest.labels.filter((label) => label !== 'none').length;
  return `${manifest.name} v${manifest.version} — ${letters} signs, macro-F1 ${manifest.metrics.macroF1.toFixed(3)}`;
}
