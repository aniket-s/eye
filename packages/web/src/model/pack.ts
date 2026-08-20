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
 * Optional index listing every installed pack.
 *
 * ```json
 * { "packs": ["asl-fingerspell", "isl-fingerspell", "asl-words"], "default": "asl-fingerspell" }
 * ```
 *
 * Without it a single pack at {@link DEFAULT_PACK_ID} is assumed, which is what a
 * one-language install looks like. With it the app can offer a picker — the mechanism
 * that makes Indian Sign Language a *new pack* rather than a code change (ADR 0003).
 */
export const REGISTRY_URL = `${PACK_ROOT}/index.json`;

/** Remembers the user's choice between visits. */
const SELECTED_PACK_KEY = 'mudrapragyan.selectedPack';

export interface PackRegistry {
  readonly packs: readonly string[];
  readonly defaultPack: string;
}

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
 * Read the pack index, if one is published.
 *
 * A missing index is normal, not an error: it simply means one pack is installed.
 */
export async function loadRegistry(): Promise<PackRegistry | null> {
  try {
    const response = await fetch(REGISTRY_URL);
    if (!response.ok) return null;
    if (!(response.headers.get('content-type') ?? '').includes('json')) return null;

    const raw = (await response.json()) as { packs?: unknown; default?: unknown };
    if (!Array.isArray(raw.packs) || raw.packs.length === 0) return null;

    const packs = raw.packs.filter((id): id is string => typeof id === 'string');
    if (packs.length === 0) return null;

    const preferred = typeof raw.default === 'string' ? raw.default : packs[0]!;
    return { packs, defaultPack: packs.includes(preferred) ? preferred : packs[0]! };
  } catch {
    return null;
  }
}

/** The pack to load: the user's choice if still installed, else the registry default. */
export async function resolvePackId(): Promise<string> {
  const registry = await loadRegistry();
  if (registry === null) return DEFAULT_PACK_ID;

  let remembered: string | null = null;
  try {
    remembered = localStorage.getItem(SELECTED_PACK_KEY);
  } catch {
    // Storage can be unavailable in private modes; fall back to the default.
  }

  return remembered !== null && registry.packs.includes(remembered)
    ? remembered
    : registry.defaultPack;
}

/** Remember which pack the user picked. */
export function rememberPackId(id: string): void {
  try {
    localStorage.setItem(SELECTED_PACK_KEY, id);
  } catch {
    // Not worth surfacing: the choice simply will not persist.
  }
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

  // The version is stamped into the URL so the weights become genuinely immutable:
  // publishing a retrained pack changes the URL, which is what lets the service worker
  // cache them permanently without ever pinning a user to a superseded model. The
  // manifest itself is fetched at a stable URL and served network-first.
  const modelUrl = `${PACK_ROOT}/${id}/${manifest.modelFile}?v=${encodeURIComponent(manifest.version)}`;
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
