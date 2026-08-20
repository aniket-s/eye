/**
 * Locations of the self-hosted MediaPipe assets.
 *
 * Populated by `npm run setup:assets`, which copies the WASM out of node_modules and
 * downloads the `.task` models against `models.lock.json`. Nothing is loaded from a
 * CDN at runtime, so a deployment is reproducible and works offline once cached
 * (docs/AUDIT.md, V1 and V2).
 */

/** Directory holding `vision_wasm_internal.{js,wasm}`. */
export const WASM_BASE_PATH = `${import.meta.env.BASE_URL}mediapipe/wasm`;

/** Two-handed landmark model. */
export const HAND_MODEL_PATH = `${import.meta.env.BASE_URL}models/hand_landmarker.task`;

/**
 * Upper-body pose model.
 *
 * The `lite` variant is deliberate: `full` and `heavy` cost far more of the frame
 * budget than body context is worth at this stage, and `heavy` would also breach
 * Cloudflare Pages' 25 MiB per-asset limit.
 */
export const POSE_MODEL_PATH = `${import.meta.env.BASE_URL}models/pose_landmarker_lite.task`;

/**
 * Guidance shown when an asset is missing, rather than a raw 404.
 *
 * The models are not committed to the repository — a fresh clone genuinely does not
 * have them until setup runs, so this is a first-run experience worth handling well.
 */
export const MISSING_ASSETS_MESSAGE =
  'Vision models are not installed. Run `npm run setup:assets` and reload.';

/** True if the asset appears to be present and is not an HTML error page. */
export async function assetExists(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    if (!response.ok) return false;
    // A dev server may answer any path with index.html; a `.task` file is not HTML.
    const type = response.headers.get('content-type') ?? '';
    return !type.includes('text/html');
  } catch {
    return false;
  }
}
