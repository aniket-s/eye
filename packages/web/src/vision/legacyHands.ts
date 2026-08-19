/**
 * ⚠️ LEGACY — REPLACED IN PHASE 1.
 *
 * Thin wrapper around the end-of-life `@mediapipe/hands` and `@mediapipe/camera_utils`
 * scripts. Google ended support for the legacy MediaPipe Solutions on 1 March 2023;
 * they remain downloadable "on an as-is basis" with no fixes (docs/AUDIT.md, V1).
 *
 * Two changes from v1, both safety rather than behaviour:
 * - **Versions are pinned.** v1 loaded `@latest`, so an upstream publish could break
 *   production with no rollback (V2). The pins below are the final legacy releases.
 * - Loading is promise-based and reports failures, instead of relying on the globals
 *   happening to exist by the time the user clicks Start.
 *
 * Phase 1 replaces this file with `@mediapipe/tasks-vision@1.0.1`, self-hosted, with
 * two hands and pose landmarks. See docs/adr/0001-client-side-inference.md.
 */
import type { Landmark } from '@mudrapragyan/core';

/** Final published version of the legacy hands solution. */
const HANDS_VERSION = '0.4.1675469240';
/** Final published version of the legacy camera helper. */
const CAMERA_UTILS_VERSION = '0.3.1675466862';

const HANDS_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${HANDS_VERSION}`;
const CAMERA_UTILS_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@${CAMERA_UTILS_VERSION}/camera_utils.js`;

/** Results object emitted by the legacy hands solution. */
export interface HandsResults {
  readonly multiHandLandmarks?: Landmark[][];
}

interface HandsOptions {
  maxNumHands: number;
  modelComplexity: number;
  minDetectionConfidence: number;
  minTrackingConfidence: number;
}

interface HandsInstance {
  setOptions(options: HandsOptions): void;
  onResults(callback: (results: HandsResults) => void): void;
  send(input: { image: HTMLVideoElement }): Promise<void>;
  close(): Promise<void>;
}

interface CameraInstance {
  start(): Promise<void>;
  stop(): void;
}

interface LegacyGlobals {
  Hands: new (config: { locateFile: (file: string) => string }) => HandsInstance;
  Camera: new (
    video: HTMLVideoElement,
    config: { onFrame: () => Promise<void>; width: number; height: number },
  ) => CameraInstance;
}

const loadedScripts = new Map<string, Promise<void>>();

/** Inject a classic script once, resolving when it has executed. */
function loadScript(url: string): Promise<void> {
  const existing = loadedScripts.get(url);
  if (existing !== undefined) return existing;

  const pending = new Promise<void>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = url;
    script.crossOrigin = 'anonymous';
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () =>
      reject(new Error(`Failed to load ${url}. Check your network connection.`));
    document.head.append(script);
  });

  loadedScripts.set(url, pending);
  return pending;
}

/** Options for {@link startHandTracking}. */
export interface HandTrackingOptions {
  readonly video: HTMLVideoElement;
  readonly onResults: (results: HandsResults) => void;
  readonly width?: number;
  readonly height?: number;
}

/** A running hand-tracking session. */
export interface HandTrackingSession {
  stop(): Promise<void>;
}

/**
 * Load MediaPipe, open the camera, and begin emitting hand landmarks.
 *
 * @throws Error if the scripts cannot be loaded or camera access is refused. The
 *   caller is responsible for showing the message; v1 used a bare `alert()`.
 */
export async function startHandTracking(
  options: HandTrackingOptions,
): Promise<HandTrackingSession> {
  await Promise.all([loadScript(`${HANDS_BASE}/hands.js`), loadScript(CAMERA_UTILS_URL)]);

  const globals = window as unknown as Partial<LegacyGlobals>;
  if (globals.Hands === undefined || globals.Camera === undefined) {
    throw new Error('MediaPipe loaded but did not register its globals.');
  }

  const hands = new globals.Hands({ locateFile: (file) => `${HANDS_BASE}/${file}` });
  hands.setOptions({
    maxNumHands: 1, // ⚠️ Blocks two-handed word signs. Raised to 2 in Phase 1 (V3).
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });
  hands.onResults(options.onResults);

  const camera = new globals.Camera(options.video, {
    onFrame: async () => {
      await hands.send({ image: options.video });
    },
    width: options.width ?? 1280,
    height: options.height ?? 720,
  });

  await camera.start();

  return {
    async stop(): Promise<void> {
      camera.stop();
      await hands.close();
    },
  };
}

/** Turn a camera failure into something worth showing a user. */
export function describeCameraError(error: unknown): string {
  if (!(error instanceof Error)) return 'Could not start the camera.';

  switch (error.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera access was blocked. Allow camera permission in your browser, then try again.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No camera was found. Connect a webcam and try again.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'Your camera is in use by another app. Close it and try again.';
    case 'OverconstrainedError':
      return 'Your camera does not support the requested resolution.';
    default:
      return error.message || 'Could not start the camera.';
  }
}

/** Camera access requires a secure context; `localhost` counts as secure. */
export function isSecureContextForCamera(): boolean {
  return window.isSecureContext;
}
