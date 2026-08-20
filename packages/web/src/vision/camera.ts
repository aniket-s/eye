/**
 * Camera capture and the per-frame loop.
 *
 * Replaces `@mediapipe/camera_utils`, which is part of the end-of-life legacy bundle
 * and offers nothing that ~60 lines here do not. Two deliberate differences from v1:
 *
 * - **`requestVideoFrameCallback`** where available, so the loop is driven by actual
 *   decoded video frames rather than the display refresh. That gives a real media
 *   timestamp per frame, which is what makes downstream timing frame-rate independent
 *   (docs/AUDIT.md, J2). It falls back to `requestAnimationFrame`.
 * - **640×480 capture** instead of 1280×720. Hand landmarking downsamples to 192×192
 *   internally, so the extra pixels bought nothing and cost a meaningful slice of the
 *   33 ms frame budget (V5).
 */

/** A frame ready to be processed. */
export interface CameraFrame {
  readonly video: HTMLVideoElement;
  /** Media time of this frame in milliseconds. Monotonic, not wall-clock. */
  readonly timestampMs: number;
}

export interface CameraOptions {
  readonly video: HTMLVideoElement;
  readonly onFrame: (frame: CameraFrame) => void;
  readonly width?: number;
  readonly height?: number;
}

/** A live camera session. */
export interface CameraSession {
  stop(): void;
  /** Actual capture resolution, once negotiated with the device. */
  readonly resolution: { width: number; height: number };
}

interface VideoFrameCallbackMetadata {
  mediaTime: number;
}

type VideoWithFrameCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (
    callback: (now: number, metadata: VideoFrameCallbackMetadata) => void,
  ) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

const DEFAULT_WIDTH = 640;
const DEFAULT_HEIGHT = 480;

/**
 * Open the camera and begin delivering frames.
 *
 * @throws DOMException from `getUserMedia` if permission is refused or no device
 *   exists. Callers should render {@link describeCameraError}.
 */
export async function startCamera(options: CameraOptions): Promise<CameraSession> {
  const video = options.video as VideoWithFrameCallback;

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: options.width ?? DEFAULT_WIDTH },
      height: { ideal: options.height ?? DEFAULT_HEIGHT },
      facingMode: 'user',
    },
    audio: false,
  });

  video.srcObject = stream;
  await video.play();

  let stopped = false;
  let rafHandle = 0;
  let vfcHandle = 0;

  const settings = stream.getVideoTracks()[0]?.getSettings() ?? {};
  const resolution = {
    width: settings.width ?? video.videoWidth,
    height: settings.height ?? video.videoHeight,
  };

  const requestVideoFrame = video.requestVideoFrameCallback?.bind(video);
  if (requestVideoFrame !== undefined) {
    const onVideoFrame = (_now: number, metadata: VideoFrameCallbackMetadata): void => {
      if (stopped) return;
      options.onFrame({ video, timestampMs: metadata.mediaTime * 1000 });
      vfcHandle = requestVideoFrame(onVideoFrame);
    };
    vfcHandle = requestVideoFrame(onVideoFrame);
  } else {
    // Safari before 15.4 and older Firefox. `currentTime` is coarser but still a
    // media clock, so timing stays independent of the display refresh rate.
    const onAnimationFrame = (): void => {
      if (stopped) return;
      options.onFrame({ video, timestampMs: video.currentTime * 1000 });
      rafHandle = requestAnimationFrame(onAnimationFrame);
    };
    rafHandle = requestAnimationFrame(onAnimationFrame);
  }

  return {
    resolution,
    stop(): void {
      if (stopped) return;
      stopped = true;
      if (rafHandle !== 0) cancelAnimationFrame(rafHandle);
      if (vfcHandle !== 0 && typeof video.cancelVideoFrameCallback === 'function') {
        video.cancelVideoFrameCallback(vfcHandle);
      }
      for (const track of stream.getTracks()) track.stop();
      video.srcObject = null;
    },
  };
}

/**
 * Turn a camera failure into something worth showing a user.
 *
 * v1 surfaced these through `alert(e.message)`, which produced strings like
 * "Permission denied" with no indication of what to do about it (A7).
 */
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
    case 'AbortError':
      return 'The camera stopped unexpectedly. Try again.';
    default:
      return error.message || 'Could not start the camera.';
  }
}

/** Camera access requires a secure context. `localhost` counts as secure. */
export function isSecureContextForCamera(): boolean {
  return window.isSecureContext;
}
