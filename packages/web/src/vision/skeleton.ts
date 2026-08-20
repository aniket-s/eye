/**
 * Hand-skeleton overlay drawing, shared by the Translator and the recorder.
 *
 * Seeing the landmarks is what makes tracking legible: when the skeleton is on the
 * hand, the pipeline is seeing what the user thinks it is seeing; when it drops or
 * sticks to a face, the user knows to move rather than blame the model. The canvas
 * carries the same CSS `scaleX(-1)` mirror as the video under it, so drawing in raw
 * landmark coordinates lines up with the mirrored preview.
 */
import type { HandObservation } from '@mudrapragyan/core';

/** MediaPipe hand topology. */
export const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

/**
 * Draw one frame's hands onto `canvas`, sized to the video's pixel grid.
 *
 * The primary hand is drawn in `accent`; any other hand is dimmed, so two hands in
 * frame make it obvious which one the recogniser is reading.
 */
export function drawHandSkeleton(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  hands: readonly HandObservation[],
  primary: HandObservation | null,
  accent: string,
): void {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (width === 0 || height === 0) return;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const context = canvas.getContext('2d');
  if (context === null) return;
  context.clearRect(0, 0, width, height);

  for (const hand of hands) {
    const colour = hand === primary ? accent : 'rgba(241, 245, 249, 0.35)';
    context.strokeStyle = colour;
    context.fillStyle = colour;
    context.lineWidth = 2;

    for (const [from, to] of HAND_CONNECTIONS) {
      const a = hand.landmarks[from];
      const b = hand.landmarks[to];
      if (a === undefined || b === undefined) continue;
      context.beginPath();
      context.moveTo(a.x * width, a.y * height);
      context.lineTo(b.x * width, b.y * height);
      context.stroke();
    }
    for (const point of hand.landmarks) {
      context.beginPath();
      context.arc(point.x * width, point.y * height, 3, 0, Math.PI * 2);
      context.fill();
    }
  }
}

/** Wipe the overlay, e.g. when the camera stops or the hand leaves frame. */
export function clearSkeleton(canvas: HTMLCanvasElement): void {
  canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
}
