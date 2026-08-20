/**
 * Transfer a whole {@link VisionFrame} across the Worker boundary.
 *
 * {@link packLandmarks} moves one hand, which is all fingerspelling needs. Word signs
 * need both hands *and* upper-body pose, because position relative to the torso is part
 * of the meaning. Rather than structured-cloning nested objects 30 times a second, the
 * frame is flattened into one transferable `Float32Array`.
 *
 * Layout (150 floats):
 * ```
 * [0]         dominant hand present (0 or 1)
 * [1]         other hand present
 * [2]         pose present
 * [3..65]     dominant hand, 21 × (x, y, z)
 * [66..128]   other hand
 * [129..149]  pose, 7 × (x, y, z)
 * ```
 *
 * Presence flags rather than NaN sentinels: NaN propagates silently through arithmetic,
 * so a missing hand would poison the features instead of being handled.
 */
import { LANDMARKS_PER_HAND, type Landmark } from '../types.js';
import { POSE_POINTS } from '../features/normaliseBody.js';
import type { Handedness, HandObservation, VisionFrame } from './frame.js';

const HAND_VALUES = LANDMARKS_PER_HAND * 3;
const POSE_VALUES = POSE_POINTS * 3;

const DOMINANT_PRESENT = 0;
const OTHER_PRESENT = 1;
const POSE_PRESENT = 2;
const DOMINANT_OFFSET = 3;
const OTHER_OFFSET = DOMINANT_OFFSET + HAND_VALUES;
const POSE_OFFSET = OTHER_OFFSET + HAND_VALUES;

/** Total floats in a packed frame. */
export const FRAME_BUFFER_LENGTH = POSE_OFFSET + POSE_VALUES;

/**
 * Flatten a frame.
 *
 * Hands are stored dominant-first, so the worker does not need to know which
 * handedness was dominant to unpack them.
 */
export function packVisionFrame(
  frame: VisionFrame,
  dominant: Handedness,
  target: Float32Array = new Float32Array(FRAME_BUFFER_LENGTH),
): Float32Array {
  if (target.length !== FRAME_BUFFER_LENGTH) {
    throw new RangeError(`Expected a buffer of ${FRAME_BUFFER_LENGTH}, received ${target.length}`);
  }
  target.fill(0);

  for (const hand of frame.hands) {
    if (hand.landmarks.length !== LANDMARKS_PER_HAND) continue;
    const isDominant = hand.handedness === dominant;
    const offset = isDominant ? DOMINANT_OFFSET : OTHER_OFFSET;
    writePoints(target, offset, hand.landmarks);
    target[isDominant ? DOMINANT_PRESENT : OTHER_PRESENT] = 1;
  }

  if (frame.pose !== null && frame.pose.length > 0) {
    writePoints(target, POSE_OFFSET, frame.pose.slice(0, POSE_POINTS));
    target[POSE_PRESENT] = 1;
  }

  return target;
}

function writePoints(target: Float32Array, offset: number, points: readonly Landmark[]): void {
  for (let i = 0; i < points.length; i++) {
    const point = points[i]!;
    target[offset + i * 3] = point.x;
    target[offset + i * 3 + 1] = point.y;
    target[offset + i * 3 + 2] = point.z;
  }
}

/**
 * Reusable view over a packed frame.
 *
 * Like {@link LandmarkView}, the returned objects are reused between calls, so callers
 * must consume them synchronously. That keeps the per-frame allocation at zero.
 */
export class VisionFrameView {
  readonly #dominant: { x: number; y: number; z: number }[];
  readonly #other: { x: number; y: number; z: number }[];
  readonly #pose: { x: number; y: number; z: number }[];

  constructor() {
    const make = (count: number): { x: number; y: number; z: number }[] =>
      Array.from({ length: count }, () => ({ x: 0, y: 0, z: 0 }));
    this.#dominant = make(LANDMARKS_PER_HAND);
    this.#other = make(LANDMARKS_PER_HAND);
    this.#pose = make(POSE_POINTS);
  }

  /**
   * Rebuild a frame from a packed buffer.
   *
   * @param dominant The handedness the buffer was packed against, so hands are
   *   labelled consistently with how they were stored.
   */
  read(buffer: Float32Array, timestampMs: number, dominant: Handedness): VisionFrame {
    if (buffer.length !== FRAME_BUFFER_LENGTH) {
      throw new RangeError(
        `Expected a buffer of ${FRAME_BUFFER_LENGTH}, received ${buffer.length}`,
      );
    }

    const other: Handedness = dominant === 'right' ? 'left' : 'right';
    const hands: HandObservation[] = [];

    if (buffer[DOMINANT_PRESENT] === 1) {
      readPoints(buffer, DOMINANT_OFFSET, this.#dominant);
      hands.push({ handedness: dominant, handednessScore: 1, landmarks: this.#dominant });
    }
    if (buffer[OTHER_PRESENT] === 1) {
      readPoints(buffer, OTHER_OFFSET, this.#other);
      hands.push({ handedness: other, handednessScore: 1, landmarks: this.#other });
    }

    let pose: Landmark[] | null = null;
    if (buffer[POSE_PRESENT] === 1) {
      readPoints(buffer, POSE_OFFSET, this.#pose);
      pose = this.#pose;
    }

    return { timestampMs, hands, pose };
  }
}

function readPoints(
  buffer: Float32Array,
  offset: number,
  target: { x: number; y: number; z: number }[],
): void {
  for (let i = 0; i < target.length; i++) {
    const point = target[i]!;
    point.x = buffer[offset + i * 3]!;
    point.y = buffer[offset + i * 3 + 1]!;
    point.z = buffer[offset + i * 3 + 2]!;
  }
}
