/**
 * Zero-allocation transfer of landmarks across the Worker boundary.
 *
 * Landmarks cross that boundary 30–60 times a second. Structured-cloning an array
 * of 21 objects each time allocates on both sides and adds steady GC pressure —
 * exactly the per-frame allocation problem flagged as A5. A `Float32Array` is
 * transferable, so the buffer moves without a copy, and the receiving side reads it
 * through a reusable view that allocates nothing after construction.
 */
import { LANDMARKS_PER_HAND, type HandLandmarks, type Landmark } from '../types.js';

/** Floats needed to hold one hand: 21 landmarks × (x, y, z). */
export const HAND_BUFFER_LENGTH = LANDMARKS_PER_HAND * 3;

/**
 * Flatten landmarks into `target`, or into a fresh buffer.
 *
 * @throws RangeError if the landmark count or buffer length is wrong.
 */
export function packLandmarks(
  landmarks: HandLandmarks,
  target: Float32Array = new Float32Array(HAND_BUFFER_LENGTH),
): Float32Array {
  if (landmarks.length !== LANDMARKS_PER_HAND) {
    throw new RangeError(`Expected ${LANDMARKS_PER_HAND} landmarks, received ${landmarks.length}`);
  }
  if (target.length !== HAND_BUFFER_LENGTH) {
    throw new RangeError(`Expected a buffer of ${HAND_BUFFER_LENGTH}, received ${target.length}`);
  }
  for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
    const point = landmarks[i]!;
    const base = i * 3;
    target[base] = point.x;
    target[base + 1] = point.y;
    target[base + 2] = point.z;
  }
  return target;
}

/**
 * A reusable, mutable view over a packed landmark buffer.
 *
 * Construct once, then call {@link read} per frame. The returned array is the *same*
 * array every time, mutated in place — so callers must not retain it across frames.
 * That trade is deliberate: it removes 21 object allocations per frame from the hot
 * path, and every consumer in this codebase reads the landmarks synchronously.
 */
export class LandmarkView {
  readonly #points: Landmark[];
  readonly #mutable: { x: number; y: number; z: number }[];

  constructor() {
    this.#mutable = Array.from({ length: LANDMARKS_PER_HAND }, () => ({ x: 0, y: 0, z: 0 }));
    this.#points = this.#mutable;
  }

  /**
   * Populate the view from a packed buffer.
   *
   * @returns The view's landmarks. Valid until the next call to `read`.
   * @throws RangeError if the buffer is the wrong length.
   */
  read(buffer: Float32Array): HandLandmarks {
    if (buffer.length !== HAND_BUFFER_LENGTH) {
      throw new RangeError(`Expected a buffer of ${HAND_BUFFER_LENGTH}, received ${buffer.length}`);
    }
    for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
      const point = this.#mutable[i]!;
      const base = i * 3;
      point.x = buffer[base]!;
      point.y = buffer[base + 1]!;
      point.z = buffer[base + 2]!;
    }
    return this.#points;
  }
}
