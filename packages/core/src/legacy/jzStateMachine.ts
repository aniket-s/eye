/**
 * ⚠️ LEGACY — DELETED IN PHASE 3. See ./README.md before touching this file.
 *
 * Detects the letters J and Z, which are the only two ASL letters defined by motion
 * rather than a static handshape. J is a hooking motion from the `I` handshape; Z is
 * a zig-zag traced from the `D` handshape.
 *
 * Ported from the module-level mutable state in the original `index.html`, with the
 * state encapsulated in a class so it can be reset and unit-tested. Behaviour,
 * constants, and branch order are unchanged. The dead `jzFlicker` variable from v1
 * (assigned in five places, never read) is omitted; it had no effect.
 *
 * **Known defects**, preserved deliberately and fixed by CTC decoding in Phase 3:
 * - {@link JzStateMachine.isZShape} accepts *both* stroke directions, roughly doubling
 *   the false-positive rate; a horizontal wiggle reads as Z.
 * - Frames are counted, not timed, so every threshold is frame-rate dependent and
 *   drifts under CPU load.
 * - A confirmed detection latches the output for 32 frames regardless of what the
 *   hand does next.
 */
import { HandLandmarkIndex, type HandLandmarks } from '../types.js';

/** One frame of tracked index- and pinky-tip positions. */
interface TrackedFrame {
  readonly indexX: number;
  readonly indexY: number;
  readonly pinkyX: number;
  readonly pinkyY: number;
}

type JzState = 'idle' | 'tracking_j' | 'tracking_z' | 'confirmed_j' | 'confirmed_z';

/** Frames of motion history retained. */
const HISTORY_LIMIT = 45;
/** Minimum history before a shape test may fire. */
const MIN_HISTORY = 12;
/** Frames a tracking window stays open. */
const TRACKING_FRAMES = 65;
/** Frames a confirmed detection latches its output. */
const CONFIRMED_FRAMES = 32;
/** Minimum model confidence to open a tracking window. */
const TRACKING_CONFIDENCE = 0.55;

export class JzStateMachine {
  #state: JzState = 'idle';
  #timer = 0;
  #history: TrackedFrame[] = [];

  /** Current internal state. Exposed for tests and debug overlays. */
  get state(): string {
    return this.#state;
  }

  /** Clear all motion history and return to idle. */
  reset(): void {
    this.#state = 'idle';
    this.#timer = 0;
    this.#history = [];
  }

  /**
   * Advance one frame.
   *
   * @param letter The letter after {@link geometricFix}.
   * @param confidence The model's softmax confidence for that letter.
   * @param landmarks The frame's hand landmarks.
   * @returns The letter to display, which may be `J` or `Z`.
   */
  update(letter: string, confidence: number, landmarks: HandLandmarks): string {
    const indexTip = landmarks[HandLandmarkIndex.INDEX_TIP]!;
    const pinkyTip = landmarks[HandLandmarkIndex.PINKY_TIP]!;
    this.#history.push({
      indexX: indexTip.x,
      indexY: indexTip.y,
      pinkyX: pinkyTip.x,
      pinkyY: pinkyTip.y,
    });
    if (this.#history.length > HISTORY_LIMIT) this.#history.shift();
    this.#timer = Math.max(0, this.#timer - 1);

    // Y is unrelated to J/Z. Reset so a Z tracked from a prior D cannot capture it.
    if (letter === 'Y') {
      this.reset();
      return 'Y';
    }

    if (this.#state === 'confirmed_j') {
      if (this.#timer === 0) this.reset();
      return 'J';
    }
    if (this.#state === 'confirmed_z') {
      if (this.#timer === 0) this.reset();
      return 'Z';
    }

    if (letter === 'I') {
      this.#maybeStartTracking('tracking_j', confidence);
      if (this.#state === 'tracking_j') {
        if (this.#timer === 0) {
          this.reset();
        } else if (this.#hasHistory() && this.isJShape()) {
          return this.#confirm('confirmed_j', 'J');
        }
      }
      return 'I';
    }

    if (letter === 'D') {
      this.#maybeStartTracking('tracking_z', confidence);
      if (this.#state === 'tracking_z') {
        if (this.#timer === 0) {
          this.reset();
        } else if (this.#hasHistory() && this.isZShape()) {
          return this.#confirm('confirmed_z', 'Z');
        }
      }
      return 'D';
    }

    // A different letter arrived mid-motion. Rely on the timer rather than aborting,
    // because the handshape genuinely changes partway through a J or Z.
    if (this.#state === 'tracking_j' || this.#state === 'tracking_z') {
      if (this.#timer === 0) {
        this.reset();
      } else {
        if (this.#state === 'tracking_j' && this.#hasHistory() && this.isJShape()) {
          return this.#confirm('confirmed_j', 'J');
        }
        if (this.#state === 'tracking_z' && this.#hasHistory() && this.isZShape()) {
          return this.#confirm('confirmed_z', 'Z');
        }
        return this.#state === 'tracking_j' ? 'I' : 'D';
      }
    }

    return letter;
  }

  /** True when the pinky tip has traced the downward hook of a J. */
  isJShape(): boolean {
    const n = this.#history.length;
    if (n < MIN_HISTORY) return false;
    const start = this.#history[0]!;
    const end = this.#history[n - 1]!;
    const downward = end.pinkyY - start.pinkyY;
    const total = Math.abs(end.pinkyY - start.pinkyY) + Math.abs(end.pinkyX - start.pinkyX);
    return downward > 0.07 && total > 0.09;
  }

  /**
   * True when the index tip has traced the three strokes of a Z.
   *
   * Accepts either stroke direction, because the camera preview is mirrored while
   * MediaPipe sees the raw feed. This is the defect noted in the module docblock:
   * accepting both directions roughly doubles the false-positive rate.
   */
  isZShape(): boolean {
    const n = this.#history.length;
    if (n < MIN_HISTORY) return false;
    const third = Math.floor(n / 3);
    const p0 = this.#history[0]!;
    const p1 = this.#history[third]!;
    const p2 = this.#history[2 * third]!;
    const p3 = this.#history[n - 1]!;

    const seg1 = p1.indexX - p0.indexX;
    const seg2 = p2.indexX - p1.indexX;
    const seg3 = p3.indexX - p2.indexX;
    const total = Math.abs(p3.indexX - p0.indexX) + Math.abs(p3.indexY - p0.indexY);
    if (total < 0.04) return false;

    const forward = seg1 < -0.01 && seg2 > 0.01 && seg3 < -0.008;
    const reverse = seg1 > 0.01 && seg2 < -0.01 && seg3 > 0.008;
    return forward || reverse;
  }

  #hasHistory(): boolean {
    return this.#history.length >= MIN_HISTORY;
  }

  #maybeStartTracking(next: 'tracking_j' | 'tracking_z', confidence: number): void {
    if (this.#state === 'idle' && confidence > TRACKING_CONFIDENCE) {
      this.#state = next;
      this.#timer = TRACKING_FRAMES;
      this.#history = [];
    }
  }

  #confirm(next: 'confirmed_j' | 'confirmed_z', letter: 'J' | 'Z'): string {
    this.#state = next;
    this.#timer = CONFIRMED_FRAMES;
    this.#history = [];
    return letter;
  }
}
