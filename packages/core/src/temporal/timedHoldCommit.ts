/**
 * Time-based replacement for {@link HoldCommitDetector}.
 *
 * v1 counted *frames* (`HOLD_NEEDED = 20`). With `requestVideoFrameCallback` driving
 * the loop, the frame rate varies with camera capability and CPU load, so a frame
 * count means a different amount of real time on every machine — the same defect as
 * J2 in the J/Z state machine. Everything here is measured in milliseconds taken
 * from the frame's media timestamp, so a 60 fps laptop and a 24 fps phone behave
 * identically.
 *
 * The defaults reproduce v1's timing at its assumed 30 fps: 20 frames ≈ 667 ms hold,
 * 30 frames ≈ 1000 ms cooldown.
 *
 * Still a dwell timer, and still therefore vulnerable to `argmax` flicker. Phase 3
 * replaces the mechanism entirely with CTC blank-transition commits; this class
 * exists so Phase 1 does not make the flicker *worse* by also being frame-rate
 * dependent.
 */
import { NO_PREDICTION } from '../types.js';

export interface TimedHoldCommitOptions {
  /** How long a letter must be held, in milliseconds. */
  readonly holdMs?: number;
  /** Quiet period after a commit, in milliseconds. */
  readonly cooldownMs?: number;
  /**
   * How long a letter may disappear before progress is discarded.
   *
   * A short tolerance bridges single-frame dropouts, which are common when the hand
   * is near the edge of frame, without letting a genuinely abandoned sign commit.
   */
  readonly toleranceMs?: number;
  /**
   * How large a gap between frames means the stream stalled rather than continued.
   *
   * Distinct from {@link toleranceMs}, which covers the hand briefly vanishing while
   * frames keep arriving. This one covers frames themselves stopping — a backgrounded
   * tab, a stalled camera. Must be generous enough not to fire on a low frame rate:
   * at 10 fps the interval is already 100 ms, so a tight value here would spuriously
   * restart the hold on slow hardware.
   */
  readonly staleMs?: number;
}

export interface TimedHoldCommitResult {
  readonly committed: string | null;
  /** Progress toward the next commit, 0–100. */
  readonly progressPercent: number;
}

const DEFAULT_HOLD_MS = 667;
const DEFAULT_COOLDOWN_MS = 1000;
const DEFAULT_TOLERANCE_MS = 120;
const DEFAULT_STALE_MS = 1000;

export class TimedHoldCommitDetector {
  readonly #holdMs: number;
  readonly #cooldownMs: number;
  readonly #toleranceMs: number;
  readonly #staleMs: number;

  #letter = '';
  #heldSinceMs = 0;
  #lastSeenMs = 0;
  #cooldownUntilMs = 0;

  constructor(options: TimedHoldCommitOptions = {}) {
    this.#holdMs = options.holdMs ?? DEFAULT_HOLD_MS;
    this.#cooldownMs = options.cooldownMs ?? DEFAULT_COOLDOWN_MS;
    this.#toleranceMs = options.toleranceMs ?? DEFAULT_TOLERANCE_MS;
    this.#staleMs = options.staleMs ?? DEFAULT_STALE_MS;
    if (this.#holdMs <= 0) throw new RangeError('holdMs must be positive');
  }

  reset(): void {
    this.#letter = '';
    this.#heldSinceMs = 0;
    this.#lastSeenMs = 0;
    this.#cooldownUntilMs = 0;
  }

  /**
   * Advance to `timestampMs`.
   *
   * @param letter The displayed letter, or {@link NO_PREDICTION}.
   * @param timestampMs Media time of the frame. Must be non-decreasing.
   */
  update(letter: string, timestampMs: number): TimedHoldCommitResult {
    if (letter === NO_PREDICTION) {
      // Tolerate a brief dropout rather than discarding progress immediately.
      if (this.#letter !== '' && timestampMs - this.#lastSeenMs > this.#toleranceMs) {
        this.#letter = '';
        this.#heldSinceMs = 0;
      }
      return { committed: null, progressPercent: this.#progress(timestampMs) };
    }

    // A gap this large means frames stopped arriving, so we cannot claim the letter
    // was held across it. `toleranceMs` is deliberately NOT used here: it measures a
    // vanishing hand, not a vanishing frame, and reusing it would restart the hold on
    // any device slower than ~8 fps.
    const stalled = this.#letter !== '' && timestampMs - this.#lastSeenMs > this.#staleMs;
    if (letter !== this.#letter || stalled) {
      this.#letter = letter;
      this.#heldSinceMs = timestampMs;
    }
    this.#lastSeenMs = timestampMs;

    const progressPercent = this.#progress(timestampMs);

    if (timestampMs - this.#heldSinceMs >= this.#holdMs && timestampMs >= this.#cooldownUntilMs) {
      this.#cooldownUntilMs = timestampMs + this.#cooldownMs;
      this.#heldSinceMs = timestampMs;
      return { committed: letter, progressPercent: 100 };
    }

    return { committed: null, progressPercent };
  }

  #progress(timestampMs: number): number {
    if (this.#letter === '') return 0;
    const elapsed = timestampMs - this.#heldSinceMs;
    return Math.max(0, Math.min(100, Math.round((elapsed / this.#holdMs) * 100)));
  }
}
