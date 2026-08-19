/**
 * ⚠️ LEGACY BEHAVIOUR — REPLACED IN PHASE 3.
 *
 * Decides when a detected letter has been held long enough to commit to the sentence.
 *
 * This is a **dwell timer**: it counts consecutive frames carrying the identical label.
 * Because the v1 model's `argmax` flickers on noisy input, the counter resets constantly
 * and the user must hold unnaturally still. The correct fix is to smooth *probabilities*
 * rather than labels, then commit on evidence rather than elapsed time — see
 * `docs/AUDIT.md` §1.5. Phase 3 replaces this with CTC blank-transition detection wrapped
 * in LocalAgreement-2.
 *
 * Extracted here as a pure state machine so the replacement can be A/B tested against it.
 */
import { NO_PREDICTION } from '../types.js';

export interface HoldCommitOptions {
  /** Consecutive identical frames required before committing. */
  readonly framesToCommit?: number;
  /** Frames to wait after a commit before another may occur. */
  readonly cooldownFrames?: number;
}

export interface HoldCommitResult {
  /** The letter committed on this frame, or `null` if none was. */
  readonly committed: string | null;
  /** Progress toward the next commit, 0–100, for the UI progress bar. */
  readonly progressPercent: number;
}

const DEFAULT_FRAMES_TO_COMMIT = 20;
const DEFAULT_COOLDOWN_FRAMES = 30;

export class HoldCommitDetector {
  readonly #framesToCommit: number;
  readonly #cooldownFrames: number;

  #heldFrames = 0;
  #lastLetter = '';
  #cooldown = 0;

  constructor(options: HoldCommitOptions = {}) {
    this.#framesToCommit = options.framesToCommit ?? DEFAULT_FRAMES_TO_COMMIT;
    this.#cooldownFrames = options.cooldownFrames ?? DEFAULT_COOLDOWN_FRAMES;
    if (this.#framesToCommit <= 0) {
      throw new RangeError('framesToCommit must be positive');
    }
  }

  /** Discard progress, e.g. when the camera stops or the sentence is cleared. */
  reset(): void {
    this.#heldFrames = 0;
    this.#lastLetter = '';
    this.#cooldown = 0;
  }

  /**
   * Advance one frame.
   *
   * @param letter The displayed letter, or {@link NO_PREDICTION} when no hand is visible.
   */
  update(letter: string): HoldCommitResult {
    if (letter === NO_PREDICTION) {
      this.#heldFrames = 0;
      this.#lastLetter = '';
      return { committed: null, progressPercent: 0 };
    }

    this.#cooldown = Math.max(0, this.#cooldown - 1);

    if (letter === this.#lastLetter) {
      this.#heldFrames++;
    } else {
      this.#heldFrames = 0;
      this.#lastLetter = letter;
    }

    const progressPercent = Math.min(
      100,
      Math.round((this.#heldFrames / this.#framesToCommit) * 100),
    );

    if (this.#heldFrames >= this.#framesToCommit && this.#cooldown === 0) {
      this.#heldFrames = 0;
      this.#cooldown = this.#cooldownFrames;
      return { committed: letter, progressPercent };
    }

    return { committed: null, progressPercent };
  }
}
