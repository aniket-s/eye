/**
 * Deciding where one sign ends and the next begins.
 *
 * Fingerspelling avoids this problem: CTC decodes a continuous stream and never needs
 * boundaries. Isolated word signs are different — the classifier takes a *complete*
 * clip and returns one label, so something must decide which frames form that clip.
 *
 * The signal is motion. A signer at rest holds still or drops their hands; producing a
 * sign means moving. So: hands appear and start moving → a sign is underway; motion
 * falls away or the hands leave → the sign is over, classify what was captured.
 *
 * Three details make this work rather than fire constantly:
 *
 * - **Hysteresis.** Separate thresholds to start and to stop. A single threshold makes
 *   the state flap every time motion hovers near it, splitting one sign into several.
 * - **Minimum duration.** A brief twitch is not a sign. Segments shorter than the
 *   floor are discarded rather than classified into whatever they happen to resemble.
 * - **Millisecond timing.** Like everywhere else after Phase 1, thresholds are in real
 *   time, so behaviour does not change with frame rate.
 *
 * This runs **causally** — it only ever sees the past, because at frame *t* that is all
 * a live camera has. The classifier it feeds does not need to be causal, since by then
 * the segment is complete.
 */
import type { Handedness, VisionFrame } from '../vision/frame.js';

export interface SignSegmenterOptions {
  /** Motion above this starts a segment. Units: normalised hand widths per second. */
  readonly startMotion?: number;
  /** Motion below this ends one. Must be lower than `startMotion`. */
  readonly stopMotion?: number;
  /** How long motion must stay low before a segment closes. */
  readonly quietMs?: number;
  /** Segments shorter than this are discarded. */
  readonly minDurationMs?: number;
  /** Segments are force-closed at this length, so one never grows unbounded. */
  readonly maxDurationMs?: number;
}

export type SegmenterState = 'idle' | 'active';

export interface SegmenterUpdate {
  readonly state: SegmenterState;
  /** True on the frame a segment closes and should be classified. */
  readonly segmentComplete: boolean;
  /** Duration of the closed segment, in milliseconds. */
  readonly segmentDurationMs: number;
  /** Smoothed motion, for the debug overlay. */
  readonly motion: number;
}

const DEFAULT_START_MOTION = 0.9;
const DEFAULT_STOP_MOTION = 0.35;
const DEFAULT_QUIET_MS = 320;
const DEFAULT_MIN_DURATION_MS = 400;
const DEFAULT_MAX_DURATION_MS = 4000;

/** Smoothing factor for the motion estimate. Roughly a 100 ms time constant at 30 fps. */
const MOTION_SMOOTHING = 0.3;

export class SignSegmenter {
  readonly #startMotion: number;
  readonly #stopMotion: number;
  readonly #quietMs: number;
  readonly #minDurationMs: number;
  readonly #maxDurationMs: number;

  #state: SegmenterState = 'idle';
  #motion = 0;
  #previous: { x: number; y: number } | null = null;
  #previousMs = 0;
  #startedMs = 0;
  #quietSinceMs: number | null = null;

  constructor(options: SignSegmenterOptions = {}) {
    this.#startMotion = options.startMotion ?? DEFAULT_START_MOTION;
    this.#stopMotion = options.stopMotion ?? DEFAULT_STOP_MOTION;
    this.#quietMs = options.quietMs ?? DEFAULT_QUIET_MS;
    this.#minDurationMs = options.minDurationMs ?? DEFAULT_MIN_DURATION_MS;
    this.#maxDurationMs = options.maxDurationMs ?? DEFAULT_MAX_DURATION_MS;

    if (this.#stopMotion >= this.#startMotion) {
      throw new RangeError(
        'stopMotion must be below startMotion, or the state will flap on every frame',
      );
    }
  }

  get state(): SegmenterState {
    return this.#state;
  }

  /** Smoothed motion estimate. */
  get motion(): number {
    return this.#motion;
  }

  reset(): void {
    this.#state = 'idle';
    this.#motion = 0;
    this.#previous = null;
    this.#quietSinceMs = null;
  }

  /**
   * Advance one frame.
   *
   * @param frame Hands and pose for this frame.
   * @param dominant The signer's dominant hand, used to pick which hand to track.
   */
  update(frame: VisionFrame, dominant: Handedness = 'right'): SegmenterUpdate {
    const timestampMs = frame.timestampMs;
    const anchor = this.#anchor(frame, dominant);

    // ── Motion estimate ──
    if (anchor !== null && this.#previous !== null && timestampMs > this.#previousMs) {
      const seconds = (timestampMs - this.#previousMs) / 1000;
      const distance = Math.hypot(anchor.x - this.#previous.x, anchor.y - this.#previous.y);
      const speed = distance / Math.max(seconds, 1e-3);
      this.#motion += (speed - this.#motion) * MOTION_SMOOTHING;
    } else if (anchor === null) {
      // Hands gone: decay toward zero rather than freezing, so a segment still closes.
      this.#motion *= 1 - MOTION_SMOOTHING;
    }

    this.#previous = anchor;
    this.#previousMs = timestampMs;

    // ── State machine ──
    if (this.#state === 'idle') {
      if (anchor !== null && this.#motion > this.#startMotion) {
        this.#state = 'active';
        this.#startedMs = timestampMs;
        this.#quietSinceMs = null;
      }
      return this.#report(false, 0);
    }

    if (this.#motion < this.#stopMotion || anchor === null) {
      this.#quietSinceMs ??= timestampMs;
      if (timestampMs - this.#quietSinceMs >= this.#quietMs) {
        // Duration is measured to the moment motion *stopped*, not to the moment we
        // concluded it had. Including the quiet period would inflate every segment by
        // `quietMs`, so `minDurationMs` would not mean what it says and brief twitches
        // would clear the floor.
        return this.#close(this.#quietSinceMs - this.#startedMs);
      }
    } else {
      this.#quietSinceMs = null;
    }

    // A segment that never settles is force-closed, so the buffer cannot grow forever.
    const elapsed = timestampMs - this.#startedMs;
    if (elapsed >= this.#maxDurationMs) return this.#close(elapsed);

    return this.#report(false, 0);
  }

  #close(duration: number): SegmenterUpdate {
    this.#state = 'idle';
    this.#quietSinceMs = null;
    // A twitch is not a sign. Discarding rather than classifying avoids emitting
    // whatever word a fragment happens to resemble.
    const usable = duration >= this.#minDurationMs;
    return this.#report(usable, usable ? duration : 0);
  }

  #report(segmentComplete: boolean, segmentDurationMs: number): SegmenterUpdate {
    return {
      state: this.#state,
      segmentComplete,
      segmentDurationMs,
      motion: this.#motion,
    };
  }

  /**
   * The point whose movement is tracked.
   *
   * The dominant hand's wrist, falling back to whichever hand is present. Tracking a
   * single stable point beats averaging every landmark, which would report motion for
   * a hand merely changing shape in place.
   */
  #anchor(frame: VisionFrame, dominant: Handedness): { x: number; y: number } | null {
    if (frame.hands.length === 0) return null;
    const preferred = frame.hands.find((hand) => hand.handedness === dominant) ?? frame.hands[0]!;
    const wrist = preferred.landmarks[0];
    return wrist === undefined ? null : { x: wrist.x, y: wrist.y };
  }
}
