/**
 * J and Z: the two letters a static classifier cannot represent.
 *
 * Every other letter in the manual alphabet is a handshape. These two are a handshape
 * *plus a path* — J hooks the pinky down and round, Z draws the letter with the index —
 * and a single frame carries no path at all. So they are absent from every
 * `static-handshape` pack, which is correct: a classifier that claimed to read them
 * would be guessing.
 *
 * They were also, until this module, absent from the app. The v1 pipeline had a
 * detector for them (`legacy/jzStateMachine.ts`), but it is wired only into
 * `LegacyRecognizer`, which never runs once a pack is installed — and a pack ships with
 * the repository. The result was that J could not be produced by any code path:
 * not by the static pack, which has no such label; not by the CTC path, whose sequence
 * fixtures are built from a handshape table that has no J either; and not by the legacy
 * detector, which was unreachable.
 *
 * This is the v2 replacement. It runs *after* the handshape recogniser and consumes its
 * verdict, so `HandshapeRecognizer` keeps its guarantee of having no per-letter special
 * cases: the classifier's job is unchanged, and this only asks "while the model was
 * reading I, did the pinky trace a hook?"
 *
 * ## What is different from the v1 detector
 *
 * Its own docblock listed three defects, all preserved deliberately at the time. All
 * three are fixed here, because this one is meant to be kept:
 *
 * 1. **Frames were counted, not timed.** Every threshold moved with the frame rate, so
 *    the same gesture registered on a 60 fps laptop and not on a 24 fps phone. Windows
 *    here are milliseconds taken from the frame's own timestamp.
 * 2. **Distances were in frame units.** A hand at arm's length traces a shorter path in
 *    the image than the same hand close up, so the thresholds only held at one
 *    distance. Everything here is measured in *hand widths*, the same quantity the
 *    normaliser divides by, so it means one thing at any distance.
 * 3. **Z accepted both stroke directions**, roughly doubling false positives — a
 *    horizontal wiggle read as Z. Direction is genuinely ambiguous without a real
 *    recording to check against, so rather than guess, this keeps both directions and
 *    earns the accuracy back where it is not ambiguous: strokes must *alternate*, each
 *    must be horizontally dominant, and the whole trace must descend. A wiggle does
 *    none of those.
 *
 * A fourth difference: the v1 detector latched its output for 32 frames, so a confirmed
 * J kept overwriting whatever the hand did next. This emits the letter exactly once,
 * as a commit, and then requires the hand to leave the shape before it can fire again.
 */
import { HandLandmarkIndex, type HandLandmarks } from '../types.js';
import type { Handedness } from '../vision/frame.js';

/** A letter defined by motion, and how to recognise its path. */
export interface MotionLetter {
  readonly letter: string;
  /** The handshape the model must be reading for this trace to be armed. */
  readonly from: string;
  /** Which fingertip draws it. */
  readonly tip: number;
  /** Does this path match? Points are canonicalised and scaled to hand widths. */
  readonly matches: (path: readonly Point[]) => boolean;
}

/** One tracked sample: position in hand widths, and when it was seen. */
export interface Point {
  readonly x: number;
  readonly y: number;
  readonly timestampMs: number;
}

export interface MotionLetterOptions {
  /**
   * Which motion letters to watch for. Defaults to {@link MOTION_LETTERS}.
   *
   * Configurable so a pack can declare its own: a `temporal-ctc` pack decodes J and Z
   * as ordinary labels, and running this alongside it would emit each letter twice.
   */
  readonly letters?: readonly MotionLetter[];
  /** How long a trace may take before the attempt is abandoned. */
  readonly windowMs?: number;
  /** Quiet period after emitting, so one gesture cannot fire twice. */
  readonly cooldownMs?: number;
  /** Gap between frames that means the stream stalled rather than continued. */
  readonly staleMs?: number;
  /** Samples to keep. At 60 fps this is about a second and a half. */
  readonly historyLimit?: number;
}

const DEFAULT_WINDOW_MS = 1400;
const DEFAULT_COOLDOWN_MS = 700;
const DEFAULT_STALE_MS = 400;
const DEFAULT_HISTORY = 90;

/** Shortest trace worth testing. Below this it is a tremor, not a letter. */
const MIN_SAMPLES = 6;

/**
 * How far the pinky must fall during J's first half, in hand widths.
 *
 * A hand width here is the RMS spread the normaliser uses, so roughly a third of the
 * hand's full height. J's drop is most of the hand's height, hence a threshold above 1.
 */
const J_DROP = 1.0;
/** How far it must then curl sideways, in hand widths. The hook, as opposed to a drop. */
const J_CURL = 0.45;

/** Shortest horizontal stroke Z will accept, in hand widths. */
const Z_STROKE = 0.6;
/** How far a Z must descend overall. Without this, a horizontal wiggle qualifies. */
const Z_DESCENT = 0.5;

/**
 * J: the pinky drops, then hooks toward the thumb side.
 *
 * Two phases rather than one displacement. The v1 test compared only the first and last
 * sample, which a straight downward drop satisfies — and a hand simply being lowered
 * out of frame is a straight downward drop.
 *
 * The hook runs toward negative x because points arrive canonicalised to a right hand,
 * where the radial — thumb — side sits at smaller x. That is the same convention
 * `normaliseHand` and the training projection use.
 */
function isHook(path: readonly Point[]): boolean {
  const half = Math.floor(path.length / 2);
  const start = path[0];
  const middle = path[half];
  const end = path[path.length - 1];
  if (start === undefined || middle === undefined || end === undefined) return false;

  const dropped = middle.y - start.y;
  const curled = middle.x - end.x;
  const fellFurther = end.y - middle.y;

  // The drop must dominate the first half and the curl the second, so a diagonal swipe
  // — which satisfies both totals — does not qualify.
  return (
    dropped > J_DROP &&
    Math.abs(middle.x - start.x) < dropped &&
    curled > J_CURL &&
    Math.abs(fellFurther) < curled * 1.6
  );
}

/**
 * Z: three strokes that alternate horizontally while the whole trace descends.
 *
 * Split by index rather than by arc length. Signers slow down at the corners, so
 * sampling is denser there, and equal-count thirds land nearer the true corners than
 * equal-length thirds would.
 */
function isZigZag(path: readonly Point[]): boolean {
  const third = Math.floor(path.length / 3);
  if (third < 2) return false;

  const corners = [path[0], path[third], path[2 * third], path[path.length - 1]];
  if (corners.some((point) => point === undefined)) return false;
  const [a, b, c, d] = corners as [Point, Point, Point, Point];

  const strokes = [
    { dx: b.x - a.x, dy: b.y - a.y },
    { dx: c.x - b.x, dy: c.y - b.y },
    { dx: d.x - c.x, dy: d.y - c.y },
  ];

  // Every stroke long enough to be deliberate, and horizontal rather than vertical.
  if (strokes.some((stroke) => Math.abs(stroke.dx) < Z_STROKE)) return false;
  if (strokes.some((stroke) => Math.abs(stroke.dy) > Math.abs(stroke.dx))) return false;

  // Alternating, in either direction — see the module docblock on why both are kept.
  if (Math.sign(strokes[0]!.dx) === Math.sign(strokes[1]!.dx)) return false;
  if (Math.sign(strokes[1]!.dx) === Math.sign(strokes[2]!.dx)) return false;

  // And the letter descends. This is what a wiggle cannot fake.
  return d.y - a.y > Z_DESCENT;
}

/** The motion letters this recogniser knows, and the handshape each starts from. */
export const MOTION_LETTERS: readonly MotionLetter[] = [
  { letter: 'J', from: 'I', tip: HandLandmarkIndex.PINKY_TIP, matches: isHook },
  { letter: 'Z', from: 'D', tip: HandLandmarkIndex.INDEX_TIP, matches: isZigZag },
];

/** A fingertip and the apparent hand size it was seen at, in raw image units. */
interface Observation {
  readonly x: number;
  readonly y: number;
  readonly scale: number;
}

interface Attempt {
  readonly motion: MotionLetter;
  readonly startedMs: number;
  /**
   * Where the traced fingertip was when the attempt began, and how big the hand looked
   * then. Fixed for the life of the attempt — see {@link measure}.
   */
  readonly origin: Observation;
  readonly path: Point[];
}

/**
 * Watches for a motion letter being traced, given the handshape recogniser's verdict.
 *
 * Stateful across frames and not safe to share between streams. `reset()` clears it.
 */
export class MotionLetterRecognizer {
  readonly #windowMs: number;
  readonly #cooldownMs: number;
  readonly #staleMs: number;
  readonly #historyLimit: number;
  readonly #known: readonly MotionLetter[];

  #attempt: Attempt | null = null;
  #lastFrameMs: number | null = null;
  #quietUntilMs = 0;

  constructor(options: MotionLetterOptions = {}) {
    this.#known = options.letters ?? MOTION_LETTERS;
    this.#windowMs = options.windowMs ?? DEFAULT_WINDOW_MS;
    this.#cooldownMs = options.cooldownMs ?? DEFAULT_COOLDOWN_MS;
    this.#staleMs = options.staleMs ?? DEFAULT_STALE_MS;
    this.#historyLimit = options.historyLimit ?? DEFAULT_HISTORY;
  }

  /** Which letter is being traced, if any. Exposed for the debug overlay. */
  get tracking(): string | null {
    return this.#attempt?.motion.letter ?? null;
  }

  reset(): void {
    this.#attempt = null;
    this.#lastFrameMs = null;
  }

  /**
   * Advance one frame.
   *
   * @param letter The handshape the model accepted this frame, or `null` when it
   *   rejected the frame. A rejected frame does not abandon a trace in progress — the
   *   handshape genuinely stops being readable partway through a J, as the hand turns.
   * @param landmarks This frame's hand, or `null` if none was detected.
   * @param handedness Which hand it is. Left hands are mirrored to match, the same
   *   canonicalisation the classifier's features use.
   * @param timestampMs The frame's timestamp.
   * @returns The motion letter to commit, or `null`. Emitted **once**, as a commit:
   *   these letters cannot be held, so there is nothing for a dwell timer to measure.
   */
  update(
    letter: string | null,
    landmarks: HandLandmarks | null,
    handedness: Handedness,
    timestampMs: number,
  ): string | null {
    if (landmarks === null || landmarks.length === 0) {
      this.reset();
      return null;
    }

    // A gap in the frames means the stream stalled — a backgrounded tab, a stuttering
    // camera — and a path with a hole in it is not evidence of anything.
    const previous = this.#lastFrameMs;
    this.#lastFrameMs = timestampMs;
    if (previous !== null && timestampMs - previous > this.#staleMs) this.#attempt = null;

    if (timestampMs < this.#quietUntilMs) return null;

    this.#armIfShapeMatches(letter, landmarks, handedness, timestampMs);
    const attempt = this.#attempt;
    if (attempt === null) return null;

    if (timestampMs - attempt.startedMs > this.#windowMs) {
      this.#attempt = null;
      return null;
    }

    const seen = frameOf(landmarks, attempt.motion.tip, handedness);
    attempt.path.push(measure(seen, attempt.origin, timestampMs));
    if (attempt.path.length > this.#historyLimit) attempt.path.shift();
    if (attempt.path.length < MIN_SAMPLES) return null;

    if (!attempt.motion.matches(attempt.path)) return null;

    this.#attempt = null;
    this.#quietUntilMs = timestampMs + this.#cooldownMs;
    return attempt.motion.letter;
  }

  /**
   * Start tracking when the model reads a motion letter's starting handshape.
   *
   * Only from idle: re-arming on every frame the shape holds would keep resetting the
   * path back to nothing, so the trace could never accumulate — the gesture *leaves*
   * the starting shape almost immediately.
   */
  #armIfShapeMatches(
    letter: string | null,
    landmarks: HandLandmarks,
    handedness: Handedness,
    timestampMs: number,
  ): void {
    if (this.#attempt !== null || letter === null) return;
    const motion = this.#known.find((candidate) => candidate.from === letter);
    if (motion === undefined) return;
    this.#attempt = {
      motion,
      startedMs: timestampMs,
      origin: frameOf(landmarks, motion.tip, handedness),
      path: [],
    };
  }
}

/**
 * One fingertip and the hand it belongs to, canonicalised to a right hand.
 *
 * `scale` is the RMS spread — the same quantity `normaliseHand` divides by, so the two
 * agree about how big a hand is.
 */
function frameOf(landmarks: HandLandmarks, tip: number, handedness: Handedness): Observation {
  const flip = handedness === 'left' ? -1 : 1;

  let centroidX = 0;
  let centroidY = 0;
  for (const point of landmarks) {
    centroidX += point.x * flip;
    centroidY += point.y;
  }
  centroidX /= landmarks.length;
  centroidY /= landmarks.length;

  let sumSquares = 0;
  for (const point of landmarks) {
    const dx = point.x * flip - centroidX;
    const dy = point.y - centroidY;
    sumSquares += dx * dx + dy * dy;
  }

  const point = landmarks[tip];
  return {
    x: (point?.x ?? 0) * flip,
    y: point?.y ?? 0,
    scale: Math.max(Math.sqrt(sumSquares / landmarks.length), 1e-6),
  };
}

/**
 * How far the fingertip has travelled since the trace began, in hand widths.
 *
 * A displacement from a fixed origin, divided by the hand size measured at that same
 * moment. Both halves of that matter:
 *
 * * **Displacement, not position.** Absolute coordinates would make every threshold
 *   depend on where in the frame the signer happens to be standing.
 * * **A fixed scale, not each frame's own.** A hand's apparent size wobbles as the
 *   fingers fold and as the signer leans, so dividing by a moving quantity mixes that
 *   wobble into the measured path. A real hand cannot change size during one letter,
 *   so freezing the scale costs nothing and keeps the path a path.
 */
function measure(seen: Observation, origin: Observation, timestampMs: number): Point {
  return {
    x: (seen.x - origin.x) / origin.scale,
    y: (seen.y - origin.y) / origin.scale,
    timestampMs,
  };
}
