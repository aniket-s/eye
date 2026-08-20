import { describe, expect, it } from 'vitest';
import { SignSegmenter } from './signSegmenter.js';
import { openHand, translate } from '../test/landmarkFactory.js';
import type { VisionFrame } from '../vision/frame.js';

/** A frame with the dominant hand at a given offset. */
function frameAt(timestampMs: number, dx: number, dy = 0): VisionFrame {
  return {
    timestampMs,
    hands: [
      { handedness: 'right', handednessScore: 0.95, landmarks: translate(openHand(), dx, dy) },
    ],
    pose: null,
  };
}

/** A frame with no hands visible. */
function emptyAt(timestampMs: number): VisionFrame {
  return { timestampMs, hands: [], pose: null };
}

/**
 * A simulated signer.
 *
 * Position and time persist across calls. An earlier version of this harness reset the
 * hand to the origin on each call, which teleported it between phases and injected a
 * motion spike the segmenter correctly reacted to — the test was wrong, not the code.
 */
class Signer {
  #x = 0;
  #timeMs = 0;
  readonly completions: number[] = [];

  constructor(
    private readonly segmenter: SignSegmenter,
    private readonly fps = 30,
  ) {}

  get timeMs(): number {
    return this.#timeMs;
  }

  /** Move at `speed` normalised units per second for `durationMs`. */
  move(durationMs: number, speed: number): this {
    const step = 1000 / this.fps;
    for (let elapsed = 0; elapsed < durationMs; elapsed += step) {
      this.#x += (speed * step) / 1000;
      this.#timeMs += step;
      const result = this.segmenter.update(frameAt(this.#timeMs, this.#x));
      if (result.segmentComplete) this.completions.push(result.segmentDurationMs);
    }
    return this;
  }

  /** Hold still. */
  hold(durationMs: number): this {
    return this.move(durationMs, 0);
  }

  /** Take the hands out of frame. */
  vanish(durationMs: number): this {
    const step = 1000 / this.fps;
    for (let elapsed = 0; elapsed < durationMs; elapsed += step) {
      this.#timeMs += step;
      const result = this.segmenter.update(emptyAt(this.#timeMs));
      if (result.segmentComplete) this.completions.push(result.segmentDurationMs);
    }
    return this;
  }
}

describe('SignSegmenter — construction', () => {
  it('rejects thresholds that would make the state flap', () => {
    expect(() => new SignSegmenter({ startMotion: 0.5, stopMotion: 0.5 })).toThrow(RangeError);
    expect(() => new SignSegmenter({ startMotion: 0.3, stopMotion: 0.9 })).toThrow(RangeError);
  });

  it('starts idle', () => {
    expect(new SignSegmenter().state).toBe('idle');
  });
});

describe('SignSegmenter — detecting a sign', () => {
  it('stays idle while the hand is still', () => {
    const segmenter = new SignSegmenter();
    const signer = new Signer(segmenter).hold(2000);
    expect(segmenter.state).toBe('idle');
    expect(signer.completions).toEqual([]);
  });

  it('becomes active once the hand moves', () => {
    const segmenter = new SignSegmenter();
    new Signer(segmenter).move(500, 2.0);
    expect(segmenter.state).toBe('active');
  });

  it('closes the segment after motion settles', () => {
    const segmenter = new SignSegmenter({ minDurationMs: 300, quietMs: 250 });
    const signer = new Signer(segmenter).move(800, 2.0).hold(700);

    expect(signer.completions).toHaveLength(1);
    expect(signer.completions[0]).toBeGreaterThanOrEqual(300);
    expect(segmenter.state).toBe('idle');
  });

  it('closes the segment when the hands leave frame', () => {
    const segmenter = new SignSegmenter({ minDurationMs: 300, quietMs: 200 });
    const signer = new Signer(segmenter).move(800, 2.0).vanish(600);
    expect(signer.completions).toHaveLength(1);
  });
});

describe('SignSegmenter — avoiding spurious segments', () => {
  /**
   * A twitch is not a sign. Without a duration floor the classifier would be handed
   * fragments and would dutifully return whichever word they most resembled.
   */
  it('discards a movement too brief to be a sign', () => {
    const segmenter = new SignSegmenter({ minDurationMs: 800, quietMs: 200 });
    const signer = new Signer(segmenter).move(150, 3.0).hold(700);

    expect(segmenter.state).toBe('idle'); // it did close
    expect(signer.completions).toEqual([]); // but produced nothing to classify
  });

  /**
   * Hysteresis. With one threshold, motion hovering near it would toggle the state
   * every frame and shred a single sign into several fragments.
   */
  it('does not split one sign when motion dips briefly mid-gesture', () => {
    const segmenter = new SignSegmenter({
      startMotion: 0.9,
      stopMotion: 0.35,
      quietMs: 300,
      minDurationMs: 200,
    });

    // A brief slow patch — below the start threshold, above the stop threshold.
    const signer = new Signer(segmenter).move(600, 2.0).move(200, 0.6).move(600, 2.0);

    expect(signer.completions).toEqual([]); // still one unbroken sign
    expect(segmenter.state).toBe('active');
  });

  it('force-closes a segment that never settles', () => {
    const segmenter = new SignSegmenter({ maxDurationMs: 1500, minDurationMs: 200 });
    const signer = new Signer(segmenter).move(4000, 3.0);

    // Continuous motion for four seconds is not one sign; the buffer must not grow
    // without bound.
    expect(signer.completions.length).toBeGreaterThanOrEqual(2);
  });
});

describe('SignSegmenter — frame-rate independence', () => {
  /**
   * Motion is measured per second and quiet periods in milliseconds, so the same
   * physical gesture segments identically whatever the camera manages.
   */
  it('produces the same segmentation at 15, 30 and 60 fps', () => {
    const durations: number[][] = [];

    for (const fps of [15, 30, 60]) {
      const segmenter = new SignSegmenter({ minDurationMs: 300, quietMs: 250 });
      const signer = new Signer(segmenter, fps).move(900, 2.0).hold(700);
      durations.push(signer.completions);
    }

    expect(durations[0]).toHaveLength(1);
    expect(durations[1]).toHaveLength(1);
    expect(durations[2]).toHaveLength(1);
    // Durations should agree to within a frame or two, not scale with frame rate.
    const values = durations.map((d) => d[0] as number);
    expect(Math.max(...values) - Math.min(...values)).toBeLessThan(150);
  });
});

describe('SignSegmenter — reset', () => {
  it('returns to idle and clears motion', () => {
    const segmenter = new SignSegmenter();
    new Signer(segmenter).move(600, 2.0);
    expect(segmenter.state).toBe('active');

    segmenter.reset();
    expect(segmenter.state).toBe('idle');
    expect(segmenter.motion).toBe(0);
  });
});
