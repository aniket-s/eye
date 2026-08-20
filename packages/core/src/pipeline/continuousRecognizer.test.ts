import { describe, expect, it } from 'vitest';
import { ContinuousRecognizer, type SequenceClassifier } from './continuousRecognizer.js';
import { FEATURE_LENGTH, normaliseHand } from '../features/normalise.js';
import { openHand } from '../test/landmarkFactory.js';

const LABELS = ['', 'A', 'B', 'J', 'Z'];

/**
 * A classifier the test drives directly.
 *
 * It records every window it receives so tests can assert on buffering and ordering,
 * and emits a scripted label sequence so decoding is deterministic.
 */
function scriptedClassifier(script: number[][]): SequenceClassifier & {
  windows: { frames: number; first: number }[];
  calls: number;
} {
  const state = { windows: [] as { frames: number; first: number }[], calls: 0 };
  return {
    labels: LABELS,
    blankIndex: 0,
    windows: state.windows,
    get calls() {
      return state.calls;
    },
    classifySequence(window: Float32Array, frames: number) {
      state.windows.push({ frames, first: window[0] ?? 0 });
      const indices = script[Math.min(state.calls, script.length - 1)] ?? [];
      state.calls++;
      return Promise.resolve(
        indices.map((index) => {
          const vector = new Array<number>(LABELS.length).fill(0.01);
          vector[index] = 0.96;
          return vector;
        }),
      );
    },
  };
}

function recognizer(
  classifier: SequenceClassifier,
  overrides: Partial<{ windowFrames: number; strideFrames: number; agreement: number }> = {},
): ContinuousRecognizer {
  return new ContinuousRecognizer(classifier, {
    windowFrames: overrides.windowFrames ?? 20,
    strideFrames: overrides.strideFrames ?? 4,
    agreement: overrides.agreement ?? 2,
    featureLength: FEATURE_LENGTH,
  });
}

/** Feed `count` frames of a visible hand. */
async function feed(subject: ContinuousRecognizer, count: number): Promise<void> {
  for (let i = 0; i < count; i++) await subject.push(openHand(), 'right');
}

describe('ContinuousRecognizer — buffering and stride', () => {
  it('buffers without decoding until it has enough context', async () => {
    const classifier = scriptedClassifier([[1]]);
    const subject = recognizer(classifier, { strideFrames: 4 });

    // Minimum context is stride × 2 = 8 frames.
    for (let i = 0; i < 7; i++) {
      const result = await subject.push(openHand(), 'right');
      expect(result.decoded).toBe(false);
    }
    expect(classifier.calls).toBe(0);
  });

  it('decodes on the stride, not on every frame', async () => {
    const classifier = scriptedClassifier([[1]]);
    const subject = recognizer(classifier, { strideFrames: 5 });
    await feed(subject, 40);

    // 40 frames at stride 5 is far fewer than 40 model runs — that is the point.
    expect(classifier.calls).toBeGreaterThan(2);
    expect(classifier.calls).toBeLessThanOrEqual(8);
  });

  it('does not grow the window beyond its capacity', async () => {
    const classifier = scriptedClassifier([[1]]);
    const subject = recognizer(classifier, { windowFrames: 12, strideFrames: 3 });
    await feed(subject, 60);

    for (const window of classifier.windows) {
      expect(window.frames).toBeLessThanOrEqual(12);
    }
  });

  it('ignores frames with no hand', async () => {
    const classifier = scriptedClassifier([[1]]);
    const subject = recognizer(classifier);
    for (let i = 0; i < 10; i++) await subject.push(null, 'right');
    expect(classifier.calls).toBe(0);
  });
});

describe('ContinuousRecognizer — decoding and commit', () => {
  it('commits text once two hypotheses agree', async () => {
    const classifier = scriptedClassifier([
      [1, 0, 2],
      [1, 0, 2],
    ]);
    const subject = recognizer(classifier, { strideFrames: 4 });
    await feed(subject, 20);

    expect(subject.text).toBe('AB');
  });

  it('keeps an unstable tail provisional rather than showing it as committed', async () => {
    const classifier = scriptedClassifier([
      [1, 0, 2], // AB
      [1, 0, 1], // AA — disagrees on the second letter
    ]);
    const subject = recognizer(classifier, { strideFrames: 4 });
    // Exactly two decodes: the first at frame 8, the second at frame 12. Feeding
    // further would let the second (repeated) hypothesis agree with itself.
    await feed(subject, 12);

    expect(subject.text).toBe('A');
  });

  /**
   * The headline replacement. In v1 these two letters were handled by a bespoke
   * state machine with frame-counted thresholds, a both-directions Z test, and a
   * 32-frame output latch (docs/AUDIT.md §3). To a temporal model they are ordinary
   * labels, so all of that disappears.
   */
  it('treats J and Z as ordinary labels, with no state machine', async () => {
    const classifier = scriptedClassifier([
      [3, 0, 4],
      [3, 0, 4],
    ]);
    const subject = recognizer(classifier, { strideFrames: 4 });
    await feed(subject, 20);

    expect(subject.text).toBe('JZ');
  });

  it('spells a doubled letter, which a dwell timer cannot', async () => {
    const classifier = scriptedClassifier([
      [1, 0, 1],
      [1, 0, 1],
    ]);
    const subject = recognizer(classifier, { strideFrames: 4 });
    await feed(subject, 20);

    expect(subject.text).toBe('AA');
  });
});

describe('ContinuousRecognizer — pauses and resets', () => {
  it('flushes the window after a sustained pause but keeps committed text', async () => {
    const classifier = scriptedClassifier([[1], [1], [2], [2]]);
    const subject = new ContinuousRecognizer(classifier, {
      windowFrames: 20,
      strideFrames: 4,
      agreement: 2,
      idleResetFrames: 5,
      featureLength: FEATURE_LENGTH,
    });

    await feed(subject, 16);
    expect(subject.text).toBe('A');

    // A pause ends the word; without the flush the next letters would be spliced onto
    // the same window and decoded as one continuous spelling.
    for (let i = 0; i < 6; i++) await subject.push(null, 'right');
    const windowsBefore = classifier.windows.length;
    await feed(subject, 8);

    expect(classifier.windows[windowsBefore]?.frames).toBeLessThan(16);
    expect(subject.text.startsWith('A')).toBe(true);
  });

  it('clears everything on reset', async () => {
    const classifier = scriptedClassifier([[1], [1]]);
    const subject = recognizer(classifier, { strideFrames: 4 });
    await feed(subject, 16);
    expect(subject.text).toBe('A');

    subject.reset();
    expect(subject.text).toBe('');
  });

  it('rejects a nonsensical window size', () => {
    expect(() => new ContinuousRecognizer(scriptedClassifier([[1]]), { windowFrames: 0 })).toThrow(
      RangeError,
    );
  });
});

describe('ContinuousRecognizer — window ordering', () => {
  /**
   * The ring buffer must present frames oldest-first. If it wrapped without
   * reordering, the model would see the sequence cut and spliced at an arbitrary
   * point — which reads as fluent nonsense rather than an obvious failure.
   */
  it('presents frames chronologically after the ring wraps', async () => {
    // Capture whole windows so ordering can be checked against what was pushed.
    const captured: Float32Array[] = [];
    const classifier: SequenceClassifier = {
      labels: LABELS,
      blankIndex: 0,
      classifySequence(window: Float32Array) {
        captured.push(Float32Array.from(window));
        return Promise.resolve([]);
      },
    };

    const windowFrames = 6;
    const subject = new ContinuousRecognizer(classifier, {
      windowFrames,
      strideFrames: 2,
      featureLength: FEATURE_LENGTH,
    });

    // Vary the index fingertip so each frame normalises to distinct features.
    const handAt = (step: number): ReturnType<typeof openHand> =>
      openHand().map((point, index) =>
        index === 8 ? { ...point, y: point.y - step * 0.01 } : point,
      );

    const pushed: Float32Array[] = [];
    for (let step = 1; step <= 14; step++) {
      const hand = handAt(step);
      pushed.push(normaliseHand(hand, 'right'));
      await subject.push(hand, 'right');
    }

    const lastWindow = captured.at(-1);
    expect(lastWindow).toBeDefined();

    // The window must hold the most recent `windowFrames` pushes, oldest first. If the
    // ring wrapped without reordering, the sequence would be spliced at an arbitrary
    // point — which reads as fluent nonsense rather than an obvious failure.
    const expected = pushed.slice(-windowFrames);
    for (let frame = 0; frame < windowFrames; frame++) {
      for (let i = 0; i < FEATURE_LENGTH; i++) {
        expect(lastWindow![frame * FEATURE_LENGTH + i]).toBeCloseTo(
          expected[frame]![i] as number,
          5,
        );
      }
    }
  });
});
