import { describe, expect, it } from 'vitest';
import { IsolatedSignRecognizer, type IsolatedClassifier } from './isolatedSignRecognizer.js';
import { BODY_FEATURE_LENGTH } from '../features/normaliseBody.js';
import { openHand, translate } from '../test/landmarkFactory.js';
import type { VisionFrame } from '../vision/frame.js';

const LABELS = ['hello', 'thank you', 'please', 'sorry'];

/** Classifier the test controls; records the clips it was handed. */
function stubClassifier(scores: number[] = [0.9, 0.05, 0.03, 0.02]): IsolatedClassifier & {
  clips: { frames: number }[];
} {
  const clips: { frames: number }[] = [];
  return {
    labels: LABELS,
    clips,
    classifyClip(_clip: Float32Array, frames: number) {
      clips.push({ frames });
      return Promise.resolve(scores);
    },
  };
}

function frameAt(timestampMs: number, dx: number): VisionFrame {
  return {
    timestampMs,
    hands: [
      { handedness: 'right', handednessScore: 0.95, landmarks: translate(openHand(), dx, 0) },
    ],
    pose: null,
  };
}

/** Simulated signer with persistent position and clock. */
class Signer {
  #x = 0;
  #timeMs = 0;
  readonly results: Awaited<ReturnType<IsolatedSignRecognizer['push']>>[] = [];

  constructor(
    private readonly subject: IsolatedSignRecognizer,
    private readonly fps = 30,
  ) {}

  async move(durationMs: number, speed: number): Promise<this> {
    const step = 1000 / this.fps;
    for (let elapsed = 0; elapsed < durationMs; elapsed += step) {
      this.#x += (speed * step) / 1000;
      this.#timeMs += step;
      this.results.push(await this.subject.push(frameAt(this.#timeMs, this.#x)));
    }
    return this;
  }

  hold(durationMs: number): Promise<this> {
    return this.move(durationMs, 0);
  }

  get recognised(): string[] {
    return this.results.flatMap((r) => (r.recognised ? [r.recognised.label] : []));
  }
}

function recognizer(
  classifier: IsolatedClassifier,
  overrides: Partial<{ maxFrames: number; minProbability: number }> = {},
): IsolatedSignRecognizer {
  return new IsolatedSignRecognizer(classifier, {
    maxFrames: overrides.maxFrames ?? 96,
    minProbability: overrides.minProbability ?? 0.35,
    featureLength: BODY_FEATURE_LENGTH,
    segmenter: { minDurationMs: 300, quietMs: 250 },
  });
}

describe('IsolatedSignRecognizer', () => {
  it('classifies once per completed sign, not per frame', async () => {
    const classifier = stubClassifier();
    const signer = new Signer(recognizer(classifier));
    await signer.move(800, 2.0);
    await signer.hold(700);

    // ~45 frames were pushed; the model ran once. That is the point of segmenting.
    expect(classifier.clips).toHaveLength(1);
    expect(signer.recognised).toEqual(['hello']);
  });

  it('reports capturing while the sign is in progress', async () => {
    const signer = new Signer(recognizer(stubClassifier()));
    await signer.move(600, 2.0);
    expect(signer.results.at(-1)?.capturing).toBe(true);
  });

  it('returns runners-up for a "did you mean" affordance', async () => {
    const classifier = stubClassifier([0.55, 0.3, 0.1, 0.05]);
    const signer = new Signer(recognizer(classifier));
    await signer.move(800, 2.0);
    await signer.hold(700);

    const result = signer.results.find((r) => r.recognised !== null);
    expect(result?.recognised?.label).toBe('hello');
    expect(result?.alternatives.map((a) => a.label)).toEqual(['thank you', 'please', 'sorry']);
  });

  /**
   * A 250-class model is confidently wrong on gestures that are not signs at all —
   * scratching an ear, reaching for a cup. Reporting nothing beats guessing.
   */
  it('reports nothing when no class is above the threshold', async () => {
    const classifier = stubClassifier([0.3, 0.28, 0.22, 0.2]);
    const signer = new Signer(recognizer(classifier, { minProbability: 0.5 }));
    await signer.move(800, 2.0);
    await signer.hold(700);

    expect(signer.recognised).toEqual([]);
    expect(classifier.clips).toHaveLength(1); // it did run, and was rejected
  });

  it('does not classify a movement too brief to be a sign', async () => {
    const classifier = stubClassifier();
    const signer = new Signer(recognizer(classifier));
    await signer.move(150, 3.0);
    await signer.hold(700);

    expect(classifier.clips).toHaveLength(0);
  });

  it('stays quiet while the signer is still', async () => {
    const classifier = stubClassifier();
    const signer = new Signer(recognizer(classifier));
    await signer.hold(2000);

    expect(classifier.clips).toHaveLength(0);
    expect(signer.recognised).toEqual([]);
  });

  /**
   * Long clips are downsampled, not truncated. Cutting the end off would discard the
   * final movement, which in many signs is exactly what distinguishes it from another.
   */
  it('downsamples a long sign to the frame cap instead of truncating it', async () => {
    const classifier = stubClassifier();
    const signer = new Signer(recognizer(classifier, { maxFrames: 20 }));
    await signer.move(1500, 2.0); // ~45 frames
    await signer.hold(700);

    expect(classifier.clips[0]?.frames).toBe(20);
  });

  it('recognises consecutive signs independently', async () => {
    const classifier = stubClassifier();
    const signer = new Signer(recognizer(classifier));
    await signer.move(800, 2.0);
    await signer.hold(700);
    await signer.move(800, 2.0);
    await signer.hold(700);

    expect(classifier.clips).toHaveLength(2);
    expect(signer.recognised).toEqual(['hello', 'hello']);
  });

  it('clears state on reset', async () => {
    const classifier = stubClassifier();
    const subject = recognizer(classifier);
    const signer = new Signer(subject);
    await signer.move(600, 2.0);

    subject.reset();
    await signer.hold(700);
    expect(classifier.clips).toHaveLength(0);
  });
});
