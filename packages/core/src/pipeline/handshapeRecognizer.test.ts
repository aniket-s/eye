import { describe, expect, it } from 'vitest';
import { HandshapeRecognizer, type Classifier } from './handshapeRecognizer.js';
import { DEFAULT_THRESHOLDS, NONE_LABEL } from '../decode/rejection.js';
import { FEATURE_LENGTH } from '../features/normalise.js';
import { NO_PREDICTION } from '../types.js';
import { mirror, openHand } from '../test/landmarkFactory.js';

const LABELS = ['A', 'B', NONE_LABEL];

/** A classifier whose output the test controls, so the pipeline is what's under test. */
function stubClassifier(
  responses: number[][],
  seen: Float32Array[] = [],
): Classifier & { seen: Float32Array[] } {
  let index = 0;
  return {
    labels: LABELS,
    seen,
    classify(features: Float32Array) {
      seen.push(Float32Array.from(features));
      const probabilities = responses[Math.min(index, responses.length - 1)] as number[];
      index++;
      return Promise.resolve({
        probabilities,
        logits: probabilities.map((p) => Math.log(Math.max(p, 1e-9)) * 4),
      });
    },
  };
}

function recognizer(classifier: Classifier, window = 1): HandshapeRecognizer {
  return new HandshapeRecognizer(classifier, {
    thresholds: DEFAULT_THRESHOLDS,
    featureLength: FEATURE_LENGTH,
    smoothingWindow: window,
  });
}

describe('HandshapeRecognizer', () => {
  it('returns a letter for a confident prediction', async () => {
    const result = await recognizer(stubClassifier([[0.95, 0.03, 0.02]])).recognise(
      openHand(),
      'right',
    );
    expect(result.letter).toBe('A');
    expect(result.verdict?.accepted).toBe(true);
  });

  it('returns no prediction when no hand is present', async () => {
    const result = await recognizer(stubClassifier([[0.95, 0.03, 0.02]])).recognise(null, 'right');
    expect(result.letter).toBe(NO_PREDICTION);
    expect(result.verdict).toBeNull();
  });

  it('suppresses the none class instead of showing it as a letter', async () => {
    const result = await recognizer(stubClassifier([[0.1, 0.1, 0.8]])).recognise(
      openHand(),
      'right',
    );
    expect(result.letter).toBe(NO_PREDICTION);
    expect(result.verdict?.reason).toBe('none-class');
  });

  it('suppresses a low-confidence prediction', async () => {
    const result = await recognizer(stubClassifier([[0.45, 0.3, 0.25]])).recognise(
      openHand(),
      'right',
    );
    expect(result.letter).toBe(NO_PREDICTION);
    expect(result.verdict?.reason).toBe('low-probability');
  });

  it('feeds the classifier normalised features of the declared length', async () => {
    const classifier = stubClassifier([[0.95, 0.03, 0.02]]);
    await recognizer(classifier).recognise(openHand(), 'right');
    expect(classifier.seen[0]).toHaveLength(FEATURE_LENGTH);
  });

  /**
   * The behavioural proof of the M1 fix. v1 fed a left hand a mirrored vector it had
   * never seen; v2 canonicalises, so both hands reach the model identically.
   */
  it('presents a left hand to the model identically to the mirrored right hand', async () => {
    const rightSeen: Float32Array[] = [];
    const leftSeen: Float32Array[] = [];

    await recognizer(stubClassifier([[0.95, 0.03, 0.02]], rightSeen)).recognise(
      openHand(),
      'right',
    );
    await recognizer(stubClassifier([[0.95, 0.03, 0.02]], leftSeen)).recognise(
      mirror(openHand()),
      'left',
    );

    const right = rightSeen[0]!;
    const left = leftSeen[0]!;
    for (let i = 0; i < right.length; i++) {
      expect(left[i]).toBeCloseTo(right[i] as number, 5);
    }
  });

  it('outvotes a single flickered frame when smoothing', async () => {
    const classifier = stubClassifier([
      [0.95, 0.03, 0.02],
      [0.95, 0.03, 0.02],
      [0.95, 0.03, 0.02],
      [0.02, 0.95, 0.03], // one confidently wrong frame
    ]);
    const subject = recognizer(classifier, 5);

    let result = await subject.recognise(openHand(), 'right');
    result = await subject.recognise(openHand(), 'right');
    result = await subject.recognise(openHand(), 'right');
    result = await subject.recognise(openHand(), 'right');

    // v1 would have shown B here and reset the whole hold.
    expect(result.letter).toBe('A');
  });

  it('clears smoothing history when the hand disappears', async () => {
    const classifier = stubClassifier([[0.95, 0.03, 0.02]]);
    const subject = recognizer(classifier, 5);
    await subject.recognise(openHand(), 'right');
    await subject.recognise(null, 'right');
    const result = await subject.recognise(openHand(), 'right');
    expect(result.letter).toBe('A');
  });
});
