#!/usr/bin/env node
/**
 * Generate the cross-language normalisation fixtures.
 *
 * Runs the **TypeScript** normaliser over a set of hand poses and writes the inputs
 * and outputs to `training/fixtures/normalisation.json`. The Python test suite loads
 * that file and asserts its own implementation reproduces the same numbers.
 *
 * This is the guard against train/inference skew — the failure where a model is
 * fitted on features that differ subtly from the ones production computes, scores
 * well offline, and behaves badly on a real camera. It is silent by nature, so it
 * needs a mechanical check rather than care.
 *
 * Regenerate whenever the normaliser changes, and commit the result:
 *   npm run fixtures:parity
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { normaliseHand } from '../packages/core/dist/features/normalise.js';
import { normaliseBody } from '../packages/core/dist/features/normaliseBody.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT = join(ROOT, 'training/fixtures/normalisation.json');
const BODY_OUTPUT = join(ROOT, 'training/fixtures/normalisation-body.json');

/** Deterministic pseudo-random generator, so fixtures are reproducible. */
function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/** A plausible hand: 21 landmarks scattered around a palm centre. */
function makeHand(random) {
  const centreX = 0.3 + random() * 0.4;
  const centreY = 0.3 + random() * 0.4;
  const size = 0.08 + random() * 0.12;
  return Array.from({ length: 21 }, () => ({
    x: centreX + (random() - 0.5) * size * 2,
    y: centreY + (random() - 0.5) * size * 2,
    z: (random() - 0.5) * 0.05,
  }));
}

const random = makeRandom(20260820);
const cases = [];

for (let i = 0; i < 40; i++) {
  const landmarks = makeHand(random);
  const handedness = i % 2 === 0 ? 'right' : 'left';
  cases.push({
    name: `random-${i}`,
    handedness,
    landmarks: landmarks.map((p) => [p.x, p.y, p.z]),
    expected: Array.from(normaliseHand(landmarks, handedness)),
  });
}

// Edge cases worth pinning explicitly.
const collapsed = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
cases.push({
  name: 'degenerate-collapsed',
  handedness: 'right',
  landmarks: collapsed.map((p) => [p.x, p.y, p.z]),
  expected: Array.from(normaliseHand(collapsed, 'right')),
});

const line = Array.from({ length: 21 }, (_, i) => ({ x: 0.1 + i * 0.02, y: 0.5, z: 0 }));
cases.push({
  name: 'degenerate-collinear',
  handedness: 'left',
  landmarks: line.map((p) => [p.x, p.y, p.z]),
  expected: Array.from(normaliseHand(line, 'left')),
});

await mkdir(dirname(OUTPUT), { recursive: true });
await writeFile(
  OUTPUT,
  `${JSON.stringify(
    {
      scheme: 'centroid-rms-v2',
      generatedBy: 'scripts/generate-parity-fixtures.mjs',
      note: 'Produced by the TypeScript normaliser. Python must reproduce these to 1e-6.',
      cases,
    },
    null,
    2,
  )}\n`,
);

console.log(`Wrote ${cases.length} cases to ${OUTPUT.replace(`${ROOT}/`, '')}`);

// ── Two-handed body-relative fixtures ──

/** Seven upper-body pose points: nose, shoulders, elbows, wrists. */
function makePose(random) {
  const centreX = 0.4 + random() * 0.2;
  const width = 0.18 + random() * 0.12;
  return [
    { x: centreX, y: 0.2 + random() * 0.05, z: 0 },
    { x: centreX - width / 2, y: 0.36, z: 0 },
    { x: centreX + width / 2, y: 0.36, z: 0 },
    { x: centreX - width, y: 0.5, z: 0 },
    { x: centreX + width, y: 0.5, z: 0 },
    { x: centreX - width * 1.1, y: 0.63, z: 0 },
    { x: centreX + width * 1.1, y: 0.63, z: 0 },
  ];
}

const bodyRandom = makeRandom(773311);
const bodyCases = [];

for (let i = 0; i < 30; i++) {
  const dominant = i % 3 === 0 ? 'left' : 'right';
  const hasOther = i % 2 === 0;
  const hasPose = i % 5 !== 0; // exercise the pose-less fallback too

  const hands = [
    {
      handedness: dominant,
      handednessScore: 0.9,
      landmarks: makeHand(bodyRandom),
    },
  ];
  if (hasOther) {
    hands.push({
      handedness: dominant === 'right' ? 'left' : 'right',
      handednessScore: 0.9,
      landmarks: makeHand(bodyRandom),
    });
  }

  const pose = hasPose ? makePose(bodyRandom) : null;
  const frame = { timestampMs: i, hands, pose };

  bodyCases.push({
    name: `body-${i}`,
    dominant,
    dominantHand: hands[0].landmarks.map((p) => [p.x, p.y, p.z]),
    otherHand: hasOther ? hands[1].landmarks.map((p) => [p.x, p.y, p.z]) : null,
    pose: pose === null ? null : pose.map((p) => [p.x, p.y, p.z]),
    expected: Array.from(normaliseBody(frame, dominant)),
  });
}

await writeFile(
  BODY_OUTPUT,
  `${JSON.stringify(
    {
      scheme: 'shoulder-frame-v1',
      generatedBy: 'scripts/generate-parity-fixtures.mjs',
      note: 'Produced by the TypeScript normaliser. Python must reproduce these to 1e-5.',
      cases: bodyCases,
    },
    null,
    2,
  )}\n`,
);

console.log(`Wrote ${bodyCases.length} cases to ${BODY_OUTPUT.replace(`${ROOT}/`, '')}`);
