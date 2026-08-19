/**
 * Helpers for synthesising hand landmarks in tests.
 *
 * These build geometrically plausible hands rather than replaying recordings, so
 * unit tests stay readable. Recorded fixtures live in `test/fixtures/` and are used
 * for the golden end-to-end assertions.
 */
import { LANDMARKS_PER_HAND, type Landmark } from '../types.js';

/** A neutral open hand: wrist at origin, fingers extending upward (decreasing y). */
export function openHand(): Landmark[] {
  const hand: Landmark[] = [
    { x: 0.5, y: 0.9, z: 0 }, // 0  wrist
    { x: 0.42, y: 0.86, z: 0 }, // 1  thumb CMC
    { x: 0.37, y: 0.8, z: 0 }, // 2  thumb MCP
    { x: 0.33, y: 0.74, z: 0 }, // 3  thumb IP
    { x: 0.3, y: 0.69, z: 0 }, // 4  thumb tip
    { x: 0.46, y: 0.7, z: 0 }, // 5  index MCP
    { x: 0.45, y: 0.62, z: 0 }, // 6  index PIP
    { x: 0.45, y: 0.56, z: 0 }, // 7  index DIP
    { x: 0.44, y: 0.5, z: 0 }, // 8  index tip
    { x: 0.5, y: 0.69, z: 0 }, // 9  middle MCP
    { x: 0.5, y: 0.6, z: 0 }, // 10 middle PIP
    { x: 0.5, y: 0.54, z: 0 }, // 11 middle DIP
    { x: 0.5, y: 0.48, z: 0 }, // 12 middle tip
    { x: 0.54, y: 0.7, z: 0 }, // 13 ring MCP
    { x: 0.55, y: 0.62, z: 0 }, // 14 ring PIP
    { x: 0.55, y: 0.56, z: 0 }, // 15 ring DIP
    { x: 0.56, y: 0.5, z: 0 }, // 16 ring tip
    { x: 0.58, y: 0.72, z: 0 }, // 17 pinky MCP
    { x: 0.6, y: 0.66, z: 0 }, // 18 pinky PIP
    { x: 0.61, y: 0.61, z: 0 }, // 19 pinky DIP
    { x: 0.62, y: 0.56, z: 0 }, // 20 pinky tip
  ];
  if (hand.length !== LANDMARKS_PER_HAND) {
    throw new Error(`openHand() built ${hand.length} landmarks, expected ${LANDMARKS_PER_HAND}`);
  }
  return hand;
}

/** A closed fist: every fingertip curled below its PIP joint. */
export function closedFist(): Landmark[] {
  const hand = openHand();
  const curl = (tip: number, pip: number): void => {
    hand[tip] = { x: hand[tip]!.x, y: hand[pip]!.y + 0.06, z: hand[tip]!.z };
  };
  curl(8, 6);
  curl(12, 10);
  curl(16, 14);
  curl(20, 18);
  return hand;
}

/** Translate every landmark by a fixed offset. */
export function translate(hand: readonly Landmark[], dx: number, dy: number): Landmark[] {
  return hand.map((p) => ({ x: p.x + dx, y: p.y + dy, z: p.z }));
}

/** Scale every landmark about the wrist. */
export function scale(hand: readonly Landmark[], factor: number): Landmark[] {
  const wrist = hand[0]!;
  return hand.map((p) => ({
    x: wrist.x + (p.x - wrist.x) * factor,
    y: wrist.y + (p.y - wrist.y) * factor,
    z: wrist.z + (p.z - wrist.z) * factor,
  }));
}

/** Mirror horizontally, turning a right hand into a left hand. */
export function mirror(hand: readonly Landmark[]): Landmark[] {
  return hand.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z }));
}

/** Rotate every landmark about the wrist, in the image plane. */
export function rotate(hand: readonly Landmark[], radians: number): Landmark[] {
  const wrist = hand[0]!;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return hand.map((p) => {
    const dx = p.x - wrist.x;
    const dy = p.y - wrist.y;
    return {
      x: wrist.x + dx * cos - dy * sin,
      y: wrist.y + dx * sin + dy * cos,
      z: p.z,
    };
  });
}
