/**
 * These tests pin v1 behaviour so Phase 3 can replace the state machine safely.
 * Several assert defects rather than desired behaviour; they are labelled DEFECT.
 */
import { describe, expect, it } from 'vitest';
import { JzStateMachine } from './jzStateMachine.js';
import { openHand } from '../test/landmarkFactory.js';
import type { Landmark } from '../types.js';

const CONFIDENT = 0.9;

/** Move the whole hand so a chosen tip traces a path. */
function handAt(dx: number, dy: number): Landmark[] {
  return openHand().map((p) => ({ x: p.x + dx, y: p.y + dy, z: p.z }));
}

describe('JzStateMachine', () => {
  it('starts idle and passes unrelated letters straight through', () => {
    const jz = new JzStateMachine();
    expect(jz.state).toBe('idle');
    expect(jz.update('B', CONFIDENT, openHand())).toBe('B');
  });

  it('opens a tracking window on a confident I', () => {
    const jz = new JzStateMachine();
    jz.update('I', CONFIDENT, openHand());
    expect(jz.state).toBe('tracking_j');
  });

  it('does not open a tracking window below the confidence threshold', () => {
    const jz = new JzStateMachine();
    jz.update('I', 0.5, openHand());
    expect(jz.state).toBe('idle');
  });

  it('confirms J when the pinky tip sweeps downward from an I', () => {
    const jz = new JzStateMachine();
    let last = '';
    for (let i = 0; i < 20; i++) {
      last = jz.update('I', CONFIDENT, handAt(0, i * 0.01)); // 0.19 total descent
    }
    expect(last).toBe('J');
    expect(jz.state).toBe('confirmed_j');
  });

  it('does not confirm J when the hand is still', () => {
    const jz = new JzStateMachine();
    let last = '';
    for (let i = 0; i < 20; i++) last = jz.update('I', CONFIDENT, openHand());
    expect(last).toBe('I');
  });

  it('latches J for subsequent frames once confirmed', () => {
    const jz = new JzStateMachine();
    for (let i = 0; i < 20; i++) jz.update('I', CONFIDENT, handAt(0, i * 0.01));
    // DEFECT: the output is now pinned to J regardless of the actual handshape.
    expect(jz.update('B', CONFIDENT, openHand())).toBe('J');
  });

  it('resets on Y so a Z tracked from a prior D cannot capture it', () => {
    const jz = new JzStateMachine();
    jz.update('D', CONFIDENT, openHand());
    expect(jz.state).toBe('tracking_z');
    expect(jz.update('Y', CONFIDENT, openHand())).toBe('Y');
    expect(jz.state).toBe('idle');
  });

  /**
   * Three strokes: left, back right, left again. Amplitudes are well clear of the
   * 0.04 total-movement floor so the assertion does not sit on a threshold.
   */
  const Z_PATH: readonly number[] = [
    ...Array.from({ length: 6 }, (_, i) => -i * 0.02), //  0    → -0.10
    ...Array.from({ length: 6 }, (_, i) => -0.1 + i * 0.01), // -0.10 → -0.05
    ...Array.from({ length: 6 }, (_, i) => -0.05 - i * 0.02), // -0.05 → -0.15
  ];

  /** Open the tracking window, then trace `path`. The first frame clears history. */
  function traceFrom(seed: 'D' | 'I', path: readonly number[]): string {
    const jz = new JzStateMachine();
    jz.update(seed, CONFIDENT, handAt(0, 0));
    let last = '';
    for (const dx of path) last = jz.update(seed, CONFIDENT, handAt(dx, 0));
    return last;
  }

  it('confirms Z on a left-right-left zig-zag', () => {
    expect(traceFrom('D', Z_PATH)).toBe('Z');
  });

  it('DEFECT: confirms Z on the mirrored zig-zag too, doubling false positives', () => {
    // `isZShape` accepts `fwd || rev`, so the reversed stroke order also qualifies.
    // In practice an arbitrary horizontal wiggle is enough to fire a Z.
    expect(
      traceFrom(
        'D',
        Z_PATH.map((dx) => -dx),
      ),
    ).toBe('Z');
  });

  it('clears history on reset', () => {
    const jz = new JzStateMachine();
    for (let i = 0; i < 15; i++) jz.update('I', CONFIDENT, handAt(0, i * 0.01));
    jz.reset();
    expect(jz.state).toBe('idle');
    expect(jz.isJShape()).toBe(false);
  });

  it('DEFECT: thresholds are frame-counted, so the same gesture depends on frame rate', () => {
    // The same physical motion sampled at half the rate yields too few frames to fire.
    const fast = new JzStateMachine();
    let fastResult = '';
    for (let i = 0; i < 20; i++) fastResult = fast.update('I', CONFIDENT, handAt(0, i * 0.01));

    const slow = new JzStateMachine();
    let slowResult = '';
    for (let i = 0; i < 10; i++) slowResult = slow.update('I', CONFIDENT, handAt(0, i * 0.02));

    expect(fastResult).toBe('J');
    expect(slowResult).toBe('I'); // identical motion, fewer samples, no detection
  });
});
