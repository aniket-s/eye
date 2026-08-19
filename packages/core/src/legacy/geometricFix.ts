/**
 * ⚠️ LEGACY — DELETED IN PHASE 2. See ./README.md before touching this file.
 *
 * Hand-tuned geometric overrides that rewrite the v1 model's output for 20 of 26
 * letters. Ported verbatim from the original `index.html` so Phase 0 changes no
 * behaviour. Constants, branch order, and comparison operators are preserved
 * exactly, including the mixed scale conventions and the unreachable branches.
 *
 * Do not extend this. Do not fix its inconsistencies — a correctly-normalised
 * classifier with a `none` class makes all of it unnecessary.
 */
import { HandLandmarkIndex as L, type HandLandmarks } from '../types.js';
import { MIN_SCALE } from '../math/geometry.js';

/**
 * Rewrite a predicted letter using palm-relative geometry.
 *
 * @param letter The v1 model's `argmax` label.
 * @param landmarks The 21 hand landmarks the prediction came from.
 * @returns Either `letter` unchanged, or the letter the heuristics prefer.
 */
export function geometricFix(letter: string, landmarks: HandLandmarks): string {
  const p = landmarks;

  // NOTE: 2D palm scale. `landmarksToVector` uses the 3D distance for the same
  // conceptual quantity. This discrepancy is inherited from v1 and is one reason
  // the thresholds below do not transfer between users.
  const wrist = p[L.WRIST]!;
  const middleMcp = p[L.MIDDLE_MCP]!;
  const sc = Math.max(Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y), MIN_SCALE);

  /** True when `tip` sits clearly above `pip` (smaller y is higher on screen). */
  const tipAbovePip = (tip: number, pip: number): boolean => p[tip]!.y < p[pip]!.y - 0.06 * sc;

  /** Palm-relative 2D distance between two landmarks. */
  const dist = (a: number, b: number): number =>
    Math.hypot(p[a]!.x - p[b]!.x, p[a]!.y - p[b]!.y) / sc;

  // F / B / W
  if (letter === 'F' || letter === 'B' || letter === 'W') {
    if (!tipAbovePip(L.INDEX_TIP, L.INDEX_PIP)) return 'F';
    return tipAbovePip(L.PINKY_TIP, L.PINKY_PIP) ? 'B' : 'W';
  }

  // E / O — also rescues F (middle/ring/pinky clearly up) and P (index pointing down)
  if (letter === 'E' || letter === 'O') {
    if (p[L.INDEX_TIP]!.y > p[L.INDEX_MCP]!.y + 0.03 * sc) return 'P';
    const othersUp =
      p[L.MIDDLE_TIP]!.y < p[L.MIDDLE_PIP]!.y - 0.02 * sc &&
      p[L.RING_TIP]!.y < p[L.RING_PIP]!.y - 0.02 * sc &&
      p[L.PINKY_TIP]!.y < p[L.PINKY_PIP]!.y - 0.02 * sc;
    if (othersUp && !tipAbovePip(L.INDEX_TIP, L.INDEX_PIP)) return 'F';
    return dist(L.INDEX_TIP, L.THUMB_TIP) < 0.2 ? 'O' : 'E';
  }

  // D / X / C — also rescues L (thumb extended far) and X (tip not clearly above DIP)
  if (letter === 'D' || letter === 'X' || letter === 'C') {
    if (!tipAbovePip(L.INDEX_TIP, L.INDEX_PIP)) return 'C';
    if (dist(L.THUMB_TIP, L.MIDDLE_MCP) > 0.65) return 'L';
    return p[L.INDEX_TIP]!.y < p[L.INDEX_DIP]!.y - 0.05 * sc ? 'D' : 'X';
  }

  // G / H
  if (letter === 'G' || letter === 'H') {
    return tipAbovePip(L.MIDDLE_TIP, L.MIDDLE_PIP) ? 'H' : 'G';
  }

  // S / I / K / V / Z — looser tip-up threshold; Z included to rescue Y
  if (letter === 'S' || letter === 'I' || letter === 'K' || letter === 'V' || letter === 'Z') {
    const tipUp = (tip: number, pip: number): boolean => p[tip]!.y < p[pip]!.y - 0.02 * sc;
    if (tipUp(L.INDEX_TIP, L.INDEX_PIP) && tipUp(L.MIDDLE_TIP, L.MIDDLE_PIP)) {
      return dist(L.INDEX_TIP, L.MIDDLE_TIP) > 0.4 ? 'V' : 'K';
    }
    if (tipUp(L.PINKY_TIP, L.PINKY_PIP) && !tipUp(L.INDEX_TIP, L.INDEX_PIP)) {
      return dist(L.THUMB_TIP, L.INDEX_MCP) > 0.45 ? 'Y' : 'I';
    }
    const thumbToIndex = dist(L.THUMB_TIP, L.INDEX_MCP);
    const thumbToMiddle = dist(L.THUMB_TIP, L.MIDDLE_MCP);
    const thumbToRing = dist(L.THUMB_TIP, L.RING_MCP);
    if (Math.min(thumbToIndex, thumbToMiddle, thumbToRing) < 0.32) {
      if (thumbToIndex <= thumbToMiddle && thumbToIndex <= thumbToRing) return 'T';
      if (thumbToMiddle <= thumbToRing) return 'N';
    }
    return 'S';
  }

  // M — rescues N (ring tip clearly higher than middle tip)
  if (letter === 'M') {
    if (tipAbovePip(L.INDEX_TIP, L.INDEX_PIP)) return 'K';
    return p[L.RING_TIP]!.y < p[L.MIDDLE_TIP]!.y - 0.03 * sc ? 'N' : 'M';
  }

  // N / U
  if (letter === 'N' || letter === 'U') {
    if (tipAbovePip(L.INDEX_TIP, L.INDEX_PIP) && tipAbovePip(L.MIDDLE_TIP, L.MIDDLE_PIP))
      return 'U';
    if (tipAbovePip(L.INDEX_TIP, L.INDEX_PIP)) return 'K';
    return 'N';
  }

  // T / A — rescues B (all four fingers loosely up) and N (thumb near middle MCP)
  if (letter === 'T' || letter === 'A') {
    if (
      p[L.INDEX_TIP]!.y < p[L.INDEX_PIP]!.y &&
      p[L.MIDDLE_TIP]!.y < p[L.MIDDLE_PIP]!.y &&
      p[L.RING_TIP]!.y < p[L.RING_PIP]!.y &&
      p[L.PINKY_TIP]!.y < p[L.PINKY_PIP]!.y
    ) {
      return 'B';
    }
    if (p[L.THUMB_TIP]!.y < p[L.INDEX_MCP]!.y - 0.05 * sc) return 'A';
    const thumbToIndex = dist(L.THUMB_TIP, L.INDEX_MCP);
    const thumbToMiddle = dist(L.THUMB_TIP, L.MIDDLE_MCP);
    const thumbToRing = dist(L.THUMB_TIP, L.RING_MCP);
    if (thumbToMiddle < thumbToIndex && thumbToMiddle < thumbToRing && thumbToMiddle < 0.3) {
      return 'N';
    }
    const minFingerX = Math.min(p[L.INDEX_MCP]!.x, p[L.MIDDLE_MCP]!.x);
    const maxFingerX = Math.max(p[L.INDEX_MCP]!.x, p[L.MIDDLE_MCP]!.x);
    return p[L.THUMB_TIP]!.x > minFingerX - 0.08 * sc && p[L.THUMB_TIP]!.x < maxFingerX + 0.08 * sc
      ? 'T'
      : 'A';
  }

  // P / L / Q / R — rescues H (both fingers sideways) and M (thumb tucked)
  if (letter === 'P' || letter === 'L' || letter === 'Q' || letter === 'R') {
    if (tipAbovePip(L.INDEX_TIP, L.INDEX_PIP) && tipAbovePip(L.MIDDLE_TIP, L.MIDDLE_PIP))
      return 'R';
    const indexSignificantlyUp = p[L.INDEX_TIP]!.y < p[L.INDEX_MCP]!.y - 0.07 * sc;
    const indexDown = p[L.INDEX_TIP]!.y > p[L.INDEX_MCP]!.y;
    const middleSignificantlyDown = p[L.MIDDLE_TIP]!.y > p[L.MIDDLE_MCP]!.y + 0.11 * sc;
    if (
      !indexSignificantlyUp &&
      !indexDown &&
      dist(L.INDEX_TIP, L.INDEX_MCP) > 0.5 &&
      dist(L.MIDDLE_TIP, L.MIDDLE_MCP) > 0.4
    ) {
      return 'H';
    }
    if (indexSignificantlyUp) return dist(L.THUMB_TIP, L.RING_MCP) < 0.35 ? 'M' : 'L';
    if (indexDown && middleSignificantlyDown) return 'P';
    if (indexDown) return 'Q';
    if (dist(L.THUMB_TIP, L.RING_MCP) < 0.35) return 'M';
    return letter;
  }

  return letter;
}
