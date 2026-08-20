import { selectPrimaryHand, type HandLandmarks } from '@mudrapragyan/core';
import { requireElement } from '../dom.js';
import { describeCameraError, isSecureContextForCamera, startCamera } from '../vision/camera.js';
import { VisionLandmarker } from '../vision/landmarker.js';

/**
 * Landmark recording tool.
 *
 * Public datasets are recorded on other people's cameras in other people's rooms.
 * The gap between that and your actual users is the single largest risk in the plan,
 * and the only way to close it is to record locally. This tool exists to make that
 * cheap.
 *
 * **It records landmarks, never video.** 21 numeric points per hand leave nothing
 * identifiable, which is what makes it reasonable to ask someone to contribute
 * samples at all.
 *
 * Output is JSONL, one sample per line, carrying the metadata Phase 2's evaluation
 * needs: the signer id (for signer-independent splits, without which every accuracy
 * number is inflated) and the recording condition (for the per-slice robustness
 * reporting that stops an aggregate score hiding a broken subgroup).
 */

interface Sample {
  readonly label: string;
  readonly signer: string;
  readonly hand: string;
  readonly condition: string;
  readonly timestampMs: number;
  /** Flattened 21 × (x, y, z), rounded to 5 decimals to keep the file small. */
  readonly landmarks: number[];
}

/** Frames to discard after the prompt appears, while the hand is still moving. */
const SETTLE_FRAMES = 12;
/** Minimum spacing between captures, so a burst is not 40 copies of one frame. */
const CAPTURE_INTERVAL_MS = 60;

const elements = {
  video: requireElement<HTMLVideoElement>('video'),
  camStatus: requireElement('camStatus'),
  start: requireElement<HTMLButtonElement>('start'),
  labels: requireElement<HTMLInputElement>('labels'),
  samples: requireElement<HTMLInputElement>('samples'),
  hand: requireElement<HTMLSelectElement>('hand'),
  condition: requireElement<HTMLSelectElement>('condition'),
  signer: requireElement<HTMLInputElement>('signer'),
  prompt: requireElement('prompt'),
  hint: requireElement('hint'),
  record: requireElement<HTMLButtonElement>('record'),
  skip: requireElement<HTMLButtonElement>('skip'),
  download: requireElement<HTMLButtonElement>('download'),
  clear: requireElement<HTMLButtonElement>('clear'),
  status: requireElement('status'),
  counts: requireElement('counts'),
};

const landmarker = new VisionLandmarker();
const collected: Sample[] = [];

let queue: string[] = [];
let current: string | null = null;
let target = 0;
let capturedForCurrent = 0;
let settleRemaining = 0;
let lastCaptureMs = 0;
let latestHand: HandLandmarks | null = null;

function parseLabels(): string[] {
  return elements.labels.value
    .split(',')
    .map((value) => value.trim())
    .filter((value) => value.length > 0);
}

function renderCounts(): void {
  const byLabel = new Map<string, { total: number; conditions: Set<string> }>();
  for (const sample of collected) {
    const entry = byLabel.get(sample.label) ?? { total: 0, conditions: new Set<string>() };
    entry.total++;
    entry.conditions.add(sample.condition);
    byLabel.set(sample.label, entry);
  }

  elements.counts.replaceChildren(
    ...[...byLabel.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([label, entry]) => {
        const row = document.createElement('tr');
        for (const text of [label, String(entry.total), [...entry.conditions].join(', ')]) {
          const cell = document.createElement('td');
          cell.textContent = text;
          row.append(cell);
        }
        return row;
      }),
  );

  elements.status.textContent = `${collected.length} samples collected across ${byLabel.size} labels.`;
  elements.download.disabled = collected.length === 0;
}

function advance(): void {
  const next = queue.shift();
  if (next === undefined) {
    current = null;
    elements.prompt.textContent = '✓';
    elements.hint.textContent = 'All labels recorded. Download, then change condition and repeat.';
    elements.record.disabled = false;
    elements.skip.disabled = true;
    return;
  }

  current = next;
  capturedForCurrent = 0;
  settleRemaining = SETTLE_FRAMES;
  elements.prompt.textContent = next;
  elements.hint.textContent = `Hold the sign — capturing ${target} samples`;
}

function onFrame(video: HTMLVideoElement, timestampMs: number): void {
  if (!landmarker.isReady) return;
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

  const frame = landmarker.detect(video, timestampMs);
  const primary = selectPrimaryHand(frame.hands);
  latestHand = primary?.landmarks ?? null;

  if (current === null || latestHand === null) return;

  // Let the hand settle after the prompt changes, so the first samples are not of a
  // hand still travelling into position.
  if (settleRemaining > 0) {
    settleRemaining--;
    elements.hint.textContent = 'Hold steady…';
    return;
  }
  if (timestampMs - lastCaptureMs < CAPTURE_INTERVAL_MS) return;
  lastCaptureMs = timestampMs;

  const flat: number[] = [];
  for (const point of latestHand) {
    flat.push(round(point.x), round(point.y), round(point.z));
  }

  collected.push({
    label: current,
    signer: elements.signer.value.trim() || 'unknown',
    hand: elements.hand.value,
    condition: elements.condition.value,
    timestampMs: Math.round(timestampMs),
    landmarks: flat,
  });

  capturedForCurrent++;
  elements.hint.textContent = `${capturedForCurrent} / ${target}`;
  renderCounts();

  if (capturedForCurrent >= target) advance();
}

function round(value: number): number {
  return Math.round(value * 1e5) / 1e5;
}

elements.start.addEventListener('click', () => {
  void (async () => {
    if (!isSecureContextForCamera()) {
      elements.camStatus.textContent = 'Camera needs https:// or localhost.';
      return;
    }
    elements.start.disabled = true;
    elements.camStatus.textContent = 'Loading models…';
    try {
      await landmarker.load();
      await startCamera({
        video: elements.video,
        onFrame: ({ video, timestampMs }) => onFrame(video, timestampMs),
      });
      elements.camStatus.textContent = 'Camera running. Landmarks only — no video is stored.';
      elements.record.disabled = false;
    } catch (error) {
      elements.camStatus.textContent =
        error instanceof Error && error.message.includes('setup:assets')
          ? error.message
          : describeCameraError(error);
      elements.start.disabled = false;
    }
  })();
});

elements.record.addEventListener('click', () => {
  const labels = parseLabels();
  if (labels.length === 0) {
    elements.hint.textContent = 'Add at least one label.';
    return;
  }
  target = Math.max(1, Number(elements.samples.value) || 1);
  queue = [...labels];
  elements.record.disabled = true;
  elements.skip.disabled = false;
  advance();
});

elements.skip.addEventListener('click', () => advance());

elements.clear.addEventListener('click', () => {
  collected.length = 0;
  renderCounts();
});

elements.download.addEventListener('click', () => {
  const jsonl = collected.map((sample) => JSON.stringify(sample)).join('\n');
  const blob = new Blob([`${jsonl}\n`], { type: 'application/x-ndjson' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  const signer = elements.signer.value.trim() || 'unknown';
  link.href = url;
  link.download = `landmarks-${signer}-${elements.condition.value}.jsonl`;
  link.click();
  URL.revokeObjectURL(url);
});

renderCounts();
