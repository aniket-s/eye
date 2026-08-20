import { NO_PREDICTION, selectPrimaryHand, type HandLandmarks } from '@mudrapragyan/core';
import type { HandObservation } from '@mudrapragyan/core';
import { requireElement } from '../dom.js';
import { describeCameraError, isSecureContextForCamera, startCamera } from '../vision/camera.js';
import { VisionLandmarker } from '../vision/landmarker.js';
import { drawHandSkeleton } from '../vision/skeleton.js';
import { RecognitionClient } from '../workers/recognitionClient.js';
import {
  clearSamples,
  loadSamples,
  saveSample,
  type RecorderSample as Sample,
} from './recorderStore.js';

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
 * Three things keep the trainer oriented while recording, because a silent recorder
 * produces bad datasets: a **skeleton overlay** and state badge show whether the hand
 * is being tracked at all (the old page froze its hint when tracking dropped, and the
 * signer had no way to tell holding wrong from not being seen); a **live model check**
 * shows what the current classifier reads from the held sign, advisory only; and every
 * sample **autosaves to IndexedDB** the moment it is captured, so a refresh or crash
 * costs nothing and the page says exactly what is stored.
 *
 * Output is JSONL, one sample per line, carrying the metadata Phase 2's evaluation
 * needs: the signer id (for signer-independent splits, without which every accuracy
 * number is inflated) and the recording condition (for the per-slice robustness
 * reporting that stops an aggregate score hiding a broken subgroup).
 */

/** Frames to discard after the prompt appears, while the hand is still moving. */
const SETTLE_FRAMES = 12;
/**
 * Capture spacing choices, in milliseconds; a burst must not be 40 copies of one
 * frame. Slower spacing spans more natural hand wobble per label, which is worth
 * more to training than raw sample count.
 */
const CAPTURE_INTERVALS: readonly number[] = [60, 150, 300];
const CAPTURE_INTERVAL_KEY = 'mudrapragyan.recorder.captureMs';
/** Consecutive mismatching frames before the handedness warning shows (~1.5 s). */
const HAND_WARN_FRAMES = 45;
/** Where the classifier fallback lives; the worker prefers a model pack. */
const MODEL_URL = `${import.meta.env.BASE_URL}model_weights.json`;
/** Remembers how many samples the last download covered, across sessions. */
const DOWNLOADED_KEY = 'mudrapragyan.recorder.downloaded';

/**
 * What the recorder is doing right now, rendered as the badge on the video.
 *
 * `waiting` is the state that motivated all of this: a run is active but no hand is
 * being tracked, which used to look identical to everything working.
 */
type RecorderState = 'off' | 'ready' | 'waiting' | 'settling' | 'capturing' | 'paused' | 'done';

const STATE_LABEL: Record<RecorderState, string> = {
  off: 'Camera off',
  ready: 'Tracking',
  waiting: 'No hand detected',
  settling: 'Get ready…',
  capturing: '● REC',
  paused: 'Paused',
  done: 'Run complete',
};

const elements = {
  video: requireElement<HTMLVideoElement>('video'),
  overlay: requireElement<HTMLCanvasElement>('overlay'),
  recState: requireElement('recState'),
  camStatus: requireElement('camStatus'),
  start: requireElement<HTMLButtonElement>('start'),
  labels: requireElement<HTMLInputElement>('labels'),
  samples: requireElement<HTMLInputElement>('samples'),
  hand: requireElement<HTMLSelectElement>('hand'),
  condition: requireElement<HTMLSelectElement>('condition'),
  signer: requireElement<HTMLInputElement>('signer'),
  refArt: requireElement<HTMLImageElement>('refArt'),
  prompt: requireElement('prompt'),
  hint: requireElement('hint'),
  barFill: requireElement('barFill'),
  livePanel: requireElement('livePanel'),
  liveRead: requireElement('liveRead'),
  liveVerdict: requireElement('liveVerdict'),
  liveModel: requireElement('liveModel'),
  handWarn: requireElement('handWarn'),
  record: requireElement<HTMLButtonElement>('record'),
  pause: requireElement<HTMLButtonElement>('pause'),
  speedChips: requireElement('speedChips'),
  skip: requireElement<HTMLButtonElement>('skip'),
  download: requireElement<HTMLButtonElement>('download'),
  clear: requireElement<HTMLButtonElement>('clear'),
  status: requireElement('status'),
  storeStatus: requireElement('storeStatus'),
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

let state: RecorderState = 'off';
let paused = false;
let captureIntervalMs = readStoredCaptureInterval();
let recognition: RecognitionClient | null = null;
let liveCheckUsable = false;
let handMismatchFrames = 0;
let downloadedCount = 0;
let autosaveBroken = false;
let clearArmTimer: number | null = null;

function readStoredCaptureInterval(): number {
  try {
    const stored = Number(localStorage.getItem(CAPTURE_INTERVAL_KEY));
    if (CAPTURE_INTERVALS.includes(stored)) return stored;
  } catch {
    // Storage unavailable; use the default.
  }
  return CAPTURE_INTERVALS[0] as number;
}

function parseLabels(): string[] {
  return elements.labels.value
    .split(',')
    .map((value) => value.trim())
    .filter((value) => value.length > 0);
}

function setState(next: RecorderState): void {
  if (state === next) return;
  state = next;
  elements.recState.dataset['state'] = next;
  elements.recState.textContent = STATE_LABEL[next];

  if (next === 'waiting') {
    elements.hint.textContent = 'No hand detected — bring your hand into view.';
  } else if (next === 'paused') {
    elements.hint.textContent = 'Paused — press Resume to continue capturing.';
  } else if (next === 'settling') {
    elements.hint.textContent = 'Hold steady…';
  }
}

function renderProgress(): void {
  const percent = target === 0 ? 0 : Math.min(100, Math.round((capturedForCurrent / target) * 100));
  elements.barFill.style.width = `${percent}%`;
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

  const pending = collected.length - downloadedCount;
  elements.status.textContent =
    collected.length === 0
      ? '0 samples collected.'
      : `${collected.length} samples collected across ${byLabel.size} labels.` +
        (pending > 0 ? ` ${pending} not yet downloaded.` : ' All downloaded.');
  elements.download.disabled = collected.length === 0;
}

/** The dictionary already has a diagram for every letter and digit — show it. */
function renderReferenceArt(label: string | null): void {
  let src: string | null = null;
  if (label !== null && /^[a-z]$/i.test(label)) {
    src = `${import.meta.env.BASE_URL}alphabets/${label.toLowerCase()}.svg`;
  } else if (label !== null && /^(10|[0-9])$/.test(label)) {
    src = `${import.meta.env.BASE_URL}numbers/${label}.svg`;
  }

  if (src === null) {
    elements.refArt.hidden = true;
    return;
  }
  elements.refArt.src = src;
  elements.refArt.alt = `Reference diagram for ${label ?? ''}`;
  elements.refArt.hidden = false;
}

function advance(): void {
  const next = queue.shift();
  if (next === undefined) {
    current = null;
    paused = false;
    elements.pause.disabled = true;
    elements.pause.textContent = '⏸ Pause';
    setState('done');
    elements.prompt.textContent = '✓';
    elements.hint.textContent = 'All labels recorded. Download, then change condition and repeat.';
    elements.record.disabled = false;
    elements.skip.disabled = true;
    elements.barFill.style.width = '100%';
    renderReferenceArt(null);
    return;
  }

  current = next;
  capturedForCurrent = 0;
  settleRemaining = SETTLE_FRAMES;
  elements.prompt.textContent = next;
  elements.hint.textContent = `Hold the sign — capturing ${target} samples`;
  renderReferenceArt(next);
  renderProgress();
}

/** Skeleton overlay: the trainer can see exactly what the landmarker sees, or doesn't. */
function drawOverlay(hands: readonly HandObservation[], primary: HandObservation | null): void {
  const accent = state === 'capturing' ? '#22c55e' : state === 'settling' ? '#f0a500' : '#00d4c8';
  drawHandSkeleton(elements.overlay, elements.video, hands, primary, accent);
}

/**
 * Warn when the tracked hand is not the one the samples are being tagged with.
 *
 * Handedness metadata feeds the signer-independent evaluation; recording a left hand
 * into a file that claims `right` poisons the dataset in a way nothing downstream can
 * detect. `landmarker.ts` already corrects MediaPipe's mirrored labels, so a sustained
 * disagreement here is real, not jitter.
 */
function renderHandWarning(primary: HandObservation | null): void {
  if (primary === null) {
    handMismatchFrames = 0;
  } else if (primary.handedness !== elements.hand.value) {
    handMismatchFrames++;
  } else {
    handMismatchFrames = 0;
  }

  const show = handMismatchFrames >= HAND_WARN_FRAMES;
  if (show && elements.handWarn.hidden) {
    elements.handWarn.textContent =
      `Tracking your ${primary?.handedness ?? ''} hand, but samples are tagged ` +
      `“${elements.hand.value}”. Fix the Hand dropdown or switch hands — mislabelled ` +
      'handedness poisons the dataset.';
    elements.handWarn.hidden = false;
  } else if (!show && !elements.handWarn.hidden) {
    elements.handWarn.hidden = true;
  }
}

/**
 * Live read of the held sign, so the trainer knows in real time whether the sign they
 * are making reads as the letter they were prompted for.
 *
 * Advisory by design: capture never waits for the model, because recording exists
 * precisely to fix the letters the model is weak at, and gating on its agreement would
 * filter out the examples that matter most.
 */
function renderLiveRead(letter: string, confidence: number | null): void {
  const seesNothing = letter === NO_PREDICTION;
  elements.liveRead.textContent = seesNothing
    ? 'nothing'
    : confidence === null
      ? letter
      : `${letter} · ${Math.round(confidence * 100)}%`;

  if (current === null) {
    elements.liveVerdict.textContent = '';
    return;
  }

  const matches =
    current.toLowerCase() === 'none'
      ? seesNothing
      : !seesNothing && letter.toLowerCase() === current.toLowerCase();
  elements.liveVerdict.textContent = matches ? '✓ matches the prompt' : `✗ expected ${current}`;
  elements.liveVerdict.className = matches ? 'ok' : 'bad';
}

/**
 * Classify frames in the recognition worker, purely as feedback for the trainer.
 *
 * Uses whatever the worker settles on — the shipped pack normally, the legacy v1
 * model on a bare clone — and says which, since their reads deserve different trust.
 */
function startLiveCheck(): void {
  recognition = new RecognitionClient({
    onReady: (info) => {
      elements.livePanel.hidden = false;
      if (info.pipeline === 'v2' || info.pipeline === 'ctc') {
        liveCheckUsable = true;
        elements.liveModel.textContent = `Checked against ${info.packName ?? 'the installed pack'}.`;
      } else if (info.pipeline === 'v1') {
        liveCheckUsable = true;
        elements.liveModel.textContent =
          'Checked against the legacy v1 model — unreliable; its misreads are why you are recording.';
      } else {
        elements.liveModel.textContent = 'Live check is unavailable for word-sign packs.';
      }
    },
    onResult: (result) => {
      renderLiveRead(result.letter, result.confidence);
    },
    onContinuous: (result) => {
      const tail = result.provisional.at(-1) ?? result.committed.at(-1) ?? NO_PREDICTION;
      renderLiveRead(tail, result.confidence);
    },
    onError: (message) => {
      elements.livePanel.hidden = false;
      liveCheckUsable = false;
      elements.liveModel.textContent = `Live check unavailable: ${message}`;
    },
  });
  recognition.init(MODEL_URL);
}

function onFrame(video: HTMLVideoElement, timestampMs: number): void {
  if (!landmarker.isReady) return;
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

  const frame = landmarker.detect(video, timestampMs);
  const primary = selectPrimaryHand(frame.hands);
  latestHand = primary?.landmarks ?? null;

  renderHandWarning(primary);
  if (liveCheckUsable) {
    recognition?.submit(latestHand, timestampMs, primary?.handedness ?? 'right');
  }
  if (primary === null && !elements.livePanel.hidden) {
    renderLiveRead(NO_PREDICTION, null);
  }

  if (current === null) {
    if (state !== 'off' && state !== 'done') setState('ready');
  } else if (paused) {
    setState('paused');
  } else if (latestHand === null) {
    setState('waiting');
  } else if (settleRemaining > 0) {
    setState('settling');
    settleRemaining--;
  } else {
    setState('capturing');
    if (timestampMs - lastCaptureMs >= captureIntervalMs) {
      lastCaptureMs = timestampMs;
      capture(timestampMs);
    }
  }

  drawOverlay(frame.hands, primary);
}

function capture(timestampMs: number): void {
  if (current === null || latestHand === null) return;

  const flat: number[] = [];
  for (const point of latestHand) {
    flat.push(round(point.x), round(point.y), round(point.z));
  }

  const sample: Sample = {
    label: current,
    signer: elements.signer.value.trim() || 'unknown',
    hand: elements.hand.value,
    condition: elements.condition.value,
    timestampMs: Math.round(timestampMs),
    landmarks: flat,
  };

  collected.push(sample);
  persist(sample);

  capturedForCurrent++;
  elements.hint.textContent = `${capturedForCurrent} / ${target}`;
  renderProgress();
  renderCounts();

  if (capturedForCurrent >= target) advance();
}

/**
 * Autosave, with a visible failure.
 *
 * A trainer who fills the storage quota mid-session must find out while there is
 * still time to download, not after the refresh that would have needed the autosave.
 */
function persist(sample: Sample): void {
  if (autosaveBroken) return;
  void saveSample(sample).catch((error: unknown) => {
    autosaveBroken = true;
    elements.storeStatus.textContent =
      '⚠ Autosave failed — samples now live only in this page. Download soon. ' +
      `(${error instanceof Error ? error.message : 'storage unavailable'})`;
  });
}

function round(value: number): number {
  return Math.round(value * 1e5) / 1e5;
}

function readDownloadedCount(): number {
  try {
    return Math.max(0, Number(localStorage.getItem(DOWNLOADED_KEY)) || 0);
  } catch {
    return 0;
  }
}

function writeDownloadedCount(count: number): void {
  try {
    localStorage.setItem(DOWNLOADED_KEY, String(count));
  } catch {
    // Private mode; the "not yet downloaded" tally simply will not persist.
  }
}

/** Bring back anything a previous session autosaved, and say so. */
async function restore(): Promise<void> {
  const stored = await loadSamples();
  downloadedCount = Math.min(readDownloadedCount(), stored.length);
  if (stored.length > 0) {
    collected.push(...stored);
    renderCounts();
    elements.storeStatus.textContent =
      `Restored ${stored.length} autosaved samples from a previous session. ` +
      'Download includes them; Clear deletes them.';
  } else {
    elements.storeStatus.textContent =
      'Autosave is on — samples persist in this browser until you clear them.';
  }
}

elements.refArt.addEventListener('error', () => {
  elements.refArt.hidden = true;
});

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
      setState('ready');
      startLiveCheck();
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
  paused = false;
  elements.record.disabled = true;
  elements.pause.disabled = false;
  elements.pause.textContent = '⏸ Pause';
  elements.skip.disabled = false;
  advance();
});

/**
 * Pause capturing without losing the run. Resuming re-settles for a few frames, so
 * the first samples after a break are of a hand back in position, not travelling.
 */
elements.pause.addEventListener('click', () => {
  paused = !paused;
  elements.pause.textContent = paused ? '▶ Resume' : '⏸ Pause';
  if (!paused) settleRemaining = SETTLE_FRAMES;
});

elements.speedChips.addEventListener('click', (event) => {
  if (!(event.target instanceof HTMLElement)) return;
  const interval = Number(event.target.dataset['interval']);
  if (!CAPTURE_INTERVALS.includes(interval)) return;
  captureIntervalMs = interval;
  renderSpeedChips();
  try {
    localStorage.setItem(CAPTURE_INTERVAL_KEY, String(interval));
  } catch {
    // Private mode: the choice simply will not persist.
  }
});

function renderSpeedChips(): void {
  for (const chip of elements.speedChips.querySelectorAll<HTMLElement>('[data-interval]')) {
    chip.classList.toggle('is-active', Number(chip.dataset['interval']) === captureIntervalMs);
  }
}

elements.skip.addEventListener('click', () => advance());

/**
 * Two clicks to clear, so one slip cannot delete a session.
 *
 * The first click arms the button and names what would be deleted; it disarms itself
 * after a moment. Both the in-page samples and the autosave are cleared together —
 * anything else would leave the page and the storage disagreeing about what exists.
 */
elements.clear.addEventListener('click', () => {
  if (collected.length === 0) return;

  if (clearArmTimer === null) {
    elements.clear.textContent = `Really delete ${collected.length}?`;
    clearArmTimer = window.setTimeout(() => {
      clearArmTimer = null;
      elements.clear.textContent = 'Clear';
    }, 4000);
    return;
  }

  window.clearTimeout(clearArmTimer);
  clearArmTimer = null;
  elements.clear.textContent = 'Clear';

  collected.length = 0;
  downloadedCount = 0;
  writeDownloadedCount(0);
  autosaveBroken = false;
  void clearSamples().catch(() => {
    elements.storeStatus.textContent = '⚠ Could not empty the local autosave.';
  });
  renderCounts();
  elements.storeStatus.textContent = 'Cleared — the local autosave is empty.';
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

  downloadedCount = collected.length;
  writeDownloadedCount(downloadedCount);
  renderCounts();
});

renderCounts();
renderSpeedChips();
void restore();
