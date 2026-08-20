import {
  NO_PREDICTION,
  SentenceBuffer,
  selectPrimaryHand,
  TimedHoldCommitDetector,
} from '@mudrapragyan/core';
import { requireElement } from '../dom.js';
import { AccuracyTest } from '../features/accuracyTest.js';
import { speak } from '../speech.js';
import {
  describeCameraError,
  isSecureContextForCamera,
  startCamera,
  type CameraSession,
} from '../vision/camera.js';
import { VisionLandmarker } from '../vision/landmarker.js';
import { RecognitionClient } from '../workers/recognitionClient.js';

const SENTENCE_PLACEHOLDER = 'Detected letters appear here…';
const MODEL_URL = `${import.meta.env.BASE_URL}model_weights.json`;

/**
 * The live translation view.
 *
 * Owns presentation and wiring only. Every recognition decision lives in
 * `@mudrapragyan/core`, and the classifier itself runs in a Web Worker, so a slow
 * inference can never stall the camera or the UI.
 *
 * Frame path: camera → MediaPipe (main thread, needs the video element) → pick the
 * primary hand → post landmarks to the worker → receive a letter → commit on a
 * millisecond timer.
 */
export class TranslatorPage {
  readonly #video = requireElement<HTMLVideoElement>('video');
  readonly #camOverlay = requireElement('camOverlay');
  readonly #camMessage = requireElement('camMessage');
  readonly #startButton = requireElement<HTMLButtonElement>('startCamera');
  readonly #liveBadge = requireElement('liveBadge');
  readonly #modelStatus = requireElement('modelStatus');
  readonly #letterBig = requireElement('letterBig');
  readonly #letterLabel = requireElement('letterLabel');
  readonly #holdBar = requireElement('holdBar');
  readonly #holdPct = requireElement('holdPct');
  readonly #sentenceDisplay = requireElement('sentenceDisplay');
  readonly #debugPanel = requireElement('debugPanel');
  readonly #dbgRaw = requireElement('dbgRaw');
  readonly #dbgFixed = requireElement('dbgFixed');
  readonly #dbgConf = requireElement('dbgConf');
  readonly #dbgPerf = requireElement('dbgPerf');

  readonly #sentence = new SentenceBuffer();
  readonly #hold = new TimedHoldCommitDetector();
  readonly #accuracyTest = new AccuracyTest();
  readonly #landmarker = new VisionLandmarker();

  #recognition: RecognitionClient | null = null;
  #camera: CameraSession | null = null;
  #starting = false;
  #debugVisible = false;
  #handsSeen = 0;

  start(): void {
    this.#accuracyTest.start(() => this.#recognition?.reset());

    this.#startButton.addEventListener('click', () => void this.#startCamera());
    requireElement('addSpace').addEventListener('click', () => this.#sentence.appendSpace());
    requireElement('backspace').addEventListener('click', () => this.#sentence.backspace());
    requireElement('clearSentence').addEventListener('click', () => {
      this.#sentence.clear();
      this.#hold.reset();
    });
    requireElement('speakSentence').addEventListener('click', () => {
      speak(this.#sentence.text);
    });
    requireElement('startAccuracyTest').addEventListener('click', () => {
      if (this.#recognition?.isReady !== true) {
        this.#setStatus('Model is still loading — try again in a moment.', 'error');
        return;
      }
      this.#accuracyTest.begin();
    });

    requireElement('debugToggle').addEventListener('click', () => this.#toggleDebug());
    document.addEventListener('keydown', (event) => {
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return;
      if (event.key === 'd' || event.key === 'D') this.#toggleDebug();
    });

    this.#sentence.subscribe((text) => this.#renderSentence(text));
    this.#renderSentence(this.#sentence.text);

    this.#recognition = new RecognitionClient({
      onReady: (info) => {
        if (info.pipeline === 'words') {
          this.#setStatus(`✅ ${info.packName} — ${info.labelCount} word signs`, 'ready');
          // Word mode segments on motion, so there is no per-frame hold to display.
          this.#holdBar.style.width = '0%';
          this.#holdPct.textContent = '—';
          this.#letterLabel.textContent = 'Sign a word, then pause';
        } else if (info.pipeline === 'ctc') {
          this.#setStatus(`✅ ${info.packName} — continuous, ${info.labelCount} signs`, 'ready');
          // CTC decides commits itself, so the hold bar has nothing to show.
          this.#holdBar.style.width = '0%';
          this.#holdPct.textContent = '—';
          this.#letterLabel.textContent = 'Fingerspell continuously — no need to pause';
        } else if (info.pipeline === 'v2') {
          this.#setStatus(`✅ ${info.packName} — ${info.labelCount} signs`, 'ready');
        } else {
          // The legacy path still runs the correction heuristics. Say so, rather than
          // presenting it as equivalent to a trained pack.
          this.#setStatus(
            `⚠ Legacy model — ${info.labelCount} signs. Train a model pack for accurate results.`,
            'error',
          );
        }
        // Show which pipeline is live immediately. Waiting for the first camera frame
        // would leave the debug panel blank exactly when someone is checking whether
        // their newly installed pack took effect.
        this.#renderPerf();
      },
      onResult: (result) => {
        if (this.#debugVisible) {
          this.#dbgRaw.textContent = result.rawLabel ?? '—';
          this.#dbgFixed.textContent = result.letter === NO_PREDICTION ? '—' : result.letter;
          this.#dbgConf.textContent =
            result.confidence === null
              ? '—'
              : `${(result.confidence * 100).toFixed(1)}% (${result.reason ?? '—'})`;
        }
        if (result.letter !== NO_PREDICTION) this.#accuracyTest.observe(result.letter);
        this.#renderLetter(result.letter, result.timestampMs);
      },
      onContinuous: (result) => {
        this.#renderContinuous(result.committed, result.provisional);
        if (this.#debugVisible) {
          this.#dbgRaw.textContent = result.provisional || '—';
          this.#dbgFixed.textContent = result.committed.slice(-12) || '—';
          this.#dbgConf.textContent = `${(result.confidence * 100).toFixed(0)}% (ctc)`;
        }
      },
      onWord: (result) => {
        this.#renderWord(result);
        if (this.#debugVisible) {
          this.#dbgRaw.textContent = result.capturing ? 'capturing…' : 'idle';
          this.#dbgFixed.textContent = result.sign ?? '—';
          this.#dbgConf.textContent = `${(result.probability * 100).toFixed(0)}% · motion ${result.motion.toFixed(2)}`;
        }
      },
      onError: (message) => this.#setStatus(`⚠ ${message}`, 'error'),
    });
    this.#recognition.init(MODEL_URL);
  }

  /** Release the camera and vision models, e.g. when navigating away. */
  async stopCamera(): Promise<void> {
    this.#camera?.stop();
    this.#camera = null;
    this.#landmarker.close();
    this.#liveBadge.classList.remove('show');
    this.#camOverlay.style.display = '';
    this.#startButton.textContent = '▶ Start Camera';
    await Promise.resolve();
  }

  async #startCamera(): Promise<void> {
    if (this.#starting || this.#camera !== null) return;

    if (!isSecureContextForCamera()) {
      this.#showCameraError('Camera access needs a secure connection (https:// or localhost).');
      return;
    }

    this.#starting = true;
    this.#startButton.disabled = true;

    try {
      this.#startButton.textContent = 'Loading models…';
      if (!this.#landmarker.isReady) await this.#landmarker.load();

      this.#startButton.textContent = 'Starting camera…';
      this.#camera = await startCamera({
        video: this.#video,
        onFrame: ({ video, timestampMs }) => this.#onFrame(video, timestampMs),
      });

      this.#camOverlay.style.display = 'none';
      this.#liveBadge.classList.add('show');
    } catch (error) {
      this.#showCameraError(
        error instanceof Error && error.message.includes('setup:assets')
          ? error.message
          : describeCameraError(error),
      );
    } finally {
      this.#starting = false;
      this.#startButton.disabled = false;
      this.#startButton.textContent = '▶ Start Camera';
    }
  }

  #onFrame(video: HTMLVideoElement, timestampMs: number): void {
    if (this.#recognition === null || !this.#landmarker.isReady) return;
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

    const frame = this.#landmarker.detect(video, timestampMs);
    this.#handsSeen = frame.hands.length;

    const primary = selectPrimaryHand(frame.hands);
    const dominant = primary?.handedness ?? 'right';

    // Word signs are located relative to the body, so word mode needs the whole frame
    // — both hands and pose. Fingerspelling needs one hand, which is a third the payload.
    const submitted =
      this.#recognition.pipeline === 'words'
        ? this.#recognition.submitFrame(frame, dominant)
        : this.#recognition.submit(primary?.landmarks ?? null, frame.timestampMs, dominant);

    // A dropped frame still advances the commit timer, otherwise the hold bar would
    // stall on slow devices even though the user is holding the sign correctly.
    if (!submitted && primary === null) this.#renderLetter(NO_PREDICTION, frame.timestampMs);
    if (this.#debugVisible) this.#renderPerf();
  }

  #showCameraError(message: string): void {
    this.#camMessage.textContent = message;
    this.#camOverlay.style.display = '';
  }

  #renderLetter(letter: string, timestampMs: number): void {
    const { committed, progressPercent } = this.#hold.update(letter, timestampMs);

    if (letter === NO_PREDICTION) {
      this.#letterBig.textContent = NO_PREDICTION;
      this.#letterLabel.textContent = 'Show your hand';
    } else {
      this.#letterBig.textContent = letter;
      this.#letterLabel.textContent = `Detected: ${letter}`;
    }

    this.#holdBar.style.width = `${progressPercent}%`;
    this.#holdPct.textContent = `${progressPercent}%`;

    if (committed !== null) this.#sentence.append(committed);
  }

  #renderPerf(): void {
    const client = this.#recognition;
    if (client === null) return;
    const total = client.processedFrames + client.droppedFrames;
    const dropRate = total === 0 ? 0 : Math.round((client.droppedFrames / total) * 100);
    this.#dbgPerf.textContent =
      `${client.pipeline} · ${this.#handsSeen} hand(s) · ` +
      `pose ${this.#landmarker.hasPose ? 'on' : 'off'} · ` +
      `${client.processedFrames} frames · ${dropRate}% dropped`;
  }

  /**
   * Render committed and provisional text together.
   *
   * The provisional tail is the decoder's live guess. It is shown dimmed rather than
   * withheld, so the latency is visible instead of feeling like lag.
   */
  #renderContinuous(committed: string, provisional: string): void {
    if (committed.length === 0 && provisional.length === 0) {
      this.#sentenceDisplay.textContent = SENTENCE_PLACEHOLDER;
      this.#sentenceDisplay.className = 'sentence-text empty';
      return;
    }

    this.#sentenceDisplay.className = 'sentence-text';
    this.#sentenceDisplay.replaceChildren(document.createTextNode(committed));

    if (provisional.length > 0) {
      const tail = document.createElement('span');
      tail.className = 'sentence-provisional';
      tail.textContent = provisional;
      this.#sentenceDisplay.append(tail);
    }

    this.#letterBig.textContent = provisional.at(-1) ?? committed.at(-1) ?? NO_PREDICTION;
  }

  /**
   * Render a recognised word sign.
   *
   * Runners-up are shown alongside: with hundreds of visually similar signs, an
   * almost-right answer plus alternatives is more useful than a confident wrong one.
   */
  #renderWord(result: {
    capturing: boolean;
    sign: string | null;
    probability: number;
    alternatives: readonly string[];
  }): void {
    if (result.capturing) {
      this.#letterBig.textContent = '•';
      this.#letterLabel.textContent = 'Capturing sign…';
      return;
    }

    if (result.sign === null) {
      this.#letterLabel.textContent = 'Sign a word, then pause';
      return;
    }

    this.#letterBig.textContent = result.sign.replace(/_/g, ' ');
    this.#letterLabel.textContent =
      result.alternatives.length > 0
        ? `${(result.probability * 100).toFixed(0)}% · or ${result.alternatives
            .slice(0, 2)
            .map((a) => a.replace(/_/g, ' '))
            .join(', ')}`
        : `${(result.probability * 100).toFixed(0)}% confident`;

    this.#sentence.append(`${result.sign.replace(/_/g, ' ')} `);
  }

  #renderSentence(text: string): void {
    const isEmpty = text.length === 0;
    this.#sentenceDisplay.textContent = isEmpty ? SENTENCE_PLACEHOLDER : text;
    this.#sentenceDisplay.className = isEmpty ? 'sentence-text empty' : 'sentence-text';
  }

  #toggleDebug(): void {
    this.#debugVisible = !this.#debugVisible;
    this.#debugPanel.hidden = !this.#debugVisible;
  }

  #setStatus(message: string, state: 'ready' | 'error'): void {
    this.#modelStatus.textContent = message;
    this.#modelStatus.className = `model-status ${state}`;
  }
}
