import {
  HoldCommitDetector,
  LegacyRecognizer,
  NO_PREDICTION,
  SentenceBuffer,
  type MlpClassifier,
} from '@mudrapragyan/core';
import { requireElement } from '../dom.js';
import { AccuracyTest } from '../features/accuracyTest.js';
import { speak } from '../speech.js';
import {
  describeCameraError,
  isSecureContextForCamera,
  startHandTracking,
  type HandsResults,
  type HandTrackingSession,
} from '../vision/legacyHands.js';

const SENTENCE_PLACEHOLDER = 'Detected letters appear here…';

/**
 * The live translation view: camera in, letters and a sentence out.
 *
 * Owns only presentation and wiring. Every recognition decision lives in
 * `@mudrapragyan/core`, where it is unit-tested without a browser.
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

  readonly #sentence = new SentenceBuffer();
  readonly #hold = new HoldCommitDetector();
  readonly #accuracyTest = new AccuracyTest();

  #recognizer: LegacyRecognizer | null = null;
  #session: HandTrackingSession | null = null;
  #starting = false;
  #debugVisible = false;

  start(): void {
    this.#accuracyTest.start(() => this.#recognizer?.reset());

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
      if (this.#recognizer === null) {
        this.#setStatus('Model is still loading — try again in a moment.', 'error');
        return;
      }
      this.#accuracyTest.begin();
    });

    requireElement('debugToggle').addEventListener('click', () => this.#toggleDebug());
    document.addEventListener('keydown', (event) => {
      // Ignore the shortcut while the user is typing into a field.
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return;
      if (event.key === 'd' || event.key === 'D') this.#toggleDebug();
    });

    this.#sentence.subscribe((text) => this.#renderSentence(text));
    this.#renderSentence(this.#sentence.text);
  }

  /** Attach the classifier once it has downloaded. */
  setClassifier(classifier: MlpClassifier): void {
    this.#recognizer = new LegacyRecognizer(classifier);
    this.#setStatus(`✅ Model ready — ${classifier.labels.length} signs loaded`, 'ready');
  }

  /** Report that the model could not be loaded. */
  setModelError(message: string): void {
    this.#setStatus(`⚠ ${message}`, 'error');
  }

  /** Release the camera, e.g. when navigating away. */
  async stopCamera(): Promise<void> {
    if (this.#session === null) return;
    const session = this.#session;
    this.#session = null;
    await session.stop();
    this.#liveBadge.classList.remove('show');
    this.#camOverlay.style.display = '';
  }

  async #startCamera(): Promise<void> {
    if (this.#starting || this.#session !== null) return;

    if (!isSecureContextForCamera()) {
      this.#showCameraError('Camera access needs a secure connection (https:// or localhost).');
      return;
    }

    this.#starting = true;
    this.#startButton.disabled = true;
    this.#startButton.textContent = 'Starting…';

    try {
      this.#session = await startHandTracking({
        video: this.#video,
        onResults: (results) => this.#onResults(results),
      });
      this.#camOverlay.style.display = 'none';
      this.#liveBadge.classList.add('show');
    } catch (error) {
      this.#showCameraError(describeCameraError(error));
    } finally {
      this.#starting = false;
      this.#startButton.disabled = false;
      this.#startButton.textContent = '▶ Start Camera';
    }
  }

  #showCameraError(message: string): void {
    this.#camMessage.textContent = message;
    this.#camOverlay.style.display = '';
  }

  #onResults(results: HandsResults): void {
    if (this.#recognizer === null) return;

    const landmarks = results.multiHandLandmarks?.[0] ?? null;
    const result = this.#recognizer.recognise(landmarks);

    if (this.#debugVisible) {
      this.#dbgRaw.textContent = result.rawLabel ?? '—';
      this.#dbgFixed.textContent = result.letter === NO_PREDICTION ? '—' : result.letter;
      this.#dbgConf.textContent =
        result.confidence === null ? '—' : `${(result.confidence * 100).toFixed(1)}%`;
    }

    if (result.letter !== NO_PREDICTION) {
      this.#accuracyTest.observe(result.letter);
    }

    this.#renderLetter(result.letter);
  }

  #renderLetter(letter: string): void {
    const { committed, progressPercent } = this.#hold.update(letter);

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
