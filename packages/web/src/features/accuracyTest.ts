/**
 * ⚠️ THIS IS NOT EVALUATION — see docs/AUDIT.md §1.6 (A3).
 *
 * Prompts the user to sign each static letter and records what the model detected.
 * Because the same person, camera, lighting and session produce both the model's
 * training conditions and the "test", the resulting percentage measures nothing that
 * generalises. It is retained in Phase 0 only so behaviour is preserved, and it now
 * carries an on-screen caveat.
 *
 * Real evaluation is signer-independent, offline, and reported per class and per
 * robustness slice; it lands in Phase 2 as a CI gate (docs/AUDIT.md §5).
 */
import { requireElement } from '../dom.js';

/** The 24 letters with a static handshape. J and Z require motion. */
export const STATIC_TEST_LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'.split('');

/** Seconds the user is given to form each letter. */
const SECONDS_PER_LETTER = 3;
/** Pause between letters. */
const GAP_MS = 600;

interface LetterResult {
  readonly target: string;
  readonly detected: string;
  readonly correct: boolean;
}

export class AccuracyTest {
  readonly #modal = requireElement('testModal');
  readonly #phase1 = requireElement('testPhase1');
  readonly #phase2 = requireElement('testPhase2');
  readonly #target = requireElement('testTarget');
  readonly #countdown = requireElement('testCountdown');
  readonly #progress = requireElement('testProgress');
  readonly #live = requireElement('testLive');
  readonly #score = requireElement('testScore');
  readonly #results = requireElement('testResults');

  #active = false;
  #index = 0;
  #results_: LetterResult[] = [];
  #buffer: string[] = [];
  #timer: ReturnType<typeof setInterval> | null = null;
  #gapTimer: ReturnType<typeof setTimeout> | null = null;
  #onReset: (() => void) | null = null;

  /** True while a test is running, so the recogniser knows to feed us letters. */
  get isActive(): boolean {
    return this.#active;
  }

  start(onReset: () => void): void {
    this.#onReset = onReset;
    requireElement('cancelTest').addEventListener('click', () => this.cancel());
    requireElement('closeTest').addEventListener('click', () => this.#hide());
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && this.#active) this.cancel();
    });
  }

  /** Record a detected letter for the letter currently being prompted. */
  observe(letter: string): void {
    if (!this.#active) return;
    this.#buffer.push(letter);
    this.#live.textContent = letter;
  }

  /** Begin a run. */
  begin(): void {
    this.#active = true;
    this.#index = 0;
    this.#results_ = [];
    this.#modal.classList.add('show');
    this.#phase1.hidden = false;
    this.#phase2.hidden = true;
    this.#runLetter();
  }

  cancel(): void {
    this.#clearTimers();
    this.#active = false;
    this.#hide();
  }

  #hide(): void {
    this.#modal.classList.remove('show');
    this.#phase1.hidden = false;
    this.#phase2.hidden = true;
  }

  #clearTimers(): void {
    if (this.#timer !== null) clearInterval(this.#timer);
    if (this.#gapTimer !== null) clearTimeout(this.#gapTimer);
    this.#timer = null;
    this.#gapTimer = null;
  }

  #runLetter(): void {
    if (this.#index >= STATIC_TEST_LETTERS.length) {
      this.#finish();
      return;
    }
    this.#onReset?.();

    const target = STATIC_TEST_LETTERS[this.#index] as string;
    this.#buffer = [];
    let secondsLeft = SECONDS_PER_LETTER;

    this.#target.textContent = target;
    this.#countdown.textContent = String(secondsLeft);
    this.#live.textContent = '—';
    this.#progress.textContent = `${this.#index + 1} / ${STATIC_TEST_LETTERS.length}`;

    this.#timer = setInterval(() => {
      secondsLeft -= 1;
      this.#countdown.textContent = String(secondsLeft);
      if (secondsLeft > 0) return;

      this.#clearTimers();
      const detected = mostFrequent(this.#buffer) ?? '?';
      this.#results_.push({ target, detected, correct: detected === target });
      this.#index += 1;
      this.#gapTimer = setTimeout(() => this.#runLetter(), GAP_MS);
    }, 1000);
  }

  #finish(): void {
    this.#active = false;
    const correct = this.#results_.filter((r) => r.correct).length;
    const total = this.#results_.length;
    const percent = total === 0 ? 0 : Math.round((correct / total) * 100);

    this.#score.textContent = `${correct} / ${total} correct  (${percent}%)`;
    this.#score.style.color = percent >= 80 ? '#4caf50' : percent >= 60 ? '#f0a500' : '#e74c3c';

    this.#results.replaceChildren(
      ...this.#results_.map((result) => {
        const row = document.createElement('div');
        row.style.color = result.correct ? '#4caf50' : '#e74c3c';
        row.textContent = result.correct
          ? `✓ ${result.target}`
          : `✗ ${result.target} → detected ${result.detected}`;
        return row;
      }),
      caveat(),
    );

    this.#phase1.hidden = true;
    this.#phase2.hidden = false;
  }
}

/**
 * An explicit warning that this number does not generalise.
 *
 * Added in Phase 0 because the unqualified percentage was actively misleading.
 */
function caveat(): HTMLElement {
  const note = document.createElement('p');
  note.style.cssText =
    'margin-top:14px;padding-top:10px;border-top:1px solid #2a2a2a;color:#777;font-size:11px;line-height:1.6;';
  note.textContent =
    'This is a self-check, not a benchmark. It measures one person, one camera and one ' +
    'session, so it cannot predict how the model performs for anyone else. Signer-independent ' +
    'evaluation arrives in Phase 2.';
  return note;
}

/** The most common entry, or `null` for an empty list. Ties go to the earliest. */
function mostFrequent(values: readonly string[]): string | null {
  const counts = new Map<string, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);

  let best: string | null = null;
  let bestCount = 0;
  for (const [value, count] of counts) {
    if (count > bestCount) {
      best = value;
      bestCount = count;
    }
  }
  return best;
}
