import type { Vocabularies } from '@mudrapragyan/core';
import { requireElement } from '../dom.js';
import type { RecognitionClient } from '../workers/recognitionClient.js';

/**
 * Switching between letters and numbers.
 *
 * This is a linguistic fact wearing a UI control. In ASL several digits *are* letters —
 * 2 is V, 6 is W, 9 is F, 0 is O — and a signer distinguishes them from context, not from
 * the hand. So the model is trained on handshapes, and this decides what a handshape
 * means.
 *
 * The alternative, one flat classifier over 26 letters and 10 digits, does not merely
 * fail at the digits: it makes the *letters* worse, because probability mass that belongs
 * to V is split between V and 2 and neither clears the margin threshold.
 *
 * Hidden entirely when a pack offers only one reading, which is what a letters-only pack
 * is.
 */

/** Shown first. Letters are what most people open the app to do. */
const DEFAULT_VOCABULARY = 'letters';

export class VocabularyMode {
  readonly #switch = requireElement('modeSwitch');
  #client: RecognitionClient | null = null;
  #active = DEFAULT_VOCABULARY;

  /** Which reading is selected. */
  get active(): string {
    return this.#active;
  }

  /**
   * Show the switch, if this pack has more than one reading to switch between.
   *
   * @returns Whether the control was shown.
   */
  attach(client: RecognitionClient, vocabularies: Vocabularies | undefined): boolean {
    this.#client = client;
    const names = Object.keys(vocabularies ?? {});
    if (names.length < 2) {
      this.#switch.hidden = true;
      return false;
    }

    this.#switch.hidden = false;
    const initial = names.includes(DEFAULT_VOCABULARY) ? DEFAULT_VOCABULARY : (names[0] as string);
    this.#select(initial);

    this.#switch.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const name = target.dataset['vocabulary'];
      if (name !== undefined && names.includes(name)) this.#select(name);
    });

    return true;
  }

  #select(name: string): void {
    this.#active = name;
    for (const button of this.#switch.querySelectorAll<HTMLElement>('[data-vocabulary]')) {
      const chosen = button.dataset['vocabulary'] === name;
      button.classList.toggle('is-active', chosen);
      button.setAttribute('aria-pressed', String(chosen));
    }
    this.#client?.setVocabulary(name);
  }
}
