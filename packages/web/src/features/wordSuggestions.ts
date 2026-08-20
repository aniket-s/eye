import {
  Lexicon,
  parseWordList,
  type ConfusionProfile,
  type SentenceBuffer,
} from '@mudrapragyan/core';
import { requireElement } from '../dom.js';

/**
 * Offering real words for what the model spelled.
 *
 * Per-letter accuracy compounds badly: even at 93% a five-letter word is right about 70%
 * of the time. Constraining to real words is what makes fingerspelling usable for words
 * rather than for letters, and it improves every pack automatically — including the ones
 * that have not been trained yet.
 *
 * The word list is fetched lazily and separately (~250 KB, ~80 KB compressed): it is not
 * needed for the app to start, only for the first word to be spelled, and holding up
 * first paint for it would be the wrong trade. Everything stays on the device.
 */

/** Where the word list is served from. */
const LEXICON_URL = `${import.meta.env.BASE_URL}lexicon/en.txt`;

/** How many candidates to show. More than this is a menu, not a suggestion. */
const SUGGESTION_LIMIT = 4;

/** Below this many letters, almost everything is a plausible completion. */
const MIN_LETTERS = 2;

export class WordSuggestions {
  readonly #panel = requireElement('wordPanel');
  readonly #list = requireElement('wordList');
  readonly #status = requireElement('wordStatus');
  readonly #sentence: SentenceBuffer;

  #lexicon: Lexicon | null = null;
  #confusions: ConfusionProfile | undefined;
  #loading: Promise<void> | null = null;

  constructor(sentence: SentenceBuffer) {
    this.#sentence = sentence;
  }

  /**
   * Start loading the word list, and rebuild suggestions whenever the text changes.
   *
   * @param confusions The pack's error profile, so corrections are weighted by the
   *   mistakes this model actually makes rather than by generic edit distance.
   */
  start(confusions?: ConfusionProfile): void {
    this.#confusions = confusions;
    this.#panel.hidden = false;
    this.#sentence.subscribe(() => this.#render());
    this.#list.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const word = target.dataset['word'];
      if (word !== undefined) this.#accept(word);
    });

    // Number keys accept a suggestion. Spelling already occupies both hands, so reaching
    // for a mouse is the expensive part of the interaction.
    document.addEventListener('keydown', (event) => {
      const focused = event.target;
      if (focused instanceof HTMLInputElement || focused instanceof HTMLTextAreaElement) return;
      const index = Number.parseInt(event.key, 10);
      if (!Number.isInteger(index) || index < 1 || index > SUGGESTION_LIMIT) return;

      const button = this.#list.querySelectorAll<HTMLElement>('[data-word]')[index - 1];
      const word = button?.dataset['word'];
      if (word !== undefined) {
        event.preventDefault();
        this.#accept(word);
      }
    });

    void this.#load();
  }

  /**
   * Fetch and index the word list.
   *
   * A failure is reported but not thrown: correction is an enhancement, and losing it
   * should not stop someone spelling letter by letter.
   */
  async #load(): Promise<void> {
    if (this.#loading !== null) return this.#loading;

    this.#loading = (async () => {
      this.#setStatus('Loading word list…');
      try {
        const response = await fetch(LEXICON_URL);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const words = parseWordList(await response.text());
        this.#lexicon = new Lexicon(words, this.#confusions);
        this.#setStatus('');
        this.#render();
      } catch {
        this.#setStatus('Word suggestions unavailable — spelling still works.');
      }
    })();

    return this.#loading;
  }

  #accept(word: string): void {
    // Upper-case to match the letters the model emits, so the sentence stays consistent
    // rather than switching case halfway through.
    this.#sentence.replaceCurrentWord(word.toUpperCase());
  }

  #render(): void {
    if (this.#lexicon === null) return;

    const word = this.#sentence.currentWord;
    this.#list.textContent = '';

    if (word.length < MIN_LETTERS) {
      this.#setStatus(word.length === 0 ? '' : 'Keep spelling…');
      return;
    }

    const suggestions = this.#lexicon.suggest(word, { limit: SUGGESTION_LIMIT });
    if (suggestions.length === 0) {
      this.#setStatus('No matching word.');
      return;
    }

    this.#setStatus('');
    suggestions.forEach((suggestion, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'word-chip';
      button.dataset['word'] = suggestion.word;

      const key = document.createElement('span');
      key.className = 'word-chip-key';
      key.textContent = String(index + 1);

      const label = document.createElement('span');
      label.className = 'word-chip-text';
      label.textContent = suggestion.word.toUpperCase();

      // An exact match is not a correction, and saying so stops the user wondering
      // whether the app has silently changed what they spelled.
      if (suggestion.edits > 0) button.classList.add('is-correction');

      button.append(key, label);
      button.setAttribute(
        'aria-label',
        suggestion.edits > 0
          ? `Correct to ${suggestion.word}, press ${index + 1}`
          : `Accept ${suggestion.word}, press ${index + 1}`,
      );
      this.#list.append(button);
    });
  }

  #setStatus(message: string): void {
    this.#status.textContent = message;
  }
}
