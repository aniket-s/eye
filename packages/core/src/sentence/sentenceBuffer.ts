/**
 * The text the user has built up from committed signs.
 *
 * Deliberately dumb and synchronous so the UI can treat it as a single source of
 * truth. Subscribers are notified on every mutation.
 */
export type SentenceListener = (text: string) => void;

export class SentenceBuffer {
  #text = '';
  readonly #listeners = new Set<SentenceListener>();

  /** The current text. */
  get text(): string {
    return this.#text;
  }

  /** True when nothing has been entered yet. */
  get isEmpty(): boolean {
    return this.#text.length === 0;
  }

  /** Append a committed letter or word. */
  append(value: string): void {
    if (value.length === 0) return;
    this.#set(this.#text + value);
  }

  /** Append a single space. */
  appendSpace(): void {
    this.#set(this.#text + ' ');
  }

  /** Remove the last character. No-op when already empty. */
  backspace(): void {
    if (this.#text.length === 0) return;
    this.#set(this.#text.slice(0, -1));
  }

  /** Remove everything. */
  clear(): void {
    if (this.#text.length === 0) return;
    this.#set('');
  }

  /**
   * Subscribe to changes.
   *
   * @returns An unsubscribe function.
   */
  subscribe(listener: SentenceListener): () => void {
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  }

  #set(next: string): void {
    this.#text = next;
    for (const listener of this.#listeners) listener(next);
  }
}
