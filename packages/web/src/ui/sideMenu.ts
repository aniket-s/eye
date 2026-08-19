import { requireElement } from '../dom.js';

/**
 * The slide-out navigation drawer.
 *
 * Adds keyboard dismissal and `aria-expanded` state, neither of which v1 had.
 */
export class SideMenu {
  readonly #menu = requireElement('sideMenu');
  readonly #overlay = requireElement('overlay');
  readonly #toggle = requireElement<HTMLButtonElement>('menuToggle');

  start(): void {
    this.#toggle.addEventListener('click', () => {
      this.#setOpen(!this.isOpen);
    });
    this.#overlay.addEventListener('click', () => this.close());
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && this.isOpen) this.close();
    });
  }

  get isOpen(): boolean {
    return this.#menu.classList.contains('open');
  }

  close(): void {
    this.#setOpen(false);
  }

  #setOpen(open: boolean): void {
    this.#menu.classList.toggle('open', open);
    this.#overlay.classList.toggle('show', open);
    this.#toggle.setAttribute('aria-expanded', String(open));
    this.#toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  }
}
