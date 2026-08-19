import { queryAll, requireElement } from './dom.js';

/** The three top-level views. */
export const PAGES = ['home', 'translator', 'dictionary'] as const;
export type PageName = (typeof PAGES)[number];

export function isPageName(value: string): value is PageName {
  return (PAGES as readonly string[]).includes(value);
}

type PageListener = (page: PageName) => void;

/**
 * Show-one-hide-the-rest navigation, wired to the URL hash.
 *
 * v1 toggled pages through a global `showPage()` and inline `onclick` attributes.
 * Reading the page from the hash means the browser Back button works and a view can
 * be linked to directly — both were broken before.
 */
export class Router {
  #current: PageName = 'home';
  readonly #listeners = new Set<PageListener>();

  /** Current page. */
  get current(): PageName {
    return this.#current;
  }

  /**
   * Wire up navigation.
   *
   * Any element carrying `data-page="…"` becomes a navigation control, which removes
   * the need for global `onclick` handlers on `window`.
   */
  start(): void {
    for (const element of queryAll('[data-page]')) {
      element.addEventListener('click', (event) => {
        event.preventDefault();
        const target = element.dataset['page'];
        if (target !== undefined && isPageName(target)) this.navigate(target);
      });
      // Cards are `role="button"`, so they must also respond to Enter and Space.
      if (element.getAttribute('role') === 'button') {
        element.addEventListener('keydown', (event) => {
          if (event.key !== 'Enter' && event.key !== ' ') return;
          event.preventDefault();
          const target = element.dataset['page'];
          if (target !== undefined && isPageName(target)) this.navigate(target);
        });
      }
    }

    window.addEventListener('hashchange', () => this.#applyHash());
    this.#applyHash();
  }

  /** Switch to a page and update the hash. */
  navigate(page: PageName): void {
    if (window.location.hash !== `#${page}`) {
      window.location.hash = page;
      return; // `hashchange` re-enters and applies the change.
    }
    this.#show(page);
  }

  /** Observe page changes. Returns an unsubscribe function. */
  subscribe(listener: PageListener): () => void {
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  }

  #applyHash(): void {
    const raw = window.location.hash.replace(/^#/, '');
    this.#show(isPageName(raw) ? raw : 'home');
  }

  #show(page: PageName): void {
    for (const name of PAGES) {
      requireElement(name).classList.toggle('active', name === page);
    }
    for (const link of queryAll('.side-menu a')) {
      link.classList.toggle('active', link.dataset['page'] === page);
    }
    this.#current = page;
    for (const listener of this.#listeners) listener(page);
  }
}
