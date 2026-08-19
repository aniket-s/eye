/**
 * Typed DOM lookups that fail loudly.
 *
 * The v1 code called `document.getElementById(...)` inline everywhere and trusted the
 * result. A renamed or removed element produced a `null` dereference deep inside a
 * per-frame callback, which is difficult to trace. Resolving elements once at startup
 * turns that into a clear error before the camera ever opens.
 */

/**
 * Find an element by id.
 *
 * @throws Error naming the missing id, so a template change surfaces immediately.
 */
export function requireElement<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) {
    throw new Error(`Missing required element #${id}`);
  }
  return element as T;
}

/** Find all elements matching a selector, typed. */
export function queryAll<T extends HTMLElement = HTMLElement>(selector: string): T[] {
  return Array.from(document.querySelectorAll<T>(selector));
}

/** Attach a click handler and return a disposer. */
export function onClick(element: HTMLElement, handler: (event: MouseEvent) => void): () => void {
  element.addEventListener('click', handler as EventListener);
  return () => element.removeEventListener('click', handler as EventListener);
}
