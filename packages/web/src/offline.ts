/**
 * Service worker registration and offline status.
 *
 * Registered only in production builds: a service worker in development caches the very
 * assets you are trying to change, which turns every edit into a debugging session.
 */

/** Where the worker lives, respecting the deployment base path. */
const WORKER_URL = `${import.meta.env.BASE_URL}sw.js`;

export type OfflineListener = (online: boolean) => void;

/**
 * Register the service worker.
 *
 * Failure is not fatal — the app works online without it — so problems are logged
 * rather than surfaced.
 */
export function registerServiceWorker(): void {
  if (!('serviceWorker' in navigator)) return;
  if (import.meta.env.DEV) return;

  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register(WORKER_URL, { scope: import.meta.env.BASE_URL })
      .catch((error: unknown) => {
        console.warn('Offline support unavailable:', error);
      });
  });
}

/**
 * Observe connectivity.
 *
 * Worth surfacing because the app genuinely still works offline once cached — the user
 * should know that, rather than assuming it is broken.
 *
 * @returns An unsubscribe function.
 */
export function watchConnectivity(listener: OfflineListener): () => void {
  const online = (): void => listener(true);
  const offline = (): void => listener(false);

  window.addEventListener('online', online);
  window.addEventListener('offline', offline);
  listener(navigator.onLine);

  return () => {
    window.removeEventListener('online', online);
    window.removeEventListener('offline', offline);
  };
}
