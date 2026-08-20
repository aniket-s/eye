/**
 * Service worker: makes the app work offline.
 *
 * Offline support matters here beyond convenience. The whole architecture puts
 * inference on the device precisely so nothing depends on a network, and until now the
 * *app itself* still did — a contradiction worth closing. It also means the tool keeps
 * working on a patchy connection, which is exactly when someone might need it.
 *
 * Two strategies, chosen per request:
 *
 * - **Cache-first** for anything content-addressed or immutable: hashed build assets,
 *   the MediaPipe and ONNX Runtime WASM, model weights, fonts, dictionary diagrams. These
 *   never change under a given URL, so serving from cache is both correct and instant.
 *   They are also the bulk of the ~25 MB payload, which is what makes a repeat visit
 *   feel different from a first one.
 * - **Network-first** for navigation, pack manifests, and anything else: the user should
 *   get a fresh copy when online, and the cached one when not.
 *
 * A **pack manifest is deliberately not cache-first**, even though it sits under
 * `/models/`. Its URL is stable across retrains, so caching it by URL would pin a user
 * to whatever pack they first downloaded, with no way to tell — precisely the failure
 * this file's rules are load-bearing against. The weights it points at *are* cache-first,
 * because `loadPack` stamps the pack version into their query string, making that URL
 * genuinely immutable.
 *
 * Deliberately hand-written rather than generated. The caching rules here are load-
 * bearing — get them wrong and users are pinned to a stale model with no way to tell —
 * so they are worth reading in full.
 */

// Bumping this drops every previous cache on activate. Do it whenever a caching rule
// changes, so users are not left holding entries filed under the old policy.
const VERSION = 'v2';
const SHELL_CACHE = `mudrapragyan-shell-${VERSION}`;
const ASSET_CACHE = `mudrapragyan-assets-${VERSION}`;

/** Requests served cache-first, because their contents cannot change under the URL. */
const IMMUTABLE = [
  /\/assets\//, // Vite build output, content-hashed
  /\/mediapipe\//, // MediaPipe WASM
  /\/alphabets\//, // generated dictionary diagrams
  /\/numbers\//,
  /\.wasm$/,
  /\.task$/, // MediaPipe bundles, pinned by models.lock.json
  /\.onnx$/, // model weights, version-stamped by `loadPack`
  /\.woff2?$/,
];

function isImmutable(url) {
  return IMMUTABLE.some((pattern) => pattern.test(url.pathname));
}

self.addEventListener('install', (event) => {
  // Take over promptly; there is no old version whose state we need to preserve.
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      // Drop caches from previous versions, so a deploy cannot pin users to stale
      // models forever.
      const names = await caches.keys();
      await Promise.all(
        names
          .filter((name) => name.startsWith('mudrapragyan-') && !name.endsWith(VERSION))
          .map((name) => caches.delete(name)),
      );
      await self.clients.claim();
    })(),
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;

  // Only GET is cacheable, and only our own origin is ours to cache.
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith(isImmutable(url) ? cacheFirst(request) : networkFirst(request));
});

/** Serve from cache, falling back to the network and storing the result. */
async function cacheFirst(request) {
  const cache = await caches.open(ASSET_CACHE);
  const cached = await cache.match(request);
  if (cached !== undefined) return cached;

  const response = await fetch(request);
  // Only cache successful, complete responses. A 206 or an opaque error would poison
  // the cache with something unusable.
  if (response.ok && response.status === 200) {
    cache.put(request, response.clone());
  }
  return response;
}

/** Try the network, fall back to cache when offline. */
async function networkFirst(request) {
  const cache = await caches.open(SHELL_CACHE);

  try {
    const response = await fetch(request);
    if (response.ok && response.status === 200) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const cached = await cache.match(request);
    if (cached !== undefined) return cached;

    // A navigation with nothing cached: fall back to the app shell so the user gets
    // the app rather than a browser error page.
    if (request.mode === 'navigate') {
      const shell = await cache.match('/index.html');
      if (shell !== undefined) return shell;
    }
    throw error;
  }
}
