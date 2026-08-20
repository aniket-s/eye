import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
  /**
   * Public base path. Defaults to the domain root, which is correct for Cloudflare
   * Pages. A GitHub Pages *project* site is served from `/<repo>/`, so CI sets
   * `BASE_PATH=/mudrapragyan/` for that target. Vite rewrites root-relative URLs in
   * HTML and CSS to match, so `/bg.png` resolves correctly under either.
   */
  base: process.env['BASE_PATH'] ?? '/',
  resolve: {
    alias: {
      /**
       * Resolve the workspace package to its **source**, not its build output.
       *
       * `@mudrapragyan/core` declares `main`/`exports` pointing at `dist/`, which is
       * correct for tsc and for anything consuming it as a package — but it means a
       * fresh clone cannot start the dev server until `core` has been compiled, and
       * the failure is an opaque "failed to resolve entry for package".
       *
       * Vite compiles TypeScript itself, so pointing it at the source removes the
       * ordering requirement entirely and makes edits in `core` hot-reload instead of
       * requiring a rebuild. Type checking is unaffected: `npm run typecheck` still
       * builds the real project references.
       */
      '@mudrapragyan/core': resolve(import.meta.dirname, '../core/src/index.ts'),
    },
  },
  optimizeDeps: {
    /**
     * Keep ONNX Runtime out of dev dependency pre-bundling.
     *
     * The runtime locates its 13 MB WASM binary with `new URL('....wasm', import.meta.url)`.
     * Pre-bundling rewrites that URL to sit beside the bundled copy under
     * `node_modules/.vite/deps/` — but does not put the binary there. The dev server then
     * answers the request with the SPA fallback, and ONNX Runtime tries to compile
     * `index.html` as WebAssembly:
     *
     *     expected magic word 00 61 73 6d, found 3c 21 64 6f   ("<!do")
     *
     * Excluding it leaves the URL pointing at the real file in `node_modules`.
     *
     * Dev only. The production build resolves the asset correctly, which is exactly why
     * an end-to-end suite that runs against the built app cannot catch this — see the
     * `dev-server` project in `playwright.config.ts`.
     */
    exclude: ['onnxruntime-web'],
  },
  build: {
    target: 'es2022',
    outDir: 'dist',
    sourcemap: true,
    // The 2.5 MB model JSON must stay a separate fetch, never inlined into JS.
    assetsInlineLimit: 4096,
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        // Developer tool for capturing training data. A separate entry point keeps
        // it out of the main bundle entirely.
        recorder: resolve(import.meta.dirname, 'recorder.html'),
      },
    },
  },
  server: {
    port: 5173,
  },
  preview: {
    port: 4173,
  },
});
