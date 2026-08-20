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
