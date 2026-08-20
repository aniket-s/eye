import { expect, test } from '@playwright/test';

/**
 * The app, served by the dev server.
 *
 * A production bundle and a dev server resolve modules and assets by completely different
 * rules, so passing against one says little about the other. This suite exists because of
 * a failure that was total in dev and invisible in the build: Vite's dependency
 * pre-bundling rewrote ONNX Runtime's WASM URL into `node_modules/.vite/deps/` without
 * putting the binary there, the dev server answered with the SPA fallback, and the runtime
 * tried to compile `index.html` as WebAssembly —
 *
 *     expected magic word 00 61 73 6d, found 3c 21 64 6f
 *
 * — which is `<!do`. Recognition did not work at all for anyone running `npm run dev`,
 * and every other test passed.
 *
 * Keep this suite small. It is about *how the app is served*, not about behaviour, which
 * the built-app suites already cover.
 */

test.describe('dev server', () => {
  test('loads the model pack and runs inference', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', (message) => {
      if (message.type() === 'error') errors.push(message.text());
    });
    page.on('pageerror', (error) => errors.push(error.message));

    await page.goto('/#translator');

    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 90_000 });
    expect(errors).toEqual([]);
  });

  test('serves the ONNX Runtime WASM as WebAssembly, not as the app shell', async ({ page }) => {
    const wasm: { url: string; type: string; status: number }[] = [];
    page.on('response', (response) => {
      if (new URL(response.url()).pathname.endsWith('.wasm')) {
        wasm.push({
          url: response.url(),
          type: response.headers()['content-type'] ?? '',
          status: response.status(),
        });
      }
    });

    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 90_000 });

    expect(wasm.length).toBeGreaterThan(0);
    for (const request of wasm) {
      expect(request.status, request.url).toBe(200);
      // The exact symptom: an HTML content type here means the SPA fallback answered.
      expect(request.type, request.url).toContain('wasm');
    }
  });

  test('resolves the workspace core package without a prior build', async ({ page }) => {
    // Complements the CI step that builds the web app with core uncompiled: this proves
    // the same aliasing works when Vite is serving rather than bundling.
    await page.goto('/#dictionary');
    await expect(page.locator('.cat-pill')).toHaveCount(2);
  });
});
