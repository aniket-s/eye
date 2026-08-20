import { defineConfig, devices } from '@playwright/test';

/**
 * End-to-end smoke tests.
 *
 * These run against the built app so CI catches breakage that unit tests cannot:
 * a missing element id, a bundling failure, an unhandled console error. The camera
 * itself is granted a fake device rather than real hardware.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: Boolean(process.env['CI']),
  retries: process.env['CI'] === undefined ? 0 : 2,
  reporter: process.env['CI'] === undefined ? 'list' : [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'on-first-retry',
    /**
     * Service worker requests bypass `page.route`, so a registered worker would serve
     * its own cached copy of whatever a *previous* test installed at
     * `/models/asl-fingerspell/model.onnx` — every pack fixture shares that URL. Tests
     * would then pass or fail by execution order.
     *
     * Blocked by default; `e2e/smoke.spec.ts` re-enables it where the worker itself is
     * what is under test.
     */
    serviceWorkers: 'block',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          /**
           * Allow an environment-supplied Chromium. Sandboxes and CI images often
           * ship a browser already; without this, Playwright insists on downloading
           * a build matching its own version. Unset locally to use the managed one.
           */
          ...(process.env['CHROMIUM_PATH'] !== undefined
            ? { executablePath: process.env['CHROMIUM_PATH'] }
            : {}),
          args: [
            // Provide a synthetic camera so the permission flow can be exercised.
            '--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream',
          ],
        },
      },
    },
  ],
  webServer: {
    command: 'npm run build && npm run preview',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: process.env['CI'] === undefined,
    timeout: 120_000,
  },
});
