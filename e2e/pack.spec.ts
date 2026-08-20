import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

/**
 * End-to-end proof that the v2 pipeline works in a real browser.
 *
 * Unit tests cover the normaliser, the rejection logic and the pack manifest, but
 * none of them exercise the part most likely to break in practice: ONNX Runtime Web
 * actually initialising its WASM, loading a quantised graph, and running inference
 * inside a Web Worker. That only fails in a browser, so it is only worth testing in
 * one.
 *
 * The fixture is a synthetic pack — see `e2e/fixtures/pack/README.md`. Its predictions
 * are meaningless; what matters is that the plumbing runs end to end.
 */

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), 'fixtures', 'pack');
const manifest = readFileSync(join(FIXTURES, 'manifest.json'), 'utf8');
const modelBytes = readFileSync(join(FIXTURES, 'model.onnx'));

/** Serve the fixture pack at the path the app looks for. */
async function installPack(page: Page, manifestBody = manifest): Promise<void> {
  await page.route('**/models/asl-fingerspell/manifest.json', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: manifestBody }),
  );
  await page.route('**/models/asl-fingerspell/model.onnx', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/octet-stream',
      body: modelBytes,
    }),
  );
}

test.describe('v2 model pack', () => {
  test('loads the pack and reports the v2 pipeline', async ({ page }) => {
    await installPack(page);
    await page.goto('/#translator');

    const status = page.locator('#modelStatus');
    await expect(status).toHaveClass(/ready/, { timeout: 60_000 });
    await expect(status).toContainText('Synthetic (pipeline test)');
    // 25 labels minus `none` = 24 signs.
    await expect(status).toContainText('24 signs');
  });

  /**
   * The headline claim of Phase 2. When a pack is installed the recogniser runs with
   * no correction heuristics at all — the 85 lines of `geometricFix` are bypassed
   * entirely (docs/AUDIT.md §2).
   */
  test('switches off the legacy correction heuristics', async ({ page }) => {
    await installPack(page);
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });

    await page.keyboard.press('d');
    await expect(page.locator('#dbgPerf')).toContainText('v2');
  });

  test('initialises ONNX Runtime Web without console errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));

    await installPack(page);
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });

    expect(errors).toEqual([]);
  });

  test('falls back to the legacy pipeline when no pack is installed', async ({ page }) => {
    await page.route('**/models/asl-fingerspell/manifest.json', (route) =>
      route.fulfill({ status: 404 }),
    );
    await page.goto('/#translator');

    const status = page.locator('#modelStatus');
    await expect(status).toContainText('Legacy model', { timeout: 30_000 });
    // Presented as a warning, not as a working configuration.
    await expect(status).toHaveClass(/error/);
  });

  /**
   * A pack trained under different feature extraction than the runtime implements
   * would not error on its own — it would quietly mispredict. Refusing to load is the
   * only safe behaviour, and this proves the refusal reaches the user.
   */
  test('refuses a pack whose normalisation scheme does not match the build', async ({ page }) => {
    const tampered = JSON.parse(manifest) as Record<string, unknown>;
    (tampered['input'] as Record<string, unknown>)['normalisation'] = 'wrist-scaled-v1';

    await installPack(page, JSON.stringify(tampered));
    await page.goto('/#translator');

    const status = page.locator('#modelStatus');
    await expect(status).toHaveClass(/error/, { timeout: 30_000 });
    await expect(status).toContainText('Refusing to load');
  });

  test('refuses a pack with no held-out signers', async ({ page }) => {
    const tampered = JSON.parse(manifest) as Record<string, unknown>;
    (tampered['metrics'] as Record<string, unknown>)['testSigners'] = 0;

    await installPack(page, JSON.stringify(tampered));
    await page.goto('/#translator');

    await expect(page.locator('#modelStatus')).toContainText('has not been evaluated', {
      timeout: 30_000,
    });
  });
});
