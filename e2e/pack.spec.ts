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

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, 'fixtures', 'pack');
const manifest = readFileSync(join(FIXTURES, 'manifest.json'), 'utf8');
const modelBytes = readFileSync(join(FIXTURES, 'model.onnx'));

const WORD_FIXTURES = join(HERE, 'fixtures', 'word-pack');
const wordManifest = readFileSync(join(WORD_FIXTURES, 'manifest.json'), 'utf8');
const wordModelBytes = readFileSync(join(WORD_FIXTURES, 'model.onnx'));

const CTC_FIXTURES = join(HERE, 'fixtures', 'ctc-pack');
const ctcManifest = readFileSync(join(CTC_FIXTURES, 'manifest.json'), 'utf8');
const ctcModelBytes = readFileSync(join(CTC_FIXTURES, 'model.onnx'));

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

test.describe('continuous fingerspelling (CTC)', () => {
  /** Serve the CTC fixture pack at the id the app looks for. */
  async function installCtcPack(page: Page): Promise<void> {
    await page.route('**/models/asl-fingerspell/manifest.json', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: ctcManifest }),
    );
    await page.route('**/models/asl-fingerspell/model.onnx', (route) =>
      route.fulfill({ status: 200, contentType: 'application/octet-stream', body: ctcModelBytes }),
    );
  }

  /**
   * The critical difference from the static pack: this graph has a **dynamic frame
   * axis**. The browser feeds a partial window while its buffer fills, so a graph that
   * silently baked in the training window would work after two seconds and fail before
   * that — the kind of bug that only appears on a real camera.
   */
  test('loads a temporal-ctc pack and reports the streaming pipeline', async ({ page }) => {
    await installCtcPack(page);
    await page.goto('/#translator');

    const status = page.locator('#modelStatus');
    await expect(status).toHaveClass(/ready/, { timeout: 60_000 });
    await expect(status).toContainText('continuous');
  });

  test('retires the dwell timer, since CTC decides commits itself', async ({ page }) => {
    await installCtcPack(page);
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });

    // v1 made the user hold each letter for two thirds of a second and showed a
    // progress bar counting it down. There is nothing to count any more.
    await expect(page.locator('#holdPct')).toHaveText('—');
    await expect(page.locator('#letterLabel')).toContainText('no need to pause');
  });

  test('reports the ctc pipeline in the debug overlay', async ({ page }) => {
    await installCtcPack(page);
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });

    await page.keyboard.press('d');
    await expect(page.locator('#dbgPerf')).toContainText('ctc');
  });

  test('initialises the sequence model without console errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));

    await installCtcPack(page);
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });

    expect(errors).toEqual([]);
  });
});

test.describe('word-level signs', () => {
  async function installWordPack(page: Page): Promise<void> {
    await page.route('**/models/asl-fingerspell/manifest.json', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: wordManifest }),
    );
    await page.route('**/models/asl-fingerspell/model.onnx', (route) =>
      route.fulfill({ status: 200, contentType: 'application/octet-stream', body: wordModelBytes }),
    );
  }

  test('loads a temporal-isolated pack and reports word mode', async ({ page }) => {
    await installWordPack(page);
    await page.goto('/#translator');

    const status = page.locator('#modelStatus');
    await expect(status).toHaveClass(/ready/, { timeout: 60_000 });
    await expect(status).toContainText('word signs');
  });

  /**
   * Word mode segments on motion and classifies each completed sign once, so there is
   * no per-frame hold to display — a different interaction model from both
   * fingerspelling pipelines.
   */
  test('shows no hold progress, since it segments on motion instead', async ({ page }) => {
    await installWordPack(page);
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });

    await expect(page.locator('#holdPct')).toHaveText('—');
    await expect(page.locator('#letterLabel')).toContainText('Sign a word, then pause');
  });

  test('reports the words pipeline in the debug overlay', async ({ page }) => {
    await installWordPack(page);
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });

    await page.keyboard.press('d');
    await expect(page.locator('#dbgPerf')).toContainText('words');
  });

  /**
   * The two-handed model uses a different feature scheme from fingerspelling. Its
   * manifest declares `shoulder-frame-v1`, and the loader must accept that as readily
   * as the single-hand scheme while still rejecting anything unknown.
   */
  test('accepts the two-handed normalisation scheme', async ({ page }) => {
    const parsed = JSON.parse(wordManifest) as { input: { normalisation: string; hands: number } };
    expect(parsed.input.normalisation).toBe('shoulder-frame-v1');
    expect(parsed.input.hands).toBe(2);

    await installWordPack(page);
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });
  });

  test('initialises the clip model without console errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));

    await installWordPack(page);
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });

    expect(errors).toEqual([]);
  });
});
