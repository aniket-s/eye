import { expect, test, type Page } from '@playwright/test';

/**
 * Collect console errors so a test can assert the page loaded cleanly.
 *
 * No filtering any more. Phase 0 had to ignore blocked third-party requests because the
 * app loaded a webfont from `fonts.googleapis.com`; that font is now self-hosted, so a
 * clean load really does mean zero errors (docs/AUDIT.md, A10).
 */
function trackConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
}

test.describe('application shell', () => {
  test('loads the home page without console errors', async ({ page }) => {
    const errors = trackConsoleErrors(page);
    await page.goto('/');

    await expect(page.locator('#home')).toHaveClass(/active/);
    await expect(page.getByRole('heading', { name: 'MudraPragyan.AI' })).toBeVisible();
    expect(errors).toEqual([]);
  });

  test('navigates between pages and reflects the hash', async ({ page }) => {
    await page.goto('/');

    await page.locator('.home-card[data-page="dictionary"]').click();
    await expect(page.locator('#dictionary')).toHaveClass(/active/);
    expect(page.url()).toContain('#dictionary');

    await page.locator('#dictionary .back-btn').click();
    await expect(page.locator('#home')).toHaveClass(/active/);
  });

  test('supports deep linking straight to the translator', async ({ page }) => {
    await page.goto('/#translator');
    await expect(page.locator('#translator')).toHaveClass(/active/);
  });

  test('restores the previous page with the browser Back button', async ({ page }) => {
    await page.goto('/');
    await page.locator('.home-card[data-page="translator"]').click();
    await expect(page.locator('#translator')).toHaveClass(/active/);

    await page.goBack();
    await expect(page.locator('#home')).toHaveClass(/active/);
  });

  test('opens and closes the side menu, including with Escape', async ({ page }) => {
    await page.goto('/');
    const menu = page.locator('#sideMenu');
    const toggle = page.locator('#menuToggle');

    await toggle.click();
    await expect(menu).toHaveClass(/open/);
    await expect(toggle).toHaveAttribute('aria-expanded', 'true');

    await page.keyboard.press('Escape');
    await expect(menu).not.toHaveClass(/open/);
    await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  });
});

test.describe('the pack that ships with the repository', () => {
  /**
   * Loads the *real* artefact from `packages/web/public/models`, with nothing mocked.
   *
   * Every other pack test serves a fixture, which proves the loader works but says
   * nothing about the file actually committed. A pack that was exported wrong, quantised
   * wrong, or whose manifest disagrees with its weights would pass all of those and fail
   * for every user.
   */
  test('loads, validates and runs', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));

    await page.goto('/#translator');

    const status = page.locator('#modelStatus');
    await expect(status).toHaveClass(/ready/, { timeout: 60_000 });
    // 24 static letters plus the six digits that are not already one of them.
    await expect(status).toContainText('30 signs');
    expect(errors).toEqual([]);

    await page.keyboard.press('d');
    await expect(page.locator('#dbgPerf')).toContainText('v2');
  });

  test('declares that it was trained on simulated data', async ({ page }) => {
    // The user should be able to tell from the app, not only from the model card.
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toContainText('simulated', { timeout: 60_000 });
  });

  test('offers custom signs, so the pack exports embeddings', async ({ page }) => {
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });
    await expect(page.locator('#customRecord')).toBeEnabled();
  });
});

test.describe('letters and numbers modes', () => {
  /**
   * In ASL, 2 and V are the same handshape — as are 6/W, 9/F and 0/O. A signer separates
   * them by context, so the app does too: the model predicts handshapes and a mode
   * decides what they mean. Training 36 flat classes instead would ask the network to
   * split classes carrying no distinguishing signal, and would cost letter accuracy.
   */
  test('offers a mode switch when the pack reads its handshapes two ways', async ({ page }) => {
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });

    await expect(page.locator('#modeSwitch')).toBeVisible();
    await expect(page.locator('[data-vocabulary="letters"]')).toHaveAttribute(
      'aria-pressed',
      'true',
    );
  });

  test('switches to numbers and marks the active mode', async ({ page }) => {
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });

    await page.locator('[data-vocabulary="numbers"]').click();
    await expect(page.locator('[data-vocabulary="numbers"]')).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    await expect(page.locator('[data-vocabulary="letters"]')).toHaveAttribute(
      'aria-pressed',
      'false',
    );
  });

  test('the pack declares both readings, and numbers covers all ten digits', async ({ page }) => {
    const response = await page.request.get('/models/asl-fingerspell/manifest.json');
    const manifest = (await response.json()) as {
      labels: string[];
      vocabularies?: Record<string, Record<string, string>>;
    };

    expect(Object.keys(manifest.vocabularies ?? {})).toEqual(['letters', 'numbers']);
    expect(Object.values(manifest.vocabularies?.['numbers'] ?? {}).sort()).toEqual([
      '0',
      '1',
      '2',
      '3',
      '4',
      '5',
      '6',
      '7',
      '8',
      '9',
    ]);

    // The four shared shapes must not have been trained as separate classes.
    for (const digit of ['0', '2', '6', '9']) {
      expect(manifest.labels).not.toContain(digit);
    }
    expect(manifest.vocabularies?.['numbers']?.['V']).toBe('2');
  });
});

test.describe('word suggestions', () => {
  /**
   * Scope note: these check the *wiring* — that the panel appears, that the word list is
   * served from our own origin, and that it loads and indexes without error. The
   * behaviour of suggestion and correction is covered by 22 unit tests in
   * `packages/core/src/lexicon/lexicon.test.ts`, which can exercise it far more thoroughly
   * than a browser can.
   *
   * Driving the suggestions end to end would need committed letters, and those only come
   * from the camera. The tempting shortcut is a test-only hook on the page to inject them;
   * that trades a real backdoor in shipped code for coverage already obtained elsewhere,
   * so it is not taken.
   */
  test('shows the suggestion panel once a letter pack is loaded', async ({ page }) => {
    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });
    await expect(page.locator('#wordPanel')).toBeVisible();
  });

  test('loads and indexes the word list without error', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));

    await page.goto('/#translator');
    await expect(page.locator('#modelStatus')).toHaveClass(/ready/, { timeout: 60_000 });

    // The status clears on success and carries a message on failure.
    await expect(page.locator('#wordStatus')).toHaveText('', { timeout: 30_000 });
    expect(errors).toEqual([]);
  });

  test('serves the word list from our own origin, in frequency order', async ({ page }) => {
    const response = await page.request.get('/lexicon/en.txt');
    expect(response.ok()).toBe(true);

    const words = (await response.text()).split('\n').filter(Boolean);
    expect(words.length).toBeGreaterThan(10_000);
    // Frequency order is load-bearing: the line number is the prior that ranks
    // suggestions, so an alphabetical list would put `aardvark` ahead of `and`.
    expect(words[0]).toBe('the');
    expect(words.slice(0, 5)).toContain('and');
  });

  test('ships the confusion profile with the pack', async ({ page }) => {
    // What makes correction better than a spell-checker. Its absence is silent — the app
    // still works, just with every substitution weighted alike.
    const response = await page.request.get('/models/asl-fingerspell/manifest.json');
    const manifest = (await response.json()) as { confusions?: Record<string, unknown> };
    expect(manifest.confusions).toBeDefined();
    expect(Object.keys(manifest.confusions ?? {}).length).toBeGreaterThan(0);
  });
});

test.describe('model loading', () => {
  /**
   * Serve a 404 for the pack, so "no pack installed" is stated rather than assumed.
   *
   * The repository now ships a trained pack, so a test that relied on its absence would
   * quietly stop testing the fallback and start testing the happy path instead.
   */
  async function withoutAPack(page: Page): Promise<void> {
    await page.route('**/models/**', (route) => route.fulfill({ status: 404 }));
  }

  /**
   * With no model pack installed the app falls back to the v1 model, which still runs
   * the legacy correction heuristics. That is presented as a warning rather than a
   * working state, so nobody mistakes the fallback for a trained model.
   * `e2e/pack.spec.ts` covers the v2 path.
   */
  test('warns that the legacy model is in use when no pack is installed', async ({ page }) => {
    await withoutAPack(page);
    await page.goto('/#translator');
    const status = page.locator('#modelStatus');
    // 26 letters + space in the v1 export.
    await expect(status).toContainText('27 signs', { timeout: 30_000 });
    await expect(status).toContainText('Train a model pack');
    await expect(status).toHaveClass(/error/);
  });

  test('surfaces a clear error when the model is missing', async ({ page }) => {
    await withoutAPack(page);
    await page.route('**/model_weights.json', (route) => route.fulfill({ status: 404 }));
    await page.goto('/#translator');

    await expect(page.locator('#modelStatus')).toHaveClass(/error/, { timeout: 30_000 });
    await expect(page.locator('#modelStatus')).toContainText('404');
  });
});

test.describe('dictionary', () => {
  test('renders every category and its signs', async ({ page }) => {
    await page.goto('/#dictionary');

    // Alphabets and numbers only: the categories the recogniser can actually read.
    // Word categories return with a word-level pack and artwork to show for them.
    await expect(page.locator('.cat-pill')).toHaveCount(2);
    await expect(page.locator('#sec-alphabets .sign-card')).toHaveCount(26);
    await expect(page.locator('#sec-numbers .sign-card')).toHaveCount(11);
  });

  test('opens a sign in the detail modal and closes it with Escape', async ({ page }) => {
    await page.goto('/#dictionary');

    await page.locator('#sec-alphabets .sign-card').first().click();
    await expect(page.locator('#modal')).toHaveClass(/show/);
    await expect(page.locator('#modalName')).toHaveText('A');

    await page.keyboard.press('Escape');
    await expect(page.locator('#modal')).not.toHaveClass(/show/);
  });

  /**
   * Inverts the Phase 0 assertion. Every alphabet and number card used to fall through
   * to the 🤟 placeholder because none of the referenced images existed (A1); they are
   * now generated vector diagrams.
   */
  test('shows a diagram for every letter and number', async ({ page }) => {
    await page.goto('/#dictionary');

    for (const selector of ['#sec-alphabets', '#sec-numbers']) {
      const cards = page.locator(`${selector} .sign-card`);
      const count = await cards.count();
      for (let i = 0; i < count; i++) {
        await expect(cards.nth(i).locator('.placeholder')).toBeHidden({ timeout: 15_000 });
      }
    }
  });

  test('serves the diagrams as SVG so they stay sharp at any size', async ({ page }) => {
    await page.goto('/#dictionary');
    const image = page.locator('#sec-alphabets .sign-card img').first();
    await expect(image).toHaveAttribute('src', /alphabets\/a\.svg$/);
  });
});

test.describe('letter hold speed', () => {
  test('defaults to Fast and remembers a different choice across reloads', async ({ page }) => {
    await page.goto('/#translator');
    await expect(page.locator('[data-speed="250"]')).toHaveClass(/is-active/);

    await page.locator('[data-speed="1500"]').click();
    await expect(page.locator('[data-speed="1500"]')).toHaveClass(/is-active/);
    await expect(page.locator('[data-speed="250"]')).not.toHaveClass(/is-active/);

    await page.reload();
    await expect(page.locator('[data-speed="1500"]')).toHaveClass(/is-active/);
  });
});

test.describe('hand-tracking overlay', () => {
  /**
   * The Translator draws the same landmark skeleton the recorder shows, on a canvas
   * carrying the same CSS mirror as the video. The fake camera feeds no hand, so what
   * can be pinned here is the part that regresses silently: the canvas must adopt the
   * feed's pixel grid and sit exactly over the video box, with a clean console.
   */
  test('lays the skeleton canvas exactly over the live video', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));

    await page.goto('/#translator');
    await page.locator('#startCamera').click();
    await expect(page.locator('#liveBadge')).toHaveClass(/show/, { timeout: 60_000 });

    // First processed frame resizes the canvas from its 300x150 default to the feed.
    await expect
      .poll(() => page.locator('#canvas').evaluate((node) => (node as HTMLCanvasElement).width), {
        timeout: 30_000,
      })
      .not.toBe(300);

    const video = await page.locator('#video').boundingBox();
    const canvas = await page.locator('#canvas').boundingBox();
    expect(video).not.toBeNull();
    expect(canvas).not.toBeNull();
    for (const side of ['x', 'y', 'width', 'height'] as const) {
      expect(Math.abs((canvas?.[side] ?? 0) - (video?.[side] ?? 0))).toBeLessThan(2);
    }
    expect(errors).toEqual([]);
  });
});

test.describe('sentence builder', () => {
  test('adds, deletes and clears text', async ({ page }) => {
    await page.goto('/#translator');
    const display = page.locator('#sentenceDisplay');

    await expect(display).toHaveClass(/empty/);

    await page.locator('#addSpace').click();
    await expect(display).not.toHaveClass(/empty/);

    await page.locator('#backspace').click();
    await expect(display).toHaveClass(/empty/);

    await page.locator('#addSpace').click();
    await page.locator('#clearSentence').click();
    await expect(display).toHaveClass(/empty/);
  });
});

test.describe('debug overlay', () => {
  test('toggles with the D key and the button', async ({ page }) => {
    await page.goto('/#translator');
    const panel = page.locator('#debugPanel');

    await expect(panel).toBeHidden();
    await page.keyboard.press('d');
    await expect(panel).toBeVisible();

    await page.locator('#debugToggle').click();
    await expect(panel).toBeHidden();
  });

  test('shows a pipeline row for hands, pose and dropped frames', async ({ page }) => {
    await page.goto('/#translator');
    await page.keyboard.press('d');
    await expect(page.locator('#dbgPerf')).toBeVisible();
  });
});

test.describe('vision assets', () => {
  /**
   * The `.task` models are downloaded by `npm run setup:assets`, not committed. A
   * fresh clone genuinely lacks them, so the failure has to be actionable rather
   * than a raw 404 in the console.
   */
  test('explains how to fix missing vision models instead of failing silently', async ({
    page,
  }) => {
    await page.route('**/models/*.task', (route) => route.fulfill({ status: 404 }));
    await page.goto('/#translator');

    await page.locator('#startCamera').click();
    await expect(page.locator('#camMessage')).toContainText('npm run setup:assets', {
      timeout: 30_000,
    });
  });

  test('re-enables the start button after a failure so the user can retry', async ({ page }) => {
    await page.route('**/models/*.task', (route) => route.fulfill({ status: 404 }));
    await page.goto('/#translator');

    const start = page.locator('#startCamera');
    await start.click();
    await expect(start).toBeEnabled({ timeout: 30_000 });
    await expect(start).toHaveText('▶ Start Camera');
  });
});

test.describe('recognition worker', () => {
  /**
   * The model is fetched and parsed inside the worker, which keeps the 2.5 MB JSON
   * parse off the main thread (docs/AUDIT.md, M6 and A5).
   */
  test('loads the model in a worker, not on the main thread', async ({ page }) => {
    const workerUrls: string[] = [];
    page.on('worker', (worker) => workerUrls.push(worker.url()));

    await page.goto('/#translator');
    // Whichever pipeline wins, the model is resolved inside the worker.
    await expect(page.locator('#modelStatus')).toContainText('signs', { timeout: 30_000 });

    expect(workerUrls.some((url) => url.includes('recognition'))).toBe(true);
  });
});

test.describe('privacy and offline', () => {
  /**
   * The strongest claim the project makes is that nothing leaves the device. It is
   * worth testing rather than asserting: a single third-party font or analytics
   * request would quietly falsify it.
   */
  test('makes no third-party requests at all', async ({ page }) => {
    const external: string[] = [];
    page.on('request', (request) => {
      const url = new URL(request.url());
      if (url.origin !== 'http://127.0.0.1:4173' && url.protocol !== 'data:') {
        external.push(request.url());
      }
    });

    await page.goto('/');
    await page.locator('#modelStatus').waitFor({ state: 'attached' });
    await page.waitForTimeout(1500);

    expect(external).toEqual([]);
  });

  test('serves the font from our own origin', async ({ page }) => {
    const fontRequests: string[] = [];
    page.on('request', (request) => {
      if (request.resourceType() === 'font') fontRequests.push(request.url());
    });

    await page.goto('/');
    await page.waitForTimeout(1000);

    for (const url of fontRequests) {
      expect(url).toContain('127.0.0.1:4173');
    }
  });

  test('publishes an installable web app manifest', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('link[rel="manifest"]')).toHaveCount(1);

    const response = await page.request.get('/manifest.webmanifest');
    expect(response.ok()).toBe(true);

    const manifest = (await response.json()) as { name: string; display: string; icons: unknown[] };
    expect(manifest.name).toBe('MudraPragyan.AI');
    expect(manifest.display).toBe('standalone');
    expect(manifest.icons.length).toBeGreaterThan(0);
  });

  test('ships a service worker that caches the heavy assets', async ({ page }) => {
    const response = await page.request.get('/sw.js');
    expect(response.ok()).toBe(true);

    const source = await response.text();
    // The WASM and model weights are the bulk of the payload; caching them is what makes
    // a repeat visit — and an offline one — work at all.
    for (const pattern of ['mediapipe', 'wasm', 'onnx']) {
      expect(source).toContain(pattern);
    }
    await page.goto('/');
  });

  /**
   * A pack manifest lives at a stable URL across retrains, so caching it by URL would
   * pin a user to the first pack they ever downloaded with no way to tell. It must fall
   * through to the network-first path.
   */
  test('does not cache pack manifests by URL', async ({ page }) => {
    const source = await (await page.request.get('/sw.js')).text();
    const immutable = /const IMMUTABLE = \[(.*?)\];/s.exec(source)?.[1] ?? '';
    expect(immutable).not.toContain('/models/');
  });

  /**
   * Service workers are blocked for the rest of the suite (see `playwright.config.ts`),
   * because they intercept the routes the pack fixtures rely on. Registration is still
   * worth proving somewhere, so this block opts back in.
   */
  test.describe('with the service worker enabled', () => {
    test.use({ serviceWorkers: 'allow' });

    test('registers and takes control', async ({ page }) => {
      await page.goto('/');
      // `ready` resolves as soon as a worker is active, which can still be `activating`
      // while our `activate` handler prunes old caches and claims clients.
      const state = await page.evaluate(async () => {
        const worker = (await navigator.serviceWorker.ready).active;
        if (worker === null) return 'none';
        if (worker.state === 'activated') return worker.state;
        await new Promise<void>((resolve) => {
          worker.addEventListener('statechange', () => {
            if (worker.state === 'activated') resolve();
          });
        });
        return worker.state;
      });
      expect(state).toBe('activated');
    });
  });

  test('tells the user the app still works when offline', async ({ page }) => {
    await page.goto('/');
    const banner = page.locator('#offlineBanner');
    await expect(banner).toBeHidden();

    await page.context().setOffline(true);
    await page.evaluate(() => window.dispatchEvent(new Event('offline')));
    await expect(banner).toBeVisible();
    await expect(banner).toContainText('runs on your device');

    await page.context().setOffline(false);
    await page.evaluate(() => window.dispatchEvent(new Event('online')));
    await expect(banner).toBeHidden();
  });
});
