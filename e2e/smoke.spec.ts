import { expect, test, type Page } from '@playwright/test';

/**
 * Errors caused by a blocked or offline third-party host rather than by our code.
 *
 * The app still loads a webfont from `fonts.googleapis.com`. That is a third-party
 * request in an app whose selling point is that nothing leaves the device, and it
 * breaks the page in restricted networks. Self-hosting the font is queued for
 * Phase 5 alongside offline/PWA support; until then these are filtered so an
 * offline CI runner does not fail the suite for the wrong reason.
 */
const EXTERNAL_RESOURCE_ERROR =
  /ERR_TUNNEL_CONNECTION_FAILED|ERR_NAME_NOT_RESOLVED|ERR_INTERNET_DISCONNECTED|net::ERR_/;

/** Collect console errors so a test can assert the page loaded cleanly. */
function trackConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error' && !EXTERNAL_RESOURCE_ERROR.test(message.text())) {
      errors.push(message.text());
    }
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

test.describe('model loading', () => {
  /**
   * With no model pack installed the app falls back to the v1 model, which still runs
   * the legacy correction heuristics. That is presented as a warning rather than a
   * working state, so nobody mistakes the fallback for a trained model.
   * `e2e/pack.spec.ts` covers the v2 path.
   */
  test('warns that the legacy model is in use when no pack is installed', async ({ page }) => {
    await page.goto('/#translator');
    const status = page.locator('#modelStatus');
    // 26 letters + space in the v1 export.
    await expect(status).toContainText('27 signs', { timeout: 30_000 });
    await expect(status).toContainText('Train a model pack');
    await expect(status).toHaveClass(/error/);
  });

  test('surfaces a clear error when the model is missing', async ({ page }) => {
    await page.route('**/model_weights.json', (route) => route.fulfill({ status: 404 }));
    await page.goto('/#translator');

    await expect(page.locator('#modelStatus')).toHaveClass(/error/, { timeout: 30_000 });
    await expect(page.locator('#modelStatus')).toContainText('404');
  });
});

test.describe('dictionary', () => {
  test('renders every category and its signs', async ({ page }) => {
    await page.goto('/#dictionary');

    await expect(page.locator('.cat-pill')).toHaveCount(11);
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

  /**
   * Word categories still have no artwork — Phase 4 adds PopSign clips. Pinned so the
   * gap is visible rather than forgotten.
   */
  test('KNOWN GAP: word signs still have no artwork', async ({ page }) => {
    await page.goto('/#dictionary');
    const card = page.locator('#sec-greetings .sign-card').first();
    await expect(card.locator('.placeholder')).toBeVisible({ timeout: 15_000 });
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
