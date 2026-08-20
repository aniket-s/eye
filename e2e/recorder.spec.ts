import { expect, test, type Page } from '@playwright/test';

/**
 * The landmark recorder — the tool that produces training data.
 *
 * The failure mode these guard against is a *silent* recorder: a page that looks like
 * it is capturing while tracking has dropped, or that discards a half-hour session on
 * refresh with nothing to say about it. Camera-driven behaviour (overlay drawing, live
 * classification) cannot run against the fake video device, so these tests pin the
 * wiring: initial state, actionable failures, and the autosave lifecycle.
 */

function trackConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
}

/** Seed the recorder's IndexedDB the way a previous session's autosave would have. */
async function seedAutosavedSamples(page: Page, labels: readonly string[]): Promise<void> {
  await page.evaluate(async (seedLabels) => {
    const database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open('mudrapragyan-recorder', 1);
      request.onupgradeneeded = () => {
        request.result.createObjectStore('samples', { autoIncrement: true });
      };
      request.onsuccess = () => {
        resolve(request.result);
      };
      request.onerror = () => {
        reject(request.error ?? new Error('open failed'));
      };
    });
    const transaction = database.transaction('samples', 'readwrite');
    for (const label of seedLabels) {
      transaction.objectStore('samples').add({
        label,
        signer: 'e2e-signer',
        hand: 'right',
        condition: 'normal',
        timestampMs: 1000,
        landmarks: Array.from({ length: 63 }, () => 0.5),
      });
    }
    await new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => {
        resolve();
      };
      transaction.onerror = () => {
        reject(transaction.error ?? new Error('seed failed'));
      };
    });
    database.close();
  }, labels);
}

test.describe('landmark recorder', () => {
  test('loads cleanly, with recording gated behind the camera', async ({ page }) => {
    const errors = trackConsoleErrors(page);
    await page.goto('/recorder.html');

    await expect(page.locator('#recState')).toHaveText('Camera off');
    await expect(page.locator('#record')).toBeDisabled();
    await expect(page.locator('#pause')).toBeDisabled();
    await expect(page.locator('#download')).toBeDisabled();
    // The live model check only appears once the camera pipeline is up.
    await expect(page.locator('#livePanel')).toBeHidden();
    await expect(page.locator('#storeStatus')).toContainText('Autosave is on');
    expect(errors).toEqual([]);
  });

  test('capture speed defaults to Fast and remembers a change across reloads', async ({ page }) => {
    await page.goto('/recorder.html');
    await expect(page.locator('[data-interval="60"]')).toHaveClass(/is-active/);

    await page.locator('[data-interval="300"]').click();
    await expect(page.locator('[data-interval="300"]')).toHaveClass(/is-active/);
    await expect(page.locator('[data-interval="60"]')).not.toHaveClass(/is-active/);

    await page.reload();
    await expect(page.locator('[data-interval="300"]')).toHaveClass(/is-active/);
  });

  test('explains how to fix missing vision models instead of failing silently', async ({
    page,
  }) => {
    await page.route('**/models/*.task', (route) => route.fulfill({ status: 404 }));
    await page.goto('/recorder.html');

    await page.locator('#start').click();
    await expect(page.locator('#camStatus')).toContainText('npm run setup:assets', {
      timeout: 30_000,
    });
    // The start button recovers so the user can retry after running setup.
    await expect(page.locator('#start')).toBeEnabled();
  });

  test('restores autosaved samples from a previous session and says so', async ({ page }) => {
    await page.goto('/recorder.html');
    await seedAutosavedSamples(page, ['A', 'A', 'B']);

    await page.reload();

    await expect(page.locator('#storeStatus')).toContainText('Restored 3 autosaved samples');
    await expect(page.locator('#status')).toContainText('3 samples collected across 2 labels');
    await expect(page.locator('#download')).toBeEnabled();
    // The counts table names what is stored, per label.
    await expect(page.locator('#counts tr')).toHaveCount(2);
    await expect(page.locator('#counts tr').first()).toContainText('A');
  });

  test('requires a second click to clear, then empties the autosave too', async ({ page }) => {
    await page.goto('/recorder.html');
    await seedAutosavedSamples(page, ['C']);
    await page.reload();
    await expect(page.locator('#status')).toContainText('1 samples collected');

    const clear = page.locator('#clear');
    await clear.click();
    // First click arms rather than deletes, and names the blast radius.
    await expect(clear).toHaveText('Really delete 1?');
    await expect(page.locator('#status')).toContainText('1 samples collected');

    await clear.click();
    await expect(page.locator('#status')).toHaveText('0 samples collected.');
    await expect(page.locator('#storeStatus')).toContainText('Cleared');

    // The autosave is genuinely gone: a reload restores nothing.
    await page.reload();
    await expect(page.locator('#storeStatus')).toContainText('Autosave is on');
    await expect(page.locator('#status')).toHaveText('0 samples collected.');
  });

  test('tracks what has not yet been downloaded', async ({ page }) => {
    await page.goto('/recorder.html');
    await seedAutosavedSamples(page, ['D', 'D']);
    await page.reload();

    await expect(page.locator('#status')).toContainText('2 not yet downloaded');

    const download = page.waitForEvent('download');
    await page.locator('#download').click();
    await download;

    await expect(page.locator('#status')).toContainText('All downloaded');
  });
});
