import type { CustomSign } from '@mudrapragyan/core';

/**
 * Persistence for user-defined signs.
 *
 * IndexedDB rather than `localStorage`, for three reasons that all matter here:
 *
 * - **Size.** A single sign carries a 128-float centroid. `localStorage` is capped
 *   around 5 MB *and* stores strings, so every read would parse the whole collection.
 * - **Structured values.** Numbers survive as numbers, so a centroid never round-trips
 *   through JSON and back with a precision surprise.
 * - **Asynchrony.** `localStorage` blocks the main thread; a user with fifty recorded
 *   signs would feel it on every page load.
 *
 * Signs are namespaced by pack id **and version**. Embeddings from one model mean nothing
 * to another — the spaces are unrelated — and that is equally true of two versions of the
 * *same* pack, since retraining moves every point in the space. {@link CustomSignBook}
 * guards the obvious case by embedding width, but a retrain at the same width slips
 * straight through it and would match confidently against nonsense.
 *
 * Signs from another version are therefore hidden rather than used — and **counted**, so
 * the app can tell the user their recordings were set aside by a model update instead of
 * letting them silently disappear.
 *
 * Everything stays on the device. Nothing here is ever uploaded (docs/PRIVACY.md).
 */

const DATABASE = 'mudrapragyan';
const DATABASE_VERSION = 1;
const STORE = 'customSigns';

/** Stored shape: the sign plus the exact pack build it was recorded against. */
interface StoredSign extends CustomSign {
  readonly packId: string;
  /**
   * Pack version at recording time.
   *
   * Optional because signs written before this field existed have none; those are
   * treated as belonging to no version, and so are never matched.
   */
  readonly packVersion?: string;
}

export interface StoredSigns {
  /** Signs recorded against the pack build now loaded. */
  readonly signs: CustomSign[];
  /** How many were set aside because they belong to a different build of this pack. */
  readonly staleCount: number;
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, DATABASE_VERSION);

    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE)) {
        const store = database.createObjectStore(STORE, { keyPath: ['packId', 'id'] });
        store.createIndex('packId', 'packId', { unique: false });
      }
    };

    request.onsuccess = () => {
      resolve(request.result);
    };
    request.onerror = () => {
      reject(request.error ?? new Error('Could not open the local sign database'));
    };
    // Another tab holding an old version open. Rare, and retrying would not help.
    request.onblocked = () => {
      reject(new Error('Another tab is using an older version of this app. Close it and reload.'));
    };
  });
}

function run<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => {
      resolve(request.result);
    };
    request.onerror = () => {
      reject(request.error ?? new Error('Local sign storage failed'));
    };
  });
}

/**
 * Signs recorded against this exact pack build.
 *
 * @returns Usable signs and a count of those set aside as stale. Empty if storage is
 *   unavailable — private browsing modes and locked-down profiles block IndexedDB
 *   outright, and recognition should still work there, just without the user's own signs.
 */
export async function loadCustomSigns(packId: string, packVersion: string): Promise<StoredSigns> {
  try {
    const database = await open();
    try {
      const index = database.transaction(STORE, 'readonly').objectStore(STORE).index('packId');
      const stored = (await run(index.getAll(IDBKeyRange.only(packId)))) as StoredSign[];

      const signs: CustomSign[] = [];
      let staleCount = 0;
      for (const { packId, packVersion: storedVersion, ...sign } of stored) {
        if (storedVersion === packVersion) signs.push(sign);
        else staleCount++;
      }
      return { signs, staleCount };
    } finally {
      database.close();
    }
  } catch {
    return { signs: [], staleCount: 0 };
  }
}

/**
 * Save one sign, replacing any with the same id.
 *
 * @throws Error if storage is unavailable. Unlike reading, a failed *save* must be
 *   visible: the user has just spent half a minute recording examples and needs to know
 *   they were not kept.
 */
export async function saveCustomSign(
  packId: string,
  packVersion: string,
  sign: CustomSign,
): Promise<void> {
  const database = await open();
  try {
    const transaction = database.transaction(STORE, 'readwrite');
    const done = commit(transaction);
    transaction.objectStore(STORE).put({ ...sign, packId, packVersion } satisfies StoredSign);
    await done;
  } finally {
    database.close();
  }
}

/** Forget one sign. */
export async function deleteCustomSign(packId: string, id: string): Promise<void> {
  const database = await open();
  try {
    const transaction = database.transaction(STORE, 'readwrite');
    const done = commit(transaction);
    transaction.objectStore(STORE).delete([packId, id]);
    await done;
  } finally {
    database.close();
  }
}

/** Forget every sign for a pack, across all of its versions. */
export async function clearCustomSigns(packId: string): Promise<void> {
  const database = await open();
  try {
    const transaction = database.transaction(STORE, 'readwrite');
    const done = commit(transaction);
    const store = transaction.objectStore(STORE);
    const keys = await run(store.index('packId').getAllKeys(IDBKeyRange.only(packId)));
    for (const key of keys) store.delete(key);
    await done;
  } finally {
    database.close();
  }
}

/**
 * Resolve when the transaction actually commits.
 *
 * Waiting on the individual request is not enough: a `put` can succeed and the
 * transaction still abort, e.g. on a quota error. Only `oncomplete` means written.
 */
function commit(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => {
      resolve();
    };
    transaction.onabort = () => {
      reject(transaction.error ?? new Error('Saving was interrupted'));
    };
    transaction.onerror = () => {
      reject(transaction.error ?? new Error('Saving failed'));
    };
  });
}
