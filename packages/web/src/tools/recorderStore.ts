/**
 * Local persistence for recorded landmark samples.
 *
 * The recorder used to hold samples in a page-scoped array, so a refresh, a crash, or
 * an accidental navigation discarded the whole session — silently, discovered only
 * when the downloaded file came up short. Samples now autosave as they are captured
 * and are restored on the next visit, so the only way to lose them is to choose Clear.
 *
 * IndexedDB rather than `localStorage`, for the same reasons as `customSignStore`:
 * a session across several conditions can pass `localStorage`'s ~5 MB cap, and writes
 * happen up to sixteen times a second, where a synchronous string API would be felt.
 *
 * A separate database from the app's, deliberately: the recorder is a developer tool,
 * and coupling its schema to `customSigns` would force a version migration on every
 * user for a store only trainers ever open.
 *
 * Everything stays on the device. Nothing is ever uploaded (docs/PRIVACY.md).
 */

export interface RecorderSample {
  readonly label: string;
  readonly signer: string;
  readonly hand: string;
  readonly condition: string;
  readonly timestampMs: number;
  /** Flattened 21 × (x, y, z), rounded to 5 decimals to keep the file small. */
  readonly landmarks: number[];
}

const DATABASE = 'mudrapragyan-recorder';
const DATABASE_VERSION = 1;
const STORE = 'samples';

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, DATABASE_VERSION);

    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE)) {
        database.createObjectStore(STORE, { autoIncrement: true });
      }
    };

    request.onsuccess = () => {
      resolve(request.result);
    };
    request.onerror = () => {
      reject(request.error ?? new Error('Could not open the local sample database'));
    };
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
      reject(request.error ?? new Error('Local sample storage failed'));
    };
  });
}

function commit(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => {
      resolve();
    };
    transaction.onerror = () => {
      reject(transaction.error ?? new Error('Local sample storage failed'));
    };
    transaction.onabort = () => {
      reject(transaction.error ?? new Error('Local sample storage was aborted'));
    };
  });
}

/** Shape check for restored rows, so a corrupted row degrades to a skip, not a crash. */
function isSample(value: unknown): value is RecorderSample {
  if (typeof value !== 'object' || value === null) return false;
  const raw = value as Record<string, unknown>;
  return (
    typeof raw['label'] === 'string' &&
    typeof raw['signer'] === 'string' &&
    typeof raw['hand'] === 'string' &&
    typeof raw['condition'] === 'string' &&
    typeof raw['timestampMs'] === 'number' &&
    Array.isArray(raw['landmarks']) &&
    raw['landmarks'].every((entry) => typeof entry === 'number')
  );
}

/**
 * Samples autosaved by previous sessions, in capture order.
 *
 * @returns Empty if storage is unavailable — private browsing modes block IndexedDB
 *   outright, and the recorder must still work there, just without persistence.
 */
export async function loadSamples(): Promise<RecorderSample[]> {
  try {
    const database = await open();
    try {
      const store = database.transaction(STORE, 'readonly').objectStore(STORE);
      const rows = (await run(store.getAll())) as unknown[];
      return rows.filter(isSample);
    } finally {
      database.close();
    }
  } catch {
    return [];
  }
}

/**
 * Autosave one sample.
 *
 * @throws Error if storage is unavailable or full. Unlike reading, a failed save must
 *   be visible: someone recording for half an hour needs to know their samples are not
 *   surviving a refresh, while there is still time to download.
 */
export async function saveSample(sample: RecorderSample): Promise<void> {
  const database = await open();
  try {
    const transaction = database.transaction(STORE, 'readwrite');
    const done = commit(transaction);
    transaction.objectStore(STORE).add(sample);
    await done;
  } finally {
    database.close();
  }
}

/** Forget every autosaved sample. */
export async function clearSamples(): Promise<void> {
  const database = await open();
  try {
    const transaction = database.transaction(STORE, 'readwrite');
    const done = commit(transaction);
    transaction.objectStore(STORE).clear();
    await done;
  } finally {
    database.close();
  }
}
