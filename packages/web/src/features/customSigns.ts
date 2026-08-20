import { buildCustomSign, CustomSignBook, MIN_SAMPLES, type CustomSign } from '@mudrapragyan/core';
import { requireElement } from '../dom.js';
import {
  clearCustomSigns,
  deleteCustomSign,
  loadCustomSigns,
  saveCustomSign,
} from '../model/customSignStore.js';
import type { RecognitionClient } from '../workers/recognitionClient.js';

/**
 * Teaching the app a sign it was never trained on.
 *
 * The mechanism is described in `packages/core/src/registry/customSigns.ts`: the pack's
 * ArcFace head puts every handshape on a unit hypersphere, so averaging a handful of
 * embeddings gives a usable class centroid. No training, no server, no cost.
 *
 * This module owns only the *interaction* — recording, naming, listing, deleting. The
 * mathematics is in `core` and the matching runs in the worker.
 *
 * The flow is deliberately slow in one place: samples are taken at
 * {@link SAMPLE_INTERVAL_MS} apart rather than on consecutive frames. Consecutive frames
 * of a held hand are nearly identical, so a centroid built from them looks beautifully
 * cohesive and generalises to nothing. Spacing them out samples the natural wobble of a
 * held pose, which is what the sign will actually look like in use.
 */

/**
 * How many examples to record.
 *
 * Comfortably above `core`'s minimum, and derived from it so raising that floor can
 * never leave this asking for fewer samples than `buildCustomSign` accepts.
 */
export const SAMPLES_TO_RECORD = Math.max(6, MIN_SAMPLES);

/** Spacing between kept samples, in milliseconds. */
export const SAMPLE_INTERVAL_MS = 350;

/** Below this cohesion the samples disagree enough that the sign will match poorly. */
export const LOW_COHESION = 0.9;

/** Above this, two signs are too close to tell apart reliably. */
export const COLLISION_SIMILARITY = 0.93;

type Phase = 'idle' | 'recording';

export class CustomSignsPanel {
  readonly #book = new CustomSignBook();
  readonly #panel = requireElement('customPanel');
  readonly #list = requireElement('customList');
  readonly #status = requireElement('customStatus');
  readonly #nameInput = requireElement<HTMLInputElement>('customName');
  readonly #recordButton = requireElement<HTMLButtonElement>('customRecord');
  readonly #clearButton = requireElement<HTMLButtonElement>('customClear');

  #client: RecognitionClient | null = null;
  #packId = '';
  #packVersion = '';
  #phase: Phase = 'idle';
  #samples: number[][] = [];
  #lastSampleMs = 0;
  #pendingName = '';

  /**
   * Show the panel and load anything already recorded.
   *
   * @param pack Which pack build is loaded. The version is part of the identity: a
   *   retrain moves the embedding space, so centroids recorded against an earlier build
   *   describe nothing in this one.
   * @param supported Whether the loaded pack exposes embeddings. When it does not, the
   *   panel explains why rather than hiding, so the feature is discoverable once a
   *   capable pack is installed.
   */
  async attach(
    client: RecognitionClient,
    pack: { id: string; version: string },
    supported: boolean,
  ): Promise<void> {
    this.#client = client;
    this.#packId = pack.id;
    this.#packVersion = pack.version;
    this.#panel.hidden = false;

    if (!supported) {
      this.#recordButton.disabled = true;
      this.#nameInput.disabled = true;
      this.#clearButton.disabled = true;
      this.#setStatus(
        'This model pack cannot learn new signs. Packs trained with an embedding head can.',
      );
      return;
    }

    const { signs, staleCount } = await loadCustomSigns(pack.id, pack.version);
    this.#book.load(signs);
    client.setCustomSigns(this.#book.list());
    this.#render();

    if (staleCount > 0) {
      // Silently dropping them would look like data loss; silently *using* them would
      // match against a space that no longer exists. Say what happened instead.
      this.#setStatus(
        `${staleCount} sign${staleCount === 1 ? '' : 's'} recorded against an older ` +
          'version of this model are set aside — the model changed, so they need ' +
          're-recording.',
        true,
      );
    }
  }

  bind(): void {
    this.#recordButton.addEventListener('click', () => {
      if (this.#phase === 'recording') this.#cancel();
      else this.#begin();
    });
    this.#clearButton.addEventListener('click', () => void this.#clearAll());
    this.#list.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const id = target.dataset['removeSign'];
      if (id !== undefined) void this.#remove(id);
    });
  }

  /**
   * Feed one frame's embedding in.
   *
   * Called for every result while recording; most are ignored, because sampling is
   * paced rather than per-frame.
   */
  observe(embedding: readonly number[] | undefined, timestampMs: number): void {
    if (this.#phase !== 'recording' || embedding === undefined) return;
    if (timestampMs - this.#lastSampleMs < SAMPLE_INTERVAL_MS) return;

    this.#lastSampleMs = timestampMs;
    this.#samples.push([...embedding]);
    this.#setStatus(`Hold the sign… ${this.#samples.length} of ${SAMPLES_TO_RECORD}`);

    if (this.#samples.length >= SAMPLES_TO_RECORD) void this.#finish();
  }

  /** True while examples are being collected. */
  get isRecording(): boolean {
    return this.#phase === 'recording';
  }

  #begin(): void {
    const name = this.#nameInput.value.trim();
    if (name === '') {
      this.#setStatus('Give the sign a name first.', true);
      this.#nameInput.focus();
      return;
    }
    if (this.#client === null) {
      this.#setStatus('Start the camera first.', true);
      return;
    }

    this.#pendingName = name;
    this.#samples = [];
    // Deliberately in the past, so the first frame with an embedding is taken
    // immediately rather than after a dead interval.
    this.#lastSampleMs = Number.NEGATIVE_INFINITY;
    this.#phase = 'recording';
    this.#client.setCapture(true);
    this.#recordButton.textContent = 'Cancel';
    this.#setStatus(`Hold the sign… 0 of ${SAMPLES_TO_RECORD}`);
  }

  #cancel(): void {
    this.#stopRecording();
    this.#setStatus('Cancelled.');
  }

  #stopRecording(): void {
    this.#phase = 'idle';
    this.#samples = [];
    this.#client?.setCapture(false);
    this.#recordButton.textContent = 'Record a sign';
  }

  async #finish(): Promise<void> {
    const samples = this.#samples;
    const name = this.#pendingName;
    this.#stopRecording();

    let sign: CustomSign;
    try {
      sign = buildCustomSign(slug(name), name, samples, Date.now());
    } catch (error) {
      this.#setStatus(error instanceof Error ? error.message : 'Could not build the sign.', true);
      return;
    }

    this.#book.add(sign);

    try {
      await saveCustomSign(this.#packId, this.#packVersion, sign);
    } catch {
      // The sign still works this session; the user simply needs to know it will not
      // survive a reload. Silence here would be the worse failure.
      this.#book.remove(sign.id);
      this.#setStatus('Could not save the sign — storage is unavailable in this browser.', true);
      return;
    }

    this.#client?.setCustomSigns(this.#book.list());
    this.#nameInput.value = '';
    this.#render();
    this.#reportQuality(sign);
  }

  /**
   * Say something useful about the sign just recorded.
   *
   * Both warnings are about problems the user cannot see and would otherwise discover
   * only through unreliable recognition later.
   */
  #reportQuality(sign: CustomSign): void {
    const collision = this.#book
      .findCollisions(COLLISION_SIMILARITY)
      .find(({ a, b }) => a.id === sign.id || b.id === sign.id);

    if (collision !== undefined) {
      const other = collision.a.id === sign.id ? collision.b : collision.a;
      this.#setStatus(
        `Saved “${sign.label}”, but it looks very like “${other.label}”. ` +
          'Recognition will flip between them — try a more distinct handshape.',
        true,
      );
      return;
    }

    if (sign.cohesion < LOW_COHESION) {
      this.#setStatus(
        `Saved “${sign.label}”, but the examples varied a lot. ` +
          'Re-record it holding the shape steadier for better results.',
        true,
      );
      return;
    }

    this.#setStatus(`Saved “${sign.label}”. Show it to the camera to try it.`);
  }

  async #remove(id: string): Promise<void> {
    this.#book.remove(id);
    await deleteCustomSign(this.#packId, id);
    this.#client?.setCustomSigns(this.#book.list());
    this.#render();
    this.#setStatus('Removed.');
  }

  async #clearAll(): Promise<void> {
    if (this.#book.size === 0) return;
    this.#book.clear();
    await clearCustomSigns(this.#packId);
    this.#client?.setCustomSigns([]);
    this.#render();
    this.#setStatus('All custom signs removed.');
  }

  #render(): void {
    this.#clearButton.disabled = this.#book.size === 0;
    this.#list.textContent = '';

    if (this.#book.size === 0) {
      const empty = document.createElement('li');
      empty.className = 'custom-empty';
      empty.textContent = 'No custom signs yet.';
      this.#list.append(empty);
      return;
    }

    for (const sign of this.#book.list()) {
      const item = document.createElement('li');
      item.className = 'custom-item';

      const label = document.createElement('span');
      label.className = 'custom-item-name';
      // textContent, not innerHTML: the name is user input and goes nowhere near a
      // parser that could interpret it.
      label.textContent = sign.label;

      const meta = document.createElement('span');
      meta.className = 'custom-item-meta';
      meta.textContent = `${sign.sampleCount} examples`;
      if (sign.cohesion < LOW_COHESION) {
        meta.textContent += ' · inconsistent';
        meta.classList.add('is-weak');
      }

      const remove = document.createElement('button');
      remove.className = 'custom-item-remove';
      remove.type = 'button';
      remove.dataset['removeSign'] = sign.id;
      remove.textContent = '✕';
      remove.setAttribute('aria-label', `Remove ${sign.label}`);

      item.append(label, meta, remove);
      this.#list.append(item);
    }
  }

  #setStatus(message: string, warning = false): void {
    this.#status.textContent = message;
    this.#status.classList.toggle('is-warning', warning);
  }
}

/** A storage key from a display name. Collisions overwrite, which is the useful behaviour. */
function slug(name: string): string {
  const cleaned = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  // A name of only punctuation would otherwise produce an empty key, and every such
  // sign would overwrite the last.
  return cleaned === '' ? `sign-${Date.now().toString(36)}` : cleaned;
}
