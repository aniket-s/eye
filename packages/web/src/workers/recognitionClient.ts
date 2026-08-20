import {
  FRAME_BUFFER_LENGTH,
  HAND_BUFFER_LENGTH,
  packLandmarks,
  packVisionFrame,
  type HandLandmarks,
  type VisionFrame,
} from '@mudrapragyan/core';
import type { ConfusionProfile, CustomSign, Vocabularies } from '@mudrapragyan/core';
import type { WorkerRequest, WorkerResponse } from './protocol.js';

/**
 * Main-thread handle for the recognition worker.
 *
 * The important behaviour here is **backpressure**. Frames arrive at 30–60 Hz; if
 * inference is slower than that, queueing every frame would build an unbounded
 * backlog and the displayed letter would drift further behind the user's hand the
 * longer the camera ran. Instead only one frame is ever in flight and newer frames
 * are dropped while the worker is busy, so latency stays bounded and the result is
 * always the most recent one the worker could manage.
 *
 * Dropped frames are counted rather than ignored, because a rising drop rate is the
 * signal that the model has outgrown the device.
 */
export interface RecognitionResultMessage {
  readonly timestampMs: number;
  readonly letter: string;
  readonly rawLabel: string | null;
  readonly confidence: number | null;
  readonly reason: string | null;
  /** What the active vocabulary displays for `letter` — `2` where the model said `V`. */
  readonly display?: string;
  /** A user-defined sign that matched, when one did. */
  readonly custom?: { readonly label: string; readonly similarity: number };
  /** This frame's embedding, present only while capture is enabled. */
  readonly embedding?: readonly number[];
}

export interface ReadyInfo {
  readonly labelCount: number;
  readonly pipeline: 'v1' | 'v2' | 'ctc' | 'words';
  readonly packName?: string;
  /** Pack version — half the namespace custom signs are stored under. */
  readonly packVersion?: string;
  /** Whether the loaded pack exposes embeddings, and so can learn the user's own signs. */
  readonly supportsCustomSigns: boolean;
  /** Which letters this pack mistakes for which, for weighting word correction. */
  readonly confusions?: ConfusionProfile;
  /** Readings this pack offers, when its handshapes mean more than one thing. */
  readonly vocabularies?: Vocabularies;
}

/** Emitted by word-level (isolated) packs. */
export interface WordMessage {
  readonly timestampMs: number;
  readonly capturing: boolean;
  readonly sign: string | null;
  readonly probability: number;
  readonly alternatives: readonly string[];
  readonly motion: number;
}

/** Emitted by continuous (CTC) packs. */
export interface ContinuousMessage {
  readonly timestampMs: number;
  readonly committed: string;
  readonly provisional: string;
  readonly confidence: number;
}

export interface RecognitionClientOptions {
  readonly onReady: (info: ReadyInfo) => void;
  readonly onResult: (result: RecognitionResultMessage) => void;
  readonly onContinuous?: (result: ContinuousMessage) => void;
  readonly onWord?: (result: WordMessage) => void;
  readonly onError: (message: string) => void;
}

export class RecognitionClient {
  readonly #worker: Worker;
  readonly #options: RecognitionClientOptions;
  /** Reused so submitting a frame allocates nothing beyond the transfer itself. */
  #scratch = new Float32Array(HAND_BUFFER_LENGTH);
  #frameScratch = new Float32Array(FRAME_BUFFER_LENGTH);

  #ready = false;
  #pipeline: 'v1' | 'v2' | 'ctc' | 'words' = 'v1';
  #inFlight = false;
  #seq = 0;
  #droppedFrames = 0;
  #processedFrames = 0;

  constructor(options: RecognitionClientOptions) {
    this.#options = options;
    this.#worker = new Worker(new URL('./recognition.worker.ts', import.meta.url), {
      type: 'module',
      name: 'recognition',
    });

    this.#worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
      this.#handle(event.data);
    });
    this.#worker.addEventListener('error', (event) => {
      this.#options.onError(event.message || 'The recognition worker failed to start.');
    });
  }

  /** True once the model has loaded and frames will be processed. */
  get isReady(): boolean {
    return this.#ready;
  }

  /** Which recognition pipeline the worker settled on. */
  get pipeline(): 'v1' | 'v2' | 'ctc' | 'words' {
    return this.#pipeline;
  }

  /** Frames skipped because the worker was busy. */
  get droppedFrames(): number {
    return this.#droppedFrames;
  }

  /** Frames actually classified. */
  get processedFrames(): number {
    return this.#processedFrames;
  }

  /** Begin downloading and parsing the model, inside the worker. */
  init(modelUrl: string): void {
    this.#post({ type: 'init', modelUrl });
  }

  /** Clear motion history, e.g. when the hand leaves frame or a test restarts. */
  reset(): void {
    this.#post({ type: 'reset' });
  }

  /**
   * Choose which reading of the handshapes is active, or `null` for all of them.
   *
   * Sent to the worker rather than applied here: it changes what the model may predict,
   * and rewriting a label after the argmax has already picked an ineligible class is not
   * the same answer.
   */
  setVocabulary(name: string | null): void {
    this.#post({ type: 'vocabulary', name });
  }

  /** Install the user's own signs, so the worker can match against them. */
  setCustomSigns(signs: readonly CustomSign[]): void {
    this.#post({ type: 'customSigns', signs });
  }

  /**
   * Ask the worker to return each frame's embedding.
   *
   * Only enabled while the recorder is open: an embedding is ~128 floats, and shipping
   * one per frame for a whole session would be waste with no reader.
   */
  setCapture(enabled: boolean): void {
    this.#post({ type: 'capture', enabled });
  }

  /**
   * Submit a frame for classification.
   *
   * @returns `true` if the frame was sent, `false` if it was dropped.
   */
  submit(
    landmarks: HandLandmarks | null,
    timestampMs: number,
    handedness: 'left' | 'right' = 'right',
  ): boolean {
    if (!this.#ready) return false;

    if (this.#inFlight) {
      this.#droppedFrames++;
      return false;
    }

    this.#seq++;
    this.#inFlight = true;

    if (landmarks === null) {
      this.#post({ type: 'frame', seq: this.#seq, timestampMs, landmarks: null, handedness });
      return true;
    }

    // The buffer is transferred, so it is detached here and must be replaced.
    const buffer = packLandmarks(landmarks, this.#scratch);
    this.#scratch = new Float32Array(HAND_BUFFER_LENGTH);
    this.#worker.postMessage(
      {
        type: 'frame',
        seq: this.#seq,
        timestampMs,
        landmarks: buffer,
        handedness,
      } satisfies WorkerRequest,
      [buffer.buffer],
    );
    return true;
  }

  /**
   * Submit a whole frame, for word-level packs.
   *
   * Carries both hands and pose. Word signs are located relative to the body, so
   * sending only the primary hand would discard part of the meaning.
   *
   * @returns `true` if the frame was sent, `false` if it was dropped.
   */
  submitFrame(frame: VisionFrame, dominant: 'left' | 'right'): boolean {
    if (!this.#ready) return false;
    if (this.#inFlight) {
      this.#droppedFrames++;
      return false;
    }

    this.#seq++;
    this.#inFlight = true;

    const buffer = packVisionFrame(frame, dominant, this.#frameScratch);
    this.#frameScratch = new Float32Array(FRAME_BUFFER_LENGTH);
    this.#worker.postMessage(
      {
        type: 'frame',
        seq: this.#seq,
        timestampMs: frame.timestampMs,
        landmarks: null,
        handedness: dominant,
        frame: buffer,
      } satisfies WorkerRequest,
      [buffer.buffer],
    );
    return true;
  }

  /** Shut the worker down. */
  terminate(): void {
    this.#worker.terminate();
    this.#ready = false;
    this.#inFlight = false;
  }

  #post(message: WorkerRequest): void {
    this.#worker.postMessage(message);
  }

  #handle(message: WorkerResponse): void {
    switch (message.type) {
      case 'ready':
        this.#ready = true;
        this.#pipeline = message.pipeline;
        this.#options.onReady({
          labelCount: message.labelCount,
          pipeline: message.pipeline,
          supportsCustomSigns: message.supportsCustomSigns ?? false,
          ...(message.packName === undefined ? {} : { packName: message.packName }),
          ...(message.packVersion === undefined ? {} : { packVersion: message.packVersion }),
          ...(message.confusions === undefined ? {} : { confusions: message.confusions }),
          ...(message.vocabularies === undefined ? {} : { vocabularies: message.vocabularies }),
        });
        return;
      case 'error':
        this.#ready = false;
        this.#inFlight = false;
        this.#options.onError(message.message);
        return;
      case 'word':
        this.#inFlight = false;
        if (message.seq !== this.#seq) return;
        this.#processedFrames++;
        this.#options.onWord?.({
          timestampMs: message.timestampMs,
          capturing: message.capturing,
          sign: message.sign,
          probability: message.probability,
          alternatives: message.alternatives,
          motion: message.motion,
        });
        return;
      case 'continuous':
        this.#inFlight = false;
        if (message.seq !== this.#seq) return;
        this.#processedFrames++;
        this.#options.onContinuous?.({
          timestampMs: message.timestampMs,
          committed: message.committed,
          provisional: message.provisional,
          confidence: message.confidence,
        });
        return;
      case 'result':
        this.#inFlight = false;
        // Ignore anything but the newest request, so a late reply cannot overwrite
        // a fresher result.
        if (message.seq !== this.#seq) return;
        this.#processedFrames++;
        this.#options.onResult({
          timestampMs: message.timestampMs,
          letter: message.letter,
          rawLabel: message.rawLabel,
          confidence: message.confidence,
          reason: message.reason,
          ...(message.display === undefined ? {} : { display: message.display }),
          ...(message.custom === undefined ? {} : { custom: message.custom }),
          ...(message.embedding === undefined ? {} : { embedding: message.embedding }),
        });
    }
  }
}
