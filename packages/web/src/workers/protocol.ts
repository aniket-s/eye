/**
 * Messages exchanged with the recognition worker.
 *
 * Landmarks travel as a transferable `Float32Array` rather than an array of objects,
 * so the hot path allocates nothing per frame on either side (docs/AUDIT.md, A5).
 */

/** Main thread → worker. */
export type WorkerRequest =
  | { readonly type: 'init'; readonly modelUrl: string }
  | {
      readonly type: 'frame';
      /** Sequence number, echoed back so stale results can be discarded. */
      readonly seq: number;
      readonly timestampMs: number;
      /** Packed landmarks of the primary hand, or `null` when no hand is visible. */
      readonly landmarks: Float32Array | null;
    }
  | { readonly type: 'reset' };

/** Worker → main thread. */
export type WorkerResponse =
  | { readonly type: 'ready'; readonly labelCount: number }
  | { readonly type: 'error'; readonly message: string }
  | {
      readonly type: 'result';
      readonly seq: number;
      readonly timestampMs: number;
      readonly letter: string;
      readonly rawLabel: string | null;
      readonly confidence: number | null;
    };
