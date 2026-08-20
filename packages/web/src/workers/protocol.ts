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
      /**
       * Which of the signer's hands it is.
       *
       * Required by the v2 pipeline, which canonicalises left hands during
       * normalisation. Sending the wrong value would mirror the geometry and produce
       * confident nonsense, so it travels with the landmarks rather than being
       * inferred later.
       */
      readonly handedness: 'left' | 'right';
    }
  | { readonly type: 'reset' };

/** Worker → main thread. */
export type WorkerResponse =
  | {
      readonly type: 'ready';
      readonly labelCount: number;
      /**
       * Which pipeline is live.
       *
       * - `v1` — legacy MLP plus correction heuristics, used only when no pack exists.
       * - `v2` — static handshape pack with proper normalisation and rejection.
       * - `ctc` — continuous fingerspelling; no dwell timer, J and Z are ordinary labels.
       */
      readonly pipeline: 'v1' | 'v2' | 'ctc';
      /** Pack name, when a v2 pack is installed. */
      readonly packName?: string;
    }
  | { readonly type: 'error'; readonly message: string }
  | {
      readonly type: 'result';
      readonly seq: number;
      readonly timestampMs: number;
      readonly letter: string;
      readonly rawLabel: string | null;
      readonly confidence: number | null;
      /** Why the frame was accepted or rejected, for the debug overlay. */
      readonly reason: string | null;
    }
  | {
      /** Emitted by `temporal-ctc` packs, which decode whole windows. */
      readonly type: 'continuous';
      readonly seq: number;
      readonly timestampMs: number;
      /** Text the decoder is confident about. Only ever grows. */
      readonly committed: string;
      /** Current best guess for the tail, rendered greyed. */
      readonly provisional: string;
      readonly confidence: number;
    };
