/**
 * Messages exchanged with the recognition worker.
 *
 * Landmarks travel as a transferable `Float32Array` rather than an array of objects,
 * so the hot path allocates nothing per frame on either side (docs/AUDIT.md, A5).
 */
import type { ConfusionProfile, CustomSign, Vocabularies } from '@mudrapragyan/core';

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
      /**
       * Whole-frame payload: both hands plus upper-body pose.
       *
       * Only sent for word-level packs, which need hand position relative to the body.
       * The fingerspelling pipelines use `landmarks` alone, which is a third the size.
       */
      readonly frame?: Float32Array;
    }
  | { readonly type: 'reset' }
  | {
      /**
       * Install the user's own signs.
       *
       * Sent whenever the collection changes. The worker owns matching so it happens on
       * the same thread as inference, next to the embedding that was just produced.
       */
      readonly type: 'customSigns';
      readonly signs: readonly CustomSign[];
    }
  | {
      /**
       * Start or stop returning embeddings with each result.
       *
       * Off by default: an embedding is ~128 floats, and posting one per frame for the
       * whole session would be pure waste when nobody is recording.
       */
      readonly type: 'capture';
      readonly enabled: boolean;
    }
  | {
      /**
       * Choose which reading of the trained handshapes is active.
       *
       * `null` allows every label. Named vocabularies come from the pack manifest — see
       * `vocabularies` there for why numbers need this rather than their own classes.
       */
      readonly type: 'vocabulary';
      readonly name: string | null;
    };

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
       * - `words` — isolated word signs; segments on motion, classifies each once.
       */
      readonly pipeline: 'v1' | 'v2' | 'ctc' | 'words';
      /** Pack name, when a v2 pack is installed. */
      readonly packName?: string;
      /**
       * Pack version.
       *
       * Part of the identity of the embedding space, not decoration: custom signs
       * recorded against an earlier build of the same pack describe nothing in this one.
       */
      readonly packVersion?: string;
      /**
       * Whether this pack exposes embeddings, and so can learn the user's own signs.
       *
       * Reported rather than assumed: packs exported before the ArcFace head cannot,
       * and the UI must say so instead of offering a feature that never matches.
       */
      readonly supportsCustomSigns?: boolean;
      /**
       * Which letters this pack mistakes for which.
       *
       * Sent to the main thread because word correction lives there, next to the sentence
       * being built — and because it is a property of the pack, so the app should never
       * have to be told separately which model is loaded.
       */
      readonly confusions?: ConfusionProfile;
      /** Readings this pack offers, if more than one. Drives the mode switch in the UI. */
      readonly vocabularies?: Vocabularies;
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
      /**
       * A user-defined sign that matched this frame more confidently than the
       * built-in vocabulary, if any.
       */
      readonly custom?: { readonly label: string; readonly similarity: number };
      /** This frame's embedding. Only present while capture is enabled. */
      readonly embedding?: readonly number[];
      /**
       * What the active vocabulary displays for `letter`.
       *
       * Differs from `letter` when the handshape has more than one reading — the model
       * predicts `V`, and in numbers mode it shows as `2`.
       */
      readonly display?: string;
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
    }
  | {
      /** Emitted by `temporal-isolated` packs, once per completed sign. */
      readonly type: 'word';
      readonly seq: number;
      readonly timestampMs: number;
      /** True while a sign is being captured. */
      readonly capturing: boolean;
      /** Recognised sign, or null when nothing cleared the threshold. */
      readonly sign: string | null;
      readonly probability: number;
      /** Runners-up, for a "did you mean" affordance. */
      readonly alternatives: readonly string[];
      readonly motion: number;
    };
