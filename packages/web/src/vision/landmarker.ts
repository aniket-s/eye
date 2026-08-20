import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type Category,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import type { Handedness, HandObservation, Landmark, VisionFrame } from '@mudrapragyan/core';
import {
  assetExists,
  HAND_MODEL_PATH,
  MISSING_ASSETS_MESSAGE,
  POSE_MODEL_PATH,
  WASM_BASE_PATH,
} from './assets.js';

/**
 * Landmark extraction on MediaPipe Tasks Vision 1.0.1.
 *
 * Replaces the legacy `@mediapipe/hands` solution, unsupported since March 2023
 * (docs/AUDIT.md, V1). Three substantive changes beyond the version bump:
 *
 * - **Two hands** instead of one, unblocking two-handed word signs (V3).
 * - **Pose landmarks** alongside hands, giving body-relative context for the
 *   word-level model in Phase 4 (V4).
 * - **Pull, not push.** The legacy API pushed results through `onResults`; Tasks
 *   Vision returns them synchronously from `detectForVideo`, so this class owns no
 *   callbacks and the caller controls when detection happens.
 *
 * A note on `HolisticLandmarker`: it exists in the JS bundle and would give face
 * landmarks (ASL non-manual markers are grammatically real), but it has no published
 * web documentation and a much larger model. Hand + Pose is the documented path;
 * Holistic is a Phase 4 decision.
 */

/** Pose landmark indices worth keeping — head and arms, not legs. */
const UPPER_BODY_POSE_INDICES = [
  0, // nose
  11,
  12, // shoulders
  13,
  14, // elbows
  15,
  16, // wrists
] as const;

export interface LandmarkerOptions {
  /** Detect pose as well as hands. Disable to save ~8 ms per frame. */
  readonly includePose?: boolean;
  readonly maxHands?: number;
}

export class VisionLandmarker {
  #hands: HandLandmarker | null = null;
  #pose: PoseLandmarker | null = null;
  #lastTimestampMs = -1;

  /** True once {@link load} has completed successfully. */
  get isReady(): boolean {
    return this.#hands !== null;
  }

  /** True if pose detection is running. */
  get hasPose(): boolean {
    return this.#pose !== null;
  }

  /**
   * Download and initialise the models.
   *
   * @throws Error with actionable guidance if the self-hosted assets are absent.
   */
  async load(options: LandmarkerOptions = {}): Promise<void> {
    if (!(await assetExists(HAND_MODEL_PATH))) {
      throw new Error(MISSING_ASSETS_MESSAGE);
    }

    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE_PATH);

    this.#hands = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: HAND_MODEL_PATH,
        // WebGL2. MediaPipe 1.0.1 has no WebGPU backend — verified by grepping the
        // shipped WASM for `navigator.gpu`, which appears zero times.
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: options.maxHands ?? 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    // Pose is optional: a missing model degrades the feature rather than the app.
    if (options.includePose !== false && (await assetExists(POSE_MODEL_PATH))) {
      this.#pose = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: POSE_MODEL_PATH, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numPoses: 1,
      });
    }
  }

  /**
   * Detect landmarks in one video frame.
   *
   * @param timestampMs Media time. Must strictly increase — MediaPipe rejects a
   *   repeated or regressing timestamp, which happens naturally when the camera
   *   delivers a duplicate frame, so those are nudged forward rather than dropped.
   */
  detect(video: HTMLVideoElement, timestampMs: number): VisionFrame {
    if (this.#hands === null) {
      throw new Error('VisionLandmarker.detect called before load()');
    }

    const timestamp = timestampMs > this.#lastTimestampMs ? timestampMs : this.#lastTimestampMs + 1;
    this.#lastTimestampMs = timestamp;

    const handResult = this.#hands.detectForVideo(video, timestamp);
    const hands: HandObservation[] = [];
    for (let i = 0; i < handResult.landmarks.length; i++) {
      const landmarks = handResult.landmarks[i];
      if (landmarks === undefined) continue;
      hands.push({
        handedness: readHandedness(handResult.handedness[i]),
        handednessScore: handResult.handedness[i]?.[0]?.score ?? 0,
        landmarks: landmarks.map(toLandmark),
      });
    }

    let pose: Landmark[] | null = null;
    if (this.#pose !== null) {
      const poseResult = this.#pose.detectForVideo(video, timestamp);
      const first = poseResult.landmarks[0];
      if (first !== undefined) {
        pose = UPPER_BODY_POSE_INDICES.map((index) => toLandmark(first[index]));
      }
    }

    return { timestampMs: timestamp, hands, pose };
  }

  /** Release GPU and WASM resources. */
  close(): void {
    this.#hands?.close();
    this.#pose?.close();
    this.#hands = null;
    this.#pose = null;
    this.#lastTimestampMs = -1;
  }
}

/**
 * Read MediaPipe's handedness label.
 *
 * MediaPipe assigns handedness *assuming the input image is mirrored*, as a selfie
 * preview would be. It is not: `getUserMedia` delivers the raw camera feed, and the
 * mirroring in this app is a CSS `scaleX(-1)` on the video element, which changes what
 * the user sees and nothing about the pixels sampled here. So its labels must be
 * swapped, which is what this does.
 *
 * Correcting it here means the rest of the codebase can treat `handedness` as the
 * signer's own hand, which is what the v2 normaliser canonicalises on (M1).
 *
 * See {@link RAW_FRAME_CONVENTION} for the other half of this, which is easier to get
 * wrong and far more expensive.
 */
function readHandedness(categories: Category[] | undefined): Handedness {
  const label = categories?.[0]?.categoryName ?? '';
  return label === 'Left' ? 'right' : 'left';
}

/**
 * **Landmarks are in raw camera coordinates, not preview coordinates.**
 *
 * Documented as a constant rather than a comment because training has to agree with it,
 * and the two live in different languages and different directories.
 *
 * A camera looks *at* the signer, so the signer's right side lands on the left of the
 * frame. In the coordinates MediaPipe emits, therefore, a **right hand has its thumb at
 * a smaller x than its pinky**. The mirrored preview shows the opposite, which is why
 * this is so easy to get backwards.
 *
 * Training data must be generated in this same frame. It once was not, and the failure
 * was silent and total: the simulator was internally consistent, scored 0.968 on its own
 * held-out signers, and rejected 69% of real letters as `none` because every hand it saw
 * at inference was a mirror image of everything it had trained on. Normalisation cannot
 * absorb it — that canonicalises left hands to right, not raw frames to mirrored ones.
 *
 * Pinned by `test_a_right_hand_puts_its_thumb_on_the_left_of_the_image` in
 * `training/tests/test_handmodel.py`.
 */
export const RAW_FRAME_CONVENTION =
  "MediaPipe landmarks are in raw camera coordinates: a right hand's thumb sits at a " +
  'smaller x than its pinky. The mirrored preview is CSS only.';

function toLandmark(point: NormalizedLandmark | undefined): Landmark {
  return { x: point?.x ?? 0, y: point?.y ?? 0, z: point?.z ?? 0 };
}
