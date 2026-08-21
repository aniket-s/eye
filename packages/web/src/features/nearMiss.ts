/**
 * Telling the signer *why* a letter did not appear.
 *
 * A rejected frame produced nothing on screen: the big letter stayed at `—` and the
 * label read "Show your hand", which is what it says when there is no hand at all. So
 * the two situations a signer most needs to tell apart — "the camera cannot see you"
 * and "you are one small adjustment away from a G" — looked identical, and there was
 * no way to discover which was happening or what to change. Letters that fail this way
 * do not read as near misses; they read as letters the app cannot do.
 *
 * That failure mode is not hypothetical. Measured on the pack shipped before this
 * change, a G held at the orientation the app's own dictionary illustrated was accepted
 * 0% of the time and fell to the `none` class in every frame. The recogniser was
 * working exactly as designed and the signer had no way to know what to change.
 *
 * Everything here is derived from the verdict the pipeline already produces
 * (`decode/rejection.ts`) plus the pack's own acceptance profile. Nothing is guessed,
 * and no letter gets a special case in the recogniser.
 */
/** The parts of a frame's verdict a hint is built from. */
export interface Verdict {
  /** The model's own top label, before thresholds. */
  readonly rawLabel: string | null;
  /** Its probability. */
  readonly confidence: number | null;
  /** Which check rejected the frame, from `judge()`. */
  readonly reason: string | null;
}

/**
 * Letters whose orientation is their whole identity, and what to say about each.
 *
 * These four are the same handshape as another letter at a different angle — H is U
 * rolled, P is K rolled, Q is G rolled, and G is the one the others are measured
 * against — so for them a rejection is far more often "not turned far enough" than
 * "wrong shape". Telling a signer to adjust their *shape* when the shape is already
 * right is worse than saying nothing.
 */
const ORIENTATION_HINTS: Readonly<Record<string, string>> = {
  G: 'lay your hand on its side so the finger points across',
  H: 'lay your hand on its side so both fingers point across',
  P: 'turn your wrist further down, so the middle finger points at the floor',
  Q: 'turn your wrist further down, so the finger points at the floor',
};

/** What each rejection reason means, in the signer's terms rather than the decoder's. */
const REASONS: Readonly<Record<string, string>> = {
  'none-class': 'not reading as a letter yet',
  'low-probability': 'not sure enough yet',
  'low-margin': 'torn between two letters',
  'high-energy': 'this pose is unlike anything it was trained on',
};

export interface NearMiss {
  /** The letter it is closest to, or `null` when there is nothing worth reporting. */
  readonly letter: string | null;
  /** One line for the signer. */
  readonly message: string;
  /** Longer detail for the debug panel. */
  readonly detail: string;
}

/** Nothing useful to say — the hand is absent, or the frame is simply idle. */
const SILENT: NearMiss = { letter: null, message: 'Show your hand', detail: '—' };

/**
 * Below this the model is not "nearly" anything, it is idle.
 *
 * A hand resting between letters produces a low-probability winner every frame, and
 * announcing that as a near miss would put a different letter on screen thirty times a
 * second — noise that is worse than the silence it replaced.
 */
const WORTH_MENTIONING = 0.3;

/**
 * Explain a rejected frame.
 *
 * @param verdict This frame's verdict.
 * @param acceptance The pack's acceptance profile, if it carries one. Optional: packs
 *   exported before it existed have none, and the hint is still useful without it.
 */
export function explain(
  verdict: Verdict,
  acceptance?: Readonly<Record<string, number>> | null,
): NearMiss {
  const { rawLabel, confidence, reason } = verdict;
  if (rawLabel === null || rawLabel === 'none' || confidence === null) return SILENT;
  if (confidence < WORTH_MENTIONING) return SILENT;

  const percent = Math.round(confidence * 100);
  const because = REASONS[reason ?? ''] ?? 'not accepted';
  const hint = ORIENTATION_HINTS[rawLabel];

  const message =
    hint === undefined ? `Nearly ${rawLabel} — hold it steadier` : `Nearly ${rawLabel} — ${hint}`;

  return {
    letter: rawLabel,
    message,
    detail: `${rawLabel} ${percent}% · ${because}${describeAcceptance(rawLabel, acceptance)}`,
  };
}

/**
 * What this pack manages on this letter, when it says.
 *
 * A signer struggling with one letter deserves to know whether they are fighting their
 * own hand or the model: "this pack gets R right 61% of the time" is an answer, and it
 * is one that would otherwise take a retrain to discover.
 */
function describeAcceptance(
  letter: string,
  acceptance?: Readonly<Record<string, number>> | null,
): string {
  const share = acceptance?.[letter];
  if (share === undefined) return '';
  return ` · this pack accepts ${letter} ${Math.round(share * 100)}% of the time`;
}
