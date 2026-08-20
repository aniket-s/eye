# ADR 0005 — User-defined signs by embedding similarity, not retraining

**Status:** Accepted · **Date:** 2026-08-20

## Context

No public dataset will ever contain a user's name sign, their family's shorthand, or the
regional variant used in their city. Sign languages also vary by region in ways a single
trained pack cannot cover — ASL alone has documented regional lexical variation, and the gap
widens for languages with less data.

So the vocabulary has to be extensible by the person using it. The obvious approaches all
fail against this project's constraints:

- **Fine-tune in the browser.** Needs a training runtime (ONNX Runtime Web is inference-only),
  gradient computation, hyperparameters, and a way to avoid catastrophic forgetting. It would
  also mean every user holds a different set of weights, so a pack update either destroys
  their additions or cannot be applied.
- **Fine-tune on a server.** There is no budget for one, and it would send hand data off the
  device, contradicting ADR 0001.
- **Ship a second, user-trained classifier.** Same training problem, plus an arbitration
  question between two models with incomparable score scales.

## Decision

Train the base classifier with an **ArcFace (additive angular margin)** head, and let a user
define a sign by the **centroid of a handful of embeddings**.

ArcFace normalises both the embedding and the class weights, then applies an angular margin.
The result is that classes become tight, well-separated clusters on the unit hypersphere — the
property face-recognition systems rely on to identify people the network never trained on. A
class is then well described by the _mean_ of its examples.

So adding a sign is:

1. Record ~6 embeddings while the user holds the shape.
2. Average them; renormalise to unit length.
3. Classify by cosine similarity to that centroid.

No gradients, no server, no cost. Matching is a dot product per stored sign — microseconds for
any realistic collection, run in the recognition worker next to the embedding that produced it.

The pack exports both `logits` and `embedding` from one graph, so the feature costs one extra
output tensor and no extra inference.

### Arbitration

A custom sign is consulted **only when the trained vocabulary declines the frame**
(`CustomSignBook.matchIfUnrecognised`). A trained class rests on thousands of examples from
many signers; a custom sign on six from one. Letting six examples override a confident trained
prediction would trade the better evidence for the worse — and since users record custom signs
precisely _because_ they are absent from the vocabulary, the case where a custom sign should
beat a confident letter barely arises.

### Storage

IndexedDB, namespaced by pack id **and version**. Embeddings from one model are meaningless to
another, and that is equally true of two builds of the _same_ pack — retraining moves every
point in the space. `CustomSignBook` refuses to compare centroids of differing width, but a
retrain at the same width slips straight through that check and would match confidently against
nonsense, so the namespace has to carry the version too.

Signs from another version are hidden rather than used, and **counted**, so the app tells the
user their recordings were set aside by a model update. Silently dropping them would look like
data loss; silently using them would be worse.

## Consequences

**Good**

- New vocabulary at zero marginal cost, entirely on-device.
- Additions are plain data rather than weights, so they can be inspected, exported, or
  migrated without touching the model.
- The same mechanism generalises to a whole user-supplied lexicon later.

**Bad**

- **Weaker than training.** A centroid from six examples in one session, one lighting
  condition, one camera, will not generalise as well as a trained class. Mitigated by spacing
  samples ~350 ms apart, which captures the natural wobble of a held pose rather than six
  near-identical frames, and by reporting cohesion so the user is told when their examples
  disagreed.
- **Static handshapes only.** The embedding describes one frame. Signs that depend on movement
  need the temporal pipelines, which do not expose per-frame embeddings.
- **Collisions are the user's problem to resolve.** Two similar handshapes will flip. Detected
  at recording time and reported rather than silently accepted.
- **A pack update costs the user their recordings.** New weights mean new embeddings, so
  stored centroids stop describing anything and must be re-recorded. The store detects this by
  version and says so, but it cannot migrate them: there is no correspondence between two
  independently trained embedding spaces. Re-recording six examples is a half-minute per sign,
  which is the honest price of not running training on the device.
- **Requires an ArcFace-trained pack.** Older packs export no embedding. The app reports this
  rather than offering a feature that silently never matches.

## Alternatives considered

- **k-NN over stored embeddings** rather than a centroid. Slightly more expressive for
  multi-modal signs, but storage grows with every example and a single bad recording becomes a
  permanent false-positive source. The centroid averages that noise away.
- **Prototypical networks / episodic meta-training.** Better few-shot accuracy in principle,
  but it requires an episodic training regime over a large multi-class corpus, and the gain
  over an ArcFace centroid is small at this vocabulary size.
- **Plain cross-entropy plus a linear probe.** The penultimate layer of a cross-entropy model
  is not angularly structured, so cosine similarity there is much less reliable — which is the
  whole reason for the ArcFace head.
