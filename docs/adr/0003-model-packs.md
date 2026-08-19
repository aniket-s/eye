# ADR 0003 — Models ship as versioned, self-describing packs

**Status:** Accepted · **Date:** 2026-08-19

## Context

v1 hardcoded one model, its label list, its confidence threshold (`0.45`), its normalisation,
and 85 lines of correction heuristics across the application. Adding a second model — Indian
Sign Language, a word-level model, a user's own signs — meant editing application code.

Worse, the operating point lived in the app rather than with the model, so retraining silently
invalidated thresholds tuned for the previous weights. That coupling is the root cause of the
`geometricFix` trap described in `docs/AUDIT.md` §2.

## Decision

A model is a **pack**: a directory carrying everything needed to run it.

```
models/asl-fingerspell/v1.2.0/
├── manifest.json   # id, version, task type, labels, input spec, normalisation
│                   # constants, thresholds, metrics, licence, sha256
├── model.onnx      # int8-quantised
└── card.md         # data, splits, metrics, limitations, intended use
```

The app reads a registry index and offers every pack whose runtime requirements it satisfies.

## Consequences

**Good**

- Adding ISL, a 250-word model, or user-recorded signs is **a new pack, not a code change**.
- **Thresholds and normalisation travel with the weights.** A retrained model ships its own
  operating point, making the v1 class of bug structurally impossible.
- Packs are content-addressed by `sha256`, so CI can verify what shipped.
- Every pack carries a **model card**, so limitations are documented rather than discovered.

**Bad**

- Indirection: the app can no longer assume one label set, so the UI must be driven by the
  manifest.
- The manifest schema becomes a compatibility surface and needs its own versioning.
