# ADR 0004 — A monorepo with a DOM-free `core`

**Status:** Accepted · **Date:** 2026-08-19

## Context

v1 was a single 1,325-line HTML file mixing markup, styles, DOM wiring, feature extraction,
matrix arithmetic, geometric heuristics, and a motion state machine. None of it could be tested
without a browser and a camera, so none of it was tested.

The immediate task is to delete 85 lines of geometric heuristics and replace a model. Doing that
safely requires being able to assert what the current system does.

## Decision

An npm-workspaces monorepo:

- **`packages/core`** — pure TypeScript, **no DOM imports whatsoever**. Normalisation, features,
  models, temporal decoding, commit policy.
- **`packages/web`** — the application. UI, camera, workers. Contains no recognition decisions.
- **`training/`** — Python. Data ingest, landmark extraction, training, evaluation, ONNX export.

## Rationale

The split is not cosmetic. Because `core` cannot touch the DOM, every recognition decision is a
pure function testable in Node against recorded landmark fixtures. That is what makes the
`geometricFix` deletion a measurable change rather than a hopeful one: its current behaviour is
pinned by 41 assertions, and the replacement must pass the golden fixtures without it.

It also means the same logic can later run in a Web Worker, a Node benchmark, or a CI latency
harness without modification.

## Consequences

**Good**

- 103 unit tests run in ~2 s with no browser.
- Normalisation can be verified against the Python training code with shared test vectors — the
  train/inference skew that silently breaks deployed ML becomes a failing test.
- Clear ownership: a bug is either recognition (`core`) or presentation (`web`).

**Bad**

- Build ordering: `core` must compile before `web`. Handled by TypeScript project references.
- Slightly more ceremony for small changes.

## Rule

If a file in `packages/core` needs `window`, `document`, or `navigator`, the design is wrong.
Pass the data in instead.
