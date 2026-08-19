# ADR 0001 — All inference runs on the client

**Status:** Accepted · **Date:** 2026-08-19

## Context

The project has no budget: no paid APIs, no paid hosting, no paid compute. It also handles
camera video of people, which is sensitive by nature, and it targets users who may have poor
or intermittent connectivity.

The obvious alternative is a server that receives frames and returns predictions.

## Decision

**Every model runs in the user's browser.** No video, landmarks, or predictions leave the
device. The app is a static bundle deployable to any static host.

## Consequences

**Good**

- **Privacy is structural, not promised.** There is no server to trust, log, or breach. This is
  the honest version of a claim most competitors can only assert.
- **Zero marginal cost.** Serving the ten-thousandth user costs the same as the first.
- **Works offline** once assets are cached, which matters for the intended users.
- **No cold starts.** A free-tier backend would add 2–10 s to the first prediction.

**Bad**

- **Hard model-size ceiling.** ~10–20 MB of cold-cache assets is the practical limit, which
  rules out large video models and forces landmark-based architectures.
- **Compute budget is the user's device.** At 30 fps we have 33 ms per frame, of which
  MediaPipe takes 15–25 ms. See ADR 0002 for how that constrains model size.
- **No central telemetry.** Model regressions cannot be detected from production; offline
  evaluation and CI gates have to carry that weight instead.

## Notes

This is not a compromise forced by the budget. For a camera-based accessibility tool,
client-side inference is the architecture a well-funded team should also choose.
