# ADR 0002 — ONNX Runtime Web for custom models, not TensorFlow.js

**Status:** Accepted · **Date:** 2026-08-19

## Context

Given ADR 0001, we need a browser runtime for our own trained classifiers. The candidates are
TensorFlow.js, ONNX Runtime Web, and LiteRT.js.

## Decision

**`onnxruntime-web@1.27.0`, WASM execution provider, `numThreads = 1`, in a Web Worker.**
Import the `onnxruntime-web/wasm` entry point specifically.

## Rationale

**TensorFlow.js is in de-facto maintenance mode.** Its last stable release was **4.22.0 in
October 2024** — roughly 22 months, spanning the entire period in which WebGPU shipped. There is
no deprecation notice, but a browser runtime that silent for that long is not a foundation.
Relatedly, `@tensorflow-models/hand-pose-detection` last shipped in July 2023.

**ORT ships every 6–8 weeks**, `torch.onnx.export` is a one-liner from our PyTorch training
code, and int8 dynamic quantisation is mature.

**Bundle economics.** `ort.wasm.min.mjs` is 49 KB; the WASM binary is 12.9 MB raw / **3.3 MB
gzipped**. The WebGPU (`jsep`) binary is 25.6 MB / 5.9 MB gzipped — nearly double, for no
benefit at our model size.

**`numThreads = 1` is deliberate.** Multi-threaded WASM needs `SharedArrayBuffer`, which needs
cross-origin isolation (COOP/COEP), which **GitHub Pages cannot provide**. Our models are small
enough that thread-pool spin-up would exceed a 10 ms inference anyway. Setting 1 thread makes
the entire COOP/COEP question disappear and keeps GitHub Pages viable as a mirror.

**LiteRT.js** is the credible runner-up — smaller WASM, real WebGPU. Rejected because its value
is WebGPU (which we do not need), `.tflite` export adds a fragile `ai-edge-torch` step to CI,
it is young, and its docs steer toward pairing it with TF.js tensors.

## Compute budget

At 30 fps we have 33 ms/frame. MediaPipe takes 15–25 ms on a laptop CPU, leaving **≤10 ms**:

| Model              | Params | int8 size | Latency | Stride      |
| ------------------ | ------ | --------- | ------- | ----------- |
| Static handshape   | ~200 k | ~250 KB   | <1 ms   | every frame |
| Fingerspelling CTC | ~1.5 M | ~2 MB     | ~8 ms   | 5 frames    |
| Word-level (250)   | ~1.8 M | ~2 MB     | ~10 ms  | 10 frames   |

This is why temporal models are capped at ~2M parameters and run on a stride rather than every
frame. A 5M-parameter transformer over 60 frames would cost 50–120 ms — three frames' worth.

## Consequences

- Training must target ONNX-exportable operations.
- JS and WASM **must come from the same build**; mixing a CDN JS with a self-hosted WASM fails
  at init with missing-symbol errors. Both are copied from the same `node_modules` install.
