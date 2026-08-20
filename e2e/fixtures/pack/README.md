# Test fixture — synthetic model pack

**Not a usable model.** Trained on procedurally generated handshapes
(`training/mudra_train/ingest/synthetic.py`), so it will fail on real hands.

It exists so `e2e/pack.spec.ts` can prove the v2 pipeline works in a real browser:
that a pack manifest validates, that ONNX Runtime Web initialises, that inference
runs, and that the app switches off the legacy correction heuristics.

Regenerate with:

```bash
cd training && PYTHONPATH=. python3 -m mudra_train.train --synthetic --out ../e2e/fixtures/pack
```
