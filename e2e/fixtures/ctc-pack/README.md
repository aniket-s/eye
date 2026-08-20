# Test fixture — synthetic CTC model pack

**Not a usable model.** Trained on procedurally generated sequences
(`training/mudra_train/ingest/sequences.py`).

It exists so `e2e/pack.spec.ts` can prove the continuous pipeline works in a real
browser: that a `temporal-ctc` manifest validates, that ONNX Runtime Web runs a model
with a _dynamic frame axis_, and that the app switches to the streaming decoder with no
dwell timer and no J/Z state machine.

Regenerate with:

```bash
cd training && PYTHONPATH=. python3 -m mudra_train.train_ctc --synthetic --out ../e2e/fixtures/ctc-pack
```
