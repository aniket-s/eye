# Test fixture — synthetic word-sign model pack

**Not a usable model.** Fifteen procedurally generated signs
(`training/mudra_train/ingest/words.py`), each with a distinct handshape and motion
path, so they are trivially separable. Real word recognition is far harder.

It exists so `e2e/pack.spec.ts` can prove the word pipeline works in a browser: that a
`temporal-isolated` manifest validates, that a two-handed body-relative pack loads, and
that the app switches to segment-then-classify.

Regenerate with:

```bash
cd training && PYTHONPATH=. python3 -m mudra_train.train_words --synthetic --out ../e2e/fixtures/word-pack
```
