# `legacy/` — scheduled for deletion

Everything in this directory is **v1 behaviour preserved verbatim** so that Phase 0 could
restructure the codebase without changing what the app does. None of it should be extended,
and no new code should depend on it outside `packages/web/src/vision/legacyPipeline.ts`.

| Module                              | Replaced by                                           | Phase |
| ----------------------------------- | ----------------------------------------------------- | ----- |
| `geometricFix.ts`                   | A correctly-normalised classifier with a `none` class | 2     |
| `jzStateMachine.ts`                 | CTC decoding, where J and Z are ordinary labels       | 3     |
| `holdCommit.ts` (in `../temporal/`) | Energy-score rejection + LocalAgreement-2 commit      | 3     |

## Why `geometricFix` must be deleted rather than improved

It is ~85 lines of hand-tuned `if`/`else` that override the model for 20 of 26 letters,
with 15+ magic constants. See `docs/AUDIT.md` §1.2 for the full analysis. In short:

1. **It is self-inconsistent.** The same physical handshape gets different answers depending
   on which class the model guessed. `M` can be rewritten to `K`; so can `N`/`U` — via a
   different threshold. The `S/I/K/V/Z` branch emits `T`/`N`/`Y` using constants that differ
   from the `T`/`A` branch that also emits `N` (`0.30` vs `0.32`).
2. **It makes the model un-retrainable.** Improving the model changes its error distribution,
   which invalidates every branch. You cannot touch one without re-tuning the other by hand.
3. **It is overfit to one person** — one hand, one camera, one distance, one lighting setup.
4. **Its units are mixed.** `tipAbovePip` _multiplies_ by palm scale; `dist` _divides_ by it,
   so thresholds are not comparable across branches.

The tests in `geometricFix.test.ts` pin the current behaviour exactly. They exist to make
deletion safe and measurable, **not** to protect this logic: when Phase 2 lands, the tests
are deleted along with the module, and the golden-fixture suite in `test/fixtures/` becomes
the contract instead.
