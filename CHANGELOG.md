# Changelog

## 0.2.0, November 23, 2026

- Add pointwise adjusted-effect intervals using a participant pairs bootstrap within strata and a fixed-design Rademacher wild bootstrap.
- Tune ridge penalties inside each outer training fold and repeat tuning for each permutation draw. Per-fold audits report the selected penalty and candidate errors.
- Add optional Matplotlib views of adjusted effects and out-of-fold predictions, plus a synthetic neurovascular workflow tutorial.
- Test the plotting extra in CI and include the PEP 561 marker in source and wheel distributions.

## 0.1.0 local draft, November 14, 2026

The Round 1 corrections unify the retained nuisance SVD space across CV fits, rank reporting, held-out prediction and permutation refits. CLI validation now checks optional subject labels and fold/group compatibility and rejects unsupported, empty and corrupt containers with exit code 2. Deterministic regressions cover nearly duplicated confounds and the reported malformed inputs. Usage documentation explicitly excludes a predicted physiology outcome and its derived interactions from predictors.

This first local version includes rank-aware Freedman-Lane inference, exhaustive and Monte Carlo within-block permutations, ordinary/HC3 studentization, shared max-T and BH adjustment, training-only ridge and signed-network CPM, fold-restricted conditional nulls, neurovascular feature extraction, a controlled synthetic generator and a JSON/CSV CLI.

The release includes direct numerical comparisons, null calibration and held-out mutation tests, a reproducible synthetic benchmark, JOSS manuscript draft and study documentation. Human review, public development history, real-cohort validation and publication remain future steps.

