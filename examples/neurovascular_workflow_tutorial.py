"""Worked neurovascular workflow on one fixed synthetic cohort.

This tutorial is an API walkthrough on the package's own synthetic generator.
It is not a real-cohort result and makes no performance claim for real data.
Inference direction differs by stage: Freedman-Lane and the bootstraps treat
pulse pressure as the exposure and each connectome feature as an outcome,
while the cross-validation predicts pulse pressure from the feature matrix.
Freedman-Lane requires exchangeable reduced-model errors within sites.
The pairs bootstrap resamples independent participants within sites, while
the wild bootstrap holds the design fixed and allows row-varying variance.
The bootstrap intervals are approximate and pointwise.

Run ``python examples/neurovascular_workflow_tutorial.py`` to print a summary.
Pass ``--output-dir DIR`` to also write two PNG figures; that path needs the
optional ``plot`` extra and nothing is written without the flag.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from nvperm import (
    ConnectomePhysiologyCV, freedman_lane, generate_cohort,
    motion_sensitivity, neurovascular_features, pulse_pressure,
)
from nvperm.bootstrap import bootstrap_effects, wild_bootstrap_effects

SEED = 20261123
TARGET_COLUMN = "pulse_pressure_mmhg"


def target_safe_columns(names):
    """Keep predictors that cannot leak the pulse-pressure target.

    The target column itself and every interaction built from it must be
    removed; neurovascular_features documents this exclusion in its docstring.
    """
    keep = [
        index for index, name in enumerate(names)
        if name != TARGET_COLUMN and not name.endswith("_x_pulse_pressure")
    ]
    dropped = [name for name in names
               if name == TARGET_COLUMN or name.endswith("_x_pulse_pressure")]
    return keep, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="write the two tutorial PNGs here (plot extra)")
    args = parser.parse_args()

    cohort = generate_cohort(n_subjects=150, n_rois=15, effect_size=0.8,
                             confounding=1.0, random_state=SEED)
    target = pulse_pressure(cohort.systolic, cohort.diastolic)
    features = neurovascular_features(
        cohort.connectomes, cohort.network_labels, cohort.systolic, cohort.diastolic,
    )
    keep, dropped = target_safe_columns(features.names)
    x = features.values[:, keep]
    names = [features.names[index] for index in keep]
    n, p = x.shape
    print(f"cohort: {n} synthetic subjects, {p} target-safe predictors")
    print(f"excluded target-derived columns: {dropped}")
    if np.any(~features.segregation_defined):
        raise ValueError(
            "this tutorial requires defined segregation for every participant")

    fl = freedman_lane(x, target, cohort.confounds, blocks=cohort.site,
                       n_permutations=999, random_state=SEED)
    print(f"\nFreedman-Lane (site-stratified shuffles): "
          f"{int(np.count_nonzero(fl.pvalues_max_t <= 0.05))} of {p} features "
          f"significant by max-T, "
          f"{int(np.count_nonzero(fl.qvalues <= 0.05))} by BH q at 0.05")
    boot = bootstrap_effects(x, target, covariates=cohort.confounds,
                             blocks=cohort.site, n_resamples=999,
                             random_state=SEED)
    wild = wild_bootstrap_effects(x, target, covariates=cohort.confounds,
                                  n_resamples=999, random_state=SEED)
    for label, result in (("stratified pairs bootstrap", boot), ("wild bootstrap", wild)):
        print(f"{label}, {names[0]} pointwise 95% CI "
              f"[{result.ci_low[0]:.4g}, {result.ci_high[0]:.4g}]")
    print("the wild interval holds the design fixed and does not resample sites")

    cv = ConnectomePhysiologyCV(n_splits=5, model="ridge",
                              alpha_grid=(0.5, 2.0, 10.0, 50.0, 200.0),
                              inner_splits=3, random_state=SEED)
    cv.fit(x, target, cohort.confounds)
    chosen = [audit.selected_alpha for audit in cv.fold_audits_]
    print(f"\nnested CV: selected alpha per outer fold {chosen}")
    print(f"mse improvement over nuisance baseline: "
          f"{cv.result_.mse_improvement:.4f} "
          f"(incremental R2 {cv.result_.incremental_r2:.3f})")
    null = cv.permutation_test(199, blocks=cohort.site, random_state=SEED + 1)
    print(f"fold-restricted permutation p for the improvement: "
          f"{null.pvalue:.4f} (199 site- and fold-restricted shuffles, retuned per draw)")

    fd_column = cohort.confound_names.index("mean_fd")
    covariates = np.delete(cohort.confounds, fd_column, axis=1)
    sensitivity = motion_sensitivity(
        x, target, cohort.framewise_displacement, covariates=covariates,
        n_permutations=199, random_state=SEED, blocks=cohort.site,
    )
    strongest = int(np.argmax(np.abs(sensitivity["correlation_before_fd"])))
    print(f"\nmotion sensitivity (descriptive, not causal): {names[strongest]} "
          f"partial r {sensitivity['correlation_before_fd'][strongest]:.3f} -> "
          f"{sensitivity['correlation_after_fd'][strongest]:.3f} after FD, "
          f"attenuation {sensitivity['relative_attenuation'][strongest]:.3f}")

    if args.output_dir is None:
        print("\nno --output-dir given; skipping figures")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nvperm.plotting import plot_adjusted_effects, plot_cv_predictions

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.28 * p + 1.5)), layout="constrained")
    plot_adjusted_effects(fl, labels=names, intervals=boot, kind="forest", ax=ax)
    first = args.output_dir / "adjusted_effects_forest.png"
    fig.savefig(first, dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), layout="constrained")
    plot_cv_predictions(target, cv.result_, ax=ax)
    second = args.output_dir / "cv_predictions.png"
    fig.savefig(second, dpi=200)
    plt.close(fig)
    print(f"\nwrote {first} and {second}")


if __name__ == "__main__":
    main()
