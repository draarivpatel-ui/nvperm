"""Deterministic calibration, selection leakage, power and CV-null benchmarks."""
from __future__ import annotations

import csv
from pathlib import Path
import time
import numpy as np
from scipy.stats import t as student_t

from ._utils import correlation, nuisance_basis, positive_integer, residualize
from .cv_connectome import ConnectomePhysiologyCV
from .freedman_lane import freedman_lane
from .synthetic import generate_cohort


def _wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(max(0, center - half)), float(min(1, center + half))]


def _naive_screen(cohort, folds):
    """Deliberately invalid full-data screen, clearly identified in every output."""
    q, rank = nuisance_basis(cohort.confounds, len(cohort.target))
    x = residualize(cohort.features, q)
    y = residualize(cohort.target, q)
    denom = np.linalg.norm(x, axis=0) * np.linalg.norm(y)
    r = np.divide(x.T @ y, denom, out=np.zeros(x.shape[1]), where=denom > 1e-12)
    df = len(y) - rank - 1
    tstat = r * np.sqrt(df / np.maximum(1 - r * r, 1e-14))
    uncorrected_p = float(np.min(2 * student_t.sf(np.abs(tstat), df)))
    # The selected edges and nuisance/scaler fits have already seen test targets.
    selected = np.argsort(np.abs(r))[-min(8, len(r)):]
    x = x[:, selected]
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)
    predictions = np.empty(len(y))
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        weights = np.linalg.solve(x[train].T @ x[train] + 10 * np.eye(x.shape[1]), x[train].T @ y[train])
        predictions[test] = x[test] @ weights
    return uncorrected_p, correlation(predictions, y)


def run_benchmark(*, n_replicates=80, n_permutations=199, n_subjects=90, n_rois=12,
                  n_cv_permutations=49, random_state=20261114):
    """Run null and power experiments; return every draw and Wilson intervals.

    Null errors are homoskedastic but age and FD confounding is strong. The
    invalid baseline includes full-data edge selection as well as global
    residualization. It does not attribute all inflation to residualization.
    """
    repetitions = positive_integer(n_replicates, "n_replicates", 2)
    draws = positive_integer(n_permutations, "n_permutations")
    cv_draws = positive_integer(n_cv_permutations, "n_cv_permutations")
    start = time.perf_counter()
    rows, power_rows = [], []
    for i in range(repetitions):
        seed = int(random_state) + i
        cohort = generate_cohort(n_subjects, n_rois, confounding=1.5, random_state=seed)
        fl = freedman_lane(cohort.features, cohort.target, cohort.confounds,
                           n_permutations=draws, random_state=seed + 10000)
        cv = ConnectomePhysiologyCV(n_splits=5, model="partial_correlation", alpha=10,
                                    random_state=seed).fit(cohort.features, cohort.target, cohort.confounds)
        naive_p, naive_r = _naive_screen(cohort, cv.fold_ids_)
        cv_null = cv.permutation_test(cv_draws, random_state=seed + 20000)
        rows.append({
            "replicate": i, "seed": seed, "naive_selected_edge_p": naive_p,
            "max_t_min_p": float(np.min(fl.pvalues_max_t)),
            "bh_min_q": float(np.min(fl.qvalues)),
            "naive_global_screen_oof_r": naive_r,
            "training_only_oof_r": cv.result_.residual_prediction_correlation,
            "cv_permutation_p": cv_null.pvalue,
            "cv_mse_improvement": cv.result_.mse_improvement,
        })
    levels = (0.0, 0.3, 0.6, 1.0)
    power_reps = max(12, repetitions // 2)
    for effect in levels:
        for i in range(power_reps):
            seed = int(random_state) + 30000 + i  # Paired random cohorts across effect levels.
            cohort = generate_cohort(n_subjects, n_rois, effect_size=effect,
                                     confounding=1.5, random_state=seed)
            fl = freedman_lane(cohort.features, cohort.target, cohort.confounds,
                               n_permutations=draws, random_state=seed + 10000)
            signal = cohort.ground_truth != 0
            detected = bool(np.any(fl.pvalues_max_t[signal] <= 0.05)) if np.any(signal) else bool(
                np.any(fl.pvalues_max_t <= 0.05))
            power_rows.append({"effect_size": effect, "replicate": i,
                               "seed": seed, "detected": int(detected)})
    metrics = {}
    for name, key in (("naive_selected_edge", "naive_selected_edge_p"),
                      ("freedman_lane_max_t", "max_t_min_p"),
                      ("benjamini_hochberg_any_rejection", "bh_min_q"),
                      ("fold_respecting_cv_null", "cv_permutation_p")):
        rejected = sum(row[key] <= 0.05 for row in rows)
        metrics[name] = {"rejections": rejected, "replicates": repetitions,
                         "rate": rejected / repetitions, "wilson_95ci": _wilson(rejected, repetitions)}
    power = []
    for effect in levels:
        selected = [row for row in power_rows if row["effect_size"] == effect]
        rejected = sum(row["detected"] for row in selected)
        power.append({"effect_size": effect, "rate": rejected / len(selected),
                      "rejections": rejected, "replicates": len(selected),
                      "wilson_95ci": _wilson(rejected, len(selected))})
    return {
        "schema_version": 1, "seed": int(random_state), "n_subjects": n_subjects,
        "n_rois": n_rois, "n_edges": n_rois * (n_rois - 1) // 2,
        "n_permutations": draws, "n_cv_permutations": cv_draws,
        "n_replicates": repetitions, "confounding": 1.5, "heteroskedasticity": 0,
        "alpha": 0.05, "calibration": metrics,
        "leakage": {
            "naive_mean_residual_oof_r": float(np.mean([r["naive_global_screen_oof_r"] for r in rows])),
            "training_only_mean_residual_oof_r": float(np.mean([r["training_only_oof_r"] for r in rows])),
        },
        "power": power, "null_replicates": rows, "power_replicates": power_rows,
        "elapsed_seconds": time.perf_counter() - start,
        "interpretation": (
            "Synthetic demonstration with Wilson uncertainty. The invalid comparator fits nuisance "
            "effects and selects edges using all targets before evaluation. Full-data screening "
            "is a source of inflation; global residualization alone is not asserted to always inflate. "
            "Exact enumeration and HC3 do not guarantee validity under arbitrary nuisance or variance patterns."
        ),
    }


def write_benchmark_csv(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = report["null_replicates"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_benchmark(report, path):
    """Write a 300 dpi three-panel diagnostic figure; requires the plot extra."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Full-data screen\nuncorrected", "Freedman-Lane\nmax-T", "Fold-restricted\nCV null"]
    names = ["naive_selected_edge", "freedman_lane_max_t", "fold_respecting_cv_null"]
    entries = [report["calibration"][name] for name in names]
    rates = np.asarray([entry["rate"] for entry in entries])
    intervals = np.asarray([entry["wilson_95ci"] for entry in entries])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    colors = ["#b45136", "#176b87", "#478257"]
    axes[0].bar(np.arange(3), rates, color=colors, width=0.65,
                yerr=np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates)), capsize=4)
    axes[0].axhline(0.05, color="#444444", linestyle="--", linewidth=1)
    axes[0].set(xticks=np.arange(3), xticklabels=labels, ylabel="Null rejection rate",
                ylim=(0, 1.05), title="A  Null calibration")
    rows = report["null_replicates"]
    arrays = [[r["naive_global_screen_oof_r"] for r in rows], [r["training_only_oof_r"] for r in rows]]
    boxes = axes[1].boxplot(arrays, patch_artist=True)
    axes[1].set_xticks([1, 2], ["Global fit +\nfull-data screen", "Training-only\nCPM"])
    for box, color in zip(boxes["boxes"], colors[:2]):
        box.set_facecolor(color)
        box.set_alpha(0.75)
    axes[1].axhline(0, color="#444444", linestyle="--", linewidth=1)
    axes[1].set(ylabel="Residual out-of-fold correlation", title="B  Selection leakage under null")
    power = report["power"]
    effects = np.asarray([r["effect_size"] for r in power])
    prates = np.asarray([r["rate"] for r in power])
    pci = np.asarray([r["wilson_95ci"] for r in power])
    axes[2].errorbar(effects, prates, yerr=np.vstack((prates - pci[:, 0], pci[:, 1] - prates)),
                     color=colors[1], marker="o", capsize=4)
    axes[2].set(xlabel="Injected edge effect / noise SD", ylabel="Family detection rate",
                ylim=(0, 1.05), title="C  Max-T power")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, metadata={"Software": "nvperm synthetic benchmark"})
    plt.close(fig)

