"""Optional Matplotlib views of permutation inference and CV predictions.

Matplotlib is imported lazily inside each function, so importing this module
never requires it and no backend, ``show`` call or file write is forced.
Install the dependency with the ``plot`` extra.
"""
from __future__ import annotations

from collections.abc import Sequence
import numbers
from typing import TYPE_CHECKING, Literal
import numpy as np

from .cv_connectome import CVResult
from .freedman_lane import PermutationResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from .bootstrap import BootstrapResult


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "nvperm plotting requires the optional matplotlib dependency; "
            "install it from the repository with python -m pip install '.[plot]'"
        ) from exc
    return plt


def _axes(ax):
    """Return the given axes or create one without touching global state."""
    if ax is not None:
        return ax
    _, ax = _pyplot().subplots()
    return ax


def _finite_vector(value, name, n=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a nonempty one-dimensional finite vector") from exc
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a nonempty one-dimensional finite vector")
    if n is not None and len(a) != n:
        raise ValueError(f"{name} must have {n} entries")
    return a


def _probability_vector(value, name, n):
    a = _finite_vector(value, name, n)
    if np.any((a < 0) | (a > 1)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return a


def _level(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        raise ValueError("alpha must be a probability in (0, 1)")
    level = float(alpha)
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("alpha must be a probability in (0, 1)")
    return level


def _intervals(intervals, n):
    """Validate bootstrap interval endpoints aligned with the features."""
    low = getattr(intervals, "ci_low", None)
    high = getattr(intervals, "ci_high", None)
    if low is None or high is None:
        raise ValueError("intervals must expose ci_low and ci_high arrays")
    low = _finite_vector(low, "intervals.ci_low", n)
    high = _finite_vector(high, "intervals.ci_high", n)
    if np.any(low > high):
        raise ValueError("intervals must satisfy ci_low <= ci_high")
    return low, high


def _labels(labels, n):
    if labels is None:
        return [f"feature {i}" for i in range(n)]
    if isinstance(labels, (str, bytes)):
        raise ValueError("labels must be a sequence with one entry per feature")
    try:
        labels = [str(label) for label in labels]
    except TypeError as exc:
        raise ValueError("labels must be an iterable with one entry per feature") from exc
    if len(labels) != n:
        raise ValueError("labels must have one entry per feature")
    return labels


def _significance(result, level):
    """Validate result arrays and build the two-criterion categories."""
    effect = _finite_vector(result.effect, "result.effect")
    n = len(effect)
    pvalues = _probability_vector(result.pvalues, "result.pvalues", n)
    p_max = _probability_vector(result.pvalues_max_t, "result.pvalues_max_t", n)
    qvalues = _probability_vector(result.qvalues, "result.qvalues", n)
    max_t = p_max <= level
    bh = qvalues <= level
    categories = [
        (max_t & bh, "s", "#b45136", f"max-T and BH q <= {level:g}"),
        (max_t & ~bh, "^", "#176b87", f"max-T only <= {level:g}"),
        (~max_t & bh, "D", "#478257", f"BH q only <= {level:g}"),
        (~max_t & ~bh, "o", "#999999", f"not significant at {level:g}"),
    ]
    return n, effect, pvalues, categories


def plot_adjusted_effects(result: PermutationResult, *, labels: Sequence[str] | None = None,
                          intervals: BootstrapResult | None = None,
                          kind: Literal["forest", "volcano"] = "forest",
                          alpha: float = 0.05, ax: Axes | None = None) -> Axes:
    """Plot covariate-adjusted effects with adjusted significance marks.

    ``kind='forest'`` draws each adjusted coefficient on the x axis with a
    zero reference line; ``intervals`` adds horizontal bootstrap intervals
    drawn straight from their endpoints, which may not contain the estimate.
    ``kind='volcano'`` plots ``-log10`` of the raw p value against the effect.
    Significance uses the familywise max-T p and the BH q value, each at
    ``alpha``; categories share a distinguishable legend. Because inference
    is adjusted, no raw-p threshold line is drawn on the volcano.
    """
    if not isinstance(result, PermutationResult):
        raise ValueError("result must be a PermutationResult")
    level = _level(alpha)
    if kind not in ("forest", "volcano"):
        raise ValueError("kind must be 'forest' or 'volcano'")
    n, effect, pvalues, categories = _significance(result, level)
    labels = _labels(labels, n)
    low = high = None
    if intervals is not None:
        if kind != "forest":
            raise ValueError("intervals are supported only for kind='forest'")
        low, high = _intervals(intervals, n)
    ax = _axes(ax)
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1, zorder=1)
    if kind == "forest":
        y = np.arange(n)
        if low is not None:
            ax.hlines(y, low, high, color="#888888", linewidth=1.2, zorder=2)
        for mask, marker, color, label in categories:
            if np.any(mask):
                ax.scatter(effect[mask], y[mask], marker=marker, color=color,
                           label=label, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.yaxis.set_inverted(True)
        ax.set_xlabel("Covariate-adjusted effect")
    else:
        positive = pvalues[pvalues > 0]
        min_positive = np.finfo(float).smallest_subnormal
        floor = max(min_positive, 0.5 * positive.min()) if positive.size else min_positive
        height = -np.log10(np.where(pvalues > 0, pvalues, floor))
        for mask, marker, color, label in categories:
            if np.any(mask):
                ax.scatter(effect[mask], height[mask], marker=marker, color=color,
                           label=label, zorder=3)
        ax.set_xlabel("Covariate-adjusted effect")
        ax.set_ylabel("-log10 raw p value")
    ax.legend()
    return ax


def plot_cv_predictions(target: ArrayLike, result: CVResult, *,
                        ax: Axes | None = None) -> Axes:
    """Plot out-of-fold predictions against the observed target.

    Both the nuisance-only baseline and the full connectome model are drawn
    on the same axes, sharing observed-target units, with an identity line
    for reference.
    """
    if not isinstance(result, CVResult):
        raise ValueError("result must be a CVResult")
    observed = _finite_vector(target, "target")
    n = len(observed)
    baseline = _finite_vector(result.baseline_predictions,
                              "result.baseline_predictions", n)
    full = _finite_vector(result.predictions, "result.predictions", n)
    ax = _axes(ax)
    low = min(observed.min(), baseline.min(), full.min())
    high = max(observed.max(), baseline.max(), full.max())
    ax.plot([low, high], [low, high], linestyle="--", color="#444444",
            linewidth=1, label="identity")
    ax.scatter(observed, baseline, marker="o", color="#999999",
               label="nuisance baseline")
    ax.scatter(observed, full, marker="s", color="#176b87",
               label="connectome model")
    ax.set_xlabel("Observed target")
    ax.set_ylabel("Out-of-fold prediction")
    ax.legend()
    return ax
