"""Pairs and wild bootstrap intervals for a single exposure effect.

Both estimators return raw-unit effects on the same footing as
:func:`nvperm.freedman_lane`. The pairs bootstrap resamples independent
participants within declared strata; the wild bootstrap holds the design
fixed and flips Rademacher signs on HC2-scaled residuals. Neither is a
permutation test: they approximate the sampling variability of the point
estimate, not a null distribution, so no p value is returned.
"""
from __future__ import annotations

import numbers
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._utils import (
    generator, label_groups, matrix, nuisance_basis, positive_integer,
    residualize, vector,
)

__all__ = ["BootstrapResult", "bootstrap_effects", "wild_bootstrap_effects"]


@dataclass(frozen=True)
class BootstrapResult:
    """Pointwise bootstrap interval for every outcome, in raw units.

    Attributes
    ----------
    effect : ndarray
        Observed exposure coefficient per outcome.
    ci_low, ci_high : ndarray
        Interval endpoints per outcome.
    bootstrap_effects : ndarray
        Retained resample effects with shape (n_valid, outcomes).
    confidence_level : float
        Requested interval level.
    n_resamples : int
        Requested draws.
    n_valid : int
        Estimable draws retained; equals n_resamples unless resampled
        designs were degenerate (pairs only).
    method : {'pairs', 'wild_rademacher'}
    nuisance_rank, residual_df : int
        Identified nuisance rank and residual degrees of freedom of the
        observed design.
    block_sizes : tuple of int or None
        Stratum sizes in label order for the pairs bootstrap; None for
        the wild bootstrap.
    """
    effect: NDArray[np.float64]
    ci_low: NDArray[np.float64]
    ci_high: NDArray[np.float64]
    bootstrap_effects: NDArray[np.float64]
    confidence_level: float
    n_resamples: int
    n_valid: int
    method: str
    nuisance_rank: int
    residual_df: int
    block_sizes: tuple[int, ...] | None

    def to_dict(self, include_draws: bool = False) -> dict[str, object]:
        """Return JSON-compatible metadata, with resamples on request."""
        assumptions = {
            "pairs": ("Rows are independent participants, exchangeable "
                      "within strata; intervals condition on retained "
                      "estimable draws"),
            "wild_rademacher": ("Rows are independent participants with "
                                "row-varying error variance; pointwise "
                                "HC2 Rademacher wild bootstrap"),
        }
        result: dict[str, object] = {
            "effect": self.effect.tolist(),
            "ci_low": self.ci_low.tolist(),
            "ci_high": self.ci_high.tolist(),
            "confidence_level": self.confidence_level,
            "n_resamples": self.n_resamples,
            "n_valid": self.n_valid,
            "method": self.method,
            "nuisance_rank": self.nuisance_rank,
            "residual_df": self.residual_df,
            "block_sizes": (None if self.block_sizes is None
                            else list(self.block_sizes)),
            "assumption": assumptions[self.method],
        }
        if include_draws:
            result["bootstrap_effects"] = self.bootstrap_effects.tolist()
        return result


@dataclass(frozen=True)
class _PreparedDesign:
    """Validated observed-data quantities shared by both bootstraps."""
    n: int
    outcomes: NDArray[np.float64]
    exposure: NDArray[np.float64]
    covariates: NDArray[np.float64]
    q: NDArray[np.float64]
    rank: int
    df: int
    exposure_residual: NDArray[np.float64]
    exposure_ssr: float
    scale_ratio: NDArray[np.float64]
    outcome_residuals: NDArray[np.float64]
    beta_conditioned: NDArray[np.float64]
    effect: NDArray[np.float64]


def _confidence_level(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            "confidence_level must be a real scalar strictly between 0 and 1")
    level = float(value)
    if not 0.0 < level < 1.0:
        raise ValueError(
            "confidence_level must be a real scalar strictly between 0 and 1")
    return level


def _exposure_terms(exposure: NDArray[np.float64], q: NDArray[np.float64]):
    """Unit-sd nuisance-residualized exposure, or None if unidentifiable."""
    n = len(exposure)
    max_abs = float(np.max(np.abs(exposure)))
    if max_abs == 0.0:
        return None
    unit = exposure / max_abs
    centered = unit - unit.mean()
    norm_unit = float(np.linalg.norm(centered) / np.sqrt(n))
    if norm_unit == 0.0:
        return None
    residual = residualize(centered / norm_unit, q)
    ssr = float(residual @ residual)
    if ssr <= np.finfo(float).eps * n:
        return None
    return max_abs, norm_unit, residual, ssr


def _conditioned_outcomes(outcomes: NDArray[np.float64],
                          q: NDArray[np.float64]):
    """Unit-variance nuisance residuals plus overflow-safe scale factors."""
    max_abs = np.max(np.abs(outcomes), axis=0)
    safe_max = np.where(max_abs > 0.0, max_abs, 1.0)
    unit = outcomes / safe_max
    centered = unit - unit.mean(axis=0)
    norm_unit = np.sqrt(np.mean(centered * centered, axis=0))
    safe_norm = np.where(norm_unit > 0.0, norm_unit, 1.0)
    residuals = residualize(centered / safe_norm, q)
    return residuals, safe_max, safe_norm


def _scale_ratio(y_max: NDArray[np.float64], y_norm: NDArray[np.float64],
                 x_max: float, x_norm: float) -> NDArray[np.float64]:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return (y_max / x_max) * (y_norm / x_norm)


def _prepare(y: ArrayLike, x: ArrayLike,
             covariates: ArrayLike | None) -> _PreparedDesign:
    outcomes = matrix(y, "y")
    n = len(outcomes)
    exposure = vector(x, "x", n)
    if outcomes.shape[1] == 0:
        raise ValueError("at least one outcome is required")
    covar = (np.empty((n, 0)) if covariates is None
             else matrix(covariates, "covariates", n))
    q, rank = nuisance_basis(None if covariates is None else covar, n)
    df = n - rank - 1
    if df < 2:
        raise ValueError(
            "at least two residual degrees of freedom are required")
    terms = _exposure_terms(exposure, q)
    if terms is None:
        raise ValueError(
            "exposure is constant or collinear with nuisance covariates")
    x_max, x_norm, exposure_residual, exposure_ssr = terms
    residuals, y_max, y_norm = _conditioned_outcomes(outcomes, q)
    ratio = _scale_ratio(y_max, y_norm, x_max, x_norm)
    beta_conditioned = exposure_residual @ residuals / exposure_ssr
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        effect = beta_conditioned * ratio
    if not np.all(np.isfinite(effect)):
        raise ValueError("observed exposure effect is not finite")
    return _PreparedDesign(
        n, outcomes, exposure, covar, q, rank, df,
        exposure_residual, exposure_ssr, ratio, residuals,
        beta_conditioned, effect)


def bootstrap_effects(
    y: ArrayLike,
    x: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    blocks: ArrayLike | None = None,
    n_resamples: int = 999,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = 42,
) -> BootstrapResult:
    """Pairs (row-resampling) bootstrap intervals for the exposure effects.

    Each draw resamples complete (outcome, exposure, covariate) rows with
    replacement, independently within every declared stratum and
    preserving its size; all outcome columns share the resample. The
    nuisance basis, exposure identifiability and the raw-unit effect are
    recomputed per draw. Intervals are pointwise percentiles at the
    requested level.

    ``blocks`` labels independent strata (for example acquisition sites)
    whose rows are exchangeable participants resampled independently.
    They are strata, not dependent clusters: no within-cluster dependence
    is modeled, so clustered or longitudinal rows lie outside its
    assumptions. Singleton strata are held fixed; if every stratum is a
    singleton no resampling variation remains and the call is rejected.

    Draws whose resampled design is degenerate (nuisance rank change or
    a constant or nuisance-collinear resampled exposure) are skipped and
    counted; ``n_valid`` reports how many were retained, and a single
    RuntimeWarning notes that intervals then condition on the retained
    estimable draws. Exactly ``n_resamples`` draws are attempted and
    none are substituted.

    This is a sampling-variability interval, not a null-hypothesis test:
    no p value is returned.
    """
    count = positive_integer(n_resamples, "n_resamples", minimum=2)
    level = _confidence_level(confidence_level)
    design = _prepare(y, x, covariates)
    groups = label_groups(blocks, design.n)
    if all(len(group) == 1 for group in groups):
        raise ValueError(
            "all strata are singletons; no resampling variation remains")
    rng = generator(random_state)
    has_covariates = covariates is not None
    draws = np.empty((count, design.outcomes.shape[1]))
    n_valid = 0
    for _ in range(count):
        indices = np.empty(design.n, dtype=int)
        for group in groups:
            indices[group] = rng.choice(
                group, size=len(group), replace=True)
        q_b, rank_b = nuisance_basis(
            design.covariates[indices] if has_covariates else None, design.n)
        terms = (_exposure_terms(design.exposure[indices], q_b)
                 if rank_b == design.rank else None)
        if terms is None:
            continue
        x_max_b, x_norm_b, residual_b, ssr_b = terms
        residuals_b, y_max_b, y_norm_b = _conditioned_outcomes(
            design.outcomes[indices], q_b)
        ratio_b = _scale_ratio(y_max_b, y_norm_b, x_max_b, x_norm_b)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            beta = (residual_b @ residuals_b / ssr_b) * ratio_b
        if not np.all(np.isfinite(beta)):
            raise ValueError(
                "non-finite pairs bootstrap effects; check input scaling")
        draws[n_valid] = beta
        n_valid += 1
    if n_valid < 2:
        raise ValueError(
            "fewer than two estimable bootstrap resamples were retained")
    boot = draws[:n_valid]
    ci_low, ci_high = np.quantile(
        boot, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    if not (np.all(np.isfinite(ci_low)) and np.all(np.isfinite(ci_high))):
        raise ValueError("non-finite bootstrap interval; check input scaling")
    if n_valid < count:
        warnings.warn(
            f"{count - n_valid} of {count} bootstrap resamples were not "
            "estimable (nuisance rank change or unidentifiable resampled "
            "exposure); intervals condition "
            f"on the {n_valid} retained estimable draws",
            RuntimeWarning, stacklevel=2)
    return BootstrapResult(
        effect=design.effect, ci_low=ci_low, ci_high=ci_high,
        bootstrap_effects=boot, confidence_level=level,
        n_resamples=count, n_valid=n_valid, method="pairs",
        nuisance_rank=design.rank, residual_df=design.df,
        block_sizes=tuple(len(group) for group in groups))


def wild_bootstrap_effects(
    y: ArrayLike,
    x: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    n_resamples: int = 999,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = 42,
) -> BootstrapResult:
    """Rademacher wild bootstrap intervals for the exposure effects.

    The design is held fixed: each draw gives every participant an
    independent Rademacher sign (one sign per participant, shared by all
    outcome columns), multiplies the participant's HC2-scaled full-model
    residual by it and reprojects onto the residualized exposure, adding
    that fluctuation to the observed effect. Endpoints are pointwise
    basic (reflected) intervals: ``effect - quantile(delta, upper)`` and
    ``effect - quantile(delta, lower)`` over the signed fluctuation
    distribution ``delta``.

    This is an approximate pointwise sensitivity check for independent
    participants whose error variance may differ across rows. It is not a
    null-hypothesis test and returns no p value, it is not simultaneous
    across outcomes, and it has no stratum or cluster handling. A
    participant whose full-model leverage reaches one leaves no residual
    variation to reweight and is rejected up front.
    """
    count = positive_integer(n_resamples, "n_resamples", minimum=2)
    level = _confidence_level(confidence_level)
    design = _prepare(y, x, covariates)
    residual = design.exposure_residual
    ssr = design.exposure_ssr
    leverage = (np.sum(design.q * design.q, axis=1)
                + residual * residual / ssr)
    if np.any(leverage >= 1 - 1e-10):
        raise ValueError(
            "wild bootstrap requires full-model leverage below one")
    errors = (design.outcome_residuals
              - residual[:, None] * design.beta_conditioned[None, :])
    weights = (residual[:, None] * errors
               / (ssr * np.sqrt(1 - leverage[:, None])))
    rng = generator(random_state)
    signs = 2 * rng.integers(0, 2, size=(count, design.n), dtype=np.int8) - 1
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        delta = np.einsum("bn,nt->bt", signs, weights) * design.scale_ratio
    boot = design.effect[None, :] + delta
    if not np.all(np.isfinite(boot)):
        raise ValueError(
            "non-finite wild bootstrap effects; check input scaling")
    lower, upper = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    ci_low = design.effect - np.quantile(delta, upper, axis=0)
    ci_high = design.effect - np.quantile(delta, lower, axis=0)
    if not (np.all(np.isfinite(ci_low)) and np.all(np.isfinite(ci_high))):
        raise ValueError("non-finite bootstrap interval; check input scaling")
    return BootstrapResult(
        effect=design.effect, ci_low=ci_low, ci_high=ci_high,
        bootstrap_effects=boot, confidence_level=level,
        n_resamples=count, n_valid=count, method="wild_rademacher",
        nuisance_rank=design.rank, residual_df=design.df, block_sizes=None)
