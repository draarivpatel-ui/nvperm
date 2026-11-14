"""Rank-aware Freedman-Lane residual permutation tests for one regressor.

All targets share each permutation. Exact mode enumerates the declared group;
with estimated nuisance effects this does not guarantee finite-sample exactness.
Exchangeable reduced-model errors are required. HC3 studentization does not
make arbitrary heteroskedastic errors exchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import numpy as np

from ._utils import (
    generator, label_groups, matrix, nuisance_basis, permutation_indices,
    positive_integer, residualize, vector,
)


def bh_fdr(pvalues):
    """Benjamini-Hochberg adjusted p values, preserving input shape and ties."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim == 0 or p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("pvalues must be a nonempty array of finite probabilities")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must lie in [0, 1]")
    flat = p.ravel()
    order = np.argsort(flat, kind="stable")
    ranked = flat[order] * len(flat) / np.arange(1, len(flat) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(flat)
    result[order] = np.clip(adjusted, 0, 1)
    return result.reshape(p.shape)


@dataclass(frozen=True)
class PermutationResult:
    statistic: np.ndarray
    effect: np.ndarray
    standard_error: np.ndarray
    pvalues: np.ndarray
    pvalues_max_t: np.ndarray
    qvalues: np.ndarray
    null_statistics: np.ndarray
    residual_df: int
    nuisance_rank: int
    n_permutations: int
    exact: bool
    statistic_name: str
    alternative: str
    block_sizes: tuple[int, ...]

    def to_dict(self, include_null=False):
        result = {
            "statistic": self.statistic.tolist(), "effect": self.effect.tolist(),
            "standard_error": self.standard_error.tolist(),
            "pvalues": self.pvalues.tolist(), "pvalues_max_t": self.pvalues_max_t.tolist(),
            "qvalues": self.qvalues.tolist(), "residual_df": self.residual_df,
            "nuisance_rank": self.nuisance_rank, "n_permutations": self.n_permutations,
            "exact": self.exact, "statistic_name": self.statistic_name,
            "alternative": self.alternative, "block_sizes": list(self.block_sizes),
            "assumption": "Reduced-model errors must be exchangeable within blocks",
        }
        if include_null:
            result["null_statistics"] = self.null_statistics.tolist()
        return result


class FreedmanLane:
    """Test a single exposure against a matrix of outcomes.

    Parameters
    ----------
    n_permutations : int
        Monte Carlo draws; ignored by exact enumeration.
    mode : {'monte_carlo', 'exact'}
        Exact enumeration includes identity and has a factorial size limit.
    statistic : {'studentized', 'hc3', 'partial_correlation'}
        OLS t, HC3 t, or nuisance-adjusted correlation.
    alternative : {'two-sided', 'greater', 'less'}
        Max-T uses the same orientation as the marginal test.
    random_state : int or numpy.random.Generator
        Seed or a supplied, deliberately consumed generator.
    """

    def __init__(self, n_permutations=999, *, mode="monte_carlo",
                 statistic="studentized", alternative="two-sided",
                 random_state=42, batch_size=128, max_exact=40320):
        self.n_permutations = positive_integer(n_permutations, "n_permutations")
        self.batch_size = positive_integer(batch_size, "batch_size")
        self.max_exact = positive_integer(max_exact, "max_exact")
        if mode not in ("monte_carlo", "exact"):
            raise ValueError("mode must be monte_carlo or exact")
        if statistic not in ("studentized", "hc3", "partial_correlation"):
            raise ValueError("unknown statistic")
        if alternative not in ("two-sided", "greater", "less"):
            raise ValueError("unknown alternative")
        self.mode, self.statistic, self.alternative = mode, statistic, alternative
        self.random_state = random_state

    def fit(self, y, x, covariates=None, *, blocks=None):
        outcomes = matrix(y, "y")
        n = len(outcomes)
        exposure = vector(x, "x", n)
        if outcomes.shape[1] == 0:
            raise ValueError("at least one outcome is required")
        q, rank = nuisance_basis(covariates, n)
        df = n - rank - 1
        if df < 2:
            raise ValueError("at least two residual degrees of freedom are required")
        # Standardize only for numerical conditioning, then return raw-unit effects.
        centered_exposure = exposure - exposure.mean()
        exposure_scale = float(np.linalg.norm(centered_exposure) / np.sqrt(n))
        if exposure_scale == 0:
            raise ValueError("exposure is constant or collinear with nuisance covariates")
        xr = residualize(centered_exposure / exposure_scale, q)
        xss = float(xr @ xr)
        if xss <= np.finfo(float).eps * n:
            raise ValueError("exposure is constant or collinear with nuisance covariates")
        outcome_scale = outcomes.std(axis=0)
        outcome_scale = np.where(outcome_scale > 0, outcome_scale, 1.0)
        yr = residualize((outcomes - outcomes.mean(axis=0)) / outcome_scale, q)
        leverage = np.sum(q * q, axis=1) + xr * xr / xss
        if self.statistic == "hc3" and np.any(leverage >= 1 - 1e-10):
            raise ValueError("HC3 requires full-model leverage below one")

        def calculate(residual_outcomes):
            # Input shape is (batch, subjects, targets). Reproject after shuffling.
            z = residual_outcomes - np.einsum(
                "nk,bkt->bnt", q, np.einsum("nk,bnt->bkt", q, residual_outcomes)
            )
            beta = np.einsum("n,bnt->bt", xr, z) / xss
            errors = z - xr[None, :, None] * beta[:, None, :]
            scale_floor = np.finfo(float).eps * np.maximum(np.sum(z * z, axis=1), 1.0)
            if self.statistic == "hc3":
                weights = xr / xss / (1 - leverage)
                variance = np.sum((errors * weights[None, :, None]) ** 2, axis=1)
            else:
                variance = np.sum(errors * errors, axis=1) / df / xss
            se = np.sqrt(np.maximum(variance, scale_floor / df / xss))
            if self.statistic == "partial_correlation":
                denom = np.sqrt(xss * np.sum(z * z, axis=1))
                stat = np.divide(beta * xss, denom, out=np.zeros_like(beta), where=denom > 1e-12)
                stat = np.clip(stat, -1, 1)
            else:
                stat = beta / se
            # Numerically zero outcomes carry no evidence.
            stat = np.where(np.sum(z * z, axis=1) > 1e-24, stat, 0.0)
            return stat, beta, se

        observed, beta, se = calculate(yr[None, :, :])
        groups = label_groups(blocks, n)
        rng = generator(self.random_state)
        if self.mode == "exact":
            count = math.prod(math.factorial(len(group)) for group in groups)
            if count > self.max_exact:
                raise ValueError("exact permutation group exceeds max_exact; choose Monte Carlo")
            group_permutations = [list(itertools.permutations(g.tolist())) for g in groups]

            def exact_rows():
                for choices in itertools.product(*group_permutations):
                    row = np.arange(n)
                    for group, choice in zip(groups, choices):
                        row[group] = choice
                    yield row

            rows = exact_rows()
        else:
            count, rows = self.n_permutations, None
        null = np.empty((count, outcomes.shape[1]))
        for start in range(0, count, self.batch_size):
            size = min(self.batch_size, count - start)
            indices = (np.asarray(list(itertools.islice(rows, size))) if rows is not None
                       else permutation_indices(rng, groups, size, n))
            null[start:start + size] = calculate(yr[indices])[0]

        orient = (np.abs if self.alternative == "two-sided" else
                  (lambda a: a) if self.alternative == "greater" else (lambda a: -a))
        obs, perm = orient(observed[0]), orient(null)
        tolerance = 1e-10 * np.maximum(1, np.abs(obs))
        exceed = np.sum(perm >= obs[None, :] - tolerance, axis=0)
        exceed_max = np.sum(np.max(perm, axis=1)[:, None] >= obs[None, :] - tolerance, axis=0)
        correction = 0 if self.mode == "exact" else 1
        p = (exceed + correction) / (count + correction)
        p_max = (exceed_max + correction) / (count + correction)
        self.result_ = PermutationResult(
            observed[0], beta[0] * outcome_scale / exposure_scale,
            se[0] * outcome_scale / exposure_scale, p, p_max, bh_fdr(p), null, df, rank,
            count, self.mode == "exact", self.statistic, self.alternative,
            tuple(len(g) for g in groups),
        )
        return self.result_


def freedman_lane(y, x, covariates=None, *, blocks=None, **kwargs):
    """Convenience interface returning a :class:`PermutationResult`."""
    return FreedmanLane(**kwargs).fit(y, x, covariates, blocks=blocks)

