"""Training-only nuisance regression, scaling, ridge and signed-network CPM.

Fixed hyperparameters prevent implicit use of held-out scores for tuning.
When an alpha grid is declared, each outer training set runs an additional
inner cross-validation over it, so tuning never touches held-out values.
Permutation tests restrict each shuffle to outer-fold and exchangeability cells,
fit reduced nuisance models inside each training set, and refit target-dependent
selection, tuning and regression for every null draw.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import numbers
import numpy as np
from scipy.stats import t as student_t

from ._utils import (
    correlation, generator, label_groups, matrix, permutation_indices,
    nuisance_svd, positive_integer, vector,
)


@dataclass(frozen=True)
class FoldAudit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    confound_mean: np.ndarray
    confound_scale: np.ndarray
    feature_nuisance_coef: np.ndarray
    target_nuisance_coef: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    selected_features: np.ndarray
    nuisance_rank: int
    nuisance_basis: np.ndarray
    nuisance_transform: np.ndarray
    nuisance_tolerance: float
    selected_alpha: float | None = None
    inner_scores: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class CVResult:
    predictions: np.ndarray
    baseline_predictions: np.ndarray
    fold_ids: np.ndarray
    mse: float
    baseline_mse: float
    mse_improvement: float
    incremental_r2: float
    prediction_correlation: float
    residual_prediction_correlation: float

    def to_dict(self):
        return {
            "predictions": self.predictions.tolist(),
            "baseline_predictions": self.baseline_predictions.tolist(),
            "fold_ids": self.fold_ids.tolist(), "mse": self.mse,
            "baseline_mse": self.baseline_mse, "mse_improvement": self.mse_improvement,
            "incremental_r2": self.incremental_r2,
            "prediction_correlation": self.prediction_correlation,
            "residual_prediction_correlation": self.residual_prediction_correlation,
        }


@dataclass(frozen=True)
class CVPermutationResult:
    observed_score: float
    null_scores: np.ndarray
    pvalue: float
    n_permutations: int
    fold_ids: np.ndarray
    block_sizes: tuple[int, ...]

    def to_dict(self, include_null=True):
        result = {
            "observed_score": self.observed_score, "pvalue": self.pvalue,
            "n_permutations": self.n_permutations,
            "fold_ids": self.fold_ids.tolist(), "block_sizes": list(self.block_sizes),
            "score": "out-of-fold baseline MSE minus connectome MSE",
            "assumption": "Reduced-model errors exchangeable within outer-fold and block cells",
        }
        if include_null:
            result["null_scores"] = self.null_scores.tolist()
        return result


def make_folds(n, n_splits=5, *, random_state=42, groups=None):
    """Construct folds independent of targets and confounds, keeping groups intact."""
    count = positive_integer(n_splits, "n_splits", 2)
    if count > n:
        raise ValueError("n_splits exceeds number of subjects")
    rng = generator(random_state)
    folds = np.empty(n, dtype=int)
    if groups is None:
        for fold, indices in enumerate(np.array_split(rng.permutation(n), count)):
            folds[indices] = fold
    else:
        cells = label_groups(groups, n, name="groups")
        if len(cells) < count:
            raise ValueError("there must be at least n_splits independent groups")
        sizes = np.zeros(count, dtype=int)
        shuffled = rng.permutation(len(cells))
        ordered = sorted(shuffled, key=lambda i: -len(cells[i]))
        for index in ordered:
            fold = int(np.argmin(sizes))
            folds[cells[index]] = fold
            sizes[fold] += len(cells[index])
    return folds


def validate_folds(n, fold_ids=None, *, n_splits=5, random_state=42, groups=None):
    """Validate subject labels and group isolation and return numeric folds."""
    if fold_ids is None:
        folds = make_folds(n, n_splits, random_state=random_state, groups=groups)
    else:
        cells = label_groups(fold_ids, n, name="fold_ids")
        if len(cells) < 2:
            raise ValueError("at least two nonempty folds are required")
        folds = np.empty(n, int)
        for i, cell in enumerate(cells):
            folds[cell] = i
        if groups is not None:
            for group in label_groups(groups, n, name="groups"):
                if len(np.unique(folds[group])) != 1:
                    raise ValueError("an independent group crosses held-out folds")
    if any(np.count_nonzero(folds != fold) < 4 for fold in np.unique(folds)):
        raise ValueError("each training fold needs at least four subjects")
    return folds


def _ridge_solve(design, target, penalty):
    """Solve ridge regression via thin SVD so tiny penalties on rank-deficient designs stay finite."""
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        factor = np.where(s > 0.0, 1.0 / (s + penalty / s), 0.0)
    return vt.T @ (factor * (u.T @ target))


def _mean_squared_error(residuals):
    """Compute mean squared error with max-magnitude scaling to prevent sum-of-squares overflow."""
    r = np.asarray(residuals, dtype=float)
    if not np.all(np.isfinite(r)):
        return float("inf")
    scale = float(np.max(np.abs(r))) if r.size else 0.0
    if scale == 0.0:
        return 0.0
    scaled = r / scale
    mean_sq = float(np.mean(scaled * scaled))
    with np.errstate(over="ignore"):
        return float(scale * (scale * mean_sq))


def _mse_difference(base_err, pred_err):
    """Compute MSE(base_err) - MSE(pred_err) without sum-of-squares overflow."""
    b = np.asarray(base_err, dtype=float)
    p = np.asarray(pred_err, dtype=float)
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(p))):
        return float("nan")
    scale = float(max(np.max(np.abs(b)), np.max(np.abs(p)))) if b.size else 0.0
    if scale == 0.0:
        return 0.0
    bs, ps = b / scale, p / scale
    mean_diff = float(np.mean((bs - ps) * (bs + ps)))
    with np.errstate(over="ignore"):
        return float(scale * (scale * mean_diff))


def _check_alpha_grid(alpha_grid):
    """Return a sorted tuple of unique finite positive penalties, or None."""
    if alpha_grid is None:
        return None
    if isinstance(alpha_grid, (str, bytes)):
        raise ValueError("alpha_grid must be a sequence of real numbers, not a string")
    try:
        values = list(alpha_grid)
    except TypeError as exc:
        raise ValueError("alpha_grid must be a sequence of real numbers") from exc
    grid = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError("alpha_grid values must be real numbers, not bool or text")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("alpha_grid values must be finite and strictly positive")
        grid.append(value)
    if not grid:
        raise ValueError("alpha_grid must contain at least one value")
    if len(set(grid)) != len(grid):
        raise ValueError("alpha_grid values must be unique")
    return tuple(sorted(grid))


class ConnectomePhysiologyCV:
    """Out-of-fold connectome prediction with auditable training state.

    Confounds are numeric; encode site using declared reference categories.
    ``model='partial_correlation'`` selects positive and negative edges using
    training partial-correlation p values and fits ridge to their two sums.
    ``alpha`` fixes the ridge penalty. ``alpha_grid`` instead tunes it inside
    each outer training set with ``inner_splits``-fold cross-validation, so no
    held-out subject influences selection; an exact tie takes the smallest
    grid value.
    """

    def __init__(self, n_splits=5, *, alpha=10.0, model="ridge",
                 selection_p=0.05, random_state=42,
                 alpha_grid: Sequence[float] | None = None,
                 inner_splits: int = 3):
        self.n_splits = positive_integer(n_splits, "n_splits", 2)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and strictly positive")
        if model not in ("ridge", "partial_correlation"):
            raise ValueError("model must be ridge or partial_correlation")
        if not np.isfinite(selection_p) or not 0 < selection_p <= 1:
            raise ValueError("selection_p must lie in (0, 1]")
        self.alpha, self.model = float(alpha), model
        self.selection_p, self.random_state = float(selection_p), random_state
        self.inner_splits = positive_integer(inner_splits, "inner_splits", 2)
        self.alpha_grid = _check_alpha_grid(alpha_grid)

    def _folds(self, n, fold_ids, groups):
        return validate_folds(n, fold_ids, n_splits=self.n_splits,
                              random_state=self.random_state, groups=groups)

    def _transform(self, x, y, c, x_eval, c_eval):
        """Fit nuisance and feature transforms on one training subset.

        All estimates come from the training rows passed in ``x``, ``y`` and
        ``c``; ``x_eval`` and ``c_eval`` are only projected through them.
        Returns the pieces shared by outer folds, inner tuning folds and
        permutation refits.
        """
        cmean = c.mean(axis=0)
        cscale = c.std(axis=0)
        cscale = np.where(cscale > 0, cscale, 1)
        train_design = np.column_stack((np.ones(len(c)), (c - cmean) / cscale))
        eval_design = np.column_stack((np.ones(len(c_eval)), (c_eval - cmean) / cscale))
        q, transform, tolerance = nuisance_svd(train_design)
        rank = q.shape[1]
        if len(x) - rank < 3:
            raise ValueError("training nuisance design leaves too few residual degrees of freedom")
        heldout_basis = eval_design @ transform
        scoresx, scoresy = q.T @ x, q.T @ y
        xtr = x - q @ scoresx
        xte = x_eval - heldout_basis @ scoresx
        ytr = y - q @ scoresy
        mean, scale = xtr.mean(axis=0), xtr.std(axis=0)
        precision = 100 * np.finfo(float).eps * np.maximum(
            np.max(np.abs(x), axis=0), np.finfo(float).tiny
        )
        scale = np.where(scale > precision, scale, 1)
        return {
            "confound_mean": cmean, "confound_scale": cscale,
            "q": q, "transform": transform, "tolerance": tolerance, "rank": rank,
            "heldout_basis": heldout_basis, "scoresx": scoresx, "scoresy": scoresy,
            "feature_nuisance_coef": transform @ scoresx,
            "target_nuisance_coef": transform @ scoresy,
            "xtr": (xtr - mean) / scale, "xte": (xte - mean) / scale,
            "ytr": ytr, "feature_mean": mean, "feature_scale": scale,
        }

    def _fit_weights(self, xtrain, ytrain, nuisance_rank, *, alpha=None):
        penalty = self.alpha if alpha is None else alpha
        p = xtrain.shape[1]
        if self.model == "ridge":
            weights = _ridge_solve(xtrain, ytrain, penalty)
            return weights, np.ones(p, dtype=bool)
        df = len(ytrain) - nuisance_rank - 1
        if df < 2:
            raise ValueError("training fold has insufficient partial-correlation degrees of freedom")
        denom = np.linalg.norm(xtrain, axis=0) * np.linalg.norm(ytrain)
        r = np.divide(xtrain.T @ ytrain, denom, out=np.zeros(p), where=denom > 1e-12)
        r = np.clip(r, -1, 1)
        tstat = r * np.sqrt(df / np.maximum(1 - r * r, 1e-14))
        selected = 2 * student_t.sf(np.abs(tstat), df) <= self.selection_p
        mapping = np.column_stack((selected & (r > 0), selected & (r < 0))).astype(float)
        sums = xtrain @ mapping
        sum_weights = _ridge_solve(sums, ytrain, penalty)
        return mapping @ sum_weights, selected

    def _inner_folds(self, train, groups):
        """Generate the cached inner split of one outer training subset.

        Only training row positions and their group labels are used, so
        held-out values and labels never shape the tuning folds.
        """
        inner_groups = None if groups is None else np.asarray(groups)[train]
        return make_folds(len(train), self.inner_splits,
                          random_state=self.random_state, groups=inner_groups)

    def _select_alpha(self, x, y, c, inner_folds):
        """Tune alpha on an outer training subset and return (alpha, scores).

        ``x``, ``y`` and ``c`` hold training rows only; each inner fold refits
        the shared transformation on its own training rows. A candidate's
        score is the pooled inner-validation sum of squared errors divided by
        the outer training count, so unequal inner folds weight each subject
        equally rather than each fold equally. Scores are returned in
        ascending grid order and an exact tie selects the smallest alpha.
        """
        predictions = np.empty((len(self.alpha_grid), len(y)))
        for fold in np.unique(inner_folds):
            inner_train, inner_test = inner_folds != fold, inner_folds == fold
            state = self._transform(x[inner_train], y[inner_train], c[inner_train],
                                    x[inner_test], c[inner_test])
            baseline = state["heldout_basis"] @ state["scoresy"]
            for j, alpha in enumerate(self.alpha_grid):
                weights, _ = self._fit_weights(state["xtr"], state["ytr"],
                                               state["rank"], alpha=alpha)
                predictions[j, inner_test] = baseline + state["xte"] @ weights
        scores = np.array([_mean_squared_error(y - row) for row in predictions])
        if not np.all(np.isfinite(scores)):
            raise ValueError("inner validation MSE is not finite; check input scaling")
        return self.alpha_grid[int(np.argmin(scores))], scores

    def fit(self, features, target, confounds=None, *, fold_ids=None, groups=None):
        """Fit each outer fold and record its auditable training state.

        With ``alpha_grid`` set, every outer training set first runs the inner
        search described by :meth:`_select_alpha`; the price is roughly
        ``inner_splits * len(alpha_grid)`` extra refits per outer fold, and the
        same retuning repeats inside every permutation draw. Inner folds are
        created once per outer training subset and reused for the whole run.
        When ``groups`` are supplied, inner folds keep those group boundaries
        intact, which requires every outer training subset to contain at least
        ``inner_splits`` independent groups.
        """
        x = matrix(features, "features")
        n = len(x)
        if x.shape[1] == 0:
            raise ValueError("features must have at least one column")
        y = vector(target, "target", n)
        c = np.empty((n, 0)) if confounds is None else matrix(confounds, "confounds", n)
        folds = self._folds(n, fold_ids, groups)
        predictions, baseline = np.empty(n), np.empty(n)
        audits, cached = [], []
        for fold in np.unique(folds):
            train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            inner_folds, inner_scores = None, ()
            if self.alpha_grid is not None:
                inner_folds = self._inner_folds(train, groups)
                alpha, scores = self._select_alpha(x[train], y[train], c[train], inner_folds)
                inner_scores = tuple(
                    (float(a), float(s)) for a, s in zip(self.alpha_grid, scores)
                )
            else:
                alpha = self.alpha
            state = self._transform(x[train], y[train], c[train], x[test], c[test])
            if self.alpha_grid is None:
                weights, selected = self._fit_weights(state["xtr"], state["ytr"], state["rank"])
            else:
                weights, selected = self._fit_weights(state["xtr"], state["ytr"],
                                                      state["rank"], alpha=alpha)
            baseline[test] = state["heldout_basis"] @ state["scoresy"]
            predictions[test] = baseline[test] + state["xte"] @ weights
            audits.append(FoldAudit(
                train.copy(), test.copy(), state["confound_mean"].copy(),
                state["confound_scale"].copy(), state["feature_nuisance_coef"].copy(),
                state["target_nuisance_coef"].copy(), state["feature_mean"].copy(),
                state["feature_scale"].copy(), weights.copy(), selected.copy(),
                state["rank"], state["q"].copy(), state["transform"].copy(),
                state["tolerance"], selected_alpha=float(alpha),
                inner_scores=inner_scores,
            ))
            # No cached fitting operation depends on held-out values or targets.
            cached_state = {"train": train, "test": test, "q": state["q"],
                            "heldout_basis": state["heldout_basis"],
                            "xtr": state["xtr"], "xte": state["xte"],
                            "scoresy": state["scoresy"], "rank": state["rank"],
                            "inner_folds": inner_folds}
            if inner_folds is not None:
                cached_state.update(x=x[train], c=c[train])
            cached.append(cached_state)
        mse = _mean_squared_error(y - predictions)
        baseline_mse = _mean_squared_error(y - baseline)
        improvement = _mse_difference(y - baseline, y - predictions)
        if not (np.isfinite(mse) and np.isfinite(baseline_mse) and np.isfinite(improvement)):
            raise ValueError("outer cross-validation MSE is not finite; check input scaling")
        self.result_ = CVResult(
            predictions, baseline, folds, mse, baseline_mse, improvement,
            float(improvement / baseline_mse) if baseline_mse > 1e-14 else 0.0,
            correlation(predictions, y), correlation(predictions - baseline, y - baseline),
        )
        self.fold_audits_, self._cached = tuple(audits), cached
        self._target, self.fold_ids_ = y, folds.copy()
        return self

    def fit_predict(self, features, target, confounds=None, **kwargs):
        return self.fit(features, target, confounds, **kwargs).result_

    def permutation_test(self, n_permutations=999, *, blocks=None, random_state=42):
        """Conditional, fold-restricted residual null for incremental prediction.

        Reuse fitted folds; training and held-out residuals never cross. The
        approximation inherits Freedman-Lane exchangeability assumptions.
        Grouped CV controls train/test isolation; dependent subjects additionally
        require valid exchangeability blocks, rather than unrestricted shuffles.
        When ``alpha_grid`` was configured, each pseudo dataset repeats the
        inner tuning on its pseudo outer-training target with the cached inner
        fold assignments, so the null carries the tuning step instead of
        inheriting the observed alpha; pseudo held-out targets never enter it.
        """
        if not hasattr(self, "result_"):
            raise ValueError("fit the model before requesting a permutation test")
        count = positive_integer(n_permutations, "n_permutations")
        n = len(self._target)
        original_blocks = label_groups(blocks, n)
        cells = [g[self.fold_ids_[g] == fold] for g in original_blocks
                 for fold in np.unique(self.fold_ids_[g])]
        rng = generator(random_state)
        indices = permutation_indices(rng, cells, count, n)
        scores = np.zeros(count)
        for state in self._cached:
            train, test = state["train"], state["test"]
            q, heldout_basis = state["q"], state["heldout_basis"]
            reduced = np.empty(n)
            reduced[train] = q @ state["scoresy"]
            reduced[test] = heldout_basis @ state["scoresy"]
            residuals = self._target - reduced
            for b, index in enumerate(indices):
                pseudo_train = reduced[train] + residuals[index[train]]
                pseudo_test = reduced[test] + residuals[index[test]]
                new_scores = q.T @ pseudo_train
                ytrain = pseudo_train - q @ new_scores
                if state["inner_folds"] is None:
                    weights, _ = self._fit_weights(state["xtr"], ytrain, state["rank"])
                else:
                    alpha, _ = self._select_alpha(state["x"], pseudo_train,
                                                  state["c"], state["inner_folds"])
                    weights, _ = self._fit_weights(state["xtr"], ytrain,
                                                  state["rank"], alpha=alpha)
                base = heldout_basis @ new_scores
                predicted = base + state["xte"] @ weights
                fold_diff = _mse_difference(pseudo_test - base, pseudo_test - predicted)
                scores[b] += (len(test) / n) * fold_diff
        observed = self.result_.mse_improvement
        if not (np.isfinite(observed) and np.all(np.isfinite(scores))):
            raise ValueError("non-finite cross-validation permutation scores; check input scaling")
        mse_scale = max(abs(observed), self.result_.baseline_mse, self.result_.mse)
        tolerance = 1e-10 * mse_scale
        pvalue = float((1 + np.count_nonzero(scores >= observed - tolerance)) / (count + 1))
        return CVPermutationResult(
            observed, scores, pvalue, count, self.fold_ids_.copy(), tuple(len(g) for g in cells)
        )
