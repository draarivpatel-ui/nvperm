"""Training-only nuisance regression, scaling, ridge and signed-network CPM.

Fixed hyperparameters prevent implicit use of held-out scores for tuning.
Permutation tests restrict each shuffle to outer-fold and exchangeability cells,
fit reduced nuisance models inside each training set, and refit target-dependent
selection and regression for every null draw.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import t as student_t

from ._utils import (
    correlation, generator, label_groups, matrix, permutation_indices,
    positive_integer, vector,
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
        cells = label_groups(groups, n)
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


class ConnectomePhysiologyCV:
    """Out-of-fold connectome prediction with auditable training state.

    Confounds are numeric; encode site using declared reference categories.
    ``model='partial_correlation'`` selects positive and negative edges using
    training partial-correlation p values and fits ridge to their two sums.
    Alpha and selection thresholds are fixed before evaluation. Nested tuning
    is intentionally left to a caller-provided training-only procedure.
    """

    def __init__(self, n_splits=5, *, alpha=10.0, model="ridge",
                 selection_p=0.05, random_state=42):
        self.n_splits = positive_integer(n_splits, "n_splits", 2)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and strictly positive")
        if model not in ("ridge", "partial_correlation"):
            raise ValueError("model must be ridge or partial_correlation")
        if not np.isfinite(selection_p) or not 0 < selection_p <= 1:
            raise ValueError("selection_p must lie in (0, 1]")
        self.alpha, self.model = float(alpha), model
        self.selection_p, self.random_state = float(selection_p), random_state

    def _folds(self, n, fold_ids, groups):
        if fold_ids is None:
            return make_folds(n, self.n_splits, random_state=self.random_state, groups=groups)
        cells = label_groups(fold_ids, n)
        if len(cells) < 2:
            raise ValueError("at least two nonempty folds are required")
        folds = np.empty(n, int)
        for i, cell in enumerate(cells):
            folds[cell] = i
        if groups is not None:
            for group in label_groups(groups, n):
                if len(np.unique(folds[group])) != 1:
                    raise ValueError("an independent group crosses held-out folds")
        return folds

    def _fit_weights(self, xtrain, ytrain, nuisance_rank):
        p = xtrain.shape[1]
        if self.model == "ridge":
            # The dual form avoids a large edge-by-edge matrix for p > n.
            if p > len(xtrain):
                weights = xtrain.T @ np.linalg.solve(
                    xtrain @ xtrain.T + self.alpha * np.eye(len(xtrain)), ytrain
                )
            else:
                weights = np.linalg.solve(
                    xtrain.T @ xtrain + self.alpha * np.eye(p), xtrain.T @ ytrain
                )
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
        sum_weights = np.linalg.solve(
            sums.T @ sums + self.alpha * np.eye(2), sums.T @ ytrain
        )
        return mapping @ sum_weights, selected

    def fit(self, features, target, confounds=None, *, fold_ids=None, groups=None):
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
            if len(train) < 4:
                raise ValueError("each training fold needs at least four subjects")
            cmean = c[train].mean(axis=0)
            cscale = c[train].std(axis=0)
            cscale = np.where(cscale > 0, cscale, 1)
            ctr = np.column_stack((np.ones(len(train)), (c[train] - cmean) / cscale))
            cte = np.column_stack((np.ones(len(test)), (c[test] - cmean) / cscale))
            rank = int(np.linalg.matrix_rank(ctr))
            if len(train) - rank < 3:
                raise ValueError("training nuisance design leaves too few residual degrees of freedom")
            pinv = np.linalg.pinv(ctr)
            coefx, coefy = pinv @ x[train], pinv @ y[train]
            xtr, xte = x[train] - ctr @ coefx, x[test] - cte @ coefx
            ytr = y[train] - ctr @ coefy
            mean, scale = xtr.mean(axis=0), xtr.std(axis=0)
            precision = 100 * np.finfo(float).eps * np.maximum(np.max(np.abs(x[train]), axis=0), np.finfo(float).tiny)
            scale = np.where(scale > precision, scale, 1)
            xtr, xte = (xtr - mean) / scale, (xte - mean) / scale
            weights, selected = self._fit_weights(xtr, ytr, rank)
            baseline[test] = cte @ coefy
            predictions[test] = baseline[test] + xte @ weights
            audits.append(FoldAudit(
                train.copy(), test.copy(), cmean.copy(), cscale.copy(), coefx.copy(),
                coefy.copy(), mean.copy(), scale.copy(), weights.copy(), selected.copy(), rank,
            ))
            # No cached fitting operation depends on held-out values or targets.
            cached.append({"train": train, "test": test, "ctr": ctr, "cte": cte,
                           "pinv": pinv, "xtr": xtr, "xte": xte,
                           "coefy": coefy, "rank": rank})
        mse = float(np.mean((y - predictions) ** 2))
        baseline_mse = float(np.mean((y - baseline) ** 2))
        improvement = baseline_mse - mse
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
            ctr, cte, coefy = state["ctr"], state["cte"], state["coefy"]
            reduced = np.empty(n)
            reduced[train], reduced[test] = ctr @ coefy, cte @ coefy
            residuals = self._target - reduced
            for b, index in enumerate(indices):
                pseudo_train = reduced[train] + residuals[index[train]]
                pseudo_test = reduced[test] + residuals[index[test]]
                new_coef = state["pinv"] @ pseudo_train
                ytrain = pseudo_train - ctr @ new_coef
                weights, _ = self._fit_weights(state["xtr"], ytrain, state["rank"])
                base = cte @ new_coef
                predicted = base + state["xte"] @ weights
                scores[b] += np.sum((pseudo_test - base) ** 2 - (pseudo_test - predicted) ** 2) / n
        observed = self.result_.mse_improvement
        tolerance = 1e-10 * max(1, abs(observed))
        pvalue = float((1 + np.count_nonzero(scores >= observed - tolerance)) / (count + 1))
        return CVPermutationResult(
            observed, scores, pvalue, count, self.fold_ids_.copy(), tuple(len(g) for g in cells)
        )

