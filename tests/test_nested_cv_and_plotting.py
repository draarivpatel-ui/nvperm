import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from nvperm import ConnectomePhysiologyCV, CVResult, PermutationResult
from nvperm.cv_connectome import FoldAudit, make_folds
import nvperm.plotting as plotting

GRID = (0.5, 2.0, 10.0, 50.0)


def _nested(model_name, **overrides):
    kwargs = dict(n_splits=4, model=model_name, alpha_grid=GRID,
                  inner_splits=3, random_state=5)
    kwargs.update(overrides)
    return ConnectomePhysiologyCV(**kwargs)


@pytest.mark.parametrize("model_name", ["ridge", "partial_correlation"])
@pytest.mark.parametrize("field", ["target", "confounds", "features"])
def test_nested_tuning_ignores_heldout_rows(cv_data, model_name, field):
    x, y, c, folds = [a.copy() for a in cv_data]
    before = _nested(model_name).fit(x, y, c, fold_ids=folds)
    test_rows = folds == 0
    if field == "target":
        y[test_rows] += 500
    elif field == "confounds":
        c[test_rows] += 200
    else:
        x[test_rows] *= 1000
    after = _nested(model_name).fit(x, y, c, fold_ids=folds)
    first, second = before.fold_audits_[0], after.fold_audits_[0]
    assert first.selected_alpha == second.selected_alpha
    assert first.inner_scores == second.inner_scores
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.selected_features, second.selected_features)


@pytest.mark.parametrize("model_name", ["ridge", "partial_correlation"])
def test_inner_scores_match_manual_fixed_alpha_cv(cv_data, model_name):
    x, y, c, folds = cv_data
    model = _nested(model_name).fit(x, y, c, fold_ids=folds)
    for index, audit in enumerate(model.fold_audits_):
        train = audit.train_indices
        inner_ids = make_folds(len(train), 3, random_state=5)
        scores = []
        for alpha in GRID:
            sub = ConnectomePhysiologyCV(n_splits=3, model=model_name, alpha=alpha)
            scores.append(sub.fit(x[train], y[train], c[train],
                                  fold_ids=inner_ids).result_.mse)
        assert audit.inner_scores == tuple(zip(GRID, scores))
        expected = GRID[int(np.argmin(scores))]
        assert audit.selected_alpha == expected
        refit = ConnectomePhysiologyCV(n_splits=4, model=model_name, alpha=expected)
        np.testing.assert_array_equal(
            refit.fit(x, y, c, fold_ids=folds).fold_audits_[index].weights,
            audit.weights)


@pytest.mark.parametrize("model_name", ["ridge", "partial_correlation"])
def test_inner_folds_respect_group_boundaries(cv_data, model_name):
    x, y, c, _ = cv_data
    groups = np.repeat(np.arange(16), 4)
    model = _nested(model_name, random_state=3).fit(x, y, c, groups=groups)
    for group in np.unique(groups):
        assert len(np.unique(model.fold_ids_[groups == group])) == 1
    assert all(a.selected_alpha in GRID for a in model.fold_audits_)


def test_inner_folds_keep_groups_together(cv_data):
    x, y, c, _ = cv_data
    groups = np.repeat(np.arange(16), 4)
    captured = []

    class Spy(ConnectomePhysiologyCV):
        def _inner_folds(self, train, groups_arg):
            inner = super()._inner_folds(train, groups_arg)
            captured.append((train.copy(), inner.copy()))
            return inner

    Spy(n_splits=4, alpha_grid=GRID, inner_splits=3,
        random_state=3).fit(x, y, c, groups=groups)
    assert len(captured) == 4
    for train, inner in captured:
        subset = groups[train]
        for group in np.unique(subset):
            assert len(np.unique(inner[subset == group])) == 1


def test_nested_rejects_insufficient_inner_groups(cv_data):
    x, y, c, _ = cv_data
    groups = np.repeat(np.arange(4), 16)
    folds = np.repeat(np.arange(4), 16)
    with pytest.raises(ValueError, match="independent groups"):
        _nested("ridge", inner_splits=4).fit(x, y, c, fold_ids=folds, groups=groups)


@pytest.mark.parametrize("model_name", ["ridge", "partial_correlation"])
def test_exact_tie_selects_smallest_grid_value(model_name):
    rng = np.random.default_rng(4)
    n = 24
    c = rng.normal(size=(n, 2))
    y = c[:, 0] + rng.normal(size=n)
    x = np.zeros((n, 5))
    folds = np.arange(n) % 4
    model = ConnectomePhysiologyCV(n_splits=4, model=model_name,
                                 alpha_grid=(3.0, 0.5, 9.0), inner_splits=2)
    model.fit(x, y, c, fold_ids=folds)
    for audit in model.fold_audits_:
        assert [a for a, _ in audit.inner_scores] == [0.5, 3.0, 9.0]
        scores = [s for _, s in audit.inner_scores]
        assert scores[0] == scores[1] == scores[2]
        assert audit.selected_alpha == 0.5


@pytest.mark.parametrize("grid", [
    [], (), (0.0,), (-1.0,), (np.nan,), (np.inf,), (1.0, 1.0), (1.0, 2 - 1),
    ("a",), (True,), (False, 1.0), (None,), (1 + 2j,), "0.5", 5.0, np.float64(4),
])
def test_invalid_alpha_grid(grid):
    with pytest.raises(ValueError):
        ConnectomePhysiologyCV(alpha_grid=grid)


@pytest.mark.parametrize("splits", [0, 1, -2, 2.5, "3", True, None])
def test_invalid_inner_splits(splits):
    with pytest.raises(ValueError):
        ConnectomePhysiologyCV(alpha_grid=GRID, inner_splits=splits)


def test_fixed_mode_reports_selected_alpha_and_empty_scores(cv_data):
    x, y, c, folds = cv_data
    model = ConnectomePhysiologyCV(n_splits=4, alpha=7.5).fit(x, y, c, fold_ids=folds)
    for audit in model.fold_audits_:
        assert audit.selected_alpha == 7.5
        assert audit.inner_scores == ()


def test_fold_audit_positional_construction_defaults():
    audit = FoldAudit(np.arange(3), np.arange(2), np.zeros(2), np.ones(2),
                      np.zeros((2, 3)), np.zeros(2), np.zeros(3), np.ones(3),
                      np.zeros(3), np.ones(3, bool), 2, np.zeros((3, 2)),
                      np.zeros((2, 2)), 1e-9)
    assert audit.selected_alpha is None
    assert audit.inner_scores == ()


class _SpyCV(ConnectomePhysiologyCV):
    def _select_alpha(self, x, y, c, inner_folds):
        self.tune_calls += 1
        self.seen_folds.append(inner_folds)
        self.seen_targets.append(np.asarray(y).copy())
        return super()._select_alpha(x, y, c, inner_folds)

    def _fit_weights(self, xtrain, ytrain, nuisance_rank, *, alpha=None):
        self.weight_calls.append(alpha)
        return super()._fit_weights(xtrain, ytrain, nuisance_rank, alpha=alpha)


def test_nested_permutation_retunes_every_pseudo_dataset(cv_data):
    x, y, c, folds = cv_data
    grid = (0.5, 5.0, 50.0)
    model = _SpyCV(n_splits=4, alpha_grid=grid, inner_splits=2, random_state=5)
    model.tune_calls, model.weight_calls = 0, []
    model.seen_folds, model.seen_targets = [], []
    model.fit(x, y, c, fold_ids=folds)
    assert model.tune_calls == 4
    assert len(model.weight_calls) == 4 * (2 * len(grid) + 1)
    model.tune_calls, model.weight_calls = 0, []
    model.seen_folds, model.seen_targets = [], []
    count = 7
    result = model.permutation_test(count, random_state=11)
    assert model.tune_calls == 4 * count
    assert len(model.weight_calls) == 4 * count * (2 * len(grid) + 1)
    assert set(model.weight_calls) <= set(grid)
    fold_ids = [id(f) for f in model.seen_folds]
    for block in range(4):
        assert len(set(fold_ids[block * count:(block + 1) * count])) == 1
    assert len(set(fold_ids[::count])) == 4
    for i, seen in enumerate(model.seen_targets):
        train = model.fold_audits_[i // count].train_indices
        assert not np.array_equal(seen, y[train])
    repeated = model.permutation_test(count, random_state=11)
    np.testing.assert_array_equal(result.null_scores, repeated.null_scores)
    for i in range(4 * count):
        np.testing.assert_array_equal(model.seen_targets[i],
                                      model.seen_targets[i + 4 * count])


def _permutation_result(pvalues=None):
    n = 6
    return PermutationResult(
        statistic=np.linspace(-2, 3, n),
        effect=np.array([-0.4, -0.2, 0.0, 0.15, 0.3, 0.55]),
        standard_error=np.full(n, 0.1),
        pvalues=np.array([0.8, 0.04, 0.9, 0.2, 0.01, 0.001]) if pvalues is None else pvalues,
        pvalues_max_t=np.array([0.9, 0.3, 0.95, 0.7, 0.03, 0.01]),
        qvalues=np.array([0.9, 0.02, 0.9, 0.48, 0.06, 0.006]),
        null_statistics=np.zeros((5, n)),
        residual_df=50, nuisance_rank=5, n_permutations=199,
        exact=False, statistic_name="studentized",
        alternative="two-sided", block_sizes=(n,),
    )


def _cv_result(observed):
    n = len(observed)
    baseline = 0.4 * observed
    full = 0.7 * observed
    return CVResult(full, baseline, np.arange(n) % 4,
                    float(np.mean((observed - full) ** 2)),
                    float(np.mean((observed - baseline) ** 2)),
                    0.5, 0.5, 0.9, 0.8)


@pytest.fixture(name="plt")
def _plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    return pytest.importorskip("matplotlib.pyplot")


def _legend_texts(ax):
    return {text.get_text() for text in ax.get_legend().get_texts()}


def test_forest_marks_both_significance_criteria(plt):
    labels = [f"edge {i}" for i in range(6)]
    ax = plotting.plot_adjusted_effects(_permutation_result(), labels=labels,
                                      kind="forest", alpha=0.05)
    assert isinstance(ax, plt.Axes)
    assert [t.get_text() for t in ax.get_yticklabels()] == labels
    assert _legend_texts(ax) == {
        "max-T and BH q <= 0.05", "max-T only <= 0.05",
        "BH q only <= 0.05", "not significant at 0.05",
    }


def test_forest_draws_intervals_by_endpoints(plt):
    from matplotlib.collections import LineCollection
    result = _permutation_result()
    intervals = types.SimpleNamespace(
        ci_low=result.effect - 0.1, ci_high=result.effect + 0.1)
    intervals.ci_low[0], intervals.ci_high[0] = 0.5, 0.7
    ax = plotting.plot_adjusted_effects(result, intervals=intervals, kind="forest")
    assert any(isinstance(c, LineCollection) for c in ax.collections)


def test_volcano_uses_finite_heights_without_raw_threshold(plt):
    pvalues = np.array([0.0, 0.04, 0.9, 0.2, 0.01,
                        np.finfo(float).smallest_subnormal])
    ax = plotting.plot_adjusted_effects(_permutation_result(pvalues), kind="volcano")
    heights = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections])
    assert heights.size == 6 and np.all(np.isfinite(heights))
    assert len(ax.lines) == 1
    assert "log10" in ax.get_ylabel()


@pytest.mark.parametrize("kwargs", [
    {"kind": "bad"}, {"alpha": 0.0}, {"alpha": 1.0}, {"alpha": np.nan}, {"alpha": True},
    {"alpha": "0.05"},
    {"labels": ["a", "b"]}, {"intervals": types.SimpleNamespace(ci_low=np.zeros(6), ci_high=np.ones(6)), "kind": "volcano"},
])
def test_adjusted_effects_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        plotting.plot_adjusted_effects(_permutation_result(), **kwargs)


def test_adjusted_effects_validates_intervals():
    result = _permutation_result()
    for intervals in (
        types.SimpleNamespace(ci_low=np.zeros(5), ci_high=np.ones(5)),
        types.SimpleNamespace(ci_low=np.zeros(6), ci_high=np.full(6, np.inf)),
        types.SimpleNamespace(ci_low=np.ones(6), ci_high=np.zeros(6)),
        types.SimpleNamespace(ci_low=np.zeros(6)),
    ):
        with pytest.raises(ValueError):
            plotting.plot_adjusted_effects(result, intervals=intervals)


def test_adjusted_effects_rejects_nonfinite_or_out_of_range():
    result = _permutation_result()
    for field, value in (("effect", np.nan), ("pvalues", 1.2), ("qvalues", -0.1)):
        broken = {**result.__dict__, field: np.full(6, value)}
        with pytest.raises(ValueError):
            plotting.plot_adjusted_effects(PermutationResult(**broken))
    with pytest.raises(ValueError):
        plotting.plot_adjusted_effects(_cv_result(np.ones(6)))


def test_cv_predictions_plots_baseline_full_and_identity(plt):
    observed = np.linspace(-2, 3, 30) + np.arange(30) * 0.01
    ax = plotting.plot_cv_predictions(observed, _cv_result(observed))
    assert isinstance(ax, plt.Axes)
    assert len(ax.collections) == 2
    assert len(ax.lines) == 1
    assert _legend_texts(ax) == {"identity", "nuisance baseline", "connectome model"}
    assert ax.get_xlabel() == "Observed target"
    assert ax.get_ylabel() == "Out-of-fold prediction"


def test_cv_predictions_validates_shapes_and_finiteness():
    observed = np.linspace(0, 1, 30)
    result = _cv_result(observed)
    with pytest.raises(ValueError, match="nonempty"):
        plotting.plot_cv_predictions(np.empty(0), result)
    with pytest.raises(ValueError):
        plotting.plot_cv_predictions(observed[:20], result)
    with pytest.raises(ValueError):
        plotting.plot_cv_predictions(np.full(30, np.nan), result)
    with pytest.raises(ValueError):
        plotting.plot_cv_predictions(observed, _permutation_result())


def test_plotting_import_leaves_matplotlib_unloaded():
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src)}
    code = ("import sys, nvperm.plotting; "
            "sys.exit(0 if 'matplotlib' not in sys.modules else 1)")
    outcome = subprocess.run([sys.executable, "-c", code], env=env,
                             capture_output=True, text=True, timeout=60)
    assert outcome.returncode == 0, outcome.stderr


def test_missing_matplotlib_reports_plot_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)
    observed = np.linspace(0, 1, 10)
    with pytest.raises(ImportError, match="plot"):
        plotting.plot_cv_predictions(observed, _cv_result(observed))
