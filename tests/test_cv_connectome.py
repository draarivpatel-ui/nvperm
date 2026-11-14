import numpy as np
import pytest

from nvperm import ConnectomePhysiologyCV
from nvperm.cv_connectome import make_folds


@pytest.mark.parametrize("field", ["target", "confounds", "features"])
def test_heldout_mutations_cannot_change_training_state(cv_data, field):
    x, y, c, folds = [a.copy() for a in cv_data]
    before = ConnectomePhysiologyCV(n_splits=4).fit(x, y, c, fold_ids=folds)
    test = folds == 0
    if field == "target":
        y[test] += 500
    elif field == "confounds":
        c[test] += 200
    else:
        x[test] *= 1000
    after = ConnectomePhysiologyCV(n_splits=4).fit(x, y, c, fold_ids=folds)
    for name in ("weights", "confound_mean", "confound_scale", "feature_nuisance_coef",
                 "target_nuisance_coef", "feature_mean", "feature_scale", "selected_features"):
        np.testing.assert_array_equal(getattr(before.fold_audits_[0], name), getattr(after.fold_audits_[0], name))


def test_every_subject_is_predicted_once_and_train_test_disjoint(cv_data):
    x, y, c, folds = cv_data
    model = ConnectomePhysiologyCV(n_splits=4).fit(x, y, c, fold_ids=folds)
    counts = np.zeros(len(y), int)
    for audit in model.fold_audits_:
        assert not np.intersect1d(audit.train_indices, audit.test_indices).size
        counts[audit.test_indices] += 1
    np.testing.assert_array_equal(counts, np.ones(len(y)))


def test_nuisance_baseline_matches_manual_training_fit(cv_data):
    x, y, c, folds = cv_data
    result = ConnectomePhysiologyCV(n_splits=4).fit_predict(x, y, c, fold_ids=folds)
    for fold in range(4):
        train, test = folds != fold, folds == fold
        design = np.column_stack((np.ones(train.sum()), c[train]))
        beta = np.linalg.lstsq(design, y[train], rcond=None)[0]
        expected = np.column_stack((np.ones(test.sum()), c[test])) @ beta
        np.testing.assert_allclose(result.baseline_predictions[test], expected, atol=1e-11)


def test_grouped_cv_keeps_families_together(cv_data):
    x, y, c, _ = cv_data
    groups = np.repeat(np.arange(16), 4)
    model = ConnectomePhysiologyCV(n_splits=4).fit(x, y, c, groups=groups)
    for group in np.unique(groups):
        assert len(np.unique(model.fold_ids_[groups == group])) == 1
    for audit in model.fold_audits_:
        assert not np.intersect1d(groups[audit.train_indices], groups[audit.test_indices]).size


def test_given_folds_reject_group_overlap(cv_data):
    x, y, c, folds = cv_data
    with pytest.raises(ValueError, match="crosses"):
        ConnectomePhysiologyCV(n_splits=4).fit(x, y, c, fold_ids=folds,
                                             groups=np.repeat(np.arange(16), 4))


def test_folds_do_not_depend_on_targets_or_confounds(cv_data):
    x, y, c, _ = cv_data
    one = ConnectomePhysiologyCV(random_state=83).fit(x, y, c)
    two = ConnectomePhysiologyCV(random_state=83).fit(x, -y + 400, c + 10)
    np.testing.assert_array_equal(one.fold_ids_, two.fold_ids_)


def test_singular_covariates_are_supported(cv_data):
    x, y, c, folds = cv_data
    duplicate = np.column_stack((c, c[:, 0], np.ones(len(y))))
    one = ConnectomePhysiologyCV().fit_predict(x, y, c, fold_ids=folds)
    two = ConnectomePhysiologyCV().fit_predict(x, y, duplicate, fold_ids=folds)
    np.testing.assert_allclose(one.predictions, two.predictions, atol=1e-11)


def test_partial_correlation_selection_is_training_only(cv_data):
    x, y, c, folds = [a.copy() for a in cv_data]
    before = ConnectomePhysiologyCV(model="partial_correlation").fit(x, y, c, fold_ids=folds)
    y[folds == 0] *= -100
    after = ConnectomePhysiologyCV(model="partial_correlation").fit(x, y, c, fold_ids=folds)
    np.testing.assert_array_equal(before.fold_audits_[0].selected_features, after.fold_audits_[0].selected_features)
    np.testing.assert_array_equal(before.fold_audits_[0].weights, after.fold_audits_[0].weights)


def test_no_selected_features_returns_baseline(cv_data):
    _, y, c, folds = cv_data
    x = np.ones((len(y), 8))
    result = ConnectomePhysiologyCV(model="partial_correlation", selection_p=1e-12).fit_predict(x, y, c, fold_ids=folds)
    np.testing.assert_allclose(result.predictions, result.baseline_predictions, atol=1e-12)


def test_high_dimensional_ridge_dual_is_finite():
    rng = np.random.default_rng(35)
    x, y = rng.normal(size=(36, 80)), rng.normal(size=36)
    result = ConnectomePhysiologyCV(n_splits=3).fit_predict(x, y)
    assert np.all(np.isfinite(result.predictions))


def test_permutation_determinism_blocks_and_model_immutability(cv_data):
    x, y, c, folds = cv_data
    model = ConnectomePhysiologyCV().fit(x, y, c, fold_ids=folds)
    weights = [a.weights.copy() for a in model.fold_audits_]
    one = model.permutation_test(19, blocks=np.arange(len(y)) % 2, random_state=32)
    two = model.permutation_test(19, blocks=np.arange(len(y)) % 2, random_state=32)
    np.testing.assert_array_equal(one.null_scores, two.null_scores)
    assert one.pvalue >= 1 / 20
    assert sum(one.block_sizes) == len(y)
    for before, audit in zip(weights, model.fold_audits_):
        np.testing.assert_array_equal(before, audit.weights)


def test_heldout_target_never_enters_null_training_weights(cv_data):
    class RecordingCV(ConnectomePhysiologyCV):
        def _fit_weights(self, xtrain, ytrain, nuisance_rank):
            weights, selected = super()._fit_weights(xtrain, ytrain, nuisance_rank)
            self.recorded.append(weights.copy())
            return weights, selected

    x, y, c, folds = [a.copy() for a in cv_data]
    first = RecordingCV()
    first.recorded = []
    first.fit(x, y, c, fold_ids=folds)
    first.recorded = []
    first.permutation_test(11, random_state=9)
    y[folds == 0] += np.arange(np.count_nonzero(folds == 0)) * 100
    c[folds == 0] += 200
    second = RecordingCV()
    second.recorded = []
    second.fit(x, y, c, fold_ids=folds)
    second.recorded = []
    second.permutation_test(11, random_state=9)
    np.testing.assert_array_equal(first.recorded[:11], second.recorded[:11])


def test_singleton_permutation_cells_give_p_one(cv_data):
    x, y, c, folds = cv_data
    model = ConnectomePhysiologyCV().fit(x, y, c, fold_ids=folds)
    result = model.permutation_test(9, blocks=np.arange(len(y)))
    np.testing.assert_allclose(result.null_scores, result.observed_score, atol=1e-11)
    assert result.pvalue == 1


def test_cv_permutation_requires_fit():
    with pytest.raises(ValueError, match="fit"):
        ConnectomePhysiologyCV().permutation_test(9)


def test_metrics_and_serialization(cv_data):
    x, y, c, folds = cv_data
    model = ConnectomePhysiologyCV().fit(x, y, c, fold_ids=folds)
    result = model.result_
    assert result.mse_improvement == pytest.approx(result.baseline_mse - result.mse)
    assert len(result.to_dict()["predictions"]) == len(y)
    assert len(model.permutation_test(9).to_dict()["null_scores"]) == 9


@pytest.mark.parametrize("kwargs", [
    {"alpha": 0}, {"alpha": np.nan}, {"model": "bad"}, {"n_splits": 1},
    {"selection_p": 0}, {"selection_p": 2},
])
def test_invalid_model_parameters(kwargs):
    with pytest.raises(ValueError):
        ConnectomePhysiologyCV(**kwargs)


def test_invalid_fold_inputs(cv_data):
    x, y, c, folds = cv_data
    with pytest.raises(ValueError):
        ConnectomePhysiologyCV().fit(x, y, c, fold_ids=np.zeros(len(y)))
    with pytest.raises(ValueError):
        make_folds(4, 5)
    with pytest.raises(ValueError):
        make_folds(8, 4, groups=np.repeat([0, 1], 4))

