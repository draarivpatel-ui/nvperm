import numpy as np
import pytest
from scipy.stats import t as student_t

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
                 "target_nuisance_coef", "feature_mean", "feature_scale", "selected_features",
                 "nuisance_basis", "nuisance_transform", "nuisance_tolerance", "nuisance_rank"):
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


def _near_duplicate_data():
    # Preserve the exact random draw order of the Round 1 public-audit probe.
    rng = np.random.default_rng(8)
    z = rng.normal(size=64)
    noise = rng.normal(size=64)
    x = rng.normal(size=(64, 4))
    y = z + rng.normal(size=64)
    c = np.column_stack((z, z + 1e-14 * noise))
    return x, y, c, np.arange(64) % 4


def _reference_fold(x, y, c, train, test, model, alpha=10.0, selection_p=0.05):
    """Independent retained-SVD projection and primal ridge/CPM reference."""
    mean, scale = c[train].mean(axis=0), c[train].std(axis=0)
    scale = np.where(scale > 0, scale, 1)
    design = np.column_stack((np.ones(len(train)), (c[train] - mean) / scale))
    heldout = np.column_stack((np.ones(len(test)), (c[test] - mean) / scale))
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    tolerance = np.finfo(float).eps * max(design.shape) * s[0]
    retained = s > tolerance
    q = u[:, retained]
    transform = vt[retained].T / s[retained]
    heldout_basis = heldout @ transform
    xtr = x[train] - q @ (q.T @ x[train])
    xte = x[test] - heldout_basis @ (q.T @ x[train])
    feature_mean, feature_scale = xtr.mean(axis=0), xtr.std(axis=0)
    precision = 100 * np.finfo(float).eps * np.maximum(
        np.max(np.abs(x[train]), axis=0), np.finfo(float).tiny
    )
    feature_scale = np.where(feature_scale > precision, feature_scale, 1)
    xtr = (xtr - feature_mean) / feature_scale
    xte = (xte - feature_mean) / feature_scale

    def weights_for(target):
        residual = target - q @ (q.T @ target)
        if model == "ridge":
            weights = np.linalg.solve(xtr.T @ xtr + alpha * np.eye(xtr.shape[1]),
                                      xtr.T @ residual)
            selected = np.ones(xtr.shape[1], dtype=bool)
        else:
            df = len(train) - np.count_nonzero(retained) - 1
            denom = np.linalg.norm(xtr, axis=0) * np.linalg.norm(residual)
            r = np.divide(xtr.T @ residual, denom, out=np.zeros(xtr.shape[1]),
                          where=denom > 1e-12)
            r = np.clip(r, -1, 1)
            tstat = r * np.sqrt(df / np.maximum(1 - r * r, 1e-14))
            selected = 2 * student_t.sf(np.abs(tstat), df) <= selection_p
            mapping = np.column_stack((selected & (r > 0), selected & (r < 0))).astype(float)
            sums = xtr @ mapping
            weights = mapping @ np.linalg.solve(sums.T @ sums + alpha * np.eye(2),
                                                sums.T @ residual)
        return weights, selected, residual

    weights, selected, ytr = weights_for(y[train])
    baseline = heldout_basis @ (q.T @ y[train])
    return {"design": design, "q": q, "s": s, "transform": transform,
            "tolerance": tolerance, "heldout_basis": heldout_basis,
            "xtr": xtr, "xte": xte, "ytr": ytr,
            "feature_mean": feature_mean, "feature_scale": feature_scale,
            "weights": weights, "selected": selected, "weights_for": weights_for,
            "baseline": baseline, "predictions": baseline + xte @ weights}


def test_near_duplicate_confounds_match_reported_rank_public_audit_probe():
    x, y, c, folds = _near_duplicate_data()
    model = ConnectomePhysiologyCV(n_splits=4).fit(x, y, c, fold_ids=folds)
    for audit in model.fold_audits_:
        train, test = audit.train_indices, audit.test_indices
        reference = _reference_fold(x, y, c, train, test, "ridge")
        assert audit.nuisance_rank == 2
        assert np.count_nonzero(reference["s"] > 1e-15 * reference["s"][0]) == 3
        assert audit.nuisance_rank == np.count_nonzero(reference["s"] > audit.nuisance_tolerance)
        actual = x[train] - reference["design"] @ audit.feature_nuisance_coef
        expected = x[train] - reference["q"] @ (reference["q"].T @ x[train])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(audit.nuisance_basis, reference["q"])
        np.testing.assert_array_equal(audit.nuisance_transform, reference["transform"])
        assert audit.nuisance_tolerance == reference["tolerance"]
        np.testing.assert_allclose(audit.nuisance_basis.T @ audit.nuisance_basis,
                                   np.eye(2), rtol=0, atol=1e-12)
        np.testing.assert_allclose(reference["q"].T @ reference["ytr"], 0, atol=1e-12)


@pytest.mark.parametrize("model_name", ["ridge", "partial_correlation"])
@pytest.mark.parametrize("heldout_shift", [0.0, 20.0])
def test_near_duplicate_confounds_heldout_transforms_and_screening_df(model_name, heldout_shift):
    x, y, c, folds = _near_duplicate_data()
    before = ConnectomePhysiologyCV(model=model_name).fit(x, y, c, fold_ids=folds)
    # The training design remains nearly singular even when test columns separate.
    c[folds == 0, 1] += heldout_shift
    fitted = ConnectomePhysiologyCV(model=model_name).fit(x, y, c, fold_ids=folds)
    audit = fitted.fold_audits_[0]
    train, test = audit.train_indices, audit.test_indices
    reference = _reference_fold(x, y, c, train, test, model_name)
    assert audit.nuisance_rank == 2
    for name in ("nuisance_basis", "nuisance_transform", "feature_nuisance_coef",
                 "target_nuisance_coef", "feature_mean", "feature_scale", "weights",
                 "selected_features", "nuisance_tolerance"):
        np.testing.assert_array_equal(getattr(before.fold_audits_[0], name), getattr(audit, name))
    np.testing.assert_allclose(audit.feature_mean, reference["feature_mean"], atol=1e-13)
    np.testing.assert_allclose(audit.feature_scale, reference["feature_scale"], atol=1e-13)
    np.testing.assert_array_equal(audit.selected_features, reference["selected"])
    np.testing.assert_allclose(audit.weights, reference["weights"], atol=1e-12)
    np.testing.assert_allclose(fitted.result_.baseline_predictions[test], reference["baseline"], atol=1e-12)
    np.testing.assert_allclose(fitted.result_.predictions[test], reference["predictions"], atol=1e-12)


@pytest.mark.parametrize("model_name", ["ridge", "partial_correlation"])
def test_near_duplicate_confounds_permutation_refits_same_retained_space(model_name):
    x, y, c, folds = _near_duplicate_data()
    fitted = ConnectomePhysiologyCV(model=model_name).fit(x, y, c, fold_ids=folds)
    count, seed = 11, 97
    rng = np.random.default_rng(seed)
    indices = np.tile(np.arange(len(y)), (count, 1))
    for index in indices:
        for fold in np.unique(fitted.fold_ids_):
            cell = np.flatnonzero(fitted.fold_ids_ == fold)
            index[cell] = rng.permutation(cell)
    expected = np.zeros(count)
    for audit in fitted.fold_audits_:
        train, test = audit.train_indices, audit.test_indices
        reference = _reference_fold(x, y, c, train, test, model_name)
        q, heldout_basis = reference["q"], reference["heldout_basis"]
        reduced = np.empty(len(y))
        reduced[train] = q @ (q.T @ y[train])
        reduced[test] = heldout_basis @ (q.T @ y[train])
        residuals = y - reduced
        for b, index in enumerate(indices):
            pseudo_train = reduced[train] + residuals[index[train]]
            pseudo_test = reduced[test] + residuals[index[test]]
            weights, _, _ = reference["weights_for"](pseudo_train)
            baseline = heldout_basis @ (q.T @ pseudo_train)
            predictions = baseline + reference["xte"] @ weights
            expected[b] += np.sum((pseudo_test - baseline) ** 2
                                  - (pseudo_test - predictions) ** 2) / len(y)
    actual = fitted.permutation_test(count, random_state=seed)
    np.testing.assert_allclose(actual.null_scores, expected, rtol=0, atol=1e-12)
    repeated = fitted.permutation_test(count, random_state=seed)
    np.testing.assert_array_equal(actual.null_scores, repeated.null_scores)


def test_near_duplicate_partial_correlation_uses_retained_rank_df(monkeypatch):
    x, y, c, folds = _near_duplicate_data()
    used_df = []
    original_sf = student_t.sf

    def recording_sf(value, df):
        used_df.append(df)
        return original_sf(value, df)

    monkeypatch.setattr(student_t, "sf", recording_sf)
    fitted = ConnectomePhysiologyCV(model="partial_correlation").fit(x, y, c, fold_ids=folds)
    expected = [len(a.train_indices) - a.nuisance_basis.shape[1] - 1 for a in fitted.fold_audits_]
    assert expected == [45] * 4
    assert used_df == expected
    used_df.clear()
    fitted.permutation_test(3, random_state=97)
    assert used_df == [df for df in expected for _ in range(3)]


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

