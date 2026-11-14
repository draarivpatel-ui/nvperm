import itertools
import numpy as np
import pytest
from scipy.stats import binom

from nvperm import FreedmanLane, freedman_lane, bh_fdr
from nvperm._utils import nuisance_basis, residualize, permutation_indices, label_groups


def test_effect_and_t_match_full_ols(regression_data):
    x, y, c = regression_data
    result = freedman_lane(y, x, c, n_permutations=19)
    design = np.column_stack((np.ones(len(x)), c, x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    errors = y - design @ beta
    variance = np.sum(errors ** 2, axis=0) / (len(x) - np.linalg.matrix_rank(design))
    se = np.sqrt(variance * np.linalg.inv(design.T @ design)[-1, -1])
    np.testing.assert_allclose(result.effect, beta[-1], atol=1e-12)
    np.testing.assert_allclose(result.standard_error, se, atol=1e-12)
    np.testing.assert_allclose(result.statistic, beta[-1] / se, atol=1e-12)


def test_nuisance_orthogonality(regression_data):
    x, y, c = regression_data
    q, rank = nuisance_basis(c, len(x))
    assert rank == 4
    np.testing.assert_allclose(q.T @ residualize(y, q), 0, atol=1e-12)
    np.testing.assert_allclose(q.T @ residualize(x, q), 0, atol=1e-12)
    np.testing.assert_allclose(q.T @ q, np.eye(rank), atol=1e-12)


def test_reprojection_after_permutation_is_required(regression_data):
    x, y, c = regression_data
    q, _ = nuisance_basis(c, len(x))
    shuffled = residualize(y, q)[np.random.default_rng(17).permutation(len(x))]
    assert np.linalg.norm(q.T @ shuffled) > 0.1
    np.testing.assert_allclose(q.T @ residualize(shuffled, q), 0, atol=1e-12)


def test_exact_enumeration_matches_direct_full_model():
    x = np.array([-2.0, -0.5, 0.3, 0.8, 2.1])
    c = np.array([0.0, 1, 0, 1, 0])[:, None]
    y = np.array([0.1, 1.3, -0.7, 0.4, 1.8])
    result = freedman_lane(y, x, c, mode="exact", batch_size=17)
    z = np.column_stack((np.ones(5), c))
    reduced = z @ np.linalg.lstsq(z, y, rcond=None)[0]
    residual = y - reduced
    design = np.column_stack((z, x))
    inv = np.linalg.inv(design.T @ design)
    direct = []
    for permutation in itertools.permutations(range(5)):
        pseudo = reduced + residual[list(permutation)]
        beta = np.linalg.lstsq(design, pseudo, rcond=None)[0]
        errors = pseudo - design @ beta
        se = np.sqrt(errors @ errors / 2 * inv[-1, -1])
        direct.append(beta[-1] / se)
    np.testing.assert_allclose(result.null_statistics[:, 0], direct, atol=1e-11)
    expected = np.mean(np.abs(direct) >= abs(result.statistic[0]) - 1e-10)
    assert result.pvalues[0] == pytest.approx(expected)
    assert result.n_permutations == 120 and result.exact


def test_exact_blocks_and_identity():
    x = np.arange(6, dtype=float)
    y = np.array([1.0, -2, 2, 0, -1, 3])
    result = freedman_lane(y, x, blocks=[0, 0, 1, 1, 2, 2], mode="exact")
    assert result.n_permutations == 8
    singleton = freedman_lane(y, x, blocks=np.arange(6), mode="exact")
    assert singleton.n_permutations == 1
    assert singleton.pvalues[0] == singleton.pvalues_max_t[0] == 1


def test_exact_limit_is_explicit(regression_data):
    x, y, c = regression_data
    with pytest.raises(ValueError, match="max_exact"):
        freedman_lane(y, x, c, mode="exact")


def test_seed_and_batch_size_invariance(regression_data):
    x, y, c = regression_data
    first = freedman_lane(y, x, c, n_permutations=57, random_state=79, batch_size=11)
    second = freedman_lane(y, x, c, n_permutations=57, random_state=79, batch_size=57)
    np.testing.assert_allclose(first.null_statistics, second.null_statistics, atol=1e-12)
    np.testing.assert_array_equal(first.pvalues, second.pvalues)


def test_supplied_generator_is_deterministic_and_consumed(regression_data):
    x, y, c = regression_data
    rng = np.random.default_rng(41)
    one = freedman_lane(y, x, c, n_permutations=19, random_state=rng)
    fresh = freedman_lane(y, x, c, n_permutations=19, random_state=np.random.default_rng(41))
    next_draw = freedman_lane(y, x, c, n_permutations=19, random_state=rng)
    np.testing.assert_array_equal(one.null_statistics, fresh.null_statistics)
    assert not np.array_equal(one.null_statistics, next_draw.null_statistics)


def test_monte_carlo_plus_one_bound(regression_data):
    x, y, c = regression_data
    result = freedman_lane(y, x, c, n_permutations=39)
    assert np.all(result.pvalues >= 1 / 40)
    assert np.all(result.pvalues <= 1)


def test_max_t_dominates_marginal_and_grows_with_family(regression_data):
    x, y, c = regression_data
    one = freedman_lane(y[:, :2], x, c, n_permutations=79)
    all_targets = freedman_lane(y, x, c, n_permutations=79)
    assert np.all(all_targets.pvalues_max_t >= all_targets.pvalues)
    assert np.all(all_targets.pvalues_max_t[:2] >= one.pvalues_max_t)


def test_identical_targets_share_null_draws(regression_data):
    x, y, c = regression_data
    result = freedman_lane(np.repeat(y[:, :1], 3, axis=1), x, c, n_permutations=39)
    np.testing.assert_allclose(result.null_statistics[:, 0], result.null_statistics[:, 2])
    np.testing.assert_array_equal(result.pvalues, result.pvalues_max_t)


def test_bh_known_values_shape_and_ties():
    p = np.array([[0.01, 0.04], [0.03, 0.2]])
    q = bh_fdr(p)
    np.testing.assert_allclose(q, [[0.04, 0.053333333333], [0.053333333333, 0.2]])
    np.testing.assert_allclose(bh_fdr([0.1, 0.1, 0.1]), [0.1, 0.1, 0.1], atol=1e-15)
    np.testing.assert_array_equal(bh_fdr([0, 1]), [0, 1])


def test_bh_monotonicity():
    p = np.random.default_rng(61).uniform(size=30)
    q = bh_fdr(p)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)
    assert np.all(q >= p - 1e-12)
    larger = np.minimum(p + 0.02, 1)
    assert np.all(bh_fdr(larger) >= q - 1e-12)


@pytest.mark.parametrize("bad", [[], [np.nan], [-0.01], [1.1], 0.5])
def test_bh_invalid_input(bad):
    with pytest.raises(ValueError):
        bh_fdr(bad)


def test_collinear_nuisance_preserves_inference(regression_data):
    x, y, c = regression_data
    full = np.column_stack((c, 2 * c[:, 0], np.ones(len(x))))
    one = freedman_lane(y, x, c, n_permutations=39)
    two = freedman_lane(y, x, full, n_permutations=39)
    assert one.nuisance_rank == two.nuisance_rank
    np.testing.assert_allclose(one.statistic, two.statistic, atol=1e-11)
    np.testing.assert_array_equal(one.pvalues, two.pvalues)


def test_added_nuisance_effect_does_not_change_result(regression_data):
    x, y, c = regression_data
    one = freedman_lane(y, x, c, n_permutations=39)
    two = freedman_lane(y + c @ np.full((3, 5), 30) + 700, x, c, n_permutations=39)
    np.testing.assert_allclose(one.null_statistics, two.null_statistics, atol=1e-10)
    np.testing.assert_array_equal(one.pvalues, two.pvalues)


@pytest.mark.parametrize("collinear", [True, False])
def test_nonidentifiable_exposure_rejected(regression_data, collinear):
    _, y, c = regression_data
    x = c[:, 0] if collinear else np.ones(len(y))
    with pytest.raises(ValueError, match="collinear"):
        freedman_lane(y, x, c, n_permutations=9)


def test_zero_outcome_has_no_evidence(regression_data):
    x, _, c = regression_data
    result = freedman_lane(np.zeros(len(x)), x, c, n_permutations=19)
    assert result.statistic[0] == 0 and result.pvalues[0] == 1


def test_hc3_standard_error_matches_sandwich(regression_data):
    x, y, c = regression_data
    result = freedman_lane(y, x, c, n_permutations=19, statistic="hc3")
    design = np.column_stack((np.ones(len(x)), c, x))
    pinv = np.linalg.pinv(design)
    beta = pinv @ y
    errors = y - design @ beta
    h = np.sum(design * pinv.T, axis=1)
    variance = np.sum((pinv[-1, :, None] * errors / (1 - h[:, None])) ** 2, axis=0)
    np.testing.assert_allclose(result.standard_error, np.sqrt(variance), atol=1e-12)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_alternatives_and_partial_correlation(regression_data, alternative):
    x, y, c = regression_data
    result = freedman_lane(y, x, c, n_permutations=39,
                           statistic="partial_correlation", alternative=alternative)
    assert np.all(np.abs(result.statistic) <= 1)
    assert np.all(result.pvalues_max_t >= result.pvalues)


def test_permutations_preserve_blocks():
    labels = np.array([0, 1, 0, 1, 2, 2, 2])
    indices = permutation_indices(np.random.default_rng(7), label_groups(labels, 7), 25, 7)
    for row in indices:
        np.testing.assert_array_equal(labels[row], labels)
        np.testing.assert_array_equal(np.sort(row), np.arange(7))


def test_type_i_calibration_with_heavy_confounding():
    # A predeclared upper 99% binomial bound at alpha=.05, not a guarantee of size.
    rejected = 0
    repetitions = 80
    for seed in range(400, 400 + repetitions):
        rng = np.random.default_rng(seed)
        c = rng.normal(size=(72, 2))
        x = 12 * c[:, 0] + 6 * c[:, 1] + rng.normal(size=72)
        y = 14 * c[:, 0] - 4 * c[:, 1] + rng.normal(size=72)
        rejected += freedman_lane(y, x, c, n_permutations=199, random_state=seed).pvalues[0] <= 0.05
    assert rejected <= binom.ppf(0.99, repetitions, 0.05)


def test_pure_null_familywise_calibration():
    rejected = 0
    repetitions = 48
    for seed in range(900, 900 + repetitions):
        rng = np.random.default_rng(seed)
        x, y = rng.normal(size=56), rng.normal(size=(56, 5))
        result = freedman_lane(y, x, n_permutations=99, random_state=seed)
        rejected += np.any(result.pvalues_max_t <= 0.05)
    assert rejected <= binom.ppf(0.99, repetitions, 0.05)


def test_hc3_variance_blocks_calibration():
    rejected = 0
    for seed in range(1200, 1240):
        rng = np.random.default_rng(seed)
        blocks = np.repeat([0, 1], 40)
        x = rng.normal(size=80) + blocks * 2
        y = rng.normal(size=80) * np.where(blocks == 0, 1, 4) + blocks * 10
        rejected += freedman_lane(y, x, blocks[:, None], blocks=blocks, statistic="hc3",
                                   n_permutations=99, random_state=seed).pvalues[0] <= 0.05
    assert rejected <= binom.ppf(0.99, 40, 0.05)


@pytest.mark.parametrize("kwargs", [
    {"n_permutations": 0}, {"batch_size": 0}, {"mode": "bad"},
    {"statistic": "bad"}, {"alternative": "bad"}, {"random_state": -1},
])
def test_invalid_parameters(regression_data, kwargs):
    x, y, c = regression_data
    with pytest.raises(ValueError):
        freedman_lane(y, x, c, **kwargs)


@pytest.mark.parametrize("field", ["x", "y", "covariates"])
def test_nonfinite_data_rejected(regression_data, field):
    x, y, c = [a.copy() for a in regression_data]
    {"x": x, "y": y, "covariates": c}[field].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        freedman_lane(y, x, c, n_permutations=9)


def test_result_serialization_and_estimator_state(regression_data):
    x, y, c = regression_data
    model = FreedmanLane(n_permutations=9)
    result = model.fit(y, x, c)
    assert model.result_ is result
    assert len(result.to_dict(include_null=True)["null_statistics"]) == 9


@pytest.mark.parametrize("scale_x,scale_y", [(1e-10, 1e-10), (1e6, 1e-6)])
def test_physical_unit_rescaling_preserves_inference(regression_data, scale_x, scale_y):
    x, y, c = regression_data
    reference = freedman_lane(y, x, c, n_permutations=39)
    scaled = freedman_lane(y * scale_y, x * scale_x, c, n_permutations=39)
    np.testing.assert_allclose(reference.statistic, scaled.statistic, atol=1e-11)
    np.testing.assert_allclose(reference.null_statistics, scaled.null_statistics, atol=1e-11)
    np.testing.assert_allclose(reference.effect * scale_y / scale_x, scaled.effect, atol=1e-11)
    np.testing.assert_array_equal(reference.pvalues, scaled.pvalues)


def test_large_exposure_offset_does_not_make_it_unidentified(regression_data):
    x, y, c = regression_data
    reference = freedman_lane(y, x, c, n_permutations=19)
    shifted = freedman_lane(y, x + 1e7, c, n_permutations=19)
    np.testing.assert_allclose(reference.statistic, shifted.statistic, rtol=1e-8)

