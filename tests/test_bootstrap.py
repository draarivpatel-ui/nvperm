import dataclasses
import warnings

import numpy as np
import pytest

from nvperm.bootstrap import (
    BootstrapResult, bootstrap_effects, wild_bootstrap_effects,
)
from nvperm.freedman_lane import freedman_lane
from nvperm._utils import label_groups


def _strata_draw_indices(seed, labels, count):
    """Replay the resample index stream used by bootstrap_effects."""
    labels = np.asarray(labels)
    n = len(labels)
    groups = label_groups(labels, n)
    rng = np.random.default_rng(seed)
    draws = np.empty((count, n), dtype=int)
    for row in draws:
        for group in groups:
            row[group] = rng.choice(group, size=len(group), replace=True)
    return draws


def test_pairs_resamples_match_explicit_lstsq():
    n = 9
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    data = np.random.default_rng(310)
    c = data.normal(size=(n, 2))
    x = labels * 0.6 + data.normal(size=n)
    y = 0.9 * x[:, None] + c @ np.array([[1.0], [-0.4]]) \
        + data.normal(size=(n, 2))
    count = 60
    result = bootstrap_effects(y, x, c, blocks=labels, n_resamples=count,
                               random_state=905)
    assert result.n_valid == count
    for indices, expected in zip(
            _strata_draw_indices(905, labels, count), result.bootstrap_effects):
        design = np.column_stack((np.ones(n), c[indices], x[indices]))
        beta = np.linalg.lstsq(design, y[indices], rcond=None)[0][-1]
        np.testing.assert_allclose(expected, beta, atol=1e-10)


def test_seed_reproducibility_and_generator_consumption(regression_data):
    x, y, c = regression_data
    for fn in (bootstrap_effects, wild_bootstrap_effects):
        one = fn(y, x, c, n_resamples=29, random_state=11)
        two = fn(y, x, c, n_resamples=29, random_state=11)
        np.testing.assert_array_equal(
            one.bootstrap_effects, two.bootstrap_effects)
        fresh = fn(y, x, c, n_resamples=29,
                   random_state=np.random.default_rng(11))
        np.testing.assert_array_equal(
            one.bootstrap_effects, fresh.bootstrap_effects)
        rng = np.random.default_rng(11)
        first = fn(y, x, c, n_resamples=29, random_state=rng)
        second = fn(y, x, c, n_resamples=29, random_state=rng)
        np.testing.assert_array_equal(
            one.bootstrap_effects, first.bootstrap_effects)
        assert not np.array_equal(
            one.bootstrap_effects, second.bootstrap_effects)


def test_shared_draws_preserve_linear_response(regression_data):
    x, y, c = regression_data
    repeated = np.repeat(y[:, :1], 3, axis=1)
    for fn in (bootstrap_effects, wild_bootstrap_effects):
        result = fn(repeated, x, c, n_resamples=49, random_state=5)
        np.testing.assert_array_equal(result.bootstrap_effects[:, 0],
                                      result.bootstrap_effects[:, 2])
        np.testing.assert_array_equal(result.effect,
                                      np.repeat(result.effect[0], 3))
        base = fn(y[:, :2], x, c, n_resamples=49, random_state=5)
        transformed = fn(np.column_stack((y[:, 0], -2.5 * y[:, 1] + 4.0)),
                         x, c, n_resamples=49, random_state=5)
        np.testing.assert_allclose(transformed.bootstrap_effects[:, 0],
                                   base.bootstrap_effects[:, 0], atol=1e-12)
        np.testing.assert_allclose(transformed.bootstrap_effects[:, 1],
                                   -2.5 * base.bootstrap_effects[:, 1],
                                   atol=1e-10)
        np.testing.assert_allclose(transformed.effect[1],
                                   -2.5 * base.effect[1], atol=1e-10)


def test_wild_matches_direct_full_design_formula():
    data = np.random.default_rng(204)
    n = 48
    c = data.normal(size=(n, 2))
    x = 0.5 * c[:, 0] + data.normal(size=n)
    spread = 0.5 + np.abs(x)
    y = (0.8 * x + c @ np.array([1.0, -0.6]))[:, None] \
        + data.normal(size=(n, 2)) * spread[:, None]
    count, level = 75, 0.9
    result = wild_bootstrap_effects(y, x, c, n_resamples=count,
                                    confidence_level=level, random_state=77)
    design = np.column_stack((np.ones(n), c, x))
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    beta = coef[-1]
    np.testing.assert_allclose(result.effect, beta, atol=1e-12)
    pinv = np.linalg.pinv(design)
    leverage = np.sum(design * pinv.T, axis=1)
    errors = y - design @ coef
    weights = pinv[-1][:, None] * errors / np.sqrt(1 - leverage[:, None])
    signs = 2 * np.random.default_rng(77).integers(
        0, 2, size=(count, n), dtype=np.int8) - 1
    delta = np.einsum("bn,nt->bt", signs, weights)
    np.testing.assert_allclose(result.bootstrap_effects,
                               beta[None, :] + delta, atol=1e-12)
    np.testing.assert_allclose(
        result.ci_low, beta - np.quantile(delta, (1 + level) / 2, axis=0),
        atol=1e-12)
    np.testing.assert_allclose(
        result.ci_high, beta - np.quantile(delta, (1 - level) / 2, axis=0),
        atol=1e-12)


def test_effect_matches_freedman_lane_with_degenerate_inputs(regression_data):
    x, y, c = regression_data
    covariates = np.column_stack((c, 2 * c[:, 0], np.ones(len(x))))
    targets = np.column_stack((y[:, 0], np.full(len(x), 3.5), y[:, 1]))
    reference = freedman_lane(targets, x, covariates, n_permutations=9)
    for fn in (bootstrap_effects, wild_bootstrap_effects):
        result = fn(targets, x, covariates, n_resamples=9)
        np.testing.assert_allclose(result.effect, reference.effect,
                                   atol=1e-12)
        assert result.nuisance_rank == reference.nuisance_rank
        assert result.residual_df == reference.residual_df


def test_vector_outcome_gives_single_target(regression_data):
    x, y, c = regression_data
    for fn in (bootstrap_effects, wild_bootstrap_effects):
        result = fn(y[:, 0], x, c, n_resamples=9)
        assert result.effect.shape == (1,)
        assert result.bootstrap_effects.shape == (result.n_valid, 1)


def test_result_metadata(regression_data):
    x, y, c = regression_data
    labels = np.repeat([0, 1], 36)
    pairs = bootstrap_effects(y, x, c[:, :1], blocks=labels, n_resamples=25,
                              confidence_level=0.8, random_state=3)
    assert pairs.method == "pairs" and pairs.block_sizes == (36, 36)
    assert pairs.n_resamples == 25 and pairs.n_valid <= 25
    assert pairs.confidence_level == 0.8 and pairs.residual_df == 72 - 2 - 1
    assert pairs.bootstrap_effects.shape == (pairs.n_valid, 5)
    wild = wild_bootstrap_effects(y, x, c, n_resamples=25, random_state=3)
    assert wild.method == "wild_rademacher" and wild.block_sizes is None
    assert wild.n_valid == 25
    assert isinstance(pairs, BootstrapResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pairs.effect = np.zeros(5)
    assert pairs.to_dict()["method"] == "pairs"
    assert len(pairs.to_dict(include_draws=True)["bootstrap_effects"]) \
        == pairs.n_valid


def test_length_mismatch_rejected(regression_data):
    x, y, c = regression_data
    for fn in (bootstrap_effects, wild_bootstrap_effects):
        with pytest.raises(ValueError):
            fn(y[:-1], x, c, n_resamples=9)
        with pytest.raises(ValueError):
            fn(y, x[:-1], c, n_resamples=9)
        with pytest.raises(ValueError):
            fn(y, x, c[:-1], n_resamples=9)
    with pytest.raises(ValueError):
        bootstrap_effects(y, x, c, blocks=np.arange(len(x) - 1),
                          n_resamples=9)


@pytest.mark.parametrize("blocks", [
    np.concatenate((np.zeros(71), [np.nan])),
    np.concatenate((np.zeros(71), [None])).astype(object),
    np.full(72, "  "),
    np.zeros((72, 2)),
])
def test_invalid_block_labels_rejected(regression_data, blocks):
    x, y, c = regression_data
    with pytest.raises(ValueError, match="block labels"):
        bootstrap_effects(y, x, c, blocks=blocks, n_resamples=9)


def test_all_singleton_strata_rejected(regression_data):
    x, y, c = regression_data
    with pytest.raises(ValueError, match="singleton"):
        bootstrap_effects(y, x, c, blocks=np.arange(len(x)), n_resamples=9)


@pytest.mark.parametrize("level", [
    0.0, 1.0, -0.1, 1.5, np.nan, np.inf, True, "0.95", np.array(0.95), None,
])
def test_invalid_confidence_level_rejected(regression_data, level):
    x, y, c = regression_data
    for fn in (bootstrap_effects, wild_bootstrap_effects):
        with pytest.raises(ValueError, match="confidence_level"):
            fn(y, x, c, n_resamples=9, confidence_level=level)


@pytest.mark.parametrize("count", [0, 1, -3, 2.5, "99", True, None])
def test_invalid_resample_count_rejected(regression_data, count):
    x, y, c = regression_data
    for fn in (bootstrap_effects, wild_bootstrap_effects):
        with pytest.raises(ValueError, match="n_resamples"):
            fn(y, x, c, n_resamples=count)


@pytest.mark.parametrize("state", [-1, 1.5, "abc", np.array(3)])
def test_invalid_random_state_rejected(regression_data, state):
    x, y, c = regression_data
    for fn in (bootstrap_effects, wild_bootstrap_effects):
        with pytest.raises(ValueError, match="random_state"):
            fn(y, x, c, n_resamples=9, random_state=state)


def test_constant_or_collinear_exposure_rejected(regression_data):
    _, y, c = regression_data
    for fn in (bootstrap_effects, wild_bootstrap_effects):
        with pytest.raises(ValueError, match="collinear"):
            fn(y, np.ones(len(y)), c, n_resamples=9)
        with pytest.raises(ValueError, match="collinear"):
            fn(y, c[:, 0] + 0.01, c, n_resamples=9)


def test_insufficient_residual_df_rejected():
    rng = np.random.default_rng(51)
    y = rng.normal(size=(5, 2))
    x = rng.normal(size=5)
    c = rng.normal(size=(5, 3))
    for fn in (bootstrap_effects, wild_bootstrap_effects):
        with pytest.raises(ValueError, match="degrees of freedom"):
            fn(y, x, c, n_resamples=9)


def test_wild_rejects_full_leverage_point():
    rng = np.random.default_rng(64)
    n = 14
    x = rng.normal(size=n)
    x[-1] = 1e8
    y = rng.normal(size=n)
    with pytest.raises(ValueError, match="leverage"):
        wild_bootstrap_effects(y, x, n_resamples=9)


def test_pairs_warns_and_counts_degenerate_draws():
    rng = np.random.default_rng(404)
    n = 8
    labels = np.repeat([0, 1], 4)
    # A single nonzero in stratum 0: roughly a third of resamples drop it
    # and the resampled nuisance design loses a rank.
    c = np.zeros((n, 1))
    c[0, 0] = 1.0
    x = labels + rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        result = bootstrap_effects(y, x, c, blocks=labels, n_resamples=60,
                                   random_state=99)
    runtime = [w for w in seen if issubclass(w.category, RuntimeWarning)]
    assert len(runtime) == 1
    assert "estimable" in str(runtime[0].message)
    assert 2 <= result.n_valid < 60
    assert result.bootstrap_effects.shape == (result.n_valid, 1)


def test_pairs_isolated_singleton_stratum_is_held_fixed():
    rng = np.random.default_rng(77)
    n = 9
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3])
    x = labels * 0.5 + rng.normal(size=n)
    y = 0.7 * x[:, None] + rng.normal(size=(n, 2))
    indices = _strata_draw_indices(63, labels, 30)
    np.testing.assert_array_equal(indices[:, 8], np.full(30, 8))
    result = bootstrap_effects(y, x, blocks=labels, n_resamples=30,
                               random_state=63)
    assert result.block_sizes == (3, 3, 2, 1)


def test_wild_interval_coverage_guard_heteroskedastic():
    # Broad finite-simulation guard for the pointwise 90% wild interval,
    # not a general calibration claim.
    covers = 0
    repetitions = 72
    for seed in range(5100, 5100 + repetitions):
        rng = np.random.default_rng(seed)
        c = rng.normal(size=96)
        x = 0.4 * c + rng.normal(size=96)
        y = (1.2 * c + 0.8 * x
             + (0.45 + 0.65 * np.abs(x)) * rng.normal(size=96))
        result = wild_bootstrap_effects(y, x, c[:, None], n_resamples=199,
                                        confidence_level=0.9,
                                        random_state=seed)
        covers += result.ci_low[0] <= 0.8 <= result.ci_high[0]
    assert 54 <= covers <= repetitions
