import numpy as np
import pytest

from nvperm import (
    generate_cohort, network_features, neurovascular_features,
    pulse_pressure, rmssd, motion_sensitivity,
)


def two_networks():
    labels = np.array(["A", "A", "B", "B"])
    a = np.full((3, 4, 4), 0.2)
    a[:, 0, 1] = a[:, 1, 0] = 0.6
    a[:, 2, 3] = a[:, 3, 2] = 0.6
    for row in a:
        np.fill_diagonal(row, 1)
    return a, labels


def test_network_means_and_segregation():
    a, labels = two_networks()
    result = network_features(a, labels)
    lookup = dict(zip(result.names, result.values.T))
    np.testing.assert_allclose(lookup["within_A"], 0.6)
    np.testing.assert_allclose(lookup["between_A"], 0.2)
    np.testing.assert_allclose(lookup["segregation_A"], 2 / 3)
    np.testing.assert_allclose(lookup["integration_A_B"], 0.2)
    assert np.all(result.segregation_defined)


def test_matrix_and_upper_triangle_equivalence():
    a, labels = two_networks()
    i, j = np.triu_indices(4, 1)
    np.testing.assert_allclose(network_features(a, labels).values,
                               network_features(a[:, i, j], labels).values)


def test_diagonal_never_changes_features():
    a, labels = two_networks()
    before = network_features(a, labels).values
    for row in a:
        np.fill_diagonal(row, 100)
    np.testing.assert_allclose(before, network_features(a, labels).values)


def test_negative_edge_policy_and_zero_validity():
    a, labels = two_networks()
    a[:, 0, 1] = a[:, 1, 0] = -0.2
    zeroed = network_features(a, labels)
    assert not np.any(zeroed.segregation_defined[:, 0])
    assert np.all(zeroed.values[:, 2] == 0)
    kept = network_features(a, labels, negative="keep")
    np.testing.assert_allclose(kept.values[:, 0], -0.2)


def test_fisher_transform_and_name_uniqueness():
    a, labels = two_networks()
    result = network_features(a, labels, fisher_z=True)
    np.testing.assert_allclose(result.values[:, 0], np.arctanh(0.6))
    assert len(set(result.names)) == len(result.names)


@pytest.mark.parametrize("bad", ["asymmetric", "missing_roi", "nan", "out_of_range"])
def test_invalid_connectivity_rejected(bad):
    a, labels = two_networks()
    if bad == "asymmetric":
        a[0, 0, 1] = 0.7
    elif bad == "missing_roi":
        labels = labels[:3]
    elif bad == "nan":
        a[0, 0, 1] = a[0, 1, 0] = np.nan
    else:
        a[:, 0, 1] = a[:, 1, 0] = 2
    with pytest.raises(ValueError):
        network_features(a, labels)


def test_singleton_network_rejected():
    a, _ = two_networks()
    with pytest.raises(ValueError, match="two ROIs"):
        network_features(a, ["A", "A", "A", "B"])


def test_pulse_pressure_values_and_invalid_order():
    np.testing.assert_array_equal(pulse_pressure([120, 130], [80, 85]), [40, 45])
    with pytest.raises(ValueError):
        pulse_pressure([70, 120], [80, 85])


def test_rmssd_units_and_batching():
    rr = np.array([1.0, 1.1, 1.0])
    assert rmssd(rr) == pytest.approx(100)
    assert rmssd(rr * 1000, unit="milliseconds") == pytest.approx(100)
    np.testing.assert_allclose(rmssd(np.vstack((rr, rr))), [100, 100])


@pytest.mark.parametrize("rr", [[1, 1], [1, -1, 1], [1, np.nan, 1]])
def test_bad_rr_intervals_rejected(rr):
    with pytest.raises(ValueError):
        rmssd(rr)


def test_neurovascular_products_require_no_fitted_scaling():
    a, labels = two_networks()
    rr = np.tile([1.0, 1.1, 1.0], (3, 1))
    result = neurovascular_features(a, labels, [120, 125, 130], [80, 80, 80], rr_intervals=rr)
    pp_index = result.names.index("pulse_pressure_mmhg")
    product_index = result.names.index("segregation_A_x_pulse_pressure")
    np.testing.assert_allclose(result.values[:, product_index], result.values[:, pp_index] * 2 / 3)
    assert result.values.shape[1] == len(result.names)


def test_motion_adjustment_attenuates_shared_confound():
    rng = np.random.default_rng(573)
    fd = rng.uniform(0.05, 0.5, 120)
    x = 30 * fd + rng.normal(size=120)
    y = 40 * fd + rng.normal(size=120)
    result = motion_sensitivity(x, y, fd, n_permutations=19)
    assert result["relative_attenuation"][0] > 0.75
    assert abs(result["correlation_after_fd"][0]) < 0.25
    assert result["causal_interpretation"] is False
    assert len(result["pvalues_after_fd"]) == 1


def test_synthetic_determinism_and_psd_connectomes():
    one = generate_cohort(32, 12, random_state=75)
    two = generate_cohort(32, 12, random_state=75)
    np.testing.assert_array_equal(one.features, two.features)
    np.testing.assert_array_equal(one.connectomes, two.connectomes)
    assert np.min(np.linalg.eigvalsh(one.connectomes)) > -1e-10
    np.testing.assert_allclose(np.diagonal(one.connectomes, axis1=1, axis2=2), 1)
    assert one.features.shape == (32, 66) and one.confounds.shape == (32, 5)


def test_synthetic_ground_truth_and_controlled_effect():
    null = generate_cohort(40, 12, effect_size=0, random_state=76)
    signal = generate_cohort(40, 12, effect_size=1, random_state=76)
    assert np.all(null.ground_truth == 0)
    assert np.count_nonzero(signal.ground_truth) == 6
    np.testing.assert_array_equal(null.target, signal.target)
    unchanged = signal.ground_truth == 0
    np.testing.assert_array_equal(null.features[:, unchanged], signal.features[:, unchanged])


def test_synthetic_heteroskedasticity_and_npz_roundtrip(tmp_path):
    data = generate_cohort(40, 12, heteroskedasticity=1.5, random_state=12)
    path = tmp_path / "cohort.npz"
    data.save(path)
    with np.load(path, allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded["features"], data.features)
        np.testing.assert_array_equal(loaded["ground_truth"], data.ground_truth)
    assert data.heteroskedasticity == 1.5


@pytest.mark.parametrize("kwargs", [{"n_subjects": 5}, {"n_rois": 5, "n_networks": 3},
                                   {"heteroskedasticity": -1}, {"effect_size": np.nan}])
def test_invalid_synthetic_parameters(kwargs):
    with pytest.raises(ValueError):
        generate_cohort(**kwargs)

