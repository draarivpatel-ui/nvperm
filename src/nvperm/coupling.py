"""Network summaries and physiology features without fitted cohort scaling."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ._utils import matrix, nuisance_basis, residualize, vector
from .freedman_lane import freedman_lane


@dataclass(frozen=True)
class FeatureSet:
    values: np.ndarray
    names: tuple[str, ...]
    segregation_defined: np.ndarray | None = None

    def to_dict(self):
        result = {"values": self.values.tolist(), "names": list(self.names)}
        if self.segregation_defined is not None:
            result["segregation_defined"] = self.segregation_defined.tolist()
        return result


def connectivity_matrices(connectivity, labels):
    """Accept symmetric matrices or NumPy-ordered upper triangles."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) < 4:
        raise ValueError("labels must identify at least four ROIs")
    if any(v is None or (isinstance(v, float) and not np.isfinite(v)) for v in labels.tolist()):
        raise ValueError("network labels must not be missing")
    a = np.asarray(connectivity, float)
    r = len(labels)
    if a.ndim == 2:
        if a.shape[1] != r * (r - 1) // 2:
            raise ValueError("upper triangles do not match ROI labels")
        matrices = np.zeros((len(a), r, r))
        upper = np.triu_indices(r, 1)
        matrices[:, upper[0], upper[1]] = a
        matrices[:, upper[1], upper[0]] = a
    elif a.ndim == 3 and a.shape[1:] == (r, r):
        matrices = a.copy()
        if not np.allclose(matrices, matrices.transpose(0, 2, 1), atol=1e-8, rtol=1e-8):
            raise ValueError("connectivity matrices must be symmetric")
    else:
        raise ValueError("connectivity must have shape (subjects, ROIs, ROIs) or (subjects, edges)")
    if len(matrices) < 2 or not np.all(np.isfinite(matrices)):
        raise ValueError("connectivity requires at least two finite subjects")
    upper_values = matrices[:, np.triu_indices(r, 1)[0], np.triu_indices(r, 1)[1]]
    if np.any(np.abs(upper_values) > 1 + 1e-8):
        raise ValueError("connectivity entries must be correlations in [-1, 1]")
    return matrices, labels.astype(str)


def network_features(connectivity, labels, *, negative="zero", fisher_z=False):
    """Within/between means, segregation (W-B)/W, and pairwise integration.

    Negative correlations are zeroed by default. ``negative='keep'`` preserves
    signed correlations. Zero within-network means produce zero segregation
    plus a false validity flag; users should omit those features in inference.
    A network needs at least two ROIs. Diagonals never enter a summary.
    """
    a, labels = connectivity_matrices(connectivity, labels)
    if negative not in ("zero", "keep"):
        raise ValueError("negative must be zero or keep")
    networks = list(dict.fromkeys(labels.tolist()))
    if len(networks) < 2 or any(np.count_nonzero(labels == label) < 2 for label in networks):
        raise ValueError("at least two networks with at least two ROIs each are required")
    if negative == "zero":
        a = np.maximum(a, 0)
    if fisher_z:
        a = np.arctanh(np.clip(a, -1 + 1e-7, 1 - 1e-7))
    values, names, defined = [], [], []
    for label in networks:
        inside, outside = np.flatnonzero(labels == label), np.flatnonzero(labels != label)
        ii, jj = np.triu_indices(len(inside), 1)
        within = a[:, inside[ii], inside[jj]].mean(axis=1)
        between = a[:, inside[:, None], outside[None, :]].mean(axis=(1, 2))
        valid = np.abs(within) > 1e-12
        segregation = np.divide(within - between, within, out=np.zeros_like(within), where=valid)
        values.extend((within, between, segregation))
        names.extend((f"within_{label}", f"between_{label}", f"segregation_{label}"))
        defined.append(valid)
    for i, first in enumerate(networks):
        for second in networks[i + 1:]:
            left, right = np.flatnonzero(labels == first), np.flatnonzero(labels == second)
            values.append(a[:, left[:, None], right[None, :]].mean(axis=(1, 2)))
            names.append(f"integration_{first}_{second}")
    return FeatureSet(np.column_stack(values), tuple(names), np.column_stack(defined))


def pulse_pressure(systolic, diastolic):
    """Systolic minus diastolic pressure, using matching mmHg units."""
    sbp = vector(systolic, "systolic")
    dbp = vector(diastolic, "diastolic", len(sbp))
    if np.any(dbp <= 0) or np.any(sbp < dbp):
        raise ValueError("pressure requires positive diastolic and systolic >= diastolic")
    return sbp - dbp


def rmssd(rr_intervals, *, unit="seconds"):
    """Root mean square successive NN-interval differences, returned in ms.

    A one-dimensional trace yields a float; a subject-by-interval matrix yields
    one value per subject. Accepted NN intervals must be supplied after artifact
    review. This function performs no ECG peak detection or clinical QC.
    """
    rr = np.asarray(rr_intervals, float)
    if rr.ndim not in (1, 2) or rr.shape[-1] < 3 or not np.all(np.isfinite(rr)):
        raise ValueError("RR intervals must contain at least three finite intervals per trace")
    if np.any(rr <= 0) or unit not in ("seconds", "milliseconds"):
        raise ValueError("RR intervals must be positive and units explicitly supported")
    factor = 1000 if unit == "seconds" else 1
    result = np.sqrt(np.mean(np.diff(rr, axis=-1) ** 2, axis=-1)) * factor
    return float(result) if np.ndim(result) == 0 else result


def neurovascular_features(connectivity, labels, systolic, diastolic, *, rr_intervals=None):
    """Append pulse pressure, optional RMSSD, and raw segregation interactions.

    No centering or scaling is fit here, so extraction may precede fold splitting.
    Interactions describe joint variation; they do not establish causal coupling.
    """
    network = network_features(connectivity, labels)
    pp = pulse_pressure(systolic, diastolic)
    if len(pp) != len(network.values):
        raise ValueError("physiology and connectivity subject counts differ")
    columns = [network.values, pp[:, None]]
    names = list(network.names) + ["pulse_pressure_mmhg"]
    seg_columns = [i for i, name in enumerate(network.names) if name.startswith("segregation_")]
    for i in seg_columns:
        columns.append((network.values[:, i] * pp)[:, None])
        names.append(f"{network.names[i]}_x_pulse_pressure")
    if rr_intervals is not None:
        hrv = np.asarray(rmssd(rr_intervals))
        if hrv.shape != pp.shape:
            raise ValueError("RR intervals must have one row per subject")
        columns.append(hrv[:, None])
        names.append("rmssd_ms")
        for i in seg_columns:
            columns.append((network.values[:, i] * hrv)[:, None])
            names.append(f"{network.names[i]}_x_rmssd")
    return FeatureSet(np.column_stack(columns), tuple(names), network.segregation_defined)


def motion_sensitivity(features, target, framewise_displacement, covariates=None,
                       *, n_permutations=199, random_state=42, blocks=None):
    """Partial correlations before/after FD adjustment with optional permutation p.

    Attenuation is (abs(before)-abs(after))/abs(before), and zero if before is
    numerically zero. Negative attenuation means the magnitude increased.
    These sensitivity summaries are descriptive, not causal evidence.
    """
    x = matrix(features, "features")
    y = vector(target, "target", len(x))
    fd = vector(framewise_displacement, "framewise_displacement", len(x))
    if np.any(fd < 0):
        raise ValueError("framewise displacement must be nonnegative")
    c = np.empty((len(x), 0)) if covariates is None else matrix(covariates, "covariates", len(x))
    adjusted = np.column_stack((c, fd))

    def partial(cov):
        q, _ = nuisance_basis(cov, len(x))
        xr, yr = residualize(x, q), residualize(y, q)
        denom = np.linalg.norm(xr, axis=0) * np.linalg.norm(yr)
        return np.divide(xr.T @ yr, denom, out=np.zeros(x.shape[1]), where=denom > 1e-12)

    before, after = partial(c), partial(adjusted)
    attenuation = np.divide(np.abs(before) - np.abs(after), np.abs(before),
                            out=np.zeros_like(before), where=np.abs(before) > 1e-12)
    result = {"correlation_before_fd": before.tolist(), "correlation_after_fd": after.tolist(),
              "relative_attenuation": attenuation.tolist(), "causal_interpretation": False}
    if n_permutations:
        result["pvalues_before_fd"] = freedman_lane(
            x, y, c, n_permutations=n_permutations, random_state=random_state, blocks=blocks
        ).pvalues.tolist()
        result["pvalues_after_fd"] = freedman_lane(
            x, y, adjusted, n_permutations=n_permutations, random_state=random_state, blocks=blocks
        ).pvalues.tolist()
    return result

