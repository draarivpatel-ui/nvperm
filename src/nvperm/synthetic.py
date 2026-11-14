"""Synthetic demographics, linear Fisher-z edges, and PSD connectomes.

This simulator is not fitted to participant data. Linear edge targets have known
conditional coefficients; PSD projection and network summaries are nonlinear,
so their downstream regression effects are not asserted to equal those targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ._utils import generator, positive_integer


@dataclass(frozen=True)
class SyntheticCohort:
    features: np.ndarray
    target: np.ndarray
    confounds: np.ndarray
    connectomes: np.ndarray
    network_labels: np.ndarray
    age: np.ndarray
    sex: np.ndarray
    framewise_displacement: np.ndarray
    site: np.ndarray
    systolic: np.ndarray
    diastolic: np.ndarray
    rr_intervals: np.ndarray
    ground_truth: np.ndarray
    effect_size: float
    heteroskedasticity: float
    confound_names: tuple[str, ...] = ("age", "sex", "mean_fd", "site_1", "site_2")

    def save(self, path):
        """Write a pickle-free NPZ compatible with ``nvperm run``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, features=self.features, target=self.target, confounds=self.confounds,
            connectomes=self.connectomes, network_labels=self.network_labels,
            age=self.age, sex=self.sex, framewise_displacement=self.framewise_displacement,
            site=self.site, systolic=self.systolic, diastolic=self.diastolic,
            rr_intervals=self.rr_intervals, ground_truth=self.ground_truth,
        )


def generate_cohort(n_subjects=120, n_rois=12, *, n_networks=3, effect_size=0.0,
                    confounding=1.0, heteroskedasticity=0.0, random_state=42):
    """Generate known-effect edges and plausible age/motion/physiology profiles.

    ``effect_size`` controls selected Fisher-z edges per residual physiology SD.
    ``confounding`` multiplies shared age, sex, FD and site effects.
    Heteroskedasticity changes conditional error SD with age; its default is zero
    for calibration under exchangeable errors. Age is in years and FD in mm.
    """
    n = positive_integer(n_subjects, "n_subjects", 12)
    r = positive_integer(n_rois, "n_rois", 4)
    k = positive_integer(n_networks, "n_networks", 2)
    if r < 2 * k:
        raise ValueError("each network requires at least two ROIs")
    if not all(np.isfinite(v) for v in (effect_size, confounding, heteroskedasticity)):
        raise ValueError("effect and noise parameters must be finite")
    if confounding < 0 or not 0 <= heteroskedasticity <= 2:
        raise ValueError("confounding must be nonnegative and heteroskedasticity in [0, 2]")
    rng = generator(random_state)
    older = rng.integers(0, 2, n)
    age = np.where(older, rng.uniform(59, 77, n), rng.uniform(20, 35, n))
    az = (age - 48) / 20
    sex, site = rng.integers(0, 2, n), rng.integers(0, 3, n)
    fd = 0.05 + 0.035 * (az + 1.4) + 0.06 * rng.gamma(2, 0.5, n)
    fdz = (fd - 0.14) / 0.06
    c = np.column_stack((age, sex, fd, site == 1, site == 2)).astype(float)
    residual_physiology = rng.normal(size=n)
    target = 50 + confounding * (9 * az + 7 * fdz + 2 * sex + 3 * (site == 1)
                                 - 2 * (site == 2)) + 6 * residual_physiology
    labels = np.empty(r, dtype="U12")
    for network, indices in enumerate(np.array_split(np.arange(r), k)):
        labels[indices] = f"N{network}"
    upper = np.triu_indices(r, 1)
    same = labels[upper[0]] == labels[upper[1]]
    signal = same & (labels[upper[0]] == "N0")
    loading = signal.astype(float)
    base = np.where(same, 0.48, 0.12)
    edge_age = rng.uniform(0.07, 0.14, len(base))
    edge_motion = rng.uniform(0.10, 0.20, len(base))
    nuisance = confounding * (
        az[:, None] * edge_age + fdz[:, None] * edge_motion
        + 0.02 * sex[:, None] + 0.03 * (site == 1)[:, None]
    )
    sd = 0.12 * np.exp(heteroskedasticity * np.clip(az, -1.4, 1.4) / 2)
    features = (base + nuisance + 0.12 * effect_size * residual_physiology[:, None] * loading
                + rng.normal(size=(n, len(base))) * sd[:, None])
    matrices = np.tile(np.eye(r), (n, 1, 1))
    matrices[:, upper[0], upper[1]] = np.tanh(features)
    matrices[:, upper[1], upper[0]] = np.tanh(features)
    for index in range(n):
        eigenvalues, eigenvectors = np.linalg.eigh(matrices[index])
        projected = (eigenvectors * np.maximum(eigenvalues, 1e-5)) @ eigenvectors.T
        scale = np.sqrt(np.diag(projected))
        matrices[index] = projected / scale[:, None] / scale[None, :]
    dbp = 72 + 3 * az + rng.normal(0, 3, n)
    sbp = dbp + target
    # NN intervals are already artifact-free synthetic intervals, not raw ECG.
    rr_sd = 0.020 * np.exp(-0.2 * az)
    rr = 0.86 + 0.03 * az[:, None] + rng.normal(size=(n, 128)) * rr_sd[:, None]
    return SyntheticCohort(
        features, target, c, matrices, labels, age, sex, fd, site, sbp, dbp, rr,
        0.12 * effect_size * loading / 6, float(effect_size), float(heteroskedasticity),
    )

