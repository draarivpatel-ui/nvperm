"""Internal numerical validation and exchangeability helpers."""
from __future__ import annotations

import numbers
import numpy as np


def matrix(value, name, n=None):
    a = np.asarray(value, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[0] < 2:
        raise ValueError(f"{name} must have shape (subjects, columns), with at least two subjects")
    if n is not None and a.shape[0] != n:
        raise ValueError(f"{name} has a different number of subjects")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite values")
    return a.copy()


def vector(value, name, n=None):
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or len(a) < 2:
        raise ValueError(f"{name} must be a one-dimensional subject vector")
    if n is not None and len(a) != n:
        raise ValueError(f"{name} has a different number of subjects")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite values")
    return a.copy()


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def generator(random_state):
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is not None and (
        isinstance(random_state, bool) or not isinstance(random_state, numbers.Integral)
        or random_state < 0
    ):
        raise ValueError("random_state must be a nonnegative integer, Generator, or None")
    return np.random.default_rng(random_state)


def label_groups(labels, n, *, name="block labels"):
    if labels is None:
        return [np.arange(n)]
    a = np.asarray(labels)
    if a.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional subject vector")
    if len(a) != n:
        raise ValueError(f"{name} has a different number of subjects")
    if a.dtype.kind in "biufc" and not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must not be missing or non-finite")
    if a.dtype.kind in "mM" and np.any(np.isnat(a)):
        raise ValueError(f"{name} must not be missing")
    groups = {}
    for i, label in enumerate(a.tolist()):
        if (label is None
                or (isinstance(label, (float, complex)) and not np.isfinite(label))
                or (isinstance(label, (str, bytes)) and not label.strip())):
            raise ValueError(f"{name} must not be missing or non-finite")
        try:
            groups.setdefault(label, []).append(i)
        except TypeError as exc:
            raise ValueError(f"{name} must be scalar and hashable") from exc
    return [np.asarray(indices, dtype=int) for indices in groups.values()]


def permutation_indices(rng, groups, count, n):
    indices = np.tile(np.arange(n), (count, 1))
    for row in indices:
        for group in groups:
            row[group] = rng.permutation(group)
    return indices


def nuisance_svd(design):
    """Retain one nuisance space using eps * max(shape) * largest singular value.

    Return its orthonormal training basis, the map from basis scores to design
    coefficients, and the absolute tolerance. Applying a held-out design to
    that same map extends only the retained directions; no second cutoff is
    used for fitting, rank reporting or prediction.
    """
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    tolerance = float(np.finfo(float).eps * max(design.shape) * s[0])
    rank = int(np.count_nonzero(s > tolerance))
    return u[:, :rank], vt[:rank].T / s[:rank], tolerance


def nuisance_basis(covariates, n):
    """Return an intercept-containing, rank-aware orthonormal nuisance basis."""
    c = np.empty((n, 0)) if covariates is None else matrix(covariates, "covariates", n)
    if c.shape[1]:
        mean = c.mean(axis=0)
        scale = c.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        c = (c - mean) / scale
    design = np.column_stack((np.ones(n), c))
    q, _, _ = nuisance_svd(design)
    return q, q.shape[1]


def residualize(value, q):
    return value - q @ (q.T @ value)


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    xnorm, ynorm = np.linalg.norm(x), np.linalg.norm(y)
    return float((x / xnorm) @ (y / ynorm)) if xnorm > 0 and ynorm > 0 else 0.0

