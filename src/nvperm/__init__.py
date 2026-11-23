"""Reusable permutation inference and fold-isolated neurovascular modeling."""

from .freedman_lane import FreedmanLane, PermutationResult, bh_fdr, freedman_lane
from .cv_connectome import ConnectomePhysiologyCV, CVResult, CVPermutationResult
from .bootstrap import BootstrapResult, bootstrap_effects, wild_bootstrap_effects
from .plotting import plot_adjusted_effects, plot_cv_predictions
from .coupling import (
    FeatureSet, network_features, neurovascular_features, motion_sensitivity,
    pulse_pressure, rmssd,
)
from .synthetic import SyntheticCohort, generate_cohort

__version__ = "0.2.0"
__all__ = [
    "FreedmanLane", "PermutationResult", "bh_fdr", "freedman_lane",
    "ConnectomePhysiologyCV", "CVResult", "CVPermutationResult",
    "FeatureSet", "network_features", "neurovascular_features",
    "motion_sensitivity", "pulse_pressure", "rmssd",
    "SyntheticCohort", "generate_cohort",
    "BootstrapResult", "bootstrap_effects", "wild_bootstrap_effects",
    "plot_adjusted_effects", "plot_cv_predictions",
]

