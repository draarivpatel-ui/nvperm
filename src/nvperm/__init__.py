"""Reusable permutation inference and fold-isolated neurovascular modeling."""

from .freedman_lane import FreedmanLane, PermutationResult, bh_fdr, freedman_lane
from .cv_connectome import ConnectomePhysiologyCV, CVResult, CVPermutationResult
from .coupling import (
    FeatureSet, network_features, neurovascular_features, motion_sensitivity,
    pulse_pressure, rmssd,
)
from .synthetic import SyntheticCohort, generate_cohort

__version__ = "0.1.0"
__all__ = [
    "FreedmanLane", "PermutationResult", "bh_fdr", "freedman_lane",
    "ConnectomePhysiologyCV", "CVResult", "CVPermutationResult",
    "FeatureSet", "network_features", "neurovascular_features",
    "motion_sensitivity", "pulse_pressure", "rmssd",
    "SyntheticCohort", "generate_cohort",
]

