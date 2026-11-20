"""Phase 2 Reward Validation & Calibration Suite (100% Python — M1 Offline Audits)."""

from .human_alignment import compute_human_alignment_report, compute_kendall_tau
from .reward_correlation import analyze_reward_distributions_and_correlations
from .reward_hacking import build_perturbation_suite, evaluate_perturbation_discrimination

__all__ = [
    "analyze_reward_distributions_and_correlations",
    "build_perturbation_suite",
    "evaluate_perturbation_discrimination",
    "compute_kendall_tau",
    "compute_human_alignment_report",
]
