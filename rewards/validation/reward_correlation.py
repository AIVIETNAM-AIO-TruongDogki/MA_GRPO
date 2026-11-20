"""
M1 Offline Validation: Reward Distribution, Saturation Detection & Spearman Correlation Matrix
(checks whether Local Expansion scores add signal beyond Global Chapter scores — H2).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from experiments.benchmarks.outline_bench.judge_outline import compute_spearman_correlation


_TRACKED_COLS = [
    "r_total",
    "r_local",
    "r_global",
    "event_realization",
    "causal_development",
    "meaningful_elaboration",
    "macro_adherence",
    "chapter_coherence",
    "character_consistency",
    "narrative_progression",
    "p_length",
    "p_rep",
    "p_contra",
    "p_format",
]


def _dist_stats(vals: List[float]) -> Dict[str, Any]:
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = math.sqrt(var)
    frac_above_09 = sum(1 for x in vals if x >= 0.9) / n
    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "frac_ge_0_9": round(frac_above_09, 4),
        "saturated_warning": bool(frac_above_09 >= 0.80 or std < 0.04),
    }


def analyze_reward_distributions_and_correlations(flat_logs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes distribution stats, saturation warnings, and Spearman correlation matrix
    across all Phase 2 reward components.
    """
    distributions: Dict[str, Dict[str, Any]] = {}
    available_cols: List[str] = []
    for col in _TRACKED_COLS:
        vals = [float(r[col]) for r in flat_logs if r.get(col) is not None]
        if len(vals) >= 2:
            distributions[col] = _dist_stats(vals)
            available_cols.append(col)

    corr_matrix: Dict[str, Dict[str, float]] = {}
    for c1 in available_cols:
        corr_matrix[c1] = {}
        for c2 in available_cols:
            xs, ys = [], []
            for r in flat_logs:
                if r.get(c1) is not None and r.get(c2) is not None:
                    xs.append(float(r[c1]))
                    ys.append(float(r[c2]))
            corr_matrix[c1][c2] = compute_spearman_correlation(xs, ys) if len(xs) >= 3 else 0.0

    local_vs_global_rho = (
        corr_matrix.get("r_local", {}).get("r_global")
        if "r_local" in corr_matrix and "r_global" in corr_matrix
        else None
    )
    saturated_axes = [
        k for k, st in distributions.items()
        if k not in {"p_length", "p_rep", "p_contra", "p_format"} and st["saturated_warning"]
    ]

    return {
        "num_samples": len(flat_logs),
        "distributions": distributions,
        "spearman_matrix": corr_matrix,
        "local_vs_global_spearman": local_vs_global_rho,
        "h2_complementary_signal_ok": (local_vs_global_rho is None or abs(local_vs_global_rho) < 0.88),
        "saturated_axes": saturated_axes,
    }
