"""Weighted geometric aggregation (EXP-05) preventing a high score on one axis from masking a low axis."""

from __future__ import annotations

import math
from typing import Dict, Optional

from rewards.base import GlobalScores, LocalScores
from .arithmetic import DEFAULT_GLOBAL_WEIGHTS, DEFAULT_LOCAL_WEIGHTS, _normalize_weights


def _weighted_geometric(values: Dict[str, float], weights: Dict[str, float], epsilon: float = 0.05) -> float:
    w = _normalize_weights(weights)
    eps = max(1e-6, float(epsilon))
    log_sum = 0.0
    for key, weight in w.items():
        if weight <= 0.0:
            continue
        r = max(0.0, min(1.0, float(values.get(key, 0.5))))
        log_sum += weight * math.log(eps + r)
    # Rescale from [eps, 1 + eps] back to [0.0, 1.0] cleanly
    raw = math.exp(log_sum)
    scaled = (raw - eps) / 1.0
    return round(max(0.0, min(1.0, scaled)), 4)


def aggregate_local_geometric(
    scores: LocalScores,
    weights: Optional[Dict[str, float]] = None,
    epsilon: float = 0.05,
) -> float:
    vals = {
        "event_realization": scores.event_realization,
        "causal_development": scores.causal_development,
        "meaningful_elaboration": scores.meaningful_elaboration,
    }
    return _weighted_geometric(vals, weights or DEFAULT_LOCAL_WEIGHTS, epsilon=epsilon)


def aggregate_global_geometric(
    scores: GlobalScores,
    weights: Optional[Dict[str, float]] = None,
    epsilon: float = 0.05,
) -> float:
    vals = {
        "macro_adherence": scores.macro_adherence,
        "chapter_coherence": scores.chapter_coherence,
        "character_consistency": scores.character_consistency,
        "narrative_progression": scores.narrative_progression,
    }
    return _weighted_geometric(vals, weights or DEFAULT_GLOBAL_WEIGHTS, epsilon=epsilon)
