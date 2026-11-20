"""Weighted arithmetic aggregation for Local and Global raw scores (EXP-02 / EXP-03 / EXP-04)."""

from __future__ import annotations

from typing import Dict, Optional

from rewards.base import GlobalScores, LocalScores


DEFAULT_LOCAL_WEIGHTS: Dict[str, float] = {
    "event_realization": 0.35,
    "causal_development": 0.35,
    "meaningful_elaboration": 0.30,
}

DEFAULT_GLOBAL_WEIGHTS: Dict[str, float] = {
    "macro_adherence": 0.30,
    "chapter_coherence": 0.30,
    "character_consistency": 0.20,
    "narrative_progression": 0.20,
}


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0.0:
        n = max(1, len(weights))
        return {k: 1.0 / n for k in weights}
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def aggregate_local_arithmetic(
    scores: LocalScores,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    w = _normalize_weights(weights or DEFAULT_LOCAL_WEIGHTS)
    val = (
        w.get("event_realization", 0.35) * scores.event_realization
        + w.get("causal_development", 0.35) * scores.causal_development
        + w.get("meaningful_elaboration", 0.30) * scores.meaningful_elaboration
    )
    return round(max(0.0, min(1.0, val)), 4)


def aggregate_global_arithmetic(
    scores: GlobalScores,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    w = _normalize_weights(weights or DEFAULT_GLOBAL_WEIGHTS)
    val = (
        w.get("macro_adherence", 0.30) * scores.macro_adherence
        + w.get("chapter_coherence", 0.30) * scores.chapter_coherence
        + w.get("character_consistency", 0.20) * scores.character_consistency
        + w.get("narrative_progression", 0.20) * scores.narrative_progression
    )
    return round(max(0.0, min(1.0, val)), 4)
