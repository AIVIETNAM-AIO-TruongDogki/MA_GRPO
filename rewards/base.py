"""Core typed data containers for the Phase 2 Expansion Reward Architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationInput:
    """Canonical input package passed to Local, Global, Holistic, and Constraint evaluators."""

    generated_chapter: str
    target_words: int = 1125
    chapter_title: str = ""
    chapter_goal: str = ""
    scene_outline: List[str] = field(default_factory=list)
    characters_active: List[str] = field(default_factory=list)
    context_summary: str = ""
    prev_chapter_tail: str = ""
    long_term_memory: Dict[str, Any] = field(default_factory=dict)
    prompt_text: str = ""
    chapter_index: int = 1
    premise_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocalScores:
    """Level 1 (Local Expansion) raw scores in [0.0, 1.0] from 32B LLM Judge."""

    event_realization: float
    causal_development: float
    meaningful_elaboration: float
    evidence: List[Dict[str, str]] = field(default_factory=list)
    missing_elements: List[str] = field(default_factory=list)
    parse_ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalScores:
    """Level 2 (Global Chapter) raw scores in [0.0, 1.0] + detected contradictions from 32B LLM Judge."""

    macro_adherence: float
    chapter_coherence: float
    character_consistency: float
    narrative_progression: float
    contradictions: List[Dict[str, str]] = field(default_factory=list)
    evidence: List[Dict[str, str]] = field(default_factory=list)
    parse_ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintPenalties:
    """Level 3 (Constraints & Safety) raw penalties in [0.0, 1.0] computed in Python + MiniLM."""

    p_length: float
    p_rep: float
    p_contra: float
    p_format: float = 0.0
    prose_words: int = 0
    think_words: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RewardBreakdown:
    """Complete auditable breakdown returned by the Reward Aggregator for GRPO & logging."""

    r_total: float
    r_quality: float
    r_local: Optional[float]
    r_global: Optional[float]
    r_holistic: Optional[float]
    penalty_total: float
    mode: str
    local_scores: Optional[LocalScores] = None
    global_scores: Optional[GlobalScores] = None
    constraints: Optional[ConstraintPenalties] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_flat_log(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "mode": self.mode,
            "r_total": round(self.r_total, 4),
            "r_quality": round(self.r_quality, 4),
            "r_local": round(self.r_local, 4) if self.r_local is not None else None,
            "r_global": round(self.r_global, 4) if self.r_global is not None else None,
            "r_holistic": round(self.r_holistic, 4) if self.r_holistic is not None else None,
            "penalty_total": round(self.penalty_total, 4),
        }
        if self.local_scores is not None:
            row.update(
                {
                    "event_realization": self.local_scores.event_realization,
                    "causal_development": self.local_scores.causal_development,
                    "meaningful_elaboration": self.local_scores.meaningful_elaboration,
                    "local_parse_ok": self.local_scores.parse_ok,
                }
            )
        if self.global_scores is not None:
            row.update(
                {
                    "macro_adherence": self.global_scores.macro_adherence,
                    "chapter_coherence": self.global_scores.chapter_coherence,
                    "character_consistency": self.global_scores.character_consistency,
                    "narrative_progression": self.global_scores.narrative_progression,
                    "n_contradictions": len(self.global_scores.contradictions),
                    "global_parse_ok": self.global_scores.parse_ok,
                }
            )
        if self.constraints is not None:
            row.update(
                {
                    "p_length": self.constraints.p_length,
                    "p_rep": self.constraints.p_rep,
                    "p_contra": self.constraints.p_contra,
                    "p_format": self.constraints.p_format,
                    "prose_words": self.constraints.prose_words,
                    "think_words": self.constraints.think_words,
                }
            )
        row.update(self.metadata)
        return row
