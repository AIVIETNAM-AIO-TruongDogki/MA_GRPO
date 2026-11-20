"""
Phase 2 Expansion Reward Architecture for MA-GRPO.

Design principle (from Phase 2 Reward Architecture Redesign):
  - Evaluators return strictly raw `[0, 1]` scores + structured diagnostics.
  - Aggregators compute final scalar rewards across `EXP-00 .. EXP-05`.
  - Clean separation between Local LLM Judge (32B 4-bit semantic reasoning)
    and Deterministic Python + MiniLM Embedder (length, redundancy, format, math).
"""

from .base import (
    ConstraintPenalties,
    EvaluationInput,
    GlobalScores,
    LocalScores,
    RewardBreakdown,
)

__all__ = [
    "EvaluationInput",
    "LocalScores",
    "GlobalScores",
    "ConstraintPenalties",
    "RewardBreakdown",
]
