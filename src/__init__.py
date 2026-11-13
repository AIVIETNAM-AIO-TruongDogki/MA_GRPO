"""
MA-GRPO: Core Paper Implementation.
Multi-Agent Group Relative Policy Optimization for Ultra-Long Narrative Generation.

Core Modules:
  - planning          : OutlineAgent (Event Graph + Story Arc), SubEventAgent, PlanCriticAgent, PlanningAgent
  - coordinator       : CoordinatorAgent (Two-Tier Memory) & ContextGateValidator
  - rewards           : CompositeRewardEngine (Intrinsic GRPO Length, Coherence, Lexical Quality)
  - pipeline          : VLLMWritingPolicy & MAGRPOPipeline
  - dataset_generator : PilotDatasetGenerator & Canonical Premises
"""

from .coordinator import ContextGateValidator, CoordinatorAgent
from .dataset_generator import CANONICAL_20_PREMISES, PilotDatasetGenerator
from .pipeline import MAGRPOPipeline, VLLMWritingPolicy, extract_cot_and_prose, resolve_optimal_dtype
from .planning import (
    STORY_ARC_STAGES,
    OutlineAgent,
    PlanCriticAgent,
    PlanningAgent,
    SubEventAgent,
    compute_story_arc_alignment,
)
from .rewards import CompositeRewardEngine, calculate_length_reward

__version__ = "0.2.0"

__all__ = [
    "STORY_ARC_STAGES",
    "OutlineAgent",
    "SubEventAgent",
    "PlanCriticAgent",
    "PlanningAgent",
    "compute_story_arc_alignment",
    "CoordinatorAgent",
    "ContextGateValidator",
    "CompositeRewardEngine",
    "calculate_length_reward",
    "VLLMWritingPolicy",
    "MAGRPOPipeline",
    "extract_cot_and_prose",
    "resolve_optimal_dtype",
    "PilotDatasetGenerator",
    "CANONICAL_20_PREMISES",
]
