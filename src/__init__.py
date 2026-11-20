"""
MA-GRPO: Core Paper Implementation.
Multi-Agent Group Relative Policy Optimization for Ultra-Long Narrative Generation.

Core Modules:
  - planning          : OutlineAgent (Event Graph + Story Arc), SubEventAgent, PlanCriticAgent, PlanningAgent
  - coordinator       : CoordinatorAgent (Two-Tier Memory)
  - rewards           : CompositeRewardEngine (Intrinsic GRPO Length, Coherence, Lexical Quality)
  - pipeline          : VLLMWritingPolicy, Artifact Cleaner, Deduplication Gate & MAGRPOPipeline
  - dataset_generator : PilotDatasetGenerator, ConditionDGRPODatasetGenerator & Canonical Premises
"""

from .coordinator import CoordinatorAgent
from .dataset_generator import (
    CANONICAL_20_PREMISES,
    ConditionDGRPODatasetGenerator,
    PilotDatasetGenerator,
    load_premises_pool,
    normalize_premise_text,
)
from .pipeline import (
    MAGRPOPipeline,
    VLLMWritingPolicy,
    clean_generation_artifacts,
    deduplicate_repeated_paragraphs,
    extract_cot_and_prose,
    resolve_optimal_dtype,
)
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
    "CompositeRewardEngine",
    "calculate_length_reward",
    "VLLMWritingPolicy",
    "MAGRPOPipeline",
    "clean_generation_artifacts",
    "deduplicate_repeated_paragraphs",
    "extract_cot_and_prose",
    "resolve_optimal_dtype",
    "PilotDatasetGenerator",
    "ConditionDGRPODatasetGenerator",
    "load_premises_pool",
    "normalize_premise_text",
    "CANONICAL_20_PREMISES",
]
