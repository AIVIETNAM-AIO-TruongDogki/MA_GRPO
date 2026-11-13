"""
Stage 1: Core Macro-Narrative Planning Package for MA-GRPO.

Implements the paper's Hierarchical Stage-1 Planning Architecture:
    - OutlineAgent (`outline_agent.py`): Causal Event Graph G=(V, E) with 5-stage Story Arc alignment.
    - SubEventAgent (`sub_event_agent.py`): Fine-grained causal micro-beat decomposition & DAG gate.
    - PlanCriticAgent (`plan_critic.py`): 5 structural verification gates before chapter approval.
    - PlanningAgent (`planning_agent.py`): Partitions Event Graph across K chapters (Conditions D & E).
"""

from .outline_agent import OutlineAgent
from .plan_critic import PlanCriticAgent
from .planning_agent import PlanningAgent
from .story_arc import (
    EVENT_GENERATION_PROMPT,
    STORY_ARC_STAGES,
    assign_story_arc_labels,
    compute_story_arc_alignment,
    ensure_event_ids,
    estimate_token_count,
    extract_premise_keywords_and_characters,
    infer_story_arc_stage,
    normalize_story_arc_label,
    parse_storywriter_events,
)
from .sub_event_agent import SubEventAgent, validate_and_sanitize_dependencies

__all__ = [
    "STORY_ARC_STAGES",
    "EVENT_GENERATION_PROMPT",
    "estimate_token_count",
    "infer_story_arc_stage",
    "normalize_story_arc_label",
    "assign_story_arc_labels",
    "compute_story_arc_alignment",
    "parse_storywriter_events",
    "ensure_event_ids",
    "extract_premise_keywords_and_characters",
    "validate_and_sanitize_dependencies",
    "OutlineAgent",
    "SubEventAgent",
    "PlanCriticAgent",
    "PlanningAgent",
]

