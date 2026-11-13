"""
Condition E Runner: Full Hierarchy (Macro Event Graph + SubEventAgent Micro Beats) + CoordinatorAgent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .._shared import run_structured_condition


def run_condition_e(
    pipeline: Any,
    premise: str,
    genre: str = "Speculative Fiction",
    total_target_words: int = 10000,
    num_chapters: int = 8,
    perturbation_type: Optional[str] = None,
    donor_plan: Optional[Dict[str, Any]] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_cot: bool = True,
) -> Dict[str, Any]:
    """Execute Condition E (Full Event Graph + Micro SubEvents via SubEventAgent)."""
    def _build_blueprint() -> Dict[str, Any]:
        event_graph = pipeline.outline_agent.create_event_graph(
            premise=premise,
            genre=genre,
            num_events=max(5, num_chapters),
        )
        return pipeline.planning_agent.partition_into_chapters(
            event_graph=event_graph,
            total_target_words=total_target_words,
            num_chapters=num_chapters,
            enable_micro_subevents=True,
        )

    return run_structured_condition(
        condition="E",
        pipeline=pipeline,
        blueprint_fn=_build_blueprint,
        genre=genre,
        total_target_words=total_target_words,
        num_chapters=num_chapters,
        perturbation_type=perturbation_type,
        donor_plan=donor_plan,
        temperature=temperature,
        top_p=top_p,
        enable_cot=enable_cot,
    )
