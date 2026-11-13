"""
Unified Registry & Dispatcher for Structural Ablation Conditions A, B, C, D, and E.

Each condition lives in its own subpackage:
  - condition_a: Direct Single-Pass (Zero Planning)
  - condition_b: Unstructured Prose Outline
  - condition_c: Sequential Linear Chapter Outline
  - condition_d: Macro Event Graph Only (No SubEventAgent)
  - condition_e: Full Two-Level Hierarchy (Macro Event Graph + SubEventAgent)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from .condition_a import run_condition_a
from .condition_b import UnstructuredPlanner, plan_next_chapter_b, run_condition_b
from .condition_c import LinearChapterPlanner, plan_next_chapter_c, run_condition_c
from .condition_d import plan_next_chapter_d, run_condition_d
from .condition_e import plan_next_chapter_e, run_condition_e

PLANNING_CONDITIONS: Tuple[str, ...] = ("A", "B", "C", "D", "E")

CONDITION_RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "A": run_condition_a,
    "B": run_condition_b,
    "C": run_condition_c,
    "D": run_condition_d,
    "E": run_condition_e,
}

NEXT_CHAPTER_PLANNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "B": plan_next_chapter_b,
    "C": plan_next_chapter_c,
    "D": plan_next_chapter_d,
    "E": plan_next_chapter_e,
}


def run_condition(
    pipeline: Any,
    condition: str,
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
    """Dispatch execution to the dedicated folder runner for Condition A, B, C, D, or E."""
    cond = condition.upper()
    if cond not in CONDITION_RUNNERS:
        raise ValueError(f"Condition must be one of {PLANNING_CONDITIONS} (got {condition})")
    return CONDITION_RUNNERS[cond](
        pipeline=pipeline,
        premise=premise,
        genre=genre,
        total_target_words=total_target_words,
        num_chapters=num_chapters,
        perturbation_type=perturbation_type,
        donor_plan=donor_plan,
        temperature=temperature,
        top_p=top_p,
        enable_cot=enable_cot,
    )


def plan_next_chapter_from_history(
    history_text: str,
    condition: str = "D",
    llm_generate_fn: Optional[Callable[[str, str], str]] = None,
    premise: str = "",
) -> Dict[str, Any]:
    """Track 1 VR-CLI dispatcher across Conditions B, C, D, and E."""
    cond = condition.upper()
    if cond not in NEXT_CHAPTER_PLANNERS:
        raise ValueError(f"Next-chapter planning expects condition in B/C/D/E (got {condition})")
    return NEXT_CHAPTER_PLANNERS[cond](
        history_text=history_text,
        llm_generate_fn=llm_generate_fn,
        premise=premise,
    )


__all__ = [
    "PLANNING_CONDITIONS",
    "CONDITION_RUNNERS",
    "NEXT_CHAPTER_PLANNERS",
    "UnstructuredPlanner",
    "LinearChapterPlanner",
    "run_condition",
    "run_condition_a",
    "run_condition_b",
    "run_condition_c",
    "run_condition_d",
    "run_condition_e",
    "plan_next_chapter_from_history",
]

