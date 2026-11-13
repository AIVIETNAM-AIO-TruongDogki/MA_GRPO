"""
Shared structured-condition execution helper used by Conditions C, D, and E.
Ensures CoordinatorAgent compression and WritingPolicy loop stay identical
across structured ablation conditions while keeping each condition in its own folder.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from experiments.perturbation import MacroPlanPerturbator


def run_structured_condition(
    condition: str,
    pipeline: Any,
    blueprint_fn: Callable[[], Dict[str, Any]],
    genre: str,
    total_target_words: int,
    num_chapters: int,
    perturbation_type: Optional[str] = None,
    donor_plan: Optional[Dict[str, Any]] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_cot: bool = True,
) -> Dict[str, Any]:
    """Run blueprint construction, optional perturbation, and shared Coordinator/Writer loop."""
    t_start = time.time()
    perturbation_metadata: Optional[Dict[str, Any]] = None

    blueprint = blueprint_fn()
    if perturbation_type:
        blueprint, perturbation_metadata = MacroPlanPerturbator.apply_to_blueprint(
            blueprint, perturbation_type, donor_plan=donor_plan
        )

    exec_out = pipeline.execute_chapter_loop(
        blueprint=blueprint,
        temperature=temperature,
        top_p=top_p,
        enable_cot=enable_cot,
    )
    plan_meta = blueprint.get("planning_metadata", {})
    return {
        "condition": condition,
        "title": f"Story (Condition {condition}): {genre}",
        "blueprint": blueprint,
        "chapters": exec_out["chapters"],
        "full_text": exec_out["full_text"],
        "target_words": total_target_words,
        "actual_total_words": exec_out["actual_total_words"],
        "num_chapters": num_chapters,
        "planning_tokens": plan_meta.get("planning_tokens", 0),
        "planning_latency_s": plan_meta.get("planning_latency_s", 0.0),
        "generation_latency_s": exec_out["generation_latency_s"],
        "elapsed_seconds": round(time.time() - t_start, 2),
        "context_summaries": exec_out["context_summaries"],
        "perturbation_metadata": perturbation_metadata,
    }

