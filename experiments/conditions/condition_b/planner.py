"""
Condition B Planner: Unstructured Prose Outline Baseline.
Generates a single prose summary plan without Event Graph nodes, DAG dependencies, or Story Arc tags.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from src.planning import estimate_token_count, extract_premise_keywords_and_characters


UNSTRUCTURED_PLAN_PROMPT = (
    "You are a narrative planner. Summarize the overarching story trajectory from the premise "
    "into a cohesive, unstructured prose outline (200-350 words) without chapter splits, graphs, "
    "or bulleted event schemas."
)


class UnstructuredPlanner:
    """Condition B Planner: Unstructured text outline baseline."""

    def __init__(self, llm_generate_fn: Optional[Callable[[str, str], str]] = None):
        self.generate_fn = llm_generate_fn

    def create_story_plan(
        self,
        premise: str,
        genre: str = "Science Fiction",
        total_target_words: int = 10000,
        num_chapters: int = 8,
    ) -> Dict[str, Any]:
        t0 = time.time()
        prose_summary = ""
        if self.generate_fn:
            try:
                prose_summary = self.generate_fn(
                    UNSTRUCTURED_PLAN_PROMPT,
                    f"Premise: {premise}\nGenre: {genre}\nTarget length: ~{total_target_words} words.",
                ).strip()
            except Exception:
                prose_summary = ""

        if not prose_summary:
            chars, keywords = extract_premise_keywords_and_characters(premise)
            focus = ", ".join(keywords[:3]) if keywords else "central mystery"
            prose_summary = (
                f"In this {genre} tale centered on {', '.join(chars[:2])}, the story opens with {premise.strip()} "
                f"As tensions surrounding {focus} deepen, unexpected revelations force the characters through escalating trials "
                f"culminating in a decisive turning point and ultimate transformation."
            )

        planning_tokens = estimate_token_count(prose_summary)
        per_ch_words = max(1, total_target_words // max(1, num_chapters))
        chapters = []
        for idx in range(1, num_chapters + 1):
            w_target = per_ch_words if idx < num_chapters else max(1, total_target_words - per_ch_words * (num_chapters - 1))
            chapters.append({
                "chapter_index": idx,
                "chapter_title": f"Part {idx}",
                "target_words": w_target,
                "chapter_goal": prose_summary,
                "scene_budgets": [w_target],
                "scene_outline": [f"Continue the narrative smoothly (~{w_target} words) following the prose overview."],
                "characters_active": ["Protagonist"],
            })

        return {
            "condition": "B",
            "premise": premise,
            "genre": genre,
            "unstructured_plan": prose_summary,
            "total_target_words": total_target_words,
            "num_chapters": num_chapters,
            "chapters": chapters,
            "planning_metadata": {
                "condition": "B",
                "planning_tokens": planning_tokens,
                "planning_latency_s": round(time.time() - t0, 4),
            },
        }


def plan_next_chapter_b(
    history_text: str,
    llm_generate_fn: Optional[Callable[[str, str], str]] = None,
    premise: str = "",
) -> Dict[str, Any]:
    """Track 1 VR-CLI Next-Chapter Planner for Condition B (Unstructured prose)."""
    t0 = time.time()
    tail_excerpt = " ".join(history_text.strip().split()[-600:]) if history_text else premise
    plan_text = ""
    if llm_generate_fn:
        try:
            plan_text = llm_generate_fn(
                UNSTRUCTURED_PLAN_PROMPT,
                f"Preceding chapters:\n{tail_excerpt}\nWrite an unstructured prose outline (120-200 words) for the very next chapter.",
            ).strip()
        except Exception:
            plan_text = ""
    if not plan_text:
        plan_text = (
            f"Continuing immediately from prior events ({tail_excerpt[:160]}...), "
            f"the next chapter escalates the unresolved tension and advances the protagonist toward a decisive confrontation."
        )
    return {
        "condition": "B",
        "plan_text": plan_text,
        "events": [],
        "planning_tokens": estimate_token_count(plan_text),
        "planning_latency_s": round(time.time() - t0, 4),
    }

