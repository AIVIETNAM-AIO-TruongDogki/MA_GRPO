"""
Condition C Planner (Re3 Linear Style): Sequential K-chapter titles and goals
without Event Graph nodes, DAG dependencies, Story Arc tags, or sub-events.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional

from src.planning import estimate_token_count, extract_premise_keywords_and_characters


LINEAR_CHAPTER_PROMPT = """Decompose the premise into a linear list of {num_chapters} sequential chapters.
Follow this exact format for each chapter:
Chapter 1: [Chapter Title]
Goal: [Concise narrative objective for this chapter]

Chapter 2: [Chapter Title]
Goal: ...
"""


class LinearChapterPlanner:
    """Condition C Planner: Flat sequential list of K chapter titles & goals."""

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
        raw_text = ""
        parsed_chapters: List[Dict[str, str]] = []

        if self.generate_fn:
            try:
                sys_prompt = LINEAR_CHAPTER_PROMPT.format(num_chapters=num_chapters)
                raw_text = self.generate_fn(sys_prompt, f"Premise: {premise}\nGenre: {genre}")
                blocks = re.findall(r"(Chapter \d+:.*?)(?=(?:\n\s*Chapter \d+:|\Z))", raw_text, re.DOTALL)
                for b in blocks:
                    t_m = re.search(r"Chapter \d+:\s*(.*?)(?=\n|$)", b)
                    g_m = re.search(r"Goal:\s*(.*)", b, re.DOTALL)
                    parsed_chapters.append({
                        "title": t_m.group(1).strip() if t_m else "Chapter Progression",
                        "goal": g_m.group(1).strip() if g_m else "Continue the narrative arc.",
                    })
            except Exception:
                parsed_chapters = []

        chars, keywords = extract_premise_keywords_and_characters(premise)
        if len(parsed_chapters) < num_chapters:
            focus = keywords[0] if keywords else "the central conflict"
            parsed_chapters = []
            raw_lines = []
            for idx in range(1, num_chapters + 1):
                title = f"Sequential Stage {idx}"
                if idx == 1:
                    goal = f"Introduce {chars[0]} and the premise around {focus}."
                elif idx == num_chapters:
                    goal = f"Resolve the conflicts concerning {focus} and conclude {chars[0]}'s journey."
                else:
                    goal = f"Advance chapter {idx} complications around {focus} for {', '.join(chars[:2])}."
                parsed_chapters.append({"title": title, "goal": goal})
                raw_lines.append(f"Chapter {idx}: {title}\nGoal: {goal}")
            raw_text = "\n\n".join(raw_lines)

        per_ch_words = max(1, total_target_words // max(1, num_chapters))
        chapters = []
        for idx in range(1, num_chapters + 1):
            item = parsed_chapters[idx - 1]
            w_target = per_ch_words if idx < num_chapters else max(1, total_target_words - per_ch_words * (num_chapters - 1))
            s1 = int(round(w_target * 0.25))
            s2 = int(round(w_target * 0.45))
            s3 = max(1, w_target - s1 - s2)
            chapters.append({
                "chapter_index": idx,
                "chapter_title": f"Chapter {idx}: {item['title']}",
                "target_words": w_target,
                "scene_budgets": [s1, s2, s3],
                "chapter_goal": item["goal"],
                "scene_outline": [
                    f"Segment 1 (~{s1} words): Opening follow-up toward chapter goal.",
                    f"Segment 2 (~{s2} words): Primary narrative development for: {item['goal']}",
                    f"Segment 3 (~{s3} words): Transition to next chapter.",
                ],
                "characters_active": chars[:2],
            })

        return {
            "condition": "C",
            "premise": premise,
            "genre": genre,
            "total_target_words": total_target_words,
            "num_chapters": num_chapters,
            "raw_plan_text": raw_text,
            "chapters": chapters,
            "planning_metadata": {
                "condition": "C",
                "planning_tokens": estimate_token_count(raw_text),
                "planning_latency_s": round(time.time() - t0, 4),
            },
        }


def plan_next_chapter_c(
    history_text: str,
    llm_generate_fn: Optional[Callable[[str, str], str]] = None,
    premise: str = "",
) -> Dict[str, Any]:
    """Track 1 VR-CLI Next-Chapter Planner for Condition C (Sequential Title + Goal)."""
    t0 = time.time()
    tail_excerpt = " ".join(history_text.strip().split()[-600:]) if history_text else premise
    plan_text = ""
    if llm_generate_fn:
        try:
            plan_text = llm_generate_fn(
                "Write a single sequential Chapter Title and Goal for the next chapter.",
                f"Preceding chapters:\n{tail_excerpt}",
            ).strip()
        except Exception:
            plan_text = ""
    if not plan_text:
        chars, kws = extract_premise_keywords_and_characters(tail_excerpt)
        plan_text = f"Chapter Next: Advancing {', '.join(chars[:2])}\nGoal: Resolve active thread around {kws[0] if kws else 'the conflict'}."
    return {
        "condition": "C",
        "plan_text": plan_text,
        "events": [],
        "planning_tokens": estimate_token_count(plan_text),
        "planning_latency_s": round(time.time() - t0, 4),
    }

