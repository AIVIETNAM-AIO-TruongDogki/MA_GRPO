"""
Condition B Runner: Unstructured Overview + Sliding Tail Multi-Segment Generation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from experiments.perturbation import MacroPlanPerturbator
from src.pipeline import VLLMWritingPolicy, extract_cot_and_prose
from src.rewards import count_words
from .planner import UnstructuredPlanner


def generate_unstructured_segment(
    writing_policy: VLLMWritingPolicy,
    premise: str,
    unstructured_plan: str,
    recent_tail: str,
    part_index: int,
    num_parts: int,
    target_words: int = 1200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2200,
) -> str:
    """Condition B: Multi-segment generation guided only by unstructured prose summary + sliding tail."""
    tail_block = f"\n[IMMEDIATE PRECEDING TEXT]:\n{recent_tail}\n" if recent_tail else ""
    prompt = (
        f"You are a literary novelist writing Part {part_index} of {num_parts}.\n"
        f"Target length for this part: Approximately {target_words} words.\n\n"
        f"[PREMISE]:\n{premise}\n\n"
        f"[UNSTRUCTURED STORY OVERVIEW]:\n{unstructured_plan}\n"
        f"{tail_block}\n"
        f"Continue writing immersive narrative prose directly from where the story stands:"
    )
    fake_plan = {
        "chapter_title": f"Part {part_index}",
        "chapter_goal": unstructured_plan,
        "characters_active": ["Protagonist"],
        "target_words": target_words,
        "scenes": [{"scene_index": 1, "target_words": target_words, "beat_goal": unstructured_plan[:120]}],
    }
    raw = writing_policy.generate_raw(
        prompt,
        max_tokens,
        temperature,
        top_p,
        lambda: writing_policy._mock_prose(fake_plan, {"plot_summary": unstructured_plan}),
    )
    _, prose = extract_cot_and_prose(raw)
    return prose if prose else raw.strip()


def run_condition_b(
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
    """Execute Condition B (Unstructured text plan + sliding tail)."""
    t_start = time.time()
    perturbation_metadata: Optional[Dict[str, Any]] = None

    planner_b = UnstructuredPlanner(llm_generate_fn=pipeline.planner_llm_fn)
    blueprint = planner_b.create_story_plan(
        premise=premise,
        genre=genre,
        total_target_words=total_target_words,
        num_chapters=num_chapters,
    )
    if perturbation_type:
        blueprint, perturbation_metadata = MacroPlanPerturbator.apply_to_blueprint(
            blueprint, perturbation_type, donor_plan=donor_plan
        )

    t_gen0 = time.time()
    generated_chapters: List[str] = []
    recent_tail = ""
    for ch in blueprint["chapters"]:
        seg = generate_unstructured_segment(
            writing_policy=pipeline.writing_policy,
            premise=premise,
            unstructured_plan=ch.get("chapter_goal") or blueprint.get("unstructured_plan", ""),
            recent_tail=recent_tail,
            part_index=ch["chapter_index"],
            num_parts=num_chapters,
            target_words=ch["target_words"],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max(1600, int(ch["target_words"] * 1.45)),
        )
        generated_chapters.append(seg)
        recent_tail = " ".join(seg.split()[-180:])

    divider = "\n\n" + ("=" * 80) + "\n\n"
    full_text = divider.join(generated_chapters)
    return {
        "condition": "B",
        "title": f"Story (Condition B): {genre}",
        "blueprint": blueprint,
        "chapters": generated_chapters,
        "full_text": full_text,
        "target_words": total_target_words,
        "actual_total_words": count_words(full_text),
        "num_chapters": num_chapters,
        "planning_tokens": blueprint["planning_metadata"]["planning_tokens"],
        "planning_latency_s": blueprint["planning_metadata"]["planning_latency_s"],
        "generation_latency_s": round(time.time() - t_gen0, 2),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "context_summaries": [],
        "perturbation_metadata": perturbation_metadata,
    }

