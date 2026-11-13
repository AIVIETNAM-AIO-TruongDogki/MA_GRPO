"""
Condition A: Direct Single-Pass Generation Baseline (Zero Planning).
Prompts the writing model directly to write a full story without any intermediate plan.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from src.pipeline import VLLMWritingPolicy, extract_cot_and_prose
from src.rewards import count_words


def generate_direct_story(
    writing_policy: VLLMWritingPolicy,
    premise: str,
    target_words: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 32768,
) -> str:
    """Single-turn direct generation without planning scaffold."""
    prompt = (
        f"You are an elite literary novelist. Write a complete, cohesive novel-length story "
        f"of approximately {target_words} words directly from this premise.\n\n"
        f"[PREMISE]:\n{premise}\n\n"
        f"Output ONLY the story prose:"
    )

    def _mock_direct() -> str:
        base = (
            f"Inspired by the premise—{premise[:120]}—the protagonist stepped into the unfamiliar threshold. "
            f"Without a segmented multi-chapter blueprint, events unfolded rapidly in one continuous rush toward an early close. "
        )
        sentences = [
            "The atmosphere shifted quickly as initial suspicions gave way to sudden confrontation.",
            "Characters exchanged brief warnings beneath flickering amber lamps while shadows gathered.",
            "Moments later, the central mystery resolved abruptly before its deeper threads could mature.",
        ]
        words = base.split()
        idx = 0
        while len(words) < min(1250, target_words):
            words.extend(sentences[idx % len(sentences)].split())
            idx += 1
        return " ".join(words)

    raw = writing_policy.generate_raw(prompt, max_tokens, temperature, top_p, _mock_direct)
    _, prose = extract_cot_and_prose(raw)
    return prose if prose else raw.strip()


def run_condition_a(
    pipeline: Any,
    premise: str,
    genre: str = "Speculative Fiction",
    total_target_words: int = 10000,
    num_chapters: int = 8,
    perturbation_type: str = None,
    donor_plan: Dict[str, Any] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_cot: bool = True,
) -> Dict[str, Any]:
    """Execute Condition A (Direct Generation without planning)."""
    t_start = time.time()
    t_gen0 = time.time()
    full_text = generate_direct_story(
        writing_policy=pipeline.writing_policy,
        premise=premise,
        target_words=total_target_words,
        temperature=temperature,
        top_p=top_p,
        max_tokens=pipeline.writing_policy.max_model_len,
    )
    gen_lat = round(time.time() - t_gen0, 2)
    return {
        "condition": "A",
        "title": f"Story (Condition A): {genre}",
        "blueprint": {"condition": "A", "premise": premise, "chapters": [], "event_graph": {"events": []}},
        "chapters": [full_text],
        "full_text": full_text,
        "target_words": total_target_words,
        "actual_total_words": count_words(full_text),
        "num_chapters": 1,
        "planning_tokens": 0,
        "planning_latency_s": 0.0,
        "generation_latency_s": gen_lat,
        "elapsed_seconds": round(time.time() - t_start, 2),
        "context_summaries": [],
        "perturbation_metadata": None,
    }

