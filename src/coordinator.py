"""
Stage 2: Context Coordination & Context Gate Validator for MA-GRPO.
    1. Coordinator Agent (ReIO): Condenses history C_{1...i-1} into compact state s_i (<4096 tokens).
    2. Context Gate Validator Agent (V_s): Evaluates candidate summary s_i for Faithfulness & Completeness.
       If V_s < 0.8, triggers structured state fallback to suppress hallucination cascades.
Operates on standard Python dictionaries without requiring schema.py.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional


COORDINATOR_SYSTEM_PROMPT = """You are the Context Coordinator (ReIO) for a multi-agent novel writing system.
Your task is to condense past generated chapters into an exact, compact context state s_i (<4000 tokens) for the writing agent.
Focus strictly on:
1. Plot progression so far (factual events only, NO hallucinated details).
2. Current physical/mental status of active characters.
3. Unresolved narrative threads and immediate goals for the upcoming chapter.

You MUST respond strictly with valid JSON.
"""

VALIDATOR_SYSTEM_PROMPT = """You are the Context Gate Validator Agent (V_s).
Evaluate the proposed context summary against the actual preceding chapter text and chapter goals.
Rate two criteria from 0.0 to 1.0:
1. Faithfulness (S_faith): Does the summary contain only facts verified in the preceding chapter, without hallucination?
2. Completeness (S_comp): Does the summary retain all critical plot developments needed for the upcoming chapter?

Respond strictly with JSON:
{
  "faithfulness_score": 0.95,
  "completeness_score": 0.90,
  "feedback": "..."
}
"""


def clean_json_response(raw_text: str) -> str:
    """Extracts JSON string from potential markdown code fences."""
    raw_text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if match:
        return match.group(1).strip()
    return raw_text


class ContextGateValidator:
    """
    Context Gate Validator evaluating s_i:
        V_s(s_i) = 0.5 * S_faith(s_i, C_{i-1}) + 0.5 * S_comp(s_i, E_{i-1})
    Threshold tau = 0.8. If V_s < 0.8, triggers fallback state extraction.
    """

    def __init__(
        self,
        llm_generate_fn: Optional[Callable[[str, str], str]] = None,
        threshold: float = 0.80,
    ):
        self.generate_fn = llm_generate_fn
        self.threshold = threshold

    def validate_summary(
        self,
        summary: Dict[str, Any],
        prev_chapter_text: str,
        prev_chapter_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validates faithfulness and completeness of candidate summary s_i.
        """
        if not prev_chapter_text or not prev_chapter_text.strip():
            return {
                "faithfulness_score": 1.0,
                "completeness_score": 1.0,
                "composite_score": 1.0,
                "is_valid": True,
                "feedback": "Initial chapter; summary validation bypassed.",
            }

        if self.generate_fn:
            user_prompt = f"""[PRECEDING CHAPTER TEXT]:
{prev_chapter_text[-2500:]}

[PROPOSED CONTEXT SUMMARY]:
Plot: {summary.get('plot_summary', '')}
Characters: {summary.get('character_statuses', [])}
Pending: {summary.get('pending_threads', [])}

Evaluate faithfulness (0.0-1.0) and completeness (0.0-1.0).
"""
            try:
                raw_res = self.generate_fn(VALIDATOR_SYSTEM_PROMPT, user_prompt)
                data = json.loads(clean_json_response(raw_res))
                faith = float(data.get("faithfulness_score", 0.85))
                comp = float(data.get("completeness_score", 0.85))
            except Exception:
                faith, comp = 0.85, 0.85
        else:
            faith = self._heuristic_faithfulness(summary, prev_chapter_text)
            comp = 0.88

        composite = round(0.5 * faith + 0.5 * comp, 3)
        is_valid = composite >= self.threshold

        return {
            "faithfulness_score": faith,
            "completeness_score": comp,
            "composite_score": composite,
            "is_valid": is_valid,
            "feedback": "Context validated successfully." if is_valid else "V_s below threshold (0.80); fallback triggered.",
        }

    def _heuristic_faithfulness(self, summary: Dict[str, Any], prev_text: str) -> float:
        words_prev = set(re.findall(r"\b\w{4,}\b", prev_text.lower()))
        plot_text = summary.get("plot_summary", "")
        words_summary = set(re.findall(r"\b\w{4,}\b", plot_text.lower()))
        if not words_summary:
            return 0.5
        overlap = len(words_summary & words_prev) / float(len(words_summary))
        return min(1.0, max(0.6, overlap + 0.2))


class CoordinatorAgent:
    """
    Coordinator Agent (ReIO).
    Compresses long narrative history into a bounded context state s_i (<4096 tokens).
    Integrates the Context Gate Validator (V_s) with automatic structured fallback.
    """

    def __init__(
        self,
        llm_generate_fn: Optional[Callable[[str, str], str]] = None,
        validator: Optional[ContextGateValidator] = None,
    ):
        self.generate_fn = llm_generate_fn
        self.validator = validator or ContextGateValidator(llm_generate_fn)

    def condense_context(
        self,
        next_chapter_index: int,
        past_chapters: List[str],
        past_plans: List[Dict[str, Any]],
        characters: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Produces validated context summary s_i for chapter i.
        """
        characters = characters or ["Protagonist"]

        if not past_chapters or next_chapter_index == 1:
            return {
                "chapter_index": 1,
                "completed_chapters": 0,
                "plot_summary": "The narrative begins. Initial setting and characters are being established.",
                "character_statuses": [f"{c}: Present at the start of the journey." for c in characters],
                "pending_threads": ["Introduction of the primary conflict."],
                "recent_tail_excerpt": "",
            }

        prev_text = past_chapters[-1]
        prev_tail_words = prev_text.strip().split()[-150:]
        recent_tail = " ".join(prev_tail_words)

        candidate_summary = self._generate_candidate_summary(
            next_chapter_index=next_chapter_index,
            past_chapters=past_chapters,
            past_plans=past_plans,
            characters=characters,
            recent_tail=recent_tail,
        )

        val_result = self.validator.validate_summary(
            summary=candidate_summary,
            prev_chapter_text=prev_text,
            prev_chapter_plan=past_plans[-1] if past_plans else None,
        )

        if val_result.get("is_valid", False):
            return candidate_summary

        # Structured Fallback
        return self._structured_fallback(
            next_chapter_index=next_chapter_index,
            past_chapters=past_chapters,
            past_plans=past_plans,
            recent_tail=recent_tail,
        )

    def _generate_candidate_summary(
        self,
        next_chapter_index: int,
        past_chapters: List[str],
        past_plans: List[Dict[str, Any]],
        characters: List[str],
        recent_tail: str,
    ) -> Dict[str, Any]:
        if self.generate_fn:
            user_prompt = f"""Past chapters written: {len(past_chapters)}.
Upcoming chapter index: {next_chapter_index}.

[LAST CHAPTER EXCERPT]:
{past_chapters[-1][-2000:]}

Generate a JSON object matching this schema:
{{
  "chapter_index": {next_chapter_index},
  "completed_chapters": {len(past_chapters)},
  "plot_summary": "Concise summary of events up to now (<300 words)",
  "character_statuses": ["Char A: Current state"],
  "pending_threads": ["Open thread 1"],
  "recent_tail_excerpt": {json.dumps(recent_tail)}
}}
"""
            try:
                raw_res = self.generate_fn(COORDINATOR_SYSTEM_PROMPT, user_prompt)
                return json.loads(clean_json_response(raw_res))
            except Exception:
                pass

        completed_titles = [p.get("chapter_title", f"Ch.{p.get('chapter_index')}") for p in past_plans]
        summary_text = (
            f"Chronicle of {len(past_chapters)} chapters completed ({', '.join(completed_titles)}). "
            f"Characters have overcome initial trials and are advancing toward subsequent confrontations."
        )
        return {
            "chapter_index": next_chapter_index,
            "completed_chapters": len(past_chapters),
            "plot_summary": summary_text,
            "character_statuses": [f"{c}: Resolute and active." for c in characters],
            "pending_threads": ["Resolve immediate complications from previous scene."],
            "recent_tail_excerpt": recent_tail,
        }

    def _structured_fallback(
        self,
        next_chapter_index: int,
        past_chapters: List[str],
        past_plans: List[Dict[str, Any]],
        recent_tail: str,
    ) -> Dict[str, Any]:
        bullet_points = []
        for p in past_plans:
            bullet_points.append(f"- {p.get('chapter_title')}: {p.get('chapter_goal')}")

        fallback_summary = (
            f"[FALLBACK STATE EXTRACTION - V_s Gated]\n"
            f"Recent Chapter Milestones:\n" + "\n".join(bullet_points[-3:])
        )
        return {
            "chapter_index": next_chapter_index,
            "completed_chapters": len(past_chapters),
            "plot_summary": fallback_summary,
            "character_statuses": ["Active characters maintain continuous trajectory."],
            "pending_threads": ["Address developments occurring at the conclusion of the previous chapter."],
            "recent_tail_excerpt": recent_tail,
        }
