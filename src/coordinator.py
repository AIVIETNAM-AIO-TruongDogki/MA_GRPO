"""
Stage 2: Context Coordination for MA-GRPO (ReIO with Two-Tier Memory).

Architecture:
    Coordinator Agent (ReIO with Two-Tier Memory):
      - long_term_memory: retains cumulative major milestones, character states,
        and unresolved plot threads across all past chapters.
      - short_term_memory: captures detailed summary and exact recent tail excerpt
        (last 180 words) of the immediately preceding chapter.
      - Falls back to structured state extraction strictly when no LLM callback is
        configured or JSON parsing fails.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional


COORDINATOR_TWO_TIER_PROMPT = """You are the Context Coordinator (ReIO) for a multi-agent long-form novel system.
Condense the narrative history into a structured Two-Tier Context State (<4000 tokens) for the writing agent.

Respond strictly with valid JSON conforming to this schema:
{
  "long_term_memory": {
    "major_events": ["Key milestones achieved across all past chapters"],
    "character_states": {
      "CharacterName": "Current physical location, emotional state, and active inventory/bonds"
    },
    "unresolved_plot_threads": ["Open conflicts and mysteries that remain unanswered"],
    "revealed_secrets": ["Major revelations or twists that occurred so far"]
  },
  "short_term_memory": {
    "recent_chapter_summary": "Detailed, factual summary of the immediately preceding chapter (<300 words)",
    "immediate_turning_point": "The exact transition or dilemma occurring at the end of the last chapter"
  }
}
"""


def clean_json_response(raw_text: str) -> str:
    """Extracts JSON string from potential markdown code fences."""
    raw_text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if match:
        return match.group(1).strip()
    return raw_text


class CoordinatorAgent:
    """
    Coordinator Agent (ReIO with Two-Tier Memory).
    Produces compact two-tier context states s_i (<4000 tokens) directly from
    narrative history without external validator overhead.
    """

    def __init__(
        self,
        llm_generate_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.generate_fn = llm_generate_fn

    def condense_context(
        self,
        next_chapter_index: int,
        past_chapters: List[str],
        past_plans: List[Dict[str, Any]],
        characters: Optional[List[str]] = None,
        previous_context: Optional[Dict[str, Any]] = None,
        next_chapter_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produces two-tier context summary s_i for chapter i."""
        characters = characters or ["Protagonist"]

        if not past_chapters or next_chapter_index == 1:
            return {
                "chapter_index": 1,
                "completed_chapters": 0,
                "long_term_memory": {
                    "major_events": ["Narrative begins."],
                    "character_states": {c: "Present at story inception." for c in characters},
                    "unresolved_plot_threads": ["Introduction of core premise."],
                    "revealed_secrets": [],
                },
                "short_term_memory": {
                    "recent_chapter_index": 0,
                    "recent_chapter_summary": "Initial world-building and character baseline.",
                    "recent_tail_excerpt": "",
                    "immediate_turning_point": "Embarking on primary mission.",
                },
                "plot_summary": "Initial world-building and character baseline.",
                "character_statuses": [f"{c}: Present at story inception." for c in characters],
                "pending_threads": ["Introduction of core premise."],
                "recent_tail_excerpt": "",
            }

        prev_text = past_chapters[-1]
        recent_tail = " ".join(prev_text.strip().split()[-180:])

        return self._generate_two_tier_summary(
            next_chapter_index=next_chapter_index,
            past_chapters=past_chapters,
            past_plans=past_plans,
            characters=characters,
            recent_tail=recent_tail,
            previous_context=previous_context,
            next_chapter_plan=next_chapter_plan,
        )

    def _generate_two_tier_summary(
        self,
        next_chapter_index: int,
        past_chapters: List[str],
        past_plans: List[Dict[str, Any]],
        characters: List[str],
        recent_tail: str,
        previous_context: Optional[Dict[str, Any]] = None,
        next_chapter_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calls LLM to synthesize two-tier memory, or derives it structurally when offline/parse-failed."""
        if self.generate_fn:
            user_prompt = f"""Total chapters completed: {len(past_chapters)}.
Upcoming chapter index: {next_chapter_index}.

[LAST CHAPTER PROSE (TAIL)]:
{past_chapters[-1][-2500:]}

[LAST CHAPTER PLAN GOAL]:
{(past_plans[-1] if past_plans else {}).get('chapter_goal', '')}
[LAST CHAPTER EXIT STATE]:
{(past_plans[-1] if past_plans else {}).get('exit_state', '')}

Synthesize Two-Tier Context State matching the JSON schema.
"""
            try:
                raw_res = self.generate_fn(COORDINATOR_TWO_TIER_PROMPT, user_prompt)
                data = json.loads(clean_json_response(raw_res))
                long_mem = data.get("long_term_memory", {})
                short_mem = data.get("short_term_memory", {})
                short_mem["recent_chapter_index"] = len(past_chapters)
                short_mem["recent_tail_excerpt"] = recent_tail

                plot_summary = short_mem.get("recent_chapter_summary", "")
                char_statuses = [
                    f"{k}: {v}" for k, v in long_mem.get("character_states", {}).items()
                ] or [f"{c}: Active" for c in characters]
                pending = long_mem.get("unresolved_plot_threads", [])

                return {
                    "chapter_index": next_chapter_index,
                    "completed_chapters": len(past_chapters),
                    "long_term_memory": long_mem,
                    "short_term_memory": short_mem,
                    "plot_summary": plot_summary,
                    "character_statuses": char_statuses,
                    "pending_threads": pending,
                    "recent_tail_excerpt": recent_tail,
                }
            except Exception:
                pass

        return self._structured_state_extraction(
            next_chapter_index, past_chapters, past_plans, characters, recent_tail, previous_context, next_chapter_plan
        )

    def _structured_state_extraction(
        self,
        next_chapter_index: int,
        past_chapters: List[str],
        past_plans: List[Dict[str, Any]],
        characters: List[str],
        recent_tail: str,
        previous_context: Optional[Dict[str, Any]] = None,
        next_chapter_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Factual, non-generic structured state extraction directly from completed plans,
        actual narrative exit states, and cumulative long-term facts.
        """
        major_milestones = []
        for p in past_plans:
            title = p.get("chapter_title", f"Chapter {p.get('chapter_index')}")
            goal = p.get("chapter_goal", "Executed narrative sequence")
            exit_s = p.get("exit_state", "")
            major_milestones.append(f"{title}: {goal} -> Result: {exit_s}")

        last_plan = past_plans[-1] if past_plans else {}
        recent_summary = (
            f"In Chapter {last_plan.get('chapter_index', len(past_chapters))}, the protagonists "
            f"pursued '{last_plan.get('chapter_goal', 'their objective')}', culminating in: "
            f"'{last_plan.get('exit_state', 'active transition point')}'."
        )

        prev_long_mem = (previous_context or {}).get("long_term_memory", {})
        prior_char_states = dict(prev_long_mem.get("character_states", {}))
        for c in characters:
            prior_char_states[c] = f"Active at conclusion of Chapter {len(past_chapters)}."

        open_threads = list(prev_long_mem.get("unresolved_plot_threads", []))
        if last_plan.get("exit_state") and last_plan["exit_state"] not in open_threads:
            open_threads.append(f"Consequence of: {last_plan['exit_state']}")

        long_term_mem = {
            "major_events": major_milestones,
            "character_states": prior_char_states,
            "unresolved_plot_threads": open_threads[-5:],
            "revealed_secrets": list(prev_long_mem.get("revealed_secrets", [])),
        }

        short_term_mem = {
            "recent_chapter_index": len(past_chapters),
            "recent_chapter_summary": recent_summary,
            "recent_tail_excerpt": recent_tail,
            "immediate_turning_point": last_plan.get("exit_state", "Entering next phase."),
        }

        return {
            "chapter_index": next_chapter_index,
            "completed_chapters": len(past_chapters),
            "long_term_memory": long_term_mem,
            "short_term_memory": short_term_mem,
            "plot_summary": recent_summary,
            "character_statuses": [f"{k}: {v}" for k, v in prior_char_states.items()],
            "pending_threads": open_threads[-5:],
            "recent_tail_excerpt": recent_tail,
        }
