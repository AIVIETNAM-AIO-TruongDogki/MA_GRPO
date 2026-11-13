"""
Condition D Planner: Core MA-GRPO Minimum-Structure Proposal (Macro Event Graph ONLY).
Builds causal Event Graph G=(V,E) + Story Arc Alignment via `src.planning`,
with `enable_micro_subevents=False` (skipping SubEventAgent micro beats).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from src.planning import (
    compute_story_arc_alignment,
    ensure_event_ids,
    estimate_token_count,
    extract_premise_keywords_and_characters,
    parse_storywriter_events,
)


NEXT_CHAPTER_MACRO_PROMPT = """Given the preceding story chapters, generate the Macro Event Plan for the NEXT chapter ONLY.
Include Story Arc stage, Setting, Characters, Core Action, Conflict, Plot Twist, and Causal Link to prior chapters.
Follow the exact fields:
Event 1: [Next Chapter Core Event]
Story Arc: [Exposition | Rising Action | Climax | Falling Action | Resolution]
Setting: [Time and Location]
Character: [Active Characters]
Action: [Core plot progression]
Conflict: [Central tension]
Plot Twist: [Key turn or hook]
Depends On: [Prior context elements]
"""


def build_next_chapter_macro_event(
    history_text: str,
    llm_generate_fn: Optional[Callable[[str, str], str]] = None,
    premise: str = "",
) -> tuple[List[Dict[str, Any]], str]:
    """Shared helper for Condition D and E next-chapter Event Graph extraction."""
    tail_excerpt = " ".join(history_text.strip().split()[-600:]) if history_text else premise
    events: List[Dict[str, Any]] = []
    raw_macro = ""
    if llm_generate_fn:
        try:
            raw_macro = llm_generate_fn(
                NEXT_CHAPTER_MACRO_PROMPT,
                f"Story Premise: {premise}\nPreceding Chapters Excerpt:\n{tail_excerpt}",
            )
            events = parse_storywriter_events(raw_macro)
        except Exception:
            events = []

    if not events:
        chars, kws = extract_premise_keywords_and_characters(premise or tail_excerpt)
        focus = kws[0] if kws else "the central crisis"
        events = [
            {
                "event_id": "E1",
                "title": f"Next Escalation on {focus.title()}",
                "story_arc": "Rising Action",
                "setting": "Immediate continuation setting from preceding chapter",
                "character": ", ".join(chars[:2]),
                "action": f"{chars[0]} acts directly on clues left in the previous chapter regarding {focus}.",
                "conflict": f"New resistance blocks {chars[0]}'s immediate objective.",
                "plot_twist": f"A revelation reframes {focus} and forces an immediate choice.",
                "depends_on": [],
            }
        ]
        raw_macro = (
            f"Event 1: {events[0]['title']}\n"
            f"Story Arc: {events[0]['story_arc']}\n"
            f"Setting: {events[0]['setting']}\n"
            f"Character: {events[0]['character']}\n"
            f"Action: {events[0]['action']}\n"
            f"Conflict: {events[0]['conflict']}\n"
            f"Plot Twist: {events[0]['plot_twist']}"
        )

    ensure_event_ids(events)
    return events, raw_macro


def plan_next_chapter_d(
    history_text: str,
    llm_generate_fn: Optional[Callable[[str, str], str]] = None,
    premise: str = "",
) -> Dict[str, Any]:
    """Track 1 VR-CLI Next-Chapter Planner for Condition D (Macro Event ONLY)."""
    t0 = time.time()
    events, raw_macro = build_next_chapter_macro_event(history_text, llm_generate_fn, premise)
    return {
        "condition": "D",
        "plan_text": raw_macro,
        "events": events,
        "story_arc_alignment": compute_story_arc_alignment(events),
        "planning_tokens": estimate_token_count(raw_macro),
        "planning_latency_s": round(time.time() - t0, 4),
    }

