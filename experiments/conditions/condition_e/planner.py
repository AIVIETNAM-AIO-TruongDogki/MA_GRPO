"""
Condition E Planner: Full Two-Level Hierarchical Planning (Macro Event Graph + Micro SubEvents).
Tests whether micro-level sub-event decomposition yields measurable gains over Condition D.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from experiments.conditions.condition_d.planner import build_next_chapter_macro_event
from src.planning import SubEventAgent, compute_story_arc_alignment, estimate_token_count


def plan_next_chapter_e(
    history_text: str,
    llm_generate_fn: Optional[Callable[[str, str], str]] = None,
    premise: str = "",
) -> Dict[str, Any]:
    """Track 1 VR-CLI Next-Chapter Planner for Condition E (Macro Event + Micro Beats)."""
    t0 = time.time()
    events, raw_macro = build_next_chapter_macro_event(history_text, llm_generate_fn, premise)
    sub_agent = SubEventAgent(llm_generate_fn=llm_generate_fn)
    sub_events = sub_agent.decompose({"events": events}, per_event=3)
    sub_lines = [f"- Beat {s['sub_event_id']} ({s['title']}): {s['goal']} Exit: {s['exit_state']}" for s in sub_events]
    plan_text = raw_macro + "\n\nMicro Sub-Events:\n" + "\n".join(sub_lines)
    return {
        "condition": "E",
        "plan_text": plan_text,
        "events": events,
        "sub_events": sub_events,
        "story_arc_alignment": compute_story_arc_alignment(events),
        "planning_tokens": estimate_token_count(plan_text),
        "planning_latency_s": round(time.time() - t0, 4),
    }

