"""
OutlineAgent: Generates a causal Event Graph G=(V, E) grounded in the story premise,
annotated with Freytag 5-stage Story Arc Alignment (`STORY_ARC_STAGES`).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from .story_arc import (
    EVENT_GENERATION_PROMPT,
    STORY_ARC_STAGES,
    compute_story_arc_alignment,
    ensure_event_ids,
    estimate_token_count,
    extract_premise_keywords_and_characters,
    parse_storywriter_events,
)


class OutlineAgent:
    """
    Outline Agent generating the StoryWriter-style Event Graph G=(V, E) with Story Arc alignment.
    Supports strict dataset mode (raising/quarantining failures without template injection).
    """

    def __init__(
        self,
        llm_generate_fn: Optional[Callable[[str, str], str]] = None,
        strict_mode: bool = False,
        max_retries: int = 3,
    ):
        self.generate_fn = llm_generate_fn
        self.strict_mode = strict_mode
        self.max_retries = max_retries

    def create_event_graph(
        self,
        premise: str,
        genre: str = "Science Fiction",
        num_events: int = 8,
        strict_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Generates Event Graph dictionary based on user premise, annotated with Freytag Story Arcs.
        In strict_mode, refuses generic hard-coded fallback to preserve research dataset validity.
        """
        t0 = time.time()
        is_strict = self.strict_mode if strict_mode is None else strict_mode
        num_events = max(3, num_events)

        if self.generate_fn:
            user_prompt = (
                f"Premise: {premise}\nGenre: {genre}\n"
                f"Generate exactly {num_events} consecutive events spanning the full 5-stage Story Arc "
                f"(Exposition -> Rising Action -> Climax -> Falling Action -> Resolution)."
            )
            for _ in range(self.max_retries):
                try:
                    raw_text = self.generate_fn(EVENT_GENERATION_PROMPT, user_prompt)
                    events = parse_storywriter_events(raw_text)
                    if events and len(events) >= max(3, num_events // 2):
                        ensure_event_ids(events)
                        arc_metrics = compute_story_arc_alignment(events)
                        return {
                            "premise": premise,
                            "genre": genre,
                            "num_events": len(events),
                            "events": events,
                            "story_arc_alignment": arc_metrics,
                            "raw_plan_text": raw_text,
                            "planning_tokens": estimate_token_count(raw_text),
                            "planning_latency_s": round(time.time() - t0, 4),
                            "status": "success",
                            "is_fallback": False,
                        }
                except Exception:
                    pass

        if is_strict:
            raise ValueError(
                f"[OutlineAgent Strict Mode] Failed to generate valid Event Graph for premise: "
                f"'{premise[:80]}...'. Marked for quarantine retry."
            )

        return self._generate_dynamic_events(premise, genre, num_events)

    def _generate_dynamic_events(self, premise: str, genre: str, num_events: int) -> Dict[str, Any]:
        """
        Synthesizes a dynamic Event Graph derived from the premise semantics across
        Freytag's 5 Story Arc stages matching num_events exactly.
        """
        t0 = time.time()
        chars, keywords = extract_premise_keywords_and_characters(premise)
        main_char = chars[0]
        secondary_char = chars[1] if len(chars) > 1 else "Unknown Figure"
        focus_subject = keywords[0] if keywords else "the anomaly"
        premise_snippet = premise.strip().rstrip(".")[:140]

        arc_stages = [
            ("Exposition", "Inception & Revelation", "Initial domain or sanctuary", main_char,
             f"{main_char} encounters the catalyst tied to {focus_subject} ({premise_snippet}).",
             "Hesitation between past comfort and the urge to investigate.",
             f"The discovery of {focus_subject} links directly to {main_char}'s hidden past."),
            ("Rising Action", "Inciting Threshold", "Perimeter boundary or transit zone", f"{main_char}, {secondary_char}",
             f"{main_char} and {secondary_char} cross into contested ground to trace {focus_subject}.",
             "Hostile observers and rising uncertainty impede passage.",
             f"{secondary_char} reveals contradictory motives concerning {focus_subject}."),
            ("Rising Action", "Rising Complications", "Archives or contested outpost", f"{main_char}, {secondary_char}",
             f"Searching for crucial truths required to resolve {focus_subject}.",
             "Alliances fracture as external pursuit closes in.",
             "Recovered evidence points to an unexpected betrayal from within."),
            ("Climax", "Core Confrontation", "The Central Nexus or Sanctum", f"{main_char}, {secondary_char}, Antagonist",
             f"{main_char} confronts the decisive crisis over {focus_subject}.",
             "Irreversible catastrophe threatens both protagonists unless a sacrifice is made.",
             f"{focus_subject} transforms the stakes rather than merely acting as a tool."),
            ("Falling Action", "Falling Aftermath", "The fractured perimeter", main_char,
             f"{main_char} stabilizes the immediate consequences after the confrontation.",
             "Lingering instability tests whether the fragile new truce can endure.",
             f"An unexpected echo of {focus_subject} persists despite the resolution."),
            ("Resolution", "Resolution & Legacy", "Restored horizon at dawn", f"{main_char}, {secondary_char}",
             f"{main_char} and {secondary_char} establish a new equilibrium honoring the transformed reality.",
             "Reconciling personal loss with the resolution achieved.",
             "Resolved: the primary conflict reaches closure while opening a lasting legacy."),
        ]

        events: List[Dict[str, Any]] = []
        raw_blocks: List[str] = []
        for i in range(num_events):
            stage_idx = min(len(arc_stages) - 1, int(round(i * (len(arc_stages) - 1) / max(1, num_events - 1))))
            arc_label, title_base, setting_base, char_base, action_base, conflict_base, twist_base = arc_stages[stage_idx]
            # Guarantee non-decreasing arc stage index in edge rounding cases
            if events:
                prev_rank = STORY_ARC_STAGES.index(events[-1]["story_arc"])
                cur_rank = STORY_ARC_STAGES.index(arc_label)
                if cur_rank < prev_rank:
                    arc_label = events[-1]["story_arc"]

            event_id = f"E{i + 1}"
            raw_text = (
                f"Event {i + 1}: Phase {i + 1}: {title_base}\n"
                f"Story Arc: {arc_label}\n"
                f"Setting: {setting_base} ({genre} context)\n"
                f"Character: {char_base}\n"
                f"Action: {action_base}\n"
                f"Conflict: {conflict_base}\n"
                f"Plot Twist: {twist_base}\n"
                f"Depends On: {'E' + str(i) if i > 0 else 'None'}"
            )
            raw_blocks.append(raw_text)
            events.append({
                "event_id": event_id,
                "title": f"Phase {i + 1}: {title_base}",
                "story_arc": arc_label,
                "setting": f"{setting_base} ({genre} context)",
                "character": char_base,
                "action": action_base,
                "conflict": conflict_base,
                "plot_twist": twist_base,
                "depends_on": [f"E{i}"] if i > 0 else [],
                "status": "pending",
                "raw_text": raw_text,
            })

        full_raw = "\n\n".join(raw_blocks)
        arc_metrics = compute_story_arc_alignment(events)
        return {
            "premise": premise,
            "genre": genre,
            "num_events": len(events),
            "events": events,
            "story_arc_alignment": arc_metrics,
            "raw_plan_text": full_raw,
            "planning_tokens": estimate_token_count(full_raw),
            "planning_latency_s": round(time.time() - t0, 4),
            "status": "dynamic_fallback",
            "is_fallback": True,
        }

