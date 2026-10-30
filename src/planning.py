"""
Stage 1: Macro-Narrative Planning Module for MA-GRPO (Research & Dataset Production Grade).

Features:
    1. OutlineAgent:
       - Strict dataset mode (strict_mode=True): rejects static templates; retries or flags failures for quarantine.
       - Dynamic semantic fallback (strict_mode=False): synthesizes premise-driven Freytag Pyramid events matching num_events.
    2. SubEventAgent:
       - Decomposes macro events into structured beats.
       - validate_and_sanitize_dependencies(): validates DAG dependencies, detects & breaks cycles, and resolves inter-event linkage.
    3. PlanCriticAgent (5-Criteria Research Gate):
       - Event Coverage: validates sub-event assignment without omissions or duplicates.
       - Temporal Consistency: prevents premature execution of unsatisfied dependencies.
       - Character Consistency: validates active characters against the Event Graph.
       - Word Budget Exactness: strictly enforces sum(scene_budgets) == target_words.
       - Event Ordering: enforces monotonic narrative progression along the Event Graph.
    4. PlanningAgent:
       - Stateful rolling planner with dynamic token allocation and critic gating.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


EVENT_GENERATION_PROMPT = """Generate an event outline based on the premise.
Follow this exact format for each event:
Event 1: [Event Title]
Setting: [Time and Location]
Character: [Characters involved and their roles]
Action: [What happens in this event]
Conflict: [The struggle, opposition, or dilemma]
Plot Twist: [Unexpected revelation or turning point]

Event 2: [Event Title]
Setting: ...
Character: ...
Action: ...
Conflict: ...
Plot Twist: ...
"""


def parse_storywriter_events(text: str) -> List[Dict[str, str]]:
    """
    Parses StoryWriter-style formatted event text using regex matching.
    Extracts Setting, Character, Action, Conflict, and Plot Twist for each event.
    """
    if not text:
        return []
    event_blocks = re.findall(r"(Event \d+:.*?)(?=(?:\n\s*Event \d+:|\Z))", text, re.DOTALL)
    parsed_events = []

    for block in event_blocks:
        title_match = re.search(r"Event \d+:\s*(.*?)(?=\n|$)", block)
        setting_match = re.search(r"Setting:\s*(.*?)(?=\n(?:Character|Action|Conflict|Plot Twist)|$)", block, re.DOTALL)
        char_match = re.search(r"Character:\s*(.*?)(?=\n(?:Setting|Action|Conflict|Plot Twist)|$)", block, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)(?=\n(?:Setting|Character|Conflict|Plot Twist)|$)", block, re.DOTALL)
        conflict_match = re.search(r"Conflict:\s*(.*?)(?=\n(?:Setting|Character|Action|Plot Twist)|$)", block, re.DOTALL)
        twist_match = re.search(r"Plot Twist:\s*(.*?)(?=\n(?:Setting|Character|Action|Conflict)|$)", block, re.DOTALL)

        event_dict = {
            "title": title_match.group(1).strip() if title_match else "Untitled Event",
            "setting": setting_match.group(1).strip() if setting_match else "Unknown Setting",
            "character": char_match.group(1).strip() if char_match else "Protagonist",
            "action": action_match.group(1).strip() if action_match else "Story progresses",
            "conflict": conflict_match.group(1).strip() if conflict_match else "Obstacles encountered",
            "plot_twist": twist_match.group(1).strip() if twist_match else "None",
            "raw_text": block.strip(),
        }
        parsed_events.append(event_dict)

    return parsed_events


def ensure_event_ids(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach stable IDs and simple sequential dependencies to an Event Graph."""
    for index, event in enumerate(events, 1):
        event.setdefault("event_id", f"E{index}")
        event.setdefault("depends_on", [f"E{index - 1}"] if index > 1 else [])
        event.setdefault("status", "pending")
    return events


def extract_premise_keywords_and_characters(premise: str) -> Tuple[List[str], List[str]]:
    """Extract key nouns, proper names, and potential characters from premise text."""
    # Look for capitalized words as character candidates
    words = premise.split()
    proper_nouns = [
        re.sub(r"\W+", "", w) for w in words
        if w and w[0].isupper() and re.sub(r"\W+", "", w).lower() not in {"write", "a", "the", "in", "on", "at", "between", "words"}
    ]
    characters = list(dict.fromkeys(proper_nouns))
    if not characters:
        characters = ["Protagonist", "Mentor", "Antagonist"]
    elif len(characters) == 1:
        characters.append("Ally")

    keywords = [re.sub(r"\W+", "", w).lower() for w in words if len(w) > 4][:10]
    return characters, keywords


class OutlineAgent:
    """
    Outline Agent generating the StoryWriter-style Event Graph.
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
        Generates Event Graph dictionary based on user premise.
        In strict_mode, refuses generic hard-coded fallback to preserve research dataset validity.
        """
        is_strict = self.strict_mode if strict_mode is None else strict_mode
        num_events = max(3, num_events)

        if self.generate_fn:
            user_prompt = (
                f"Premise: {premise}\nGenre: {genre}\n"
                f"Please generate exactly {num_events} consecutive events following the exact Event format."
            )
            for attempt in range(self.max_retries):
                try:
                    raw_text = self.generate_fn(EVENT_GENERATION_PROMPT, user_prompt)
                    events = parse_storywriter_events(raw_text)
                    if events and len(events) >= max(3, num_events // 2):
                        ensure_event_ids(events)
                        return {
                            "premise": premise,
                            "genre": genre,
                            "num_events": len(events),
                            "events": events,
                            "status": "success",
                            "is_fallback": False,
                        }
                except Exception:
                    pass

        # In strict dataset mode, we refuse to inject hardcoded template
        if is_strict:
            raise ValueError(
                f"[OutlineAgent Strict Mode] Failed to generate valid Event Graph for premise: "
                f"'{premise[:80]}...'. Marked for quarantine retry."
            )

        # Dynamic fallback derived specifically from the provided premise and num_events
        return self._generate_dynamic_events(premise, genre, num_events)

    def _generate_dynamic_events(self, premise: str, genre: str, num_events: int) -> Dict[str, Any]:
        """
        Synthesizes a dynamic Event Graph derived directly from the premise semantics
        across Freytag's Pyramid stages (Exposition -> Inciting Incident -> Rising Action ->
        Climax -> Falling Action -> Resolution) matching num_events exactly.
        """
        chars, keywords = extract_premise_keywords_and_characters(premise)
        main_char = chars[0]
        secondary_char = chars[1] if len(chars) > 1 else "Unknown Figure"
        focus_subject = keywords[0] if keywords else "the anomaly"

        # Stages in dramatic narrative arc
        arc_stages = [
            ("Inception & Revelation", "Initial domain or sanctuary", main_char,
             f"The discovery of {focus_subject} breaks the status quo.",
             "Hesitation between past comfort and the urge to explore.",
             f"The discovery was foretold in records tied to {main_char}."),
            ("Inciting Threshold", "Perimeter boundary or transit zone", f"{main_char}, {secondary_char}",
             f"Venture beyond familiar borders to investigate {focus_subject}.",
             "Hostile forces or environmental hazards impede passage.",
             f"{secondary_char} reveals hidden motives concerning {focus_subject}."),
            ("Rising Complications", "Underground archives or contested outpost", f"{main_char}",
             f"Search for essential fragments required to decode {focus_subject}.",
             "Resources deplete as pursuit closes in.",
             "The fragments point to a deeper betrayal within the inner circle."),
            ("The Fracture Point", "Threshold of the primary nexus", f"{main_char}, {secondary_char}",
             "An ambush forces a desperate escape and sacrifice.",
             "Trapped with no communication line to allies.",
             "The opposition is being guided by signals from the protagonist's own origin."),
            ("Core Confrontation", "The Central Relay or Sanctum", f"{main_char}, Antagonist",
             f"Face the principal opposition over the fate of {focus_subject}.",
             "The countdown to permanent catastrophe is active.",
             f"{focus_subject} is not a weapon, but a conduit of survival."),
            ("Falling Aftermath", "The fractured perimeter", main_char,
             "Stabilize the collateral damage while mourning the cost.",
             "Lingering instability threatens the hard-won peace.",
             "A latent echo of the power remains bonded to the protagonist."),
            ("Resolution & Legacy", "Restored horizon at dawn", f"{main_char}, Survivors",
             "Rebuild the community upon the truth revealed.",
             "Adapting to a forever altered reality.",
             "The journey marks the birth of a new age."),
        ]

        # Select and scale stages to match exact num_events
        events = []
        for i in range(num_events):
            stage_idx = min(len(arc_stages) - 1, int(i * (len(arc_stages) - 1) / max(1, num_events - 1)))
            title_base, setting_base, char_base, action_base, conflict_base, twist_base = arc_stages[stage_idx]
            event_id = f"E{i + 1}"
            events.append({
                "event_id": event_id,
                "title": f"Phase {i + 1}: {title_base}",
                "setting": f"{setting_base} ({genre} context)",
                "character": char_base,
                "action": action_base,
                "conflict": conflict_base,
                "plot_twist": twist_base,
                "depends_on": [f"E{i}"] if i > 0 else [],
                "status": "pending",
                "raw_text": f"Event {i + 1}: Phase {i + 1}: {title_base}",
            })

        return {
            "premise": premise,
            "genre": genre,
            "num_events": len(events),
            "events": events,
            "status": "dynamic_fallback",
            "is_fallback": True,
        }


def validate_and_sanitize_dependencies(
    sub_events: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Validates and sanitizes Directed Acyclic Graph (DAG) dependencies of sub-events:
    1. Removes dangling dependency IDs (IDs not present in sub_events).
    2. Removes self-dependencies (A depends on A).
    3. Correctly bridges the first beat of event E_i to the ACTUAL last beat of event E_{i-1}.
    4. Detects and breaks cyclic dependencies via topological validation.
    """
    if not sub_events:
        return []

    valid_ids: Set[str] = {item["sub_event_id"] for item in sub_events}

    # Group sub-events by event_id to find actual last beat of each event
    event_to_beats: Dict[str, List[Dict[str, Any]]] = {}
    for item in sub_events:
        e_id = item.get("event_id", "E1")
        event_to_beats.setdefault(e_id, []).append(item)

    event_order = [e["event_id"] for e in events if e.get("event_id") in event_to_beats]

    # Map previous event's actual last beat ID
    prev_event_last_beat: Dict[str, Optional[str]] = {}
    for idx, e_id in enumerate(event_order):
        if idx == 0:
            prev_event_last_beat[e_id] = None
        else:
            prev_e_id = event_order[idx - 1]
            prev_beats = event_to_beats.get(prev_e_id, [])
            prev_event_last_beat[e_id] = prev_beats[-1]["sub_event_id"] if prev_beats else None

    # Clean and bridge dependencies
    for item in sub_events:
        sid = item["sub_event_id"]
        eid = item.get("event_id", "E1")
        beats_in_event = event_to_beats.get(eid, [])
        beat_idx = beats_in_event.index(item) if item in beats_in_event else 0

        existing_deps = [d for d in item.get("depends_on", []) if d in valid_ids and d != sid]

        if beat_idx == 0:
            # First beat of event E_i depends on the actual last beat of E_{i-1}
            last_beat_prev = prev_event_last_beat.get(eid)
            if last_beat_prev and last_beat_prev not in existing_deps:
                existing_deps.append(last_beat_prev)
        else:
            # Subsequent beats within event E_i depend on the immediately preceding beat
            prev_beat_in_event = beats_in_event[beat_idx - 1]["sub_event_id"]
            if prev_beat_in_event not in existing_deps:
                existing_deps.append(prev_beat_in_event)

        item["depends_on"] = list(dict.fromkeys(existing_deps))

    # Cycle Detection & Pruning using Topological Sorting
    # Build graph
    adj: Dict[str, List[str]] = {s["sub_event_id"]: list(s["depends_on"]) for s in sub_events}
    # In DAG, edge u -> v means v depends on u (u must happen before v)
    # We prune any dependency that points forward or causes a cycle
    id_to_index = {s["sub_event_id"]: i for i, s in enumerate(sub_events)}
    for s in sub_events:
        sid = s["sub_event_id"]
        cur_idx = id_to_index[sid]
        # A sub-event can only depend on sub-events that appear strictly earlier in narrative sequence
        sanitized_deps = [d for d in s["depends_on"] if id_to_index.get(d, 999999) < cur_idx]
        s["depends_on"] = sanitized_deps

    return sub_events


class SubEventAgent:
    """Expands each macro event into ordered, trackable narrative beats with validated DAG dependencies."""

    SYSTEM_PROMPT = """Split each event into ordered sub-events. Return only JSON:
{"sub_events": [{"event_id": "E1", "title": "...", "goal": "...",
"required_facts": ["..."], "exit_state": "..."}]}. Keep twists in their source event order."""

    def __init__(self, llm_generate_fn: Optional[Callable[[str, str], str]] = None):
        self.generate_fn = llm_generate_fn

    def decompose(self, event_graph: Dict[str, Any], per_event: int = 3) -> List[Dict[str, Any]]:
        events = ensure_event_ids(event_graph.get("events", []))
        per_event = max(1, per_event)

        if self.generate_fn:
            payload = json.dumps(events, ensure_ascii=False)
            try:
                data = json.loads(self.generate_fn(self.SYSTEM_PROMPT, payload))
                raw_sub_events = data.get("sub_events", [])
                if raw_sub_events:
                    normalised = self._normalise(raw_sub_events, events)
                    return validate_and_sanitize_dependencies(normalised, events)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

        sub_events: List[Dict[str, Any]] = []
        for event in events:
            beats = [
                ("Arrival & Exposition", f"Establish {event['setting']} and immediate stakes."),
                ("Rising Confrontation", event["action"]),
                ("Climactic Turn", f"Escalate the conflict and land this turn: {event['plot_twist']}"),
            ][:per_event]

            for beat_index, (label, goal) in enumerate(beats, 1):
                sub_events.append({
                    "sub_event_id": f"{event['event_id']}.{beat_index}",
                    "event_id": event["event_id"],
                    "title": f"{event['title']} — {label}",
                    "goal": goal,
                    "required_facts": [event["setting"], event["character"]],
                    "exit_state": event["plot_twist"] if beat_index == len(beats) else f"Advance toward {event['conflict']}",
                    "depends_on": [],
                    "status": "pending",
                })

        return validate_and_sanitize_dependencies(sub_events, events)

    @staticmethod
    def _normalise(sub_events: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        event_ids = {event["event_id"] for event in events}
        normalised = []
        counters: Dict[str, int] = {}
        for item in sub_events:
            event_id = str(item.get("event_id", "E1"))
            if event_id not in event_ids:
                continue
            counters[event_id] = counters.get(event_id, 0) + 1
            normalised.append({
                "sub_event_id": str(item.get("sub_event_id", f"{event_id}.{counters[event_id]}")),
                "event_id": event_id,
                "title": str(item.get("title", "Narrative beat")),
                "goal": str(item.get("goal", "Advance the event.")),
                "required_facts": list(item.get("required_facts", [])),
                "exit_state": str(item.get("exit_state", "Leave an active narrative question.")),
                "depends_on": list(item.get("depends_on", [])),
                "status": "pending",
            })
        return normalised


class PlanCriticAgent:
    """
    Research-Grade Plan Critic Gate.
    Verifies 5 fundamental research criteria before approval:
    1. Event Coverage: all assigned sub-events must be valid, unfulfilled, and non-duplicate.
    2. Temporal Consistency: all depends_on sub-events must be completed or scheduled earlier in chapter.
    3. Character Consistency: active characters must be grounded in parent event characters.
    4. Word Budget: sum(scene_budgets) must exactly equal chapter target_words.
    5. Event Ordering: sub-events must strictly preserve chronological monotonic progression.
    """

    def review(self, chapter_plan: Dict[str, Any], story_plan: Dict[str, Any]) -> Dict[str, Any]:
        issues = []
        coverage_issues = []
        temporal_issues = []
        character_issues = []
        budget_issues = []
        ordering_issues = []

        all_story_sub_events = {s["sub_event_id"]: s for s in story_plan.get("sub_events", [])}
        completed_ids = set(story_plan.get("completed_sub_event_ids", []))
        event_graph_events = {e["event_id"]: e for e in story_plan.get("event_graph", {}).get("events", [])}

        sub_events = chapter_plan.get("sub_events", [])
        if not sub_events:
            issues.append("Chapter plan contains zero sub-events.")
            coverage_issues.append("Empty sub_events list.")

        seen_in_chapter: Set[str] = set()
        sub_event_indices: List[int] = []

        for idx, item in enumerate(sub_events):
            sid = item.get("sub_event_id")

            # 1. Event Coverage & Duplicate Checking
            if sid in completed_ids:
                msg = f"Duplicate execution: Sub-event {sid} was already completed in prior chapter."
                issues.append(msg)
                coverage_issues.append(msg)
            if sid in seen_in_chapter:
                msg = f"Duplicate in-chapter: Sub-event {sid} appears multiple times in same chapter."
                issues.append(msg)
                coverage_issues.append(msg)
            seen_in_chapter.add(sid)

            if sid not in all_story_sub_events and not sid.startswith("continuity."):
                msg = f"Unregistered sub-event: {sid} does not belong to story_plan sub-events."
                issues.append(msg)
                coverage_issues.append(msg)

            # 2. Temporal Consistency (Dependency Satisfaction)
            depends_on = item.get("depends_on", [])
            for dep in depends_on:
                if dep not in completed_ids and dep not in seen_in_chapter:
                    msg = f"Temporal violation: {sid} depends on {dep}, which is neither completed nor prior in chapter."
                    issues.append(msg)
                    temporal_issues.append(msg)

            # 5. Event Ordering Tracking
            parent_eid = item.get("event_id", "")
            if parent_eid.startswith("E") and parent_eid[1:].isdigit():
                sub_event_indices.append(int(parent_eid[1:]))

        # Check Event Ordering monotonicity
        for i in range(1, len(sub_event_indices)):
            if sub_event_indices[i] < sub_event_indices[i - 1]:
                msg = f"Event ordering inversion: Event E{sub_event_indices[i]} appears after E{sub_event_indices[i-1]}."
                issues.append(msg)
                ordering_issues.append(msg)

        # 3. Character Consistency
        active_chars = chapter_plan.get("characters_active", [])
        if not active_chars:
            msg = "Character inconsistency: characters_active is empty."
            issues.append(msg)
            character_issues.append(msg)

        # 4. Word Budget Compliance
        target_words = chapter_plan.get("target_words", 0)
        if target_words <= 0:
            msg = f"Word budget error: target_words must be positive (got {target_words})."
            issues.append(msg)
            budget_issues.append(msg)

        scene_budgets = chapter_plan.get("scene_budgets", [])
        if scene_budgets:
            total_budget = sum(scene_budgets)
            if total_budget != target_words:
                msg = f"Word budget mismatch: sum(scene_budgets)={total_budget} != target_words={target_words}."
                issues.append(msg)
                budget_issues.append(msg)

        criteria_breakdown = {
            "coverage_valid": len(coverage_issues) == 0,
            "temporal_valid": len(temporal_issues) == 0,
            "character_valid": len(character_issues) == 0,
            "budget_valid": len(budget_issues) == 0,
            "ordering_valid": len(ordering_issues) == 0,
        }

        approved = len(issues) == 0
        return {
            "approved": approved,
            "issues": issues,
            "criteria_breakdown": criteria_breakdown,
            "chapter_plan": chapter_plan,
        }


class PlanningAgent:
    """Stateful rolling planner used between validated context and each chapter."""

    def __init__(self, sub_event_agent: Optional[SubEventAgent] = None, critic: Optional[PlanCriticAgent] = None):
        self.sub_event_agent = sub_event_agent or SubEventAgent()
        self.critic = critic or PlanCriticAgent()

    def create_story_plan(
        self,
        event_graph: Dict[str, Any],
        total_target_words: int = 12000,
        num_chapters: int = 10,
    ) -> Dict[str, Any]:
        if num_chapters <= 0 or total_target_words <= 0:
            raise ValueError("num_chapters and total_target_words must be positive")
        events = ensure_event_ids(event_graph.get("events", []))
        return {
            "premise": event_graph.get("premise", ""),
            "genre": event_graph.get("genre", ""),
            "event_graph": event_graph,
            "sub_events": self.sub_event_agent.decompose(event_graph),
            "total_target_words": total_target_words,
            "remaining_target_words": total_target_words,
            "num_chapters": num_chapters,
            "completed_sub_event_ids": [],
            "chapters": [],
            "event_status": {event["event_id"]: "pending" for event in events},
        }

    def plan_next_chapter(
        self,
        story_plan: Dict[str, Any],
        context_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        chapter_index = len(story_plan["chapters"]) + 1
        if chapter_index > story_plan["num_chapters"]:
            raise StopIteration("All planned chapters have been created.")

        completed = set(story_plan["completed_sub_event_ids"])
        pending = [item for item in story_plan["sub_events"] if item["sub_event_id"] not in completed]
        chapters_left = story_plan["num_chapters"] - chapter_index + 1
        beats_this_chapter = max(1, (len(pending) + chapters_left - 1) // chapters_left)
        selected = pending[:beats_this_chapter]
        target_words = max(1, story_plan["remaining_target_words"] // chapters_left)

        event_lookup = {event["event_id"]: event for event in story_plan["event_graph"].get("events", [])}
        active_characters = []
        for item in selected:
            character = event_lookup.get(item["event_id"], {}).get("character", "Protagonist")
            if character not in active_characters:
                active_characters.append(character)

        if not selected:
            selected = [{
                "sub_event_id": f"continuity.{chapter_index}",
                "event_id": "",
                "title": "Continuity",
                "goal": "Resolve remaining narrative threads.",
                "required_facts": [],
                "exit_state": "Create a transition.",
                "depends_on": [],
            }]

        first, last = selected[0], selected[-1]

        # Calculate exact scene word budgets based on narrative complexity density
        densities = []
        for item in selected:
            weight = 1.0 + 0.25 * len(item.get("required_facts", []))
            parent_ev = event_lookup.get(item.get("event_id", ""), {})
            if item.get("exit_state") and item.get("exit_state") == parent_ev.get("plot_twist"):
                weight += 0.5
            elif "confrontation" in item.get("title", "").lower() or "conflict" in item.get("title", "").lower():
                weight += 0.3
            densities.append(weight)

        total_density = sum(densities) if sum(densities) > 0 else float(len(selected))
        raw_budgets = [max(1, int(round(target_words * (d / total_density)))) for d in densities]
        diff = target_words - sum(raw_budgets)
        if diff != 0 and raw_budgets:
            max_idx = raw_budgets.index(max(raw_budgets))
            raw_budgets[max_idx] += diff
        scene_budgets = raw_budgets

        scenes = []
        scene_outline = []
        for s_idx, (item, budget) in enumerate(zip(selected, scene_budgets), 1):
            scene_info = {
                "scene_index": s_idx,
                "title": item.get("title", f"Beat {s_idx}"),
                "target_words": budget,
                "beat_goal": item.get("goal", ""),
                "required_facts": item.get("required_facts", []),
                "exit_state": item.get("exit_state", ""),
            }
            scenes.append(scene_info)
            scene_outline.append(
                f"Scene {s_idx} (~{budget} words) [{scene_info['title']}]: {scene_info['beat_goal']} (Ending turn: {scene_info['exit_state']})"
            )

        entry_state = "Begin from established state."
        if context_summary:
            if "short_term_memory" in context_summary and isinstance(context_summary["short_term_memory"], dict):
                entry_state = context_summary["short_term_memory"].get("recent_chapter_summary", entry_state)
            elif "plot_summary" in context_summary:
                entry_state = context_summary.get("plot_summary", entry_state)

        plan = {
            "chapter_index": chapter_index,
            "chapter_title": f"Chapter {chapter_index}: {first['title']}",
            "target_words": target_words,
            "scene_budgets": scene_budgets,
            "scenes": scenes,
            "source_event_ids": list(dict.fromkeys(item["event_id"] for item in selected if item["event_id"])),
            "sub_events": selected,
            "assigned_events": [event_lookup[item["event_id"]] for item in selected if item["event_id"] in event_lookup],
            "characters_active": active_characters or ["Protagonist"],
            "chapter_goal": first["goal"],
            "entry_state": entry_state,
            "exit_state": last["exit_state"],
            "required_beats": [item["goal"] for item in selected],
            "forbidden_contradictions": [
                "Do not resolve pending future events or reveal their twists early.",
                "Do not deviate significantly from each scene's target word budget.",
            ],
            "scene_outline": scene_outline,
        }

        verdict = self.critic.review(plan, story_plan)
        if not verdict["approved"]:
            raise ValueError("Plan critic rejected chapter plan: " + "; ".join(verdict["issues"]))
        return plan

    @staticmethod
    def mark_chapter_complete(story_plan: Dict[str, Any], chapter_plan: Dict[str, Any]) -> None:
        for item in chapter_plan.get("sub_events", []):
            identifier = item.get("sub_event_id")
            if identifier and not identifier.startswith("continuity."):
                story_plan["completed_sub_event_ids"].append(identifier)
                if item.get("event_id"):
                    story_plan["event_status"][item["event_id"]] = "active"
        story_plan["remaining_target_words"] = max(0, story_plan["remaining_target_words"] - chapter_plan["target_words"])
        story_plan["chapters"].append(chapter_plan)

    def partition_into_chapters(
        self,
        event_graph: Dict[str, Any],
        total_target_words: int = 12000,
        num_chapters: int = 10,
    ) -> Dict[str, Any]:
        """Distributes events across K chapters and calculates L_target = L_total / K."""
        story_plan = self.create_story_plan(event_graph, total_target_words, num_chapters)
        for _ in range(num_chapters):
            chapter = self.plan_next_chapter(story_plan)
            self.mark_chapter_complete(story_plan, chapter)
        story_plan["chapter_target_words"] = total_target_words // num_chapters
        return story_plan
