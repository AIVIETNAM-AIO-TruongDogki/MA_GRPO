"""
Stage 1: Macro-Narrative Planning Module for MA-GRPO.
Inspired by StoryWriter's event-driven outline structure:
    Event n: {EVENT_NAME}
    Setting: {SETTING}
    Character: {CHARACTER}
    Action: {ACTION}
    Conflict: {CONFLICT}
    Plot Twist: {PLOT_TWIST}

Partitioned into K chapters with target length L_target = L_total / K.
Uses clean Python dict structures without requiring schema.py.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional


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


class OutlineAgent:
    """
    Outline Agent generating the StoryWriter-style Event Graph.
    """

    def __init__(self, llm_generate_fn: Optional[Callable[[str, str], str]] = None):
        self.generate_fn = llm_generate_fn

    def create_event_graph(
        self,
        premise: str,
        genre: str = "Science Fiction",
        num_events: int = 8,
    ) -> Dict[str, Any]:
        """
        Generates Event Graph dictionary based on user premise.
        """
        if self.generate_fn:
            user_prompt = f"Premise: {premise}\nGenre: {genre}\nPlease generate {num_events} consecutive events."
            raw_text = self.generate_fn(EVENT_GENERATION_PROMPT, user_prompt)
            events = parse_storywriter_events(raw_text)
            if events:
                ensure_event_ids(events)
                return {
                    "premise": premise,
                    "genre": genre,
                    "num_events": len(events),
                    "events": events,
                }

        # Fallback template adhering to StoryWriter structure
        return self._generate_template_events(premise, genre, num_events)

    def _generate_template_events(self, premise: str, genre: str, num_events: int) -> Dict[str, Any]:
        template_text = f"""Event 1: Discovery of the Anomaly
Setting: High-altitude observatory overlooking a stormy coast.
Character: Dr. Elena Vance (Astrophysicist), Dr. Marcus Rowe (Colleague)
Action: Elena detects an unrecorded transmission pulse piercing through electromagnetic storms.
Conflict: Marcus urges caution and secrecy, while Elena insists on immediately alerting the scientific council.
Plot Twist: The signal is encoded with mathematical algorithms that Elena herself designed five years ago.

Event 2: The Syndicate's Ultimatum
Setting: Neon-lit underground archive in Sector 4.
Character: Elena Vance, Agent Kaelen (Syndicate Enforcer)
Action: Elena attempts to retrieve cold-storage backup drives before her access is revoked.
Conflict: Syndicate enforcers ambush the facility, demanding the decryption keys.
Plot Twist: Kaelen reveals that the observatory's data was fabricated by Elena's mentor.

Event 3: The Descent into the Periphery
Setting: Abandoned geothermal transit tunnels beneath the desert.
Character: Elena Vance, Marcus Rowe, Kaelen (reluctant hostage)
Action: The team escapes into the sub-tunnels to reach the transmitter hub.
Conflict: The life support filters fail as subterranean tremors shake the tunnel structure.
Plot Twist: The transmission is not originating from space, but deep within the continental mantle.

Event 4: The Core Confrontation
Setting: The Central Relay Chamber at the subterranean hub.
Character: Elena Vance, Syndicate Director Thorne
Action: Elena connects the core key to the planetary transmitter.
Conflict: Thorne threatens to flood the chamber with coolant if she initiates the broadcast.
Plot Twist: Broadcaster is not a weapon—it is an automated distress call for human consciousness preservation.
"""
        events = parse_storywriter_events(template_text)
        ensure_event_ids(events)
        return {
            "premise": premise,
            "genre": genre,
            "num_events": len(events),
            "events": events,
        }


class SubEventAgent:
    """Expands each macro event into ordered, trackable narrative beats.

    The optional generator is intended to be a thin client for one shared vLLM
    server.  It is never trainable and its deterministic fallback keeps the
    pipeline testable without a model endpoint.
    """

    SYSTEM_PROMPT = """Split each event into ordered sub-events. Return only JSON:
{"sub_events": [{"event_id": "E1", "title": "...", "goal": "...",
"required_facts": ["..."], "exit_state": "..."}]}. Keep twists in their source event order."""

    def __init__(self, llm_generate_fn: Optional[Callable[[str, str], str]] = None):
        self.generate_fn = llm_generate_fn

    def decompose(self, event_graph: Dict[str, Any], per_event: int = 3) -> List[Dict[str, Any]]:
        events = ensure_event_ids(event_graph.get("events", []))
        if self.generate_fn:
            payload = json.dumps(events, ensure_ascii=False)
            try:
                data = json.loads(self.generate_fn(self.SYSTEM_PROMPT, payload))
                sub_events = data.get("sub_events", [])
                if sub_events:
                    return self._normalise(sub_events, events)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

        sub_events: List[Dict[str, Any]] = []
        for event in events:
            beats = [
                ("Arrival", f"Establish {event['setting']} and the immediate stakes."),
                ("Confrontation", event["action"]),
                ("Turn", f"Escalate the conflict and land this turn: {event['plot_twist']}"),
            ][:max(1, per_event)]
            for beat_index, (label, goal) in enumerate(beats, 1):
                sub_events.append({
                    "sub_event_id": f"{event['event_id']}.{beat_index}",
                    "event_id": event["event_id"],
                    "title": f"{event['title']} — {label}",
                    "goal": goal,
                    "required_facts": [event["setting"], event["character"]],
                    "exit_state": event["plot_twist"] if beat_index == len(beats) else f"Advance toward {event['conflict']}",
                    "depends_on": (
                        [f"{event['event_id']}.{beat_index - 1}"]
                        if beat_index > 1
                        else ([f"E{int(event['event_id'][1:]) - 1}.{per_event}"] if event["event_id"] != "E1" else [])
                    ),
                    "status": "pending",
                })
        return sub_events

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
    """Frozen gate that rejects duplicate, premature, or dependency-breaking plans."""

    def review(self, chapter_plan: Dict[str, Any], story_plan: Dict[str, Any]) -> Dict[str, Any]:
        completed = set(story_plan.get("completed_sub_event_ids", []))
        seen = set()
        issues = []
        for item in chapter_plan.get("sub_events", []):
            identifier = item.get("sub_event_id")
            if identifier in completed or identifier in seen:
                issues.append(f"Duplicate sub-event: {identifier}")
            seen.add(identifier)
            missing = set(item.get("depends_on", [])) - completed - seen
            if missing:
                issues.append(f"Unmet dependencies for {identifier}: {sorted(missing)}")
        if chapter_plan.get("target_words", 0) <= 0:
            issues.append("target_words must be positive")
        return {"approved": not issues, "issues": issues, "chapter_plan": chapter_plan}


class PlanningAgent:
    """Stateful rolling planner used between validated context and each chapter.

    ``partition_into_chapters`` remains as a compatibility wrapper, while new
    code should call ``create_story_plan`` → ``plan_next_chapter`` →
    ``mark_chapter_complete`` in a generation loop.
    """

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
            selected = [{"sub_event_id": f"continuity.{chapter_index}", "event_id": "", "title": "Continuity", "goal": "Resolve remaining narrative threads.", "required_facts": [], "exit_state": "Create a transition.", "depends_on": []}]

        first, last = selected[0], selected[-1]

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
            "entry_state": (context_summary or {}).get("plot_summary", "Begin from the established story state."),
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
                story_plan["event_status"][item["event_id"]] = "active"
        story_plan["remaining_target_words"] = max(0, story_plan["remaining_target_words"] - chapter_plan["target_words"])
        story_plan["chapters"].append(chapter_plan)

    def partition_into_chapters(
        self,
        event_graph: Dict[str, Any],
        total_target_words: int = 12000,
        num_chapters: int = 10,
    ) -> Dict[str, Any]:
        """
        Distributes events across K chapters and calculates L_target = L_total / K.
        """
        story_plan = self.create_story_plan(event_graph, total_target_words, num_chapters)
        for _ in range(num_chapters):
            chapter = self.plan_next_chapter(story_plan)
            self.mark_chapter_complete(story_plan, chapter)
        story_plan["chapter_target_words"] = total_target_words // num_chapters
        return story_plan
