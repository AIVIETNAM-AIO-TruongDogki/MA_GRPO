"""
SubEventAgent: Causal micro-beat decomposition with DAG dependency validation.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Set

from .story_arc import ensure_event_ids, estimate_token_count


def validate_and_sanitize_dependencies(
    sub_events: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Validates and sanitizes Directed Acyclic Graph (DAG) dependencies of sub-events:
    1. Removes dangling dependency IDs (IDs not present in sub_events).
    2. Removes self-dependencies (A depends on A).
    3. Correctly bridges the first beat of event E_i to the ACTUAL last beat of event E_{i-1}.
    4. Detects and breaks cyclic dependencies via topological sequence pruning.
    """
    if not sub_events:
        return []

    valid_ids: Set[str] = {item["sub_event_id"] for item in sub_events}

    event_to_beats: Dict[str, List[Dict[str, Any]]] = {}
    for item in sub_events:
        e_id = item.get("event_id", "E1")
        event_to_beats.setdefault(e_id, []).append(item)

    event_order = [e["event_id"] for e in events if e.get("event_id") in event_to_beats]

    prev_event_last_beat: Dict[str, Optional[str]] = {}
    for idx, e_id in enumerate(event_order):
        if idx == 0:
            prev_event_last_beat[e_id] = None
        else:
            prev_e_id = event_order[idx - 1]
            prev_beats = event_to_beats.get(prev_e_id, [])
            prev_event_last_beat[e_id] = prev_beats[-1]["sub_event_id"] if prev_beats else None

    for item in sub_events:
        sid = item["sub_event_id"]
        eid = item.get("event_id", "E1")
        beats_in_event = event_to_beats.get(eid, [])
        beat_idx = beats_in_event.index(item) if item in beats_in_event else 0

        existing_deps = [d for d in item.get("depends_on", []) if d in valid_ids and d != sid]

        if beat_idx == 0:
            last_beat_prev = prev_event_last_beat.get(eid)
            if last_beat_prev and last_beat_prev not in existing_deps:
                existing_deps.append(last_beat_prev)
        else:
            prev_beat_in_event = beats_in_event[beat_idx - 1]["sub_event_id"]
            if prev_beat_in_event not in existing_deps:
                existing_deps.append(prev_beat_in_event)

        item["depends_on"] = list(dict.fromkeys(existing_deps))

    id_to_index = {s["sub_event_id"]: i for i, s in enumerate(sub_events)}
    for s in sub_events:
        sid = s["sub_event_id"]
        cur_idx = id_to_index[sid]
        s["depends_on"] = [d for d in s["depends_on"] if id_to_index.get(d, 999999) < cur_idx]

    return sub_events


class SubEventAgent:
    """Expands each macro event into ordered, trackable micro narrative beats (Condition E)."""

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
                raw_resp = self.generate_fn(self.SYSTEM_PROMPT, payload)
                data = json.loads(raw_resp)
                raw_sub_events = data.get("sub_events", [])
                if raw_sub_events:
                    normalised = self._normalise(raw_sub_events, events)
                    for item in normalised:
                        item["generation_tokens"] = max(1, estimate_token_count(raw_resp) // max(1, len(normalised)))
                    return validate_and_sanitize_dependencies(normalised, events)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

        sub_events: List[Dict[str, Any]] = []
        for event in events:
            beats = [
                ("Arrival & Exposition", f"Establish {event['setting']} and immediate stakes for {event['character']}."),
                ("Rising Confrontation", event["action"]),
                ("Climactic Turn", f"Escalate {event['conflict']} and land this turn: {event['plot_twist']}"),
            ][:per_event]

            for beat_index, (label, goal) in enumerate(beats, 1):
                entry = {
                    "sub_event_id": f"{event['event_id']}.{beat_index}",
                    "event_id": event["event_id"],
                    "story_arc": event.get("story_arc", "Rising Action"),
                    "title": f"{event['title']} — {label}",
                    "goal": goal,
                    "required_facts": [event["setting"], event["character"]],
                    "exit_state": event["plot_twist"] if beat_index == len(beats) else f"Advance toward {event['conflict']}",
                    "depends_on": [],
                    "status": "pending",
                }
                entry["generation_tokens"] = estimate_token_count(json.dumps(entry, ensure_ascii=False))
                sub_events.append(entry)

        return validate_and_sanitize_dependencies(sub_events, events)

    @staticmethod
    def _normalise(sub_events: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        event_lookup = {event["event_id"]: event for event in events}
        normalised = []
        counters: Dict[str, int] = {}
        for item in sub_events:
            event_id = str(item.get("event_id", "E1"))
            if event_id not in event_lookup:
                continue
            counters[event_id] = counters.get(event_id, 0) + 1
            normalised.append({
                "sub_event_id": str(item.get("sub_event_id", f"{event_id}.{counters[event_id]}")),
                "event_id": event_id,
                "story_arc": event_lookup[event_id].get("story_arc", "Rising Action"),
                "title": str(item.get("title", "Narrative beat")),
                "goal": str(item.get("goal", "Advance the event.")),
                "required_facts": list(item.get("required_facts", [])),
                "exit_state": str(item.get("exit_state", "Leave an active narrative question.")),
                "depends_on": list(item.get("depends_on", [])),
                "status": "pending",
            })
        return normalised

