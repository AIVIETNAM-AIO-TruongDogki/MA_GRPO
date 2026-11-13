"""
PlanningAgent: Partitions the Event Graph across K chapters with proportional word budgets,
supporting Macro-Only (`enable_micro_subevents=False`) or Macro+Micro (`enable_micro_subevents=True`).
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from .plan_critic import PlanCriticAgent
from .story_arc import compute_story_arc_alignment, ensure_event_ids, estimate_token_count
from .sub_event_agent import SubEventAgent


class PlanningAgent:
    """
    Stateful rolling planner used between validated context and each chapter.
    Supports:
      - enable_micro_subevents=False -> Condition D (Macro Event Graph Only)
      - enable_micro_subevents=True  -> Condition E (Macro + Micro Sub-Events)
    """

    def __init__(
        self,
        sub_event_agent: Optional[SubEventAgent] = None,
        critic: Optional[PlanCriticAgent] = None,
        enable_micro_subevents: bool = True,
        llm_generate_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.sub_event_agent = sub_event_agent or SubEventAgent(llm_generate_fn=llm_generate_fn)
        self.critic = critic or PlanCriticAgent()
        self.enable_micro_subevents = enable_micro_subevents
        self.llm_generate_fn = llm_generate_fn

    @staticmethod
    def _build_macro_only_units(events: List[Dict[str, Any]], num_chapters: int) -> List[Dict[str, Any]]:
        """
        Condition D: Convert Macro Events directly into trackable macro units
        without decomposing into micro sub-events via SubEventAgent.
        Ensures every macro unit maps strictly to its parent event dependencies.
        """
        if not events:
            return []

        units: List[Dict[str, Any]] = []
        # If there are at least num_chapters events, each event is 1 macro unit
        if len(events) >= num_chapters:
            for idx, ev in enumerate(events):
                eid = ev["event_id"]
                deps = [f"{d}.macro" for d in ev.get("depends_on", []) if any(e["event_id"] == d for e in events)]
                units.append({
                    "sub_event_id": f"{eid}.macro",
                    "event_id": eid,
                    "story_arc": ev.get("story_arc", "Rising Action"),
                    "title": ev.get("title", eid),
                    "goal": f"{ev.get('action', '')} (Conflict: {ev.get('conflict', '')})",
                    "required_facts": [ev.get("setting", ""), ev.get("character", "")],
                    "exit_state": ev.get("plot_twist", "Advance arc"),
                    "depends_on": deps if deps else ([units[-1]["sub_event_id"]] if units else []),
                    "status": "pending",
                    "is_macro_only": True,
                })
            return units

        # Otherwise proportionally allocate macro events across num_chapters cleanly
        per_ev_slots = [1] * len(events)
        rem = num_chapters - len(events)
        idx = 0
        while rem > 0:
            per_ev_slots[idx % len(events)] += 1
            rem -= 1
            idx += 1

        for ev, slots in zip(events, per_ev_slots):
            eid = ev["event_id"]
            for s_i in range(1, slots + 1):
                phase_tag = "" if slots == 1 else (" (Setup)" if s_i == 1 else (" (Climax/Turn)" if s_i == slots else f" (Development {s_i})"))
                goal = f"{ev.get('action', '')} — Conflict: {ev.get('conflict', '')}"
                if slots > 1 and s_i == slots:
                    goal = f"Culminate event via turn: {ev.get('plot_twist', ev.get('action', ''))}"
                uid = f"{eid}.macro{'' if slots == 1 else f'_{s_i}'}"
                prev_uid = units[-1]["sub_event_id"] if units else None
                units.append({
                    "sub_event_id": uid,
                    "event_id": eid,
                    "story_arc": ev.get("story_arc", "Rising Action"),
                    "title": f"{ev.get('title', eid)}{phase_tag}",
                    "goal": goal,
                    "required_facts": [ev.get("setting", ""), ev.get("character", "")],
                    "exit_state": ev.get("plot_twist", "Transition to next stage") if s_i == slots else f"Continue {ev.get('title', eid)}",
                    "depends_on": [prev_uid] if prev_uid else [],
                    "status": "pending",
                    "is_macro_only": True,
                })
        return units

    def create_story_plan(
        self,
        event_graph: Dict[str, Any],
        total_target_words: int = 12000,
        num_chapters: int = 10,
        enable_micro_subevents: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if num_chapters <= 0 or total_target_words <= 0:
            raise ValueError("num_chapters and total_target_words must be positive")
        t0 = time.time()
        events = ensure_event_ids(event_graph.get("events", []))
        use_micro = self.enable_micro_subevents if enable_micro_subevents is None else enable_micro_subevents

        if use_micro:
            units = self.sub_event_agent.decompose(event_graph)
            sub_event_tokens = sum(u.get("generation_tokens", estimate_token_count(u.get("goal", ""))) for u in units)
            condition_tag = "E"
        else:
            units = self._build_macro_only_units(events, num_chapters)
            sub_event_tokens = 0  # Condition D skips SubEventAgent overhead!
            condition_tag = "D"

        outline_tokens = event_graph.get("planning_tokens", estimate_token_count(json.dumps(events)))
        planning_latency = event_graph.get("planning_latency_s", 0.0) + round(time.time() - t0, 4)

        return {
            "condition": condition_tag,
            "enable_micro_subevents": use_micro,
            "premise": event_graph.get("premise", ""),
            "genre": event_graph.get("genre", ""),
            "event_graph": event_graph,
            "story_arc_alignment": event_graph.get("story_arc_alignment", compute_story_arc_alignment(events)),
            "sub_events": units,
            "total_target_words": total_target_words,
            "remaining_target_words": total_target_words,
            "num_chapters": num_chapters,
            "completed_sub_event_ids": [],
            "chapters": [],
            "event_status": {event["event_id"]: "pending" for event in events},
            "planning_metadata": {
                "condition": condition_tag,
                "outline_tokens": outline_tokens,
                "sub_event_tokens": sub_event_tokens,
                "planning_tokens": outline_tokens + sub_event_tokens,
                "planning_latency_s": round(planning_latency, 4),
            },
        }

    def plan_next_chapter(
        self,
        story_plan: Dict[str, Any],
        context_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        chapter_index = len(story_plan["chapters"]) + 1
        if chapter_index > story_plan["num_chapters"]:
            raise StopIteration("All planned chapters have been created.")

        use_micro = story_plan.get("enable_micro_subevents", self.enable_micro_subevents)
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
                "story_arc": "Resolution",
                "title": "Continuity",
                "goal": "Resolve remaining narrative threads.",
                "required_facts": [],
                "exit_state": "Create a transition.",
                "depends_on": [],
            }]

        first, last = selected[0], selected[-1]

        if not use_micro and len(selected) == 1:
            # Condition D (Macro-Only): Divide single macro event goal across standard 25%/45%/30% scene budgets
            s1 = int(round(target_words * 0.25))
            s2 = int(round(target_words * 0.45))
            s3 = max(1, target_words - s1 - s2)
            scene_budgets = [s1, s2, s3]
            parent_ev = event_lookup.get(first.get("event_id", ""), {})
            scenes = [
                {
                    "scene_index": 1,
                    "title": f"{first.get('title', 'Chapter')} — Macro Opening",
                    "target_words": s1,
                    "beat_goal": f"Establish {parent_ev.get('setting', 'the setting')} and introduce {first.get('goal', '')}",
                    "required_facts": first.get("required_facts", []),
                    "exit_state": "Transition into central tension.",
                },
                {
                    "scene_index": 2,
                    "title": f"{first.get('title', 'Chapter')} — Macro Escalation",
                    "target_words": s2,
                    "beat_goal": f"Develop conflict: {parent_ev.get('conflict', first.get('goal', ''))}",
                    "required_facts": first.get("required_facts", []),
                    "exit_state": "Reach turning point.",
                },
                {
                    "scene_index": 3,
                    "title": f"{first.get('title', 'Chapter')} — Macro Turn & Transition",
                    "target_words": s3,
                    "beat_goal": f"Land turn: {last.get('exit_state', 'Resolve chapter beat')}",
                    "required_facts": first.get("required_facts", []),
                    "exit_state": last.get("exit_state", ""),
                },
            ]
            scene_outline = [
                f"Scene {s['scene_index']} (~{s['target_words']} words) [{s['title']}]: {s['beat_goal']}"
                for s in scenes
            ]
        else:
            # Condition E (or multi-unit chapter): Allocate budgets per beat/unit density
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
            "story_arc": first.get("story_arc", "Rising Action"),
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
            if identifier and not str(identifier).startswith("continuity."):
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
        enable_micro_subevents: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Distributes events across K chapters and calculates L_target = L_total / K."""
        story_plan = self.create_story_plan(
            event_graph,
            total_target_words,
            num_chapters,
            enable_micro_subevents=enable_micro_subevents,
        )
        for _ in range(num_chapters):
            chapter = self.plan_next_chapter(story_plan)
            self.mark_chapter_complete(story_plan, chapter)
        story_plan["chapter_target_words"] = total_target_words // num_chapters
        return story_plan

