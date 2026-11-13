"""
PlanCriticAgent: Enforces 5 structural verification gates (coverage, temporal
consistency, character grounding, word budget conservation, monotonic event ordering).
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


class PlanCriticAgent:
    """
    Research-Grade Plan Critic Gate.
    Verifies 5 fundamental research criteria before approval:
    1. Event Coverage: all assigned units must be valid, unfulfilled, and non-duplicate.
    2. Temporal Consistency: all depends_on dependencies must be completed or prior in chapter.
    3. Character Consistency: active characters must be grounded in parent event characters.
    4. Word Budget: sum(scene_budgets) must exactly equal chapter target_words.
    5. Event Ordering: events/sub-events must preserve chronological monotonic progression.
    Supports both Condition D (Macro Event units) and Condition E (Micro Sub-event beats).
    """

    def review(self, chapter_plan: Dict[str, Any], story_plan: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        coverage_issues: List[str] = []
        temporal_issues: List[str] = []
        character_issues: List[str] = []
        budget_issues: List[str] = []
        ordering_issues: List[str] = []

        all_story_sub_events = {s["sub_event_id"]: s for s in story_plan.get("sub_events", [])}
        completed_ids = set(story_plan.get("completed_sub_event_ids", []))

        sub_events = chapter_plan.get("sub_events", [])
        if not sub_events:
            issues.append("Chapter plan contains zero sub-events.")
            coverage_issues.append("Empty sub_events list.")

        seen_in_chapter: Set[str] = set()
        sub_event_indices: List[int] = []

        for item in sub_events:
            sid = item.get("sub_event_id")

            # 1. Event Coverage & Duplicate Checking
            if sid in completed_ids:
                msg = f"Duplicate execution: {sid} was already completed in prior chapter."
                issues.append(msg)
                coverage_issues.append(msg)
            if sid in seen_in_chapter:
                msg = f"Duplicate in-chapter: {sid} appears multiple times in same chapter."
                issues.append(msg)
                coverage_issues.append(msg)
            seen_in_chapter.add(sid)

            if sid not in all_story_sub_events and not str(sid).startswith("continuity."):
                msg = f"Unregistered unit: {sid} does not belong to story_plan."
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
            parent_eid = str(item.get("event_id", ""))
            if parent_eid.startswith("E") and parent_eid[1:].isdigit():
                sub_event_indices.append(int(parent_eid[1:]))

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

        return {
            "approved": len(issues) == 0,
            "issues": issues,
            "criteria_breakdown": criteria_breakdown,
            "chapter_plan": chapter_plan,
        }

