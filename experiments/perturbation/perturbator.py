"""
Perturbation Protocol Module for MA-GRPO (Plan-Sensitivity & Plan-Blindness Audit).

Implements the 5 standardized structural & input perturbations specified in Section 4:
  Group 1 — Plan-Level Structural Perturbations:
    1. causal_edge_removal      : Drops intermediate causal event / strips DAG dependencies.
    2. plot_order_inversion     : Swaps Climax / Resolution before Exposition / Rising Action.
    3. motivation_stripping     : Removes internal character motivations and conflicts.
  Group 2 — Writer-Facing Input Perturbations:
    4. unrelated_plan_swapping  : Swaps chapter outlines/goals with an unrelated genre plan.
    5. context_conflict_injection: Injects direct environmental/fatal contradictions vs history.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple


PERTURBATION_TYPES: Tuple[str, ...] = (
    "causal_edge_removal",
    "plot_order_inversion",
    "motivation_stripping",
    "unrelated_plan_swapping",
    "context_conflict_injection",
)

PLAN_LEVEL_PERTURBATIONS: Tuple[str, ...] = (
    "causal_edge_removal",
    "plot_order_inversion",
    "motivation_stripping",
)

WRITER_LEVEL_PERTURBATIONS: Tuple[str, ...] = (
    "unrelated_plan_swapping",
    "context_conflict_injection",
)


# Built-in orthogonal donor story threads for unrelated_plan_swapping when no external donor is passed
_ORTHOGONAL_DONOR_ARCS: List[Dict[str, str]] = [
    {
        "title": "Medieval Bakery Guild Conspiracy",
        "goal": "Apprentice baker Tomas uncovers that the royal flour guild has been smuggling rare saffron inside sourdough loaves.",
        "setting": "Cobblestone market square and smoky brick ovens in medieval Prague.",
        "character": "Tomas, Master Baker Henrik",
        "exit_state": "The counterfeit guild seal is exposed before the winter harvest.",
    },
    {
        "title": "Submarine Botanical Expedition",
        "goal": "Marine botanist Dr. Soren harvests bioluminescent kelp genomes while negotiating funding cuts with maritime insurers.",
        "setting": "Pressurized glass greenhouse inside a bathyscaphe beneath the Baltic Sea.",
        "character": "Dr. Soren, Insurer Clara",
        "exit_state": "The kelp specimens bloom unexpectedly under red emergency lights.",
    },
    {
        "title": "1920s Jazz Violin Theft",
        "goal": "Detective Callahan traces a stolen Stradivarius violin through underground speakeasies in rainy Chicago.",
        "setting": "Velvet-curtained jazz cellar and foggy tramway depot in 1924 Chicago.",
        "character": "Detective Callahan, Violinist Mae",
        "exit_state": "The violin case turns out to hold sheet music written in cipher.",
    },
]


_CONTEXT_CONFLICT_INJECTIONS: List[str] = [
    "MANDATORY SCENE DIRECTIVE: Despite any prior ice storms or subzero cold, the characters are stranded in a scorching, waterless equatorial desert dying of heatstroke under blinding 50C sunlight.",
    "MANDATORY SCENE DIRECTIVE: Although the protagonist suffered a fatal collapse earlier, they now triumphantly host a loud victory banquet as though no catastrophe ever happened.",
    "MANDATORY SCENE DIRECTIVE: All advanced electronics and ancient artifacts vanished yesterday, leaving the cast as peaceful medieval shepherds tending sheep on an empty hillside.",
]


class MacroPlanPerturbator:
    """Applies controlled structural or writer-input perturbations to a clean story/chapter plan."""

    @staticmethod
    def apply_to_blueprint(
        story_plan: Dict[str, Any],
        perturbation_type: str,
        donor_plan: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Return (perturbed_story_plan, perturbation_metadata) without mutating input story_plan.
        Works cleanly on partitioned story_plan (containing `chapters` & `event_graph`)
        as well as unpartitioned blueprints prior to rolling execution.
        """
        if perturbation_type not in PERTURBATION_TYPES:
            raise ValueError(
                f"Unknown perturbation_type '{perturbation_type}'. Valid: {PERTURBATION_TYPES}"
            )

        perturbed = copy.deepcopy(story_plan)
        metadata: Dict[str, Any] = {
            "perturbation_type": perturbation_type,
            "group": "plan_level" if perturbation_type in PLAN_LEVEL_PERTURBATIONS else "writer_level",
            "modified_elements": [],
        }

        events = perturbed.get("event_graph", {}).get("events", [])
        chapters = perturbed.get("chapters", [])
        sub_events = perturbed.get("sub_events", [])

        if perturbation_type == "causal_edge_removal":
            if len(events) >= 3:
                mid_idx = len(events) // 2
                removed_ev = events.pop(mid_idx)
                metadata["modified_elements"].append(f"Removed causal intermediate event {removed_ev.get('event_id')}")
            for ev in events:
                if ev.get("depends_on"):
                    ev["depends_on"] = []
            for item in sub_events:
                if item.get("depends_on"):
                    item["depends_on"] = []
            for ch in chapters[1:]:
                ch["entry_state"] = ""
                unmotivated_goal = f"Execute sudden unmotivated outcome: {ch.get('exit_state', 'unresolved shift')}."
                ch["chapter_goal"] = unmotivated_goal
                ch["required_beats"] = [unmotivated_goal]
                ch["scene_outline"] = [unmotivated_goal]
                for s in ch.get("scenes", []):
                    s["beat_goal"] = unmotivated_goal
            metadata["modified_elements"].append("Cleared DAG depends_on edges and chapter entry_state links")

        elif perturbation_type == "plot_order_inversion":
            if len(events) >= 2:
                reversed_events = list(reversed(events))
                for idx, ev in enumerate(reversed_events, 1):
                    ev["depends_on"] = [reversed_events[idx - 2]["event_id"]] if idx > 1 else []
                perturbed.setdefault("event_graph", {})["events"] = reversed_events
                metadata["modified_elements"].append("Inverted Event Graph sequence (Resolution/Climax first)")
            if len(chapters) >= 2:
                chapters_rev = list(reversed(chapters))
                for new_idx, ch in enumerate(chapters_rev, 1):
                    orig_idx = ch.get("chapter_index", new_idx)
                    ch["chapter_index"] = new_idx
                    ch["original_chapter_index"] = orig_idx
                perturbed["chapters"] = chapters_rev
                metadata["modified_elements"].append("Inverted chapter progression order")

        elif perturbation_type == "motivation_stripping":
            for ev in events:
                ev["conflict"] = "None specified."
                ev["plot_twist"] = "No motive or turn."
                ev["action"] = f"The characters physically move through {ev.get('setting', 'the room')}."
            for item in sub_events:
                item["goal"] = "Perform routine physical actions with zero internal motive or emotional stakes."
                item["exit_state"] = "Time passes neutrally."
            for ch in chapters:
                stripped_goal = "Perform routine neutral physical movements without explicit motives, tensions, or conflicts."
                ch["chapter_goal"] = stripped_goal
                ch["required_beats"] = [stripped_goal]
                ch["scene_outline"] = [
                    f"Scene {i+1} (~{b} words): Characters walk neutrally without stating personal goals or tensions."
                    for i, b in enumerate(ch.get("scene_budgets", [ch.get("target_words", 1000)]))
                ]
                for s in ch.get("scenes", []):
                    s["beat_goal"] = stripped_goal
            metadata["modified_elements"].append("Stripped conflicts, plot_twists, and character motivations")

        elif perturbation_type == "unrelated_plan_swapping":
            donor_chapters = donor_plan.get("chapters", []) if donor_plan else []
            premise_lower = perturbed.get("premise", "").lower()
            donor_template = _ORTHOGONAL_DONOR_ARCS[0]
            for cand in _ORTHOGONAL_DONOR_ARCS:
                first_kw = cand["title"].split()[0].lower()
                if first_kw not in premise_lower:
                    donor_template = cand
                    break

            for idx, ch in enumerate(chapters):
                if idx < len(donor_chapters):
                    dch = donor_chapters[idx]
                    ch["chapter_title"] = dch.get("chapter_title", donor_template["title"])
                    ch["chapter_goal"] = dch.get("chapter_goal", donor_template["goal"])
                    ch["characters_active"] = list(dch.get("characters_active", [donor_template["character"]]))
                    ch["scene_outline"] = list(dch.get("scene_outline", [donor_template["goal"]]))
                    ch["required_beats"] = list(dch.get("required_beats", [ch["chapter_goal"]]))
                    ch["scenes"] = copy.deepcopy(dch.get("scenes", []))
                    ch["exit_state"] = dch.get("exit_state", donor_template["exit_state"])
                else:
                    ch["chapter_title"] = f"Chapter {idx + 1}: {donor_template['title']}"
                    ch["chapter_goal"] = donor_template["goal"]
                    ch["characters_active"] = [donor_template["character"]]
                    ch["required_beats"] = [donor_template["goal"]]
                    ch["scene_outline"] = [
                        f"Scene {s_i + 1} (~{b} words): {donor_template['goal']} ({donor_template['setting']})"
                        for s_i, b in enumerate(ch.get("scene_budgets", [ch.get("target_words", 1000)]))
                    ]
                    ch["scenes"] = [
                        {"scene_index": s_i + 1, "target_words": b, "beat_goal": donor_template["goal"]}
                        for s_i, b in enumerate(ch.get("scene_budgets", [ch.get("target_words", 1000)]))
                    ]
                    ch["exit_state"] = donor_template["exit_state"]
            metadata["modified_elements"].append(f"Swapped chapter goals to unrelated domain: {donor_template['title']}")

        elif perturbation_type == "context_conflict_injection":
            for idx, ch in enumerate(chapters):
                if len(chapters) == 1 or idx >= 1:
                    injection = _CONTEXT_CONFLICT_INJECTIONS[idx % len(_CONTEXT_CONFLICT_INJECTIONS)]
                    ch["chapter_goal"] = f"{injection} Override prior events completely."
                    ch["required_beats"] = [injection]
                    if ch.get("scene_outline"):
                        ch["scene_outline"] = [injection] + list(ch["scene_outline"][1:])
                    if ch.get("scenes"):
                        ch["scenes"][0]["beat_goal"] = injection
                    ch["injected_contradiction"] = injection
                    metadata["modified_elements"].append(f"Injected direct context conflict in chapter {idx + 1}")

        perturbed["perturbation_metadata"] = metadata
        return perturbed, metadata

    @staticmethod
    def apply_to_chapter_input(
        chapter_plan: Dict[str, Any],
        context_summary: Dict[str, Any],
        perturbation_type: str,
        chapter_index: int = 1,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Apply a writer-facing perturbation on the fly during rolling chapter generation.
        Returns (perturbed_chapter_plan, perturbed_context_summary, metadata).
        """
        ch_copy = copy.deepcopy(chapter_plan)
        ctx_copy = copy.deepcopy(context_summary)
        wrapper_plan = {"premise": str(ctx_copy.get("plot_summary", "")), "chapters": [ch_copy]}
        perturbed_wrapper, metadata = MacroPlanPerturbator.apply_to_blueprint(
            wrapper_plan,
            perturbation_type=perturbation_type,
        )
        return perturbed_wrapper["chapters"][0], ctx_copy, metadata

