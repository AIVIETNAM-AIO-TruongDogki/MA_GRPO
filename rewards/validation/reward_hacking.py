"""
M1 Offline Validation: Perturbation Discrimination & Reward Hacking Resistance Test.

Constructs 4 controlled variants from clean chapters:
  1. `clean`         : Original chapter prose.
  2. `padded_filler` : Inflates chapter with repetitive hollow atmospheric padding (tests Meaningful Elaboration & P_rep).
  3. `off_plan`      : Replaces chapter with fluent prose unrelated to the chapter goal (tests Event Realization & Macro Adherence).
  4. `contradictory` : Injects explicit critical continuity contradictions (tests Character Consistency & P_contra).

Verifies that `Clean` reliably beats all 3 degraded variants AND measures group-level `std(R)`
to confirm sufficient variance for stable GRPO group-relative advantages.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Sequence

from rewards.base import EvaluationInput, RewardBreakdown


_FILLER_BLOCK = (
    "Empty filler stretched across the room while silence lingered over the quiet walls and the faint air drifted slowly "
    "without changing anything about the situation or moving any decision forward. "
    "Empty filler stretched across the room while silence lingered over the quiet walls and the faint air drifted slowly "
    "without changing anything about the situation or moving any decision forward."
)

_OFF_PLAN_PROSE = (
    "Far across an unrelated polar research station, two meteorologists spent the afternoon calibrating barometers and "
    "recording ordinary wind speed tables inside a quiet greenhouse. Nothing connected to the story's characters, "
    "setting, or central conflict occurred; they drank lukewarm tea, sorted spare cables into labeled plastic bins, "
    "and turned off the generator lamp before nightfall."
)

_CONTRADICTION_INJECTION = (
    " Suddenly, without any explanation, the dead character resurrected without explanation and walked in smiling, "
    "claiming they had never met anyone here before (critical_contradiction and major_contradiction)."
)


def build_perturbation_suite(clean_item: EvaluationInput) -> Dict[str, EvaluationInput]:
    """Produces a 4-way controlled candidate group `(clean, padded_filler, off_plan, contradictory)`."""
    clean = copy.deepcopy(clean_item)

    filler = copy.deepcopy(clean_item)
    filler.generated_chapter = f"{_FILLER_BLOCK}\n\n{_FILLER_BLOCK}\n\n{_FILLER_BLOCK}"

    off_plan = copy.deepcopy(clean_item)
    off_plan.generated_chapter = f"{_OFF_PLAN_PROSE}\n\n{_OFF_PLAN_PROSE}"

    contra = copy.deepcopy(clean_item)
    contra.generated_chapter = (clean_item.generated_chapter.strip() + "\n\n" + _CONTRADICTION_INJECTION).strip()

    return {
        "clean": clean,
        "padded_filler": filler,
        "off_plan": off_plan,
        "contradictory": contra,
    }


def evaluate_perturbation_discrimination(
    engine: Any,
    clean_items: Sequence[EvaluationInput],
) -> Dict[str, Any]:
    """Runs the 4-way discrimination test across `clean_items` and reports pass rates & group std."""
    if not clean_items:
        return {"passed": False, "reason": "No clean items provided."}

    beat_filler = 0
    beat_off_plan = 0
    beat_contra = 0
    group_stds: List[float] = []
    details: List[Dict[str, Any]] = []

    for item in clean_items:
        variants = build_perturbation_suite(item)
        ordered_keys = ["clean", "padded_filler", "off_plan", "contradictory"]
        bds: List[RewardBreakdown] = engine.evaluate_batch([variants[k] for k in ordered_keys])
        by_key = dict(zip(ordered_keys, bds))

        r_clean = by_key["clean"].r_total
        r_filler = by_key["padded_filler"].r_total
        r_off = by_key["off_plan"].r_total
        r_con = by_key["contradictory"].r_total

        if r_clean > r_filler:
            beat_filler += 1
        if r_clean > r_off:
            beat_off_plan += 1
        if r_clean > r_con:
            beat_contra += 1

        vals = [r_clean, r_filler, r_off, r_con]
        mean = sum(vals) / 4.0
        g_std = math.sqrt(sum((v - mean) ** 2 for v in vals) / 4.0)
        group_stds.append(g_std)

        details.append(
            {
                "premise_id": item.premise_id,
                "chapter_index": item.chapter_index,
                "r_clean": r_clean,
                "r_padded_filler": r_filler,
                "r_off_plan": r_off,
                "r_contradictory": r_con,
                "group_std": round(g_std, 4),
            }
        )

    n = len(clean_items)
    mean_std = round(sum(group_stds) / max(1, n), 4)
    return {
        "num_groups": n,
        "clean_beats_filler_rate": round(beat_filler / n, 4),
        "clean_beats_off_plan_rate": round(beat_off_plan / n, 4),
        "clean_beats_contradictory_rate": round(beat_contra / n, 4),
        "mean_group_reward_std": mean_std,
        "passed": (beat_filler == n and beat_off_plan == n and beat_contra == n and mean_std >= 0.05),
        "details": details,
    }
