"""
DOC + HANNA Inspired Outline Judge Evaluator, Blind Pairwise Comparison,
Perturbation Sanity Check, and HANNA Spearman Calibration for `MacroPlan-Bench`.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from experiments.benchmark.evaluator import (
    NLIScorer,
    compute_causal_coherence,
    compute_nli_progression,
)
from experiments.benchmarks._judge_utils import parse_json_object
from .metrics_pacing import evaluate_concreteness_and_pacing


OUTLINE_RUBRIC_SYSTEM_PROMPT = """You are an expert narrative editor evaluating a long-form story outline (Macro Event Graph / Chapter Outline) BEFORE prose generation.
Do NOT penalize non-linear or organic narrative structures; judge solely on narrative merit, causality, and pacing.

Score each dimension from 1.0 to 5.0 and cite concrete evidence from the outline.
Respond strictly with ONE valid JSON object matching this schema:
{
  "outline_relevance": 1.0-5.0,
  "plot_coherence": 1.0-5.0,
  "surprise_and_twists": 1.0-5.0,
  "engagement_and_complexity": 1.0-5.0,
  "weakest_chapter_index": int,
  "evidence": [
    {"criterion": "plot_coherence", "span": "short quote from outline", "reason": "brief explanation"}
  ],
  "rationale": "concise 2-sentence editorial summary"
}
"""

OUTLINE_PAIRWISE_SYSTEM_PROMPT = """You are an expert literary editor comparing two competing outlines (Outline A and Outline B) for the SAME story premise.
Judge which outline provides a stronger, more coherent, better-paced, and more compelling blueprint for an 8,000-10,000 word novel.
Ignore formatting style differences and avoid favoring longer outlines unless the extra detail adds genuine narrative value.

Respond strictly with ONE valid JSON object:
{
  "winner": "A | B | tie",
  "winning_dimensions": ["plot_coherence", "pacing", "engagement"],
  "rationale": "concise comparative justification citing specific events"
}
"""


def format_blueprint_for_judge(blueprint: Dict[str, Any]) -> str:
    """Formats either an Event Graph or chapter list into clean readable text for a Judge."""
    events = blueprint.get("event_graph", {}).get("events", [])
    if events:
        blocks = []
        for idx, ev in enumerate(events, 1):
            deps = ", ".join(ev.get("depends_on", [])) or "None"
            blocks.append(
                f"Event {ev.get('event_id', f'E{idx}')}: {ev.get('title', '')}\n"
                f"  Setting: {ev.get('setting', '')}\n"
                f"  Characters: {ev.get('character', '')}\n"
                f"  Action: {ev.get('action', '')}\n"
                f"  Conflict: {ev.get('conflict', '')}\n"
                f"  Plot Twist: {ev.get('plot_twist', '')}\n"
                f"  Depends On: {deps}"
            )
        return "\n\n".join(blocks)

    chapters = blueprint.get("chapters", [])
    return "\n\n".join(
        f"Chapter {ch.get('chapter_index', i)} ({ch.get('target_words', '')}w): {ch.get('chapter_title', '')}\n"
        f"  Goal: {ch.get('chapter_goal', '')}\n"
        f"  Exit Turn: {ch.get('exit_state', '')}"
        for i, ch in enumerate(chapters, 1)
    )


class OutlineJudgeEvaluator:
    """Pointwise DOC+HANNA rubric judge and position-debiased blind pairwise comparator."""

    def __init__(self, judge_fn: Optional[Callable[[str, str], str]] = None, max_retries: int = 2):
        self.judge_fn = judge_fn
        self.max_retries = max(1, max_retries)

    @staticmethod
    def _empty_scores() -> Dict[str, Any]:
        return {
            "judge_outline_relevance": None,
            "judge_plot_coherence": None,
            "judge_surprise_and_twists": None,
            "judge_engagement_complexity": None,
            "judge_outline_overall": None,
            "judge_weakest_chapter": None,
            "judge_evidence": [],
            "judge_rationale": None,
        }

    def evaluate(self, premise: str, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        if self.judge_fn is None:
            return self._empty_scores()

        plan_text = format_blueprint_for_judge(blueprint)[:8000]
        user_prompt = (
            f"[STORY PREMISE]:\n{premise}\n\n"
            f"[GENERATED OUTLINE]:\n{plan_text}\n\n"
            "Return strictly ONE valid JSON object."
        )

        parsed = None
        for _ in range(self.max_retries):
            try:
                raw = self.judge_fn(OUTLINE_RUBRIC_SYSTEM_PROMPT, user_prompt)
                parsed = parse_json_object(raw)
                break
            except Exception:
                parsed = None

        if parsed is None:
            return self._empty_scores()

        def _clamp(key: str) -> float:
            return round(max(1.0, min(5.0, float(parsed.get(key, 3.0)))), 2)

        rel = _clamp("outline_relevance")
        coh = _clamp("plot_coherence")
        sur = _clamp("surprise_and_twists")
        eng = _clamp("engagement_and_complexity")
        overall = round((rel + coh + sur + eng) / 4.0, 3)

        return {
            "judge_outline_relevance": rel,
            "judge_plot_coherence": coh,
            "judge_surprise_and_twists": sur,
            "judge_engagement_complexity": eng,
            "judge_outline_overall": overall,
            "judge_weakest_chapter": parsed.get("weakest_chapter_index"),
            "judge_evidence": parsed.get("evidence", []),
            "judge_rationale": parsed.get("rationale"),
        }

    def compare_pair(
        self,
        premise: str,
        blueprint_a: Dict[str, Any],
        blueprint_b: Dict[str, Any],
        label_a: str = "A",
        label_b: str = "B",
    ) -> Dict[str, Any]:
        """Blind pairwise comparison with A/B position swap to eliminate position bias."""
        if self.judge_fn is None:
            return {
                "label_a": label_a,
                "label_b": label_b,
                "winner": None,
                "position_consistent": None,
                "rationale": None,
            }

        txt_a = format_blueprint_for_judge(blueprint_a)[:6000]
        txt_b = format_blueprint_for_judge(blueprint_b)[:6000]

        def _ask(first_txt: str, second_txt: str) -> Tuple[str, str]:
            u = (
                f"[STORY PREMISE]:\n{premise}\n\n"
                f"[OUTLINE A]:\n{first_txt}\n\n"
                f"[OUTLINE B]:\n{second_txt}\n\n"
                "Return strictly ONE valid JSON object."
            )
            for _ in range(self.max_retries):
                try:
                    data = parse_json_object(self.judge_fn(OUTLINE_PAIRWISE_SYSTEM_PROMPT, u))
                    w = str(data.get("winner", "tie")).strip().upper()
                    if w not in ("A", "B", "TIE"):
                        w = "TIE"
                    return w, str(data.get("rationale", ""))
                except Exception:
                    pass
            return "TIE", "Judge parse failed."

        w_ab, rat_ab = _ask(txt_a, txt_b)
        w_ba, rat_ba = _ask(txt_b, txt_a)
        # Map swapped result back to original labels
        mapped_ba = "B" if w_ba == "A" else ("A" if w_ba == "B" else "TIE")

        consistent = w_ab == mapped_ba
        if consistent:
            final_winner = label_a if w_ab == "A" else (label_b if w_ab == "B" else "tie")
        else:
            final_winner = "tie"

        return {
            "label_a": label_a,
            "label_b": label_b,
            "winner": final_winner,
            "raw_order_ab": w_ab,
            "raw_order_ba_mapped": mapped_ba,
            "position_consistent": consistent,
            "rationale": rat_ab if consistent else f"Position-inconsistent ({w_ab} vs {mapped_ba}) -> tie. {rat_ab}",
        }


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_r
        i = j + 1
    return ranks


def compute_spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pure-Python Spearman rank correlation for HANNA / human Judge calibration."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = sum((a - mx) ** 2 for a in rx) ** 0.5
    den_y = sum((b - my) ** 2 for b in ry) ** 0.5
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return round(num / (den_x * den_y), 4)


def run_outline_perturbation_sanity_check(
    clean_blueprints: List[Dict[str, Any]],
    nli_scorer: Optional[NLIScorer] = None,
) -> Dict[str, Any]:
    """
    Meta-Evaluation Sanity Check (Part 3 — Optional):
    Applies 3 controlled corruptions to clean blueprints and verifies that
    MacroPlan-Bench metrics detect each degradation accurately:
      1. causal_break        -> S_causal must drop
      2. duplicate_events    -> inter_event_diversity & S_prog must drop
      3. sparse_middle_empty -> pacing_balance_score & mean_concreteness must drop
    """
    if not clean_blueprints:
        return {"passed": False, "reason": "No clean blueprints supplied."}

    causal_drops = 0
    dup_drops = 0
    sparse_drops = 0
    total = len(clean_blueprints)

    for bp in clean_blueprints:
        events = bp.get("event_graph", {}).get("events", [])
        if len(events) < 4:
            continue

        # Clean baselines
        clean_causal = compute_causal_coherence(bp, nli_scorer)["S_causal"]
        clean_pacing = evaluate_concreteness_and_pacing(bp)
        clean_prog = compute_nli_progression(bp, nli_scorer)["S_prog"] if nli_scorer else None

        # 1. Corrupt causal edges (isolate middle/end events)
        bp_causal = copy.deepcopy(bp)
        for ev in bp_causal["event_graph"]["events"][1:]:
            ev["depends_on"] = []
        bad_causal = compute_causal_coherence(bp_causal, nli_scorer)["S_causal"]
        if bad_causal < clean_causal:
            causal_drops += 1

        # 2. Duplicate Event 1 across all events (zero progression / diversity)
        bp_dup = copy.deepcopy(bp)
        first_ev = copy.deepcopy(bp_dup["event_graph"]["events"][0])
        for idx in range(1, len(bp_dup["event_graph"]["events"])):
            dup_ev = copy.deepcopy(first_ev)
            dup_ev["event_id"] = f"E{idx + 1}"
            bp_dup["event_graph"]["events"][idx] = dup_ev
        bad_dup_div = evaluate_concreteness_and_pacing(bp_dup)["inter_event_diversity"]
        bad_dup_prog = compute_nli_progression(bp_dup, nli_scorer)["S_prog"] if nli_scorer else None
        if bad_dup_div < clean_pacing["inter_event_diversity"] and (
            clean_prog is None or (bad_dup_prog is not None and bad_dup_prog <= clean_prog)
        ):
            dup_drops += 1

        # 3. Hollow out middle chapters to create erratic pacing
        bp_sparse = copy.deepcopy(bp)
        mid = len(bp_sparse["event_graph"]["events"]) // 2
        for idx in (mid - 1, mid):
            ev = bp_sparse["event_graph"]["events"][idx]
            ev["setting"] = ""
            ev["character"] = ""
            ev["action"] = "Stuff happens."
            ev["conflict"] = ""
            ev["plot_twist"] = ""
        bad_pacing = evaluate_concreteness_and_pacing(bp_sparse)
        if bad_pacing["pacing_balance_score"] < clean_pacing["pacing_balance_score"]:
            sparse_drops += 1

    return {
        "total_tested": total,
        "causal_drop_rate": round(causal_drops / max(1, total), 4),
        "duplicate_drop_rate": round(dup_drops / max(1, total), 4),
        "sparse_pacing_drop_rate": round(sparse_drops / max(1, total), 4),
        "passed": (causal_drops == total and dup_drops == total and sparse_drops == total),
    }
