"""EXP-01 Holistic Chapter Evaluator (single holistic quality score + contradictions via 32B Judge)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

from rewards.base import EvaluationInput
from .evaluator_schema import clamp_unit_score, normalize_contradictions, parse_first_json_object
from .local_evaluator import _clip_chapter_for_judge


HOLISTIC_SYSTEM_PROMPT = """You are a Holistic Literary Evaluator scoring the overall quality of a generated novel chapter in context (0.0 to 1.0).
Consider plan adherence, coherence, character consistency, meaningful narrative progression, and prose quality holistically.
Also list any genuine contradictions as [{"severity": "minor|major|critical", "reason": "..."}] or [].

Respond strictly with ONE compact JSON object:
{"holistic_score": 0.0-1.0, "contradictions": [], "rationale": "1-sentence summary"}
"""


class HolisticChapterEvaluator:
    """Single-scalar holistic judge used specifically for ablation `EXP-01`."""

    def __init__(self, judge_backend: Any):
        self.judge_backend = judge_backend

    def _build_request(self, item: EvaluationInput) -> Tuple[str, str]:
        ltm_str = json.dumps(item.long_term_memory, ensure_ascii=False)[:700] if item.long_term_memory else "{}"
        prose = _clip_chapter_for_judge(item.generated_chapter)
        usr_p = (
            f"[LONG-TERM MEMORY]:\n{ltm_str}\n\n"
            f"[CHAPTER GOAL]:\n{item.chapter_goal}\n\n"
            f"[GENERATED CHAPTER]:\n{prose}\n\n"
            "Return strictly ONE valid JSON object."
        )
        return HOLISTIC_SYSTEM_PROMPT, usr_p

    def evaluate_batch(self, items: Sequence[EvaluationInput]) -> List[Dict[str, Any]]:
        if not items:
            return []
        reqs = [self._build_request(it) for it in items]
        raw_outs = self.judge_backend.generate_batch(reqs, max_new_tokens=160)
        results: List[Dict[str, Any]] = []
        for raw in raw_outs:
            try:
                data = parse_first_json_object(raw)
                results.append(
                    {
                        "holistic_score": clamp_unit_score(data.get("holistic_score"), 0.5),
                        "contradictions": normalize_contradictions(data.get("contradictions")),
                        "parse_ok": True,
                    }
                )
            except Exception:
                results.append({"holistic_score": 0.5, "contradictions": [], "parse_ok": False})
        return results
