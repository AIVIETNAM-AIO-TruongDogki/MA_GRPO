"""
Single-Pass Unified Evaluator (32B LLM Judge) for High-Throughput GRPO Training.

Returns BOTH raw `LocalScores` (3 axes) and raw `GlobalScores` (4 axes + contradictions)
in ONE batched forward pass per rollout (~2x faster than separate calls during GRPO training).
For M1 Offline Calibration & H2 Independence Audits, use separate `LocalExpansionEvaluator`
and `GlobalChapterEvaluator` (`single_pass=False`) to eliminate prompt halo effect.
"""

from __future__ import annotations

import json
from typing import Any, List, Sequence, Tuple

from rewards.base import EvaluationInput, GlobalScores, LocalScores
from .evaluator_schema import clamp_unit_score, normalize_contradictions, parse_first_json_object
from .local_evaluator import _clip_chapter_for_judge


UNIFIED_PHASE2_SYSTEM_PROMPT = """You are a Unified Narrative Evaluator auditing a generated novel chapter on 7 independent criteria (each 0.0 to 1.0) plus contradictions.
Do NOT reward word count or repetitive filler. Evaluate each axis independently:

[LOCAL EXPANSION AXES]:
- event_realization: Main event and scene beats are genuinely enacted in prose.
- causal_development: Sub-events link via logical cause -> decision -> consequence.
- meaningful_elaboration: Added details create State Change, Conflict Development, and Narrative Contribution (penalize hollow padding).

[GLOBAL CHAPTER AXES]:
- macro_adherence: Whole chapter fulfills planned narrative intent and trajectory.
- chapter_coherence: Unified internal flow and smooth hook from preceding chapter tail.
- character_consistency: Matches Long-Term Memory character motives, states, and knowledge.
- narrative_progression: Advances the overarching story meaningfully.
- contradictions: List genuine errors as [{"severity": "minor|major|critical", "reason": "..."}] or [].

Respond strictly with ONE compact JSON object:
{
  "event_realization": 0.0-1.0,
  "causal_development": 0.0-1.0,
  "meaningful_elaboration": 0.0-1.0,
  "macro_adherence": 0.0-1.0,
  "chapter_coherence": 0.0-1.0,
  "character_consistency": 0.0-1.0,
  "narrative_progression": 0.0-1.0,
  "contradictions": []
}
"""


class UnifiedPhase2Evaluator:
    """Single-pass 32B Judge emitting separate raw `LocalScores` and `GlobalScores`."""

    def __init__(self, judge_backend: Any, max_retries: int = 2):
        self.judge_backend = judge_backend
        self.max_retries = max(1, int(max_retries))

    def _build_request(self, item: EvaluationInput) -> Tuple[str, str]:
        scenes_txt = "\n".join(f"- {s}" for s in item.scene_outline) if item.scene_outline else item.chapter_goal
        ltm_str = json.dumps(item.long_term_memory, ensure_ascii=False)[:800] if item.long_term_memory else "{}"
        chars_str = ", ".join(item.characters_active) if item.characters_active else "Protagonist"
        prose = _clip_chapter_for_judge(item.generated_chapter)
        usr_p = (
            f"[LONG-TERM MEMORY]:\n{ltm_str}\n\n"
            f"[ACTIVE CHARACTERS]:\n{chars_str}\n\n"
            f"[RECENT CONTEXT]:\n{item.context_summary}\n\n"
            f"[PRECEDING CHAPTER TAIL]:\n{item.prev_chapter_tail[:750]}\n\n"
            f"[CHAPTER GOAL]:\n{item.chapter_goal}\n\n"
            f"[SCENE OUTLINE]:\n{scenes_txt}\n\n"
            f"[GENERATED CHAPTER]:\n{prose}\n\n"
            "Return strictly ONE valid JSON object."
        )
        return UNIFIED_PHASE2_SYSTEM_PROMPT, usr_p

    def _parse(self, raw: str) -> Tuple[LocalScores, GlobalScores]:
        try:
            data = parse_first_json_object(raw)
            loc = LocalScores(
                event_realization=clamp_unit_score(data.get("event_realization"), 0.5),
                causal_development=clamp_unit_score(data.get("causal_development"), 0.5),
                meaningful_elaboration=clamp_unit_score(data.get("meaningful_elaboration"), 0.5),
                parse_ok=True,
            )
            glob = GlobalScores(
                macro_adherence=clamp_unit_score(data.get("macro_adherence"), 0.5),
                chapter_coherence=clamp_unit_score(data.get("chapter_coherence"), 0.5),
                character_consistency=clamp_unit_score(data.get("character_consistency"), 0.5),
                narrative_progression=clamp_unit_score(data.get("narrative_progression"), 0.5),
                contradictions=normalize_contradictions(data.get("contradictions")),
                parse_ok=True,
            )
            return loc, glob
        except Exception:
            return (
                LocalScores(0.5, 0.5, 0.5, parse_ok=False),
                GlobalScores(0.5, 0.5, 0.5, 0.5, [], parse_ok=False),
            )

    def evaluate_batch(self, items: Sequence[EvaluationInput]) -> List[Tuple[LocalScores, GlobalScores]]:
        if not items:
            return []
        reqs = [self._build_request(it) for it in items]
        raw_outs = self.judge_backend.generate_batch(reqs, max_new_tokens=240)
        results = [self._parse(r) for r in raw_outs]

        if self.max_retries > 1:
            retry_idx = [i for i, (loc, glob) in enumerate(results) if not (loc.parse_ok and glob.parse_ok)]
            if retry_idx:
                retry_reqs = [reqs[i] for i in retry_idx]
                retry_outs = self.judge_backend.generate_batch(retry_reqs, max_new_tokens=240)
                for idx, raw in zip(retry_idx, retry_outs):
                    loc, glob = self._parse(raw)
                    if loc.parse_ok and glob.parse_ok:
                        results[idx] = (loc, glob)
        return results
