"""
Level 2 — Global Chapter Evaluator (32B LLM Judge).

Evaluates strictly raw [0.0, 1.0] chapter-level narrative quality + severity-typed contradictions:
  1. macro_adherence       : Whole-chapter alignment with macro plan intent & structure.
  2. chapter_coherence     : Smooth internal unity & hook continuity from preceding chapter tail.
  3. character_consistency : Behavior/knowledge consistency against Long-Term Memory character states.
  4. narrative_progression : Genuine story advancement from entry state to exit state.
  5. contradictions        : List of `{severity: minor|major|critical, reason: ...}` for Level 3 `P_contra`.
"""

from __future__ import annotations

import json
from typing import Any, List, Sequence, Tuple

from rewards.base import EvaluationInput, GlobalScores
from .evaluator_schema import (
    clamp_unit_score,
    normalize_contradictions,
    normalize_evidence,
    parse_first_json_object,
)
from .local_evaluator import _clip_chapter_for_judge


GLOBAL_CHAPTER_SYSTEM_PROMPT_WITH_EVIDENCE = """You are a Global Chapter Evaluator auditing chapter-level narrative integrity and consistency against Long-Term Story Memory and the chapter plan.
Score strictly from 0.0 to 1.0 and list any genuine internal or cross-chapter contradictions:
1. macro_adherence (0.0-1.0): Does the whole chapter fulfill the planned narrative intent and ending trajectory?
2. chapter_coherence (0.0-1.0): Do scenes connect smoothly into one unified chapter and hook naturally from the preceding chapter tail?
3. character_consistency (0.0-1.0): Do character actions, motives, and knowledge match Long-Term Memory character states?
4. narrative_progression (0.0-1.0): Does the chapter meaningfully advance the overarching story rather than stalling?
5. contradictions: Only list genuine contradictions with severity "minor", "major", or "critical" (empty list [] if none).

Respond strictly with ONE valid JSON object:
{
  "macro_adherence": 0.0-1.0,
  "chapter_coherence": 0.0-1.0,
  "character_consistency": 0.0-1.0,
  "narrative_progression": 0.0-1.0,
  "contradictions": [{"severity": "minor|major|critical", "reason": "brief explanation"}],
  "evidence": [{"criterion": "chapter_coherence", "text_span": "short quote", "reason": "brief reason"}]
}
"""

GLOBAL_CHAPTER_SYSTEM_PROMPT_COMPACT = """You are a Global Chapter Evaluator auditing chapter-level narrative integrity against Long-Term Memory and the chapter plan.
Score strictly from 0.0 to 1.0:
- macro_adherence: Whole chapter fulfills planned narrative intent.
- chapter_coherence: Unified flow and smooth hook from preceding chapter tail.
- character_consistency: Matches established character motives, states, and knowledge.
- narrative_progression: Advances the overarching plot meaningfully.
- contradictions: List genuine errors as [{"severity": "minor|major|critical", "reason": "..."}] or [].

Respond strictly with ONE compact JSON object:
{"macro_adherence": 0.0-1.0, "chapter_coherence": 0.0-1.0, "character_consistency": 0.0-1.0, "narrative_progression": 0.0-1.0, "contradictions": []}
"""


class GlobalChapterEvaluator:
    """Level 2 LLM Judge returning strictly raw `GlobalScores` (no weighting/aggregation)."""

    def __init__(self, judge_backend: Any, include_evidence: bool = True, max_retries: int = 2):
        self.judge_backend = judge_backend
        self.include_evidence = include_evidence
        self.max_retries = max(1, int(max_retries))

    def _build_request(self, item: EvaluationInput) -> Tuple[str, str]:
        sys_p = GLOBAL_CHAPTER_SYSTEM_PROMPT_WITH_EVIDENCE if self.include_evidence else GLOBAL_CHAPTER_SYSTEM_PROMPT_COMPACT
        ltm_str = json.dumps(item.long_term_memory, ensure_ascii=False)[:900] if item.long_term_memory else "{}"
        chars_str = ", ".join(item.characters_active) if item.characters_active else "Protagonist"
        prose = _clip_chapter_for_judge(item.generated_chapter)
        usr_p = (
            f"[LONG-TERM MEMORY]:\n{ltm_str}\n\n"
            f"[ACTIVE CHARACTERS]:\n{chars_str}\n\n"
            f"[PRECEDING CHAPTER TAIL]:\n{item.prev_chapter_tail[:900]}\n\n"
            f"[CHAPTER GOAL]:\n{item.chapter_goal}\n\n"
            f"[GENERATED CHAPTER]:\n{prose}\n\n"
            "Return strictly ONE valid JSON object."
        )
        return sys_p, usr_p

    def _parse(self, raw: str) -> GlobalScores:
        try:
            data = parse_first_json_object(raw)
            return GlobalScores(
                macro_adherence=clamp_unit_score(data.get("macro_adherence"), 0.5),
                chapter_coherence=clamp_unit_score(data.get("chapter_coherence"), 0.5),
                character_consistency=clamp_unit_score(data.get("character_consistency"), 0.5),
                narrative_progression=clamp_unit_score(data.get("narrative_progression"), 0.5),
                contradictions=normalize_contradictions(data.get("contradictions")),
                evidence=normalize_evidence(data.get("evidence")) if self.include_evidence else [],
                parse_ok=True,
            )
        except Exception:
            return GlobalScores(0.5, 0.5, 0.5, 0.5, [], [], parse_ok=False)

    def evaluate_batch(self, items: Sequence[EvaluationInput]) -> List[GlobalScores]:
        if not items:
            return []
        reqs = [self._build_request(it) for it in items]
        max_tok = 360 if self.include_evidence else 180
        raw_outs = self.judge_backend.generate_batch(reqs, max_new_tokens=max_tok)
        results = [self._parse(r) for r in raw_outs]

        if self.max_retries > 1:
            retry_idx = [i for i, res in enumerate(results) if not res.parse_ok]
            if retry_idx:
                retry_reqs = [reqs[i] for i in retry_idx]
                retry_outs = self.judge_backend.generate_batch(retry_reqs, max_new_tokens=max_tok)
                for idx, raw in zip(retry_idx, retry_outs):
                    parsed = self._parse(raw)
                    if parsed.parse_ok:
                        results[idx] = parsed
        return results

    def evaluate(self, item: EvaluationInput) -> GlobalScores:
        return self.evaluate_batch([item])[0]
