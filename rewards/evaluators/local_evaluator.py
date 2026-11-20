"""
Level 1 — Local Expansion Evaluator (32B LLM Judge).

Evaluates strictly raw [0.0, 1.0] expansion quality dimensions:
  1. event_realization      : How completely main chapter beats are enacted in prose.
  2. causal_development     : Cause -> decision -> consequence linkage across sub-events.
  3. meaningful_elaboration : State Change + Conflict Development + Narrative Contribution
                              (penalizing hollow filler written solely to inflate word count).
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from rewards.base import EvaluationInput, LocalScores
from src.rewards import strip_think_tags
from .evaluator_schema import clamp_unit_score, normalize_evidence, parse_first_json_object


LOCAL_EXPANSION_SYSTEM_PROMPT_WITH_EVIDENCE = """You are a Local Expansion Evaluator auditing a single generated novel chapter against its planned chapter goal and scene outline.
Do NOT reward word count or verbose filler. Score strictly on genuine narrative expansion from 0.0 to 1.0:
1. event_realization (0.0-1.0): Are the planned main event and scene beats actually enacted in prose (not skipped or merely summarized)?
2. causal_development (0.0-1.0): Do sub-events connect via logical cause-and-effect (discovery -> reaction -> decision -> consequence) rather than disconnected chronological filler?
3. meaningful_elaboration (0.0-1.0): Do added details produce concrete State Change, Conflict Development, and Narrative Contribution rather than hollow padding?

Respond strictly with ONE valid JSON object:
{
  "event_realization": 0.0-1.0,
  "causal_development": 0.0-1.0,
  "meaningful_elaboration": 0.0-1.0,
  "evidence": [{"criterion": "causal_development", "text_span": "short quote", "reason": "brief reason"}],
  "missing_elements": ["any omitted beat or under-developed consequence"]
}
"""

LOCAL_EXPANSION_SYSTEM_PROMPT_COMPACT = """You are a Local Expansion Evaluator auditing a generated novel chapter against its planned goal and scene outline.
Do NOT reward hollow filler or length padding. Score strictly from 0.0 to 1.0:
- event_realization: Planned main event and scene beats are genuinely enacted.
- causal_development: Sub-events follow logical cause -> decision -> consequence links.
- meaningful_elaboration: Added prose delivers State Change, Conflict Development, and Narrative Contribution (NOT empty filler).

Respond strictly with ONE compact JSON object:
{"event_realization": 0.0-1.0, "causal_development": 0.0-1.0, "meaningful_elaboration": 0.0-1.0}
"""


def _clip_chapter_for_judge(text: str, max_words: int = 1600) -> str:
    clean = strip_think_tags(text).strip()
    words = clean.split()
    if len(words) <= max_words:
        return clean
    head = max_words // 2
    tail = max_words - head
    return " ".join(words[:head]) + " [...] " + " ".join(words[-tail:])


class LocalExpansionEvaluator:
    """Level 1 LLM Judge returning strictly raw `LocalScores` (no weighting/aggregation)."""

    def __init__(self, judge_backend: Any, include_evidence: bool = True, max_retries: int = 2):
        self.judge_backend = judge_backend
        self.include_evidence = include_evidence
        self.max_retries = max(1, int(max_retries))

    def _build_request(self, item: EvaluationInput) -> Tuple[str, str]:
        sys_p = LOCAL_EXPANSION_SYSTEM_PROMPT_WITH_EVIDENCE if self.include_evidence else LOCAL_EXPANSION_SYSTEM_PROMPT_COMPACT
        scenes_txt = "\n".join(f"- {s}" for s in item.scene_outline) if item.scene_outline else item.chapter_goal
        prose = _clip_chapter_for_judge(item.generated_chapter)
        usr_p = (
            f"[CHAPTER GOAL]:\n{item.chapter_goal}\n\n"
            f"[SCENE OUTLINE]:\n{scenes_txt}\n\n"
            f"[RECENT CONTEXT]:\n{item.context_summary}\n\n"
            f"[GENERATED CHAPTER]:\n{prose}\n\n"
            "Return strictly ONE valid JSON object."
        )
        return sys_p, usr_p

    def _parse(self, raw: str) -> LocalScores:
        try:
            data = parse_first_json_object(raw)
            return LocalScores(
                event_realization=clamp_unit_score(data.get("event_realization"), 0.5),
                causal_development=clamp_unit_score(data.get("causal_development"), 0.5),
                meaningful_elaboration=clamp_unit_score(data.get("meaningful_elaboration"), 0.5),
                evidence=normalize_evidence(data.get("evidence")) if self.include_evidence else [],
                missing_elements=[str(x) for x in data.get("missing_elements", [])] if isinstance(data.get("missing_elements"), list) else [],
                parse_ok=True,
            )
        except Exception:
            return LocalScores(0.5, 0.5, 0.5, [], [], parse_ok=False)

    def evaluate_batch(self, items: Sequence[EvaluationInput]) -> List[LocalScores]:
        if not items:
            return []
        reqs = [self._build_request(it) for it in items]
        max_tok = 320 if self.include_evidence else 120
        raw_outs = self.judge_backend.generate_batch(reqs, max_new_tokens=max_tok)
        results = [self._parse(r) for r in raw_outs]

        # Single targeted retry only for failed JSON parses
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

    def evaluate(self, item: EvaluationInput) -> LocalScores:
        return self.evaluate_batch([item])[0]
