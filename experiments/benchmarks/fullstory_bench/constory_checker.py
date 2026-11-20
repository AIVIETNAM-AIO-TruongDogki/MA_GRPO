"""
ConStory-Checker Evaluator (ACL Findings 2026 ConStory-Bench Inspired).

Audits 8,000–10,000 word stories for evidence-grounded contradictions across
5 Categories and 19 Fine-Grained Error Subtypes, requiring exact text quotes
from conflicting passages and computing:
  - `total_errors`
  - `errors_per_1k_words`
  - `C_weighted = 3 * n_fatal + 2 * n_moderate + 1 * n_minor`
  - per-category counts (`err_characterization`, `err_factual_detail`,
    `err_narrative_style`, `err_timeline_plot`, `err_world_setting`)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from experiments.benchmarks._judge_utils import parse_json_object
from src.rewards import strip_think_tags


CONSTORY_TAXONOMY: Dict[str, List[str]] = {
    "characterization": [
        "memory_contradiction",
        "knowledge_contradiction",
        "skill_power_inconsistency",
        "forgotten_ability_or_trait",
    ],
    "factual_detail": [
        "physical_appearance",
        "nomenclature_name_drift",
        "quantitative_mismatch",
        "object_or_inventory_state",
    ],
    "narrative_style": [
        "pov_perspective_shift",
        "tone_inconsistency",
        "style_or_voice_break",
    ],
    "timeline_plot": [
        "temporal_contradiction",
        "duration_mismatch",
        "causality_violation",
        "abandoned_plot_thread",
    ],
    "world_setting": [
        "world_rule_violation",
        "social_norm_violation",
        "geography_spatial_contradiction",
        "technology_or_magic_inconsistency",
    ],
}

_CATEGORY_ALIASES = {
    "characterization": "characterization",
    "character": "characterization",
    "factual_detail": "factual_detail",
    "factual": "factual_detail",
    "narrative_style": "narrative_style",
    "style": "narrative_style",
    "timeline_plot": "timeline_plot",
    "timeline": "timeline_plot",
    "plot": "timeline_plot",
    "world_setting": "world_setting",
    "world-building": "world_setting",
    "world_building": "world_setting",
    "world": "world_setting",
}


CONSTORY_SYSTEM_PROMPT = """You are ConStory-Checker, a rigorous literary continuity auditor evaluating a long-form story (up to 10,000 words).
Identify ONLY genuine internal contradictions or continuity errors that can be proven by quoting two conflicting passages (or a broken setup) from the text. Do NOT flag deliberate plot twists or character deception that the text itself explains.

Use strictly these 5 categories and 19 subtypes:
1. characterization: memory_contradiction | knowledge_contradiction | skill_power_inconsistency | forgotten_ability_or_trait
2. factual_detail: physical_appearance | nomenclature_name_drift | quantitative_mismatch | object_or_inventory_state
3. narrative_style: pov_perspective_shift | tone_inconsistency | style_or_voice_break
4. timeline_plot: temporal_contradiction | duration_mismatch | causality_violation | abandoned_plot_thread
5. world_setting: world_rule_violation | social_norm_violation | geography_spatial_contradiction | technology_or_magic_inconsistency

Severity definitions:
- fatal (weight 3): Breaks core plot logic, resurrects a dead character, forgets a central item/fact, or swaps protagonist identity.
- moderate (weight 2): Noticeable timeline/location/knowledge contradiction or unexplained POV shift.
- minor (weight 1): Small cosmetic inconsistency (minor prop detail, slight tone slip).

Respond strictly with ONE valid JSON object:
{
  "errors": [
    {
      "category": "characterization | factual_detail | narrative_style | timeline_plot | world_setting",
      "subtype": "one of the 19 subtypes above",
      "severity": "fatal | moderate | minor",
      "chapter_a": 1,
      "chapter_b": 5,
      "quote_a": "short verbatim quote from earlier passage",
      "quote_b": "short verbatim quote from later conflicting passage",
      "explanation": "1-sentence explanation of the contradiction"
    }
  ],
  "summary": "1-2 sentence overall continuity assessment"
}
"""


def _prepare_story_for_window(chapters: List[str], max_story_words: int) -> str:
    """Formats chapters with headers; keeps full text if within budget, else preserves head+mid+tail per chapter."""
    cleaned = [strip_think_tags(c).strip() for c in chapters]
    total_w = sum(len(c.split()) for c in cleaned)
    if total_w <= max_story_words:
        return "\n\n".join(f"=== CHAPTER {i} ===\n{txt}" for i, txt in enumerate(cleaned, 1))

    per_ch_budget = max(250, max_story_words // max(1, len(cleaned)))
    blocks = []
    for i, txt in enumerate(cleaned, 1):
        words = txt.split()
        if len(words) <= per_ch_budget:
            excerpt = txt
        else:
            part = per_ch_budget // 3
            mid_start = max(part, (len(words) - part) // 2)
            excerpt = (
                " ".join(words[:part])
                + " [...] "
                + " ".join(words[mid_start : mid_start + part])
                + " [...] "
                + " ".join(words[-part:])
            )
        blocks.append(f"=== CHAPTER {i} ===\n{excerpt}")
    return "\n\n".join(blocks)


def _normalize_category(raw_cat: str) -> str:
    key = raw_cat.strip().lower().replace(" ", "_").replace("&", "").replace("__", "_")
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    for alias, canon in _CATEGORY_ALIASES.items():
        if alias in key:
            return canon
    return "timeline_plot"


class ConStoryCheckerEvaluator:
    """Evidence-grounded 5-category / 19-subtype contradiction auditor."""

    def __init__(
        self,
        judge_fn: Optional[Callable[[str, str], str]] = None,
        max_story_words: int = 12000,
        max_retries: int = 2,
    ):
        self.judge_fn = judge_fn
        self.max_story_words = max_story_words
        self.max_retries = max(1, max_retries)

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "constory_total_errors": None,
            "constory_errors_per_1k": None,
            "constory_n_fatal": None,
            "constory_n_moderate": None,
            "constory_n_minor": None,
            "C_weighted": None,
            "err_characterization": None,
            "err_factual_detail": None,
            "err_narrative_style": None,
            "err_timeline_plot": None,
            "err_world_setting": None,
            "constory_summary": None,
            "evidence_errors": [],
        }

    def evaluate(
        self,
        premise: str,
        chapters: List[str],
        actual_words: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.judge_fn is None:
            return self._empty_result()

        story_text = _prepare_story_for_window(chapters, self.max_story_words)
        total_words = actual_words if actual_words is not None else sum(
            len(strip_think_tags(c).split()) for c in chapters
        )
        user_prompt = (
            f"[STORY PREMISE]:\n{premise}\n\n"
            f"[FULL STORY ({total_words} words, {len(chapters)} chapters)]:\n{story_text}\n\n"
            "Audit all internal contradictions with verbatim quotes. Return strictly ONE valid JSON object."
        )

        parsed = None
        for _ in range(self.max_retries):
            try:
                raw = self.judge_fn(CONSTORY_SYSTEM_PROMPT, user_prompt)
                parsed = parse_json_object(raw)
                break
            except Exception:
                parsed = None

        if parsed is None:
            return self._empty_result()

        raw_errors = parsed.get("errors", [])
        if not isinstance(raw_errors, list):
            raw_errors = []

        cat_counts = {cat: 0 for cat in CONSTORY_TAXONOMY}
        n_fatal = 0
        n_moderate = 0
        n_minor = 0
        normalized_errors: List[Dict[str, Any]] = []

        for err in raw_errors:
            if not isinstance(err, dict):
                continue
            cat = _normalize_category(str(err.get("category", "timeline_plot")))
            sev = str(err.get("severity", "minor")).strip().lower()
            if sev not in ("fatal", "moderate", "minor"):
                sev = "minor"
            cat_counts[cat] += 1
            if sev == "fatal":
                n_fatal += 1
            elif sev == "moderate":
                n_moderate += 1
            else:
                n_minor += 1

            normalized_errors.append(
                {
                    "category": cat,
                    "subtype": str(err.get("subtype", "unspecified")),
                    "severity": sev,
                    "chapter_a": err.get("chapter_a"),
                    "chapter_b": err.get("chapter_b"),
                    "quote_a": str(err.get("quote_a", "")),
                    "quote_b": str(err.get("quote_b", "")),
                    "explanation": str(err.get("explanation", "")),
                }
            )

        total_errors = len(normalized_errors)
        errors_per_1k = round((total_errors / max(1, total_words)) * 1000.0, 3)
        c_weighted = 3 * n_fatal + 2 * n_moderate + 1 * n_minor

        return {
            "constory_total_errors": total_errors,
            "constory_errors_per_1k": errors_per_1k,
            "constory_n_fatal": n_fatal,
            "constory_n_moderate": n_moderate,
            "constory_n_minor": n_minor,
            "C_weighted": c_weighted,
            "err_characterization": cat_counts["characterization"],
            "err_factual_detail": cat_counts["factual_detail"],
            "err_narrative_style": cat_counts["narrative_style"],
            "err_timeline_plot": cat_counts["timeline_plot"],
            "err_world_setting": cat_counts["world_setting"],
            "constory_summary": parsed.get("summary"),
            "evidence_errors": normalized_errors,
        }
