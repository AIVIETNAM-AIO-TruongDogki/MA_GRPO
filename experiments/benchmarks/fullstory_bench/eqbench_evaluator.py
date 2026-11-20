"""
EQ-Bench Longform Creative Writing Evaluator & Blind Pairwise Story Judge
for Suite 2 (`FullStory-Bench`).

Rubric combines:
  - 5 EQ-Bench Longform Positive Axes (0–10):
      nuanced_characters, emotionally_engaging, compelling_plot, coherence, prose_quality
  - 1 MA-GRPO Expansion Quality Axis (0–10):
      meaningful_elaboration (State Change + Conflict Development + Narrative Contribution)
  - Early-vs-Late Chapter Degradation Penalty (`max(0, early_score - late_score)`)
  - Slop Penalty (`slop_score_per_1k`) -> Composite `eqbench_score` (0–100)
  - Swapped-Order Blind Pairwise Comparison (`BlindPairwiseStoryJudge`)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from experiments.benchmarks._judge_utils import parse_json_object
from .constory_checker import _prepare_story_for_window


EQBENCH_SYSTEM_PROMPT = """You are an expert literary critic scoring a complete multi-chapter story (8,000-10,000 words) following the EQ-Bench Longform Creative Writing rubric.

Score each criterion strictly from 0.0 to 10.0 (decimals allowed):
1. nuanced_characters: Distinct voices, psychological depth, earned growth, non-caricatured dialogue.
2. emotionally_engaging: Resonance, stakes, tension, empathetic weight across chapters.
3. compelling_plot: Momentum, satisfying causal escalation, payoffs, strong ending.
4. coherence: Smooth chapter-to-chapter continuity, clear spatial/temporal flow, zero abrupt resets.
5. prose_quality: Vivid imagery, varied sentence rhythm, avoidance of repetitive LLM clichés ("slop").
6. meaningful_elaboration: Do chapters expand the outline through concrete state changes, conflict development, and narrative contribution rather than hollow filler?

Also score the opening chapters (Chapters 1-2) vs. the closing chapters (final 2 chapters) separately (0.0-10.0) to detect late-story quality degradation.

Respond strictly with ONE valid JSON object:
{
  "nuanced_characters": 0.0-10.0,
  "emotionally_engaging": 0.0-10.0,
  "compelling_plot": 0.0-10.0,
  "coherence": 0.0-10.0,
  "prose_quality": 0.0-10.0,
  "meaningful_elaboration": 0.0-10.0,
  "early_chapters_score": 0.0-10.0,
  "late_chapters_score": 0.0-10.0,
  "rationale": "concise 2-3 sentence critical evaluation citing specific chapters"
}
"""

STORY_PAIRWISE_SYSTEM_PROMPT = """You are a blind literary judge comparing two full multi-chapter stories (Story A and Story B) written from the SAME premise.
Judge which story is stronger overall across:
  (1) Global Consistency & Continuity across chapters
  (2) Plot Progression & Meaningful Elaboration (not padded filler)
  (3) Prose Quality & Character Depth
Do NOT favor a story merely for being slightly longer if the extra text is repetitive filler.

Respond strictly with ONE valid JSON object:
{
  "winner": "A | B | tie",
  "winning_axes": ["consistency", "continuity", "plot", "prose"],
  "rationale": "concise comparative judgment citing specific chapter evidence"
}
"""


class EQBenchLongformEvaluator:
    """Computes EQ-Bench Longform 6-axis quality, early-vs-late degradation, and 0–100 composite score."""

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
            "eq_nuanced_characters": None,
            "eq_emotionally_engaging": None,
            "eq_compelling_plot": None,
            "eq_coherence": None,
            "eq_prose_quality": None,
            "eq_meaningful_elaboration": None,
            "eq_early_chapters_score": None,
            "eq_late_chapters_score": None,
            "eq_degradation_penalty": None,
            "eqbench_score": None,
            "eq_rationale": None,
        }

    def evaluate(
        self,
        premise: str,
        chapters: List[str],
        slop_score_per_1k: float = 0.0,
    ) -> Dict[str, Any]:
        if self.judge_fn is None:
            return self._empty_result()

        story_text = _prepare_story_for_window(chapters, self.max_story_words)
        user_prompt = (
            f"[STORY PREMISE]:\n{premise}\n\n"
            f"[FULL STORY ({len(chapters)} chapters)]:\n{story_text}\n\n"
            "Return strictly ONE valid JSON object."
        )

        parsed = None
        for _ in range(self.max_retries):
            try:
                raw = self.judge_fn(EQBENCH_SYSTEM_PROMPT, user_prompt)
                parsed = parse_json_object(raw)
                break
            except Exception:
                parsed = None

        if parsed is None:
            return self._empty_result()

        def _clamp(k: str) -> float:
            return round(max(0.0, min(10.0, float(parsed.get(k, 5.0)))), 2)

        chars = _clamp("nuanced_characters")
        emo = _clamp("emotionally_engaging")
        plot = _clamp("compelling_plot")
        coh = _clamp("coherence")
        prose = _clamp("prose_quality")
        elab = _clamp("meaningful_elaboration")
        early = _clamp("early_chapters_score")
        late = _clamp("late_chapters_score")

        degradation = round(max(0.0, early - late), 2)
        mean_6 = (chars + emo + plot + coh + prose + elab) / 6.0
        composite = mean_6 * 10.0 - (degradation * 4.0) - min(12.0, slop_score_per_1k * 1.5)
        composite = round(max(0.0, min(100.0, composite)), 2)

        return {
            "eq_nuanced_characters": chars,
            "eq_emotionally_engaging": emo,
            "eq_compelling_plot": plot,
            "eq_coherence": coh,
            "eq_prose_quality": prose,
            "eq_meaningful_elaboration": elab,
            "eq_early_chapters_score": early,
            "eq_late_chapters_score": late,
            "eq_degradation_penalty": degradation,
            "eqbench_score": composite,
            "eq_rationale": parsed.get("rationale"),
        }


class BlindPairwiseStoryJudge:
    """Position-debiased blind pairwise comparator running both (A, B) and (B, A) orders."""

    def __init__(
        self,
        judge_fn: Optional[Callable[[str, str], str]] = None,
        max_story_words_each: int = 5500,
        max_retries: int = 2,
    ):
        self.judge_fn = judge_fn
        self.max_story_words_each = max_story_words_each
        self.max_retries = max(1, max_retries)

    def compare_pair(
        self,
        premise: str,
        chapters_a: List[str],
        chapters_b: List[str],
        label_a: str = "A",
        label_b: str = "B",
    ) -> Dict[str, Any]:
        if self.judge_fn is None:
            return {
                "label_a": label_a,
                "label_b": label_b,
                "winner": None,
                "position_consistent": None,
                "winning_axes": [],
                "rationale": None,
            }

        txt_a = _prepare_story_for_window(chapters_a, self.max_story_words_each)
        txt_b = _prepare_story_for_window(chapters_b, self.max_story_words_each)

        def _ask(first_txt: str, second_txt: str) -> Tuple[str, List[str], str]:
            u = (
                f"[STORY PREMISE]:\n{premise}\n\n"
                f"[STORY A]:\n{first_txt}\n\n"
                f"[STORY B]:\n{second_txt}\n\n"
                "Return strictly ONE valid JSON object."
            )
            for _ in range(self.max_retries):
                try:
                    data = parse_json_object(self.judge_fn(STORY_PAIRWISE_SYSTEM_PROMPT, u))
                    w = str(data.get("winner", "tie")).strip().upper()
                    if w not in ("A", "B", "TIE"):
                        w = "TIE"
                    axes = data.get("winning_axes", [])
                    if not isinstance(axes, list):
                        axes = []
                    return w, [str(x) for x in axes], str(data.get("rationale", ""))
                except Exception:
                    pass
            return "TIE", [], "Judge parse failed."

        w_ab, axes_ab, rat_ab = _ask(txt_a, txt_b)
        w_ba, _, _ = _ask(txt_b, txt_a)
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
            "winning_axes": axes_ab if consistent else [],
            "rationale": rat_ab if consistent else f"Position-inconsistent ({w_ab} vs {mapped_ba}) -> tie. {rat_ab}",
        }
