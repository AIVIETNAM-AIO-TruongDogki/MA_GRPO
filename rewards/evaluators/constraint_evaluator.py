"""
Level 3 — Deterministic Python + MiniLM Constraint & Safety Evaluator (0 LLM calls).

Computes strictly normalized penalties in [0.0, 1.0]:
  1. `p_length`: Soft piecewise violation outside [0.85 * L_target, 1.15 * L_target]
                 measured strictly on narrative prose (after stripping `<think>`).
  2. `p_rep`   : Semantic redundancy (adjacent-paragraph MiniLM cosine > threshold +
                 intra-chapter repeated sentences/paragraphs + summary 3-gram copying
                 with entity protection).
  3. `p_contra`: Severity-weighted contradiction penalty from Global/Holistic Judge list:
                 min(1.0, (3*critical + 2*major + 1*minor) / max(3, N_paragraphs)).
  4. `p_format`: CoT `<think>...</think>` hygiene (closing tag, <= 400 think words,
                 no leaked special tags `</assistant>`, `[INST]`, `Scene 1:`).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from rewards.base import ConstraintPenalties, EvaluationInput
from src.pipeline import deduplicate_repeated_paragraphs
from src.rewards import calculate_anti_context_ratio, calculate_length_reward, strip_think_tags


_LEAKED_ARTIFACT_RE = re.compile(
    r"(?:</assistant>|<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|\[/?INST\]|</s>)",
    flags=re.IGNORECASE,
)


def _split_think_and_prose(raw_text: str) -> tuple[str, int, bool]:
    """Extract (clean_prose, think_words, unclosed_think) robustly.

    Handles both standard `<think>...</think>prose` and implicit-open `...</think>prose`
    (which happens when prompts end with `Begin with <think> now:` and the model omits `<think>`).
    """
    text = raw_text or ""
    has_open = bool(re.search(r"<think>", text, flags=re.IGNORECASE))
    has_close = bool(re.search(r"</think>", text, flags=re.IGNORECASE))

    if has_open and has_close:
        think_blocks = re.findall(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE)
        think_words = sum(len(b.split()) for b in think_blocks)
        clean_prose = strip_think_tags(text)
        unclosed_think = False
    elif has_close and not has_open:
        # Prompt already cued <think>; everything before the first </think> is reasoning
        parts = re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)
        think_words = len(parts[0].split())
        clean_prose = parts[1].strip() if len(parts) > 1 else ""
        unclosed_think = False
    elif has_open and not has_close:
        think_words = len(re.sub(r"<think>", "", text, flags=re.IGNORECASE).split())
        clean_prose = ""
        unclosed_think = True
    else:
        think_words = 0
        clean_prose = text.strip()
        unclosed_think = False

    return clean_prose, think_words, unclosed_think


def _has_clean_ending(clean_prose: str) -> bool:
    """Allow standard sentence punctuation as well as markdown italics/scene breaks & ellipses."""
    trimmed = (clean_prose or "").rstrip(" \t\r\n*_~`")
    if not trimmed:
        return False
    return trimmed[-1] in '.!?"”’\'…)—-'


def _jaccard_tokens(a: str, b: str) -> float:
    sa = {w for w in re.findall(r"[a-z0-9]+", a.lower()) if len(w) >= 4}
    sb = {w for w in re.findall(r"[a-z0-9]+", b.lower()) if len(w) >= 4}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


class ConstraintEvaluator:
    """100% Python + optional MiniLM Embedder constraint auditor (0 LLM calls)."""

    def __init__(
        self,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        use_embedder: bool = True,
        redundancy_cos_threshold: float = 0.78,
        max_think_words: int = 400,
    ):
        self.embedder_name = embedder_name
        self.device = device
        self.use_embedder = use_embedder
        self.redundancy_cos_threshold = float(redundancy_cos_threshold)
        self.max_think_words = int(max_think_words)
        self._embedder = None
        self._embedder_attempted = False

    def _get_embedder(self):
        if not self.use_embedder:
            return None
        if not self._embedder_attempted:
            self._embedder_attempted = True
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.embedder_name, device=self.device)
            except Exception:
                self._embedder = None
        return self._embedder

    def _compute_semantic_redundancy(self, clean_prose: str, context_summary: str) -> Dict[str, float]:
        paras = [p.strip() for p in re.split(r"\n\s*\n+", clean_prose) if len(p.split()) >= 15]
        if len(paras) < 2:
            words = clean_prose.split()
            if len(words) >= 60:
                mid = len(words) // 2
                paras = [" ".join(words[:mid]), " ".join(words[mid:])]

        # 1. Adjacent-paragraph semantic redundancy ratio
        redundant_pairs = 0
        total_pairs = max(0, len(paras) - 1)
        if total_pairs > 0:
            embedder = self._get_embedder()
            if embedder is not None:
                try:
                    import numpy as np

                    embs = embedder.encode(paras, normalize_embeddings=True)
                    sims = [float(np.dot(embs[i], embs[i + 1])) for i in range(total_pairs)]
                except Exception:
                    sims = [_jaccard_tokens(paras[i], paras[i + 1]) for i in range(total_pairs)]
            else:
                sims = [_jaccard_tokens(paras[i], paras[i + 1]) for i in range(total_pairs)]
            # Jaccard fallback uses lower threshold (~0.55) than MiniLM cosine (0.78)
            thr = self.redundancy_cos_threshold if embedder is not None else 0.55
            redundant_pairs = sum(1 for s in sims if s >= thr)

        adj_redundancy = redundant_pairs / float(max(1, total_pairs)) if total_pairs > 0 else 0.0

        # 2. Exact/near-duplicate repeated paragraphs or intra-paragraph looping sentences
        deduped, removed_units = deduplicate_repeated_paragraphs(clean_prose, min_words=30, min_sentence_words=12)
        orig_w = max(1, len(clean_prose.split()))
        dup_word_ratio = max(0.0, (orig_w - len(deduped.split())) / float(orig_w))

        # 3. Verbatim copying from Coordinator summary (entity-protected)
        if context_summary and len(context_summary.split()) >= 8:
            anti_ctx = calculate_anti_context_ratio(clean_prose, context_summary)
            # Only penalize when anti-context novelty falls below 0.85
            copy_pen = max(0.0, (0.85 - anti_ctx) / 0.85)
        else:
            copy_pen = 0.0

        # Combine into [0.0, 1.0] penalty
        p_rep = min(1.0, round(0.50 * adj_redundancy + 0.35 * min(1.0, dup_word_ratio * 2.5) + 0.15 * copy_pen, 4))
        return {
            "p_rep": p_rep,
            "adj_redundancy": round(adj_redundancy, 4),
            "dup_word_ratio": round(dup_word_ratio, 4),
            "removed_dup_units": removed_units,
            "summary_copy_pen": round(copy_pen, 4),
            "n_paragraphs": max(1, len(paras)),
        }

    def _compute_format_penalty(self, raw_text: str, clean_prose: str) -> Dict[str, Any]:
        parsed_prose, think_words, unclosed_think = _split_think_and_prose(raw_text)
        prose_for_check = parsed_prose if parsed_prose else clean_prose

        pen = 0.0
        if unclosed_think:
            pen += 0.60  # Unclosed <think> wastes budget and corrupts prose
        if think_words > self.max_think_words:
            pen += min(0.40, ((think_words - self.max_think_words) / float(self.max_think_words)) * 0.40)
        if _LEAKED_ARTIFACT_RE.search(prose_for_check):
            pen += 0.35
        if prose_for_check and not _has_clean_ending(prose_for_check):
            pen += 0.15  # Truncated mid-sentence ending

        return {
            "p_format": round(min(1.0, pen), 4),
            "think_words": think_words,
            "unclosed_think": unclosed_think,
        }

    @staticmethod
    def compute_contradiction_penalty(contradictions: List[Dict[str, str]], n_units: int) -> float:
        if not contradictions:
            return 0.0
        severity_sum = 0.0
        for c in contradictions:
            sev = str(c.get("severity", "minor")).lower()
            if sev == "critical":
                severity_sum += 3.0
            elif sev == "major":
                severity_sum += 2.0
            else:
                severity_sum += 1.0
        return round(min(1.0, severity_sum / float(max(3, n_units))), 4)

    def evaluate(
        self,
        item: EvaluationInput,
        contradictions: Optional[List[Dict[str, str]]] = None,
    ) -> ConstraintPenalties:
        raw_text = item.generated_chapter or ""
        clean_prose = strip_think_tags(raw_text)
        prose_words = len(clean_prose.split())

        # Soft piecewise length penalty: 0.0 inside [0.85L, 1.15L], ramps smoothly to 1.0 outside
        len_reward = calculate_length_reward(raw_text, target_length=item.target_words)
        p_length = round(max(0.0, min(1.0, 1.0 - len_reward)), 4)

        rep_info = self._compute_semantic_redundancy(clean_prose, item.context_summary)
        fmt_info = self._compute_format_penalty(raw_text, clean_prose)
        p_contra = self.compute_contradiction_penalty(contradictions or [], rep_info["n_paragraphs"])

        return ConstraintPenalties(
            p_length=p_length,
            p_rep=rep_info["p_rep"],
            p_contra=p_contra,
            p_format=fmt_info["p_format"],
            prose_words=prose_words,
            think_words=fmt_info["think_words"],
            details={
                **rep_info,
                "unclosed_think": fmt_info["unclosed_think"],
            },
        )


def verify_objective_constraints(
    item: EvaluationInput,
    min_len_ratio: float = 0.85,
    max_len_ratio: float = 1.15,
    max_think_words: int = 400,
) -> Dict[str, Any]:
    """
    Minimal Orthogonal RLMR Verifier (100% pure Python, < 1ms, 0 LLM, 0 MiniLM).
    Checks strictly the 3 objective constraints orthogonal to the 32B semantic judge:
      1. V_length  : min_len_ratio * L <= prose_words <= max_len_ratio * L (post-</think>)
      2. V_format  : valid closed </think>, think_words <= max_think_words, no leaked tags, complete ending
      3. V_no_loop : zero repeated long paragraphs (>30w) or looping intra-paragraph sentences (>12w)
    """
    raw_text = item.generated_chapter or ""
    clean_prose, think_words, unclosed_think = _split_think_and_prose(raw_text)
    prose_words = len(clean_prose.split())

    target = max(1, int(item.target_words or 1125))
    l_low = int(round(target * min_len_ratio))
    l_high = int(round(target * max_len_ratio))
    v_length = l_low <= prose_words <= l_high
    length_compliance = float(calculate_length_reward(clean_prose, target_length=target, l_lower=l_low, l_upper=l_high))

    has_leak = bool(_LEAKED_ARTIFACT_RE.search(clean_prose))
    clean_end = _has_clean_ending(clean_prose)
    think_ok = think_words <= max_think_words
    v_format = (not unclosed_think) and think_ok and (not has_leak) and clean_end

    _, removed_dup_units = deduplicate_repeated_paragraphs(clean_prose, min_words=30, min_sentence_words=12)
    v_no_loop = removed_dup_units == 0

    failed: List[str] = []
    if not v_length:
        failed.append(f"length({prose_words}w not in [{l_low},{l_high}])")
    if not v_format:
        fmt_reasons: List[str] = []
        if unclosed_think:
            fmt_reasons.append("unclosed_think")
        if not think_ok:
            fmt_reasons.append(f"think={think_words}w>{max_think_words}w")
        if has_leak:
            fmt_reasons.append("leaked_tag")
        if not clean_end:
            fmt_reasons.append("truncated_end")
        failed.append(f"format({'+'.join(fmt_reasons)})")
    if not v_no_loop:
        failed.append(f"loop(removed={removed_dup_units})")

    return {
        "is_compliant": bool(v_length and v_format and v_no_loop),
        "v_length": v_length,
        "v_format": v_format,
        "v_no_loop": v_no_loop,
        "length_compliance": round(length_compliance, 4),
        "failed_constraints": failed,
        "prose_words": prose_words,
        "think_words": think_words,
        "removed_dup_units": removed_dup_units,
    }
