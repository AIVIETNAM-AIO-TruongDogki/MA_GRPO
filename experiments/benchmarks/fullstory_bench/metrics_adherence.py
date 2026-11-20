"""
Deterministic & Embedding Metrics for Suite 2 (`FullStory-Bench`):
  - Slop Score (LLM clichés per 1,000 words — EQ-Bench inspired)
  - Intra-Chapter Semantic Redundancy (adjacent-paragraph similarity)
  - Inter-Chapter Continuity (180w tail -> 150w head cosine) & Anti-Context Novelty
  - Length Compliance (`length_score_sl`), Early Termination / Overshoot & `<think>` Word Tracking
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from src.rewards import calculate_anti_context_ratio, strip_think_tags


# Curated high-signal LLM creative writing clichés ("slop" indicators inspired by EQ-Bench)
SLOP_PHRASES: List[str] = [
    "shiver down",
    "breath she didn't know",
    "breath he didn't know",
    "breath they didn't know",
    "let out a breath",
    "unspoken promise",
    "silent understanding",
    "tapestry of",
    "testament to",
    "symphony of",
    "echoed in the silence",
    "weight of the world",
    "piercing gaze",
    "sending chills",
    "knuckles turned white",
    "air grew thick",
    "hung heavy in the air",
    "heart hammered",
    "eyes widened",
    "time seemed to stop",
    "shadows danced",
    "in the dim light",
    "a mixture of",
    "flicker of",
    "stark reminder",
    "delicate dance",
    "fragile truce",
    "loomed large",
    "unwavering resolve",
    "quiet determination",
    "pang of guilt",
    "sense of unease",
    "wash over",
    "palpable tension",
    "chilling realization",
    "couldn't shake the feeling",
]


def compute_slop_score(full_text: str) -> Dict[str, Any]:
    """Counts LLM cliché hits normalized per 1,000 words."""
    clean = strip_think_tags(full_text)
    words = clean.split()
    n_words = len(words)
    if n_words == 0:
        return {"slop_hits": 0, "slop_score_per_1k": 0.0, "top_slop_phrases": []}

    lower = clean.lower()
    counts: Dict[str, int] = {}
    total_hits = 0
    for phrase in SLOP_PHRASES:
        c = len(re.findall(re.escape(phrase), lower))
        if c > 0:
            counts[phrase] = c
            total_hits += c

    per_1k = round((total_hits / n_words) * 1000.0, 3)
    top_phrases = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return {
        "slop_hits": total_hits,
        "slop_score_per_1k": per_1k,
        "top_slop_phrases": [{"phrase": p, "count": c} for p, c in top_phrases],
    }


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    toks_a = {w for w in re.findall(r"[a-z0-9]+", text_a.lower()) if len(w) >= 4}
    toks_b = {w for w in re.findall(r"[a-z0-9]+", text_b.lower()) if len(w) >= 4}
    if not toks_a and not toks_b:
        return 0.0
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / float(len(toks_a | toks_b))


def _pair_similarity(text_a: str, text_b: str, embedder: Optional[Any] = None) -> float:
    if embedder is not None:
        try:
            import numpy as np

            embs = embedder.encode([text_a, text_b], normalize_embeddings=True)
            return float(np.dot(embs[0], embs[1]))
        except Exception:
            pass
    return _jaccard_similarity(text_a, text_b)


def compute_intra_chapter_redundancy(
    chapters: List[str],
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Measures mean semantic/lexical similarity between adjacent paragraphs inside each chapter.
    Higher values indicate repetitive padding / looping narration.
    """
    per_ch: List[float] = []
    for ch in chapters:
        clean = strip_think_tags(ch)
        paras = [p.strip() for p in re.split(r"\n\s*\n+", clean) if len(p.split()) >= 15]
        if len(paras) < 2:
            # Fallback to half-split when chapter has no double newlines
            words = clean.split()
            if len(words) >= 60:
                mid = len(words) // 2
                paras = [" ".join(words[:mid]), " ".join(words[mid:])]
            else:
                per_ch.append(0.0)
                continue

        if embedder is not None:
            try:
                import numpy as np

                embs = embedder.encode(paras, normalize_embeddings=True)
                sims = [float(np.dot(embs[i], embs[i + 1])) for i in range(len(paras) - 1)]
            except Exception:
                sims = [_jaccard_similarity(paras[i], paras[i + 1]) for i in range(len(paras) - 1)]
        else:
            sims = [_jaccard_similarity(paras[i], paras[i + 1]) for i in range(len(paras) - 1)]
        per_ch.append(round(sum(sims) / max(1, len(sims)), 4))

    overall = round(sum(per_ch) / max(1, len(per_ch)), 4) if per_ch else 0.0
    return {
        "intra_chapter_redundancy": overall,
        "per_chapter_redundancy": per_ch,
    }


def compute_transition_continuity(
    chapters: List[str],
    transitions: Optional[List[Dict[str, Any]]] = None,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Computes:
      - `transition_cosine`: semantic continuity between last 180 words of Ch k-1 and first 150 words of Ch k.
      - `anti_context_novelty`: fraction of Ch k prose NOT verbatim-copied from Coordinator `context_summary`
        (when `coordinator_transitions.jsonl` is provided).
      - `fallback_rate_pct`: percentage of transitions where ContextGate Validator triggered fallback.
    """
    trans_cosines: List[float] = []
    for i in range(1, len(chapters)):
        prev_words = strip_think_tags(chapters[i - 1]).split()
        curr_words = strip_think_tags(chapters[i]).split()
        tail = " ".join(prev_words[-180:])
        head = " ".join(curr_words[:150])
        if tail and head:
            trans_cosines.append(round(_pair_similarity(tail, head, embedder=embedder), 4))

    mean_cos = round(sum(trans_cosines) / len(trans_cosines), 4) if trans_cosines else None

    anti_ctx_scores: List[float] = []
    fallback_flags: List[bool] = []
    if transitions:
        by_ch = {int(t.get("chapter_index", 0)): t for t in transitions}
        for ch_idx in range(2, len(chapters) + 1):
            tr = by_ch.get(ch_idx)
            if not tr:
                continue
            fallback_flags.append(bool(tr.get("is_fallback", False)))
            summary_txt = str(tr.get("final_summary") or tr.get("context_summary") or "").strip()
            if summary_txt:
                anti_ctx_scores.append(
                    round(calculate_anti_context_ratio(strip_think_tags(chapters[ch_idx - 1]), summary_txt), 4)
                )

    mean_anti_ctx = round(sum(anti_ctx_scores) / len(anti_ctx_scores), 4) if anti_ctx_scores else None
    fb_rate = (
        round((sum(1 for f in fallback_flags if f) / len(fallback_flags)) * 100.0, 2)
        if fallback_flags
        else None
    )

    return {
        "transition_cosine": mean_cos,
        "anti_context_novelty": mean_anti_ctx,
        "fallback_rate_pct": fb_rate,
        "per_transition_cosine": trans_cosines,
    }


def compute_reasoning_and_length(
    chapters: List[str],
    raw_chapters: Optional[List[str]],
    target_words: int,
) -> Dict[str, Any]:
    """Tracks prose length compliance, chapter length std, early termination, overshoot, and `<think>` budget."""
    clean_chapters = [strip_think_tags(c) for c in chapters]
    ch_word_counts = [len(c.split()) for c in clean_chapters]
    actual_words = sum(ch_word_counts)

    if len(ch_word_counts) > 1:
        mean_ch = actual_words / len(ch_word_counts)
        ch_std = round(math.sqrt(sum((x - mean_ch) ** 2 for x in ch_word_counts) / len(ch_word_counts)), 1)
    else:
        ch_std = 0.0

    think_words = 0
    source_raw = raw_chapters if raw_chapters else chapters
    for raw in source_raw:
        for block in re.findall(r"<think>([\s\S]*?)</think>", raw or "", flags=re.IGNORECASE):
            think_words += len(block.split())

    if target_words > 0:
        sl = max(0.0, min(100.0, round(100.0 * (1.0 - abs(actual_words - target_words) / target_words), 2)))
        early_term = actual_words < 0.5 * target_words
        overshot = actual_words > 1.35 * target_words
    else:
        sl = 100.0
        early_term = False
        overshot = False

    think_ratio = round((think_words / max(1, actual_words + think_words)) * 100.0, 2)

    return {
        "actual_words": actual_words,
        "target_words": target_words,
        "length_score_sl": sl,
        "early_terminated": early_term,
        "overshot": overshot,
        "think_words": think_words,
        "think_ratio_pct": think_ratio,
        "num_chapters": len(clean_chapters),
        "chapter_words_std": ch_std,
    }
