"""
CONCOCT-inspired Concreteness, Pacing Balance, Inter-Event Diversity & Critic Pass Rate
for standalone Outline Planning evaluation (`MacroPlan-Bench`).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from src.planning import PlanCriticAgent


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "their", "they",
    "about", "after", "before", "while", "where", "which", "when", "then", "than",
    "have", "has", "had", "will", "would", "could", "should", "must", "being",
    "chapter", "event", "scene", "story", "protagonist", "conflict", "action",
}


def _extract_outline_units(blueprint: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extracts canonical planning units (Macro Events preferred, else chapters)."""
    events = blueprint.get("event_graph", {}).get("events", [])
    if events:
        return [
            {
                "id": ev.get("event_id", f"E{i}"),
                "setting": str(ev.get("setting", "")).strip(),
                "character": str(ev.get("character", "")).strip(),
                "action": str(ev.get("action", "")).strip(),
                "conflict": str(ev.get("conflict", "")).strip(),
                "plot_twist": str(ev.get("plot_twist", "")).strip(),
                "text": " ".join(
                    str(ev.get(k, "")).strip()
                    for k in ("title", "setting", "character", "action", "conflict", "plot_twist")
                    if ev.get(k)
                ),
            }
            for i, ev in enumerate(events, 1)
        ]

    chapters = blueprint.get("chapters", [])
    return [
        {
            "id": f"Ch{ch.get('chapter_index', i)}",
            "setting": "",
            "character": ", ".join(ch.get("characters_active", [])),
            "action": str(ch.get("chapter_goal", "")).strip(),
            "conflict": "",
            "plot_twist": str(ch.get("exit_state", "")).strip(),
            "text": " ".join(
                [
                    str(ch.get("chapter_title", "")),
                    str(ch.get("chapter_goal", "")),
                    " ".join(str(b) for b in ch.get("required_beats", [])),
                    str(ch.get("exit_state", "")),
                ]
            ).strip(),
        }
        for i, ch in enumerate(chapters, 1)
    ]


def compute_unit_concreteness(unit: Dict[str, Any]) -> float:
    """
    Estimates concreteness / specificity of a single outline event (0.0 to 1.0),
    inspired by CONCOCT (EMNLP 2023):
      - Field completeness (setting, character, action, conflict, twist populated & non-generic)
      - Specific content-word density (non-stopword words >= 5 chars)
      - Named entity / proper noun richness
      - Substantive detail length (saturates cleanly around 45+ words per event)
    """
    text = unit.get("text", "").strip()
    words = re.findall(r"[A-Za-z0-9'-]+", text)
    if not words:
        return 0.0

    # 1. Structural field population (0..1)
    generic_placeholders = {"none", "unknown", "unknown setting", "protagonist", "story progresses", ""}
    populated = 0
    fields = ("setting", "character", "action", "conflict", "plot_twist")
    for k in fields:
        val = unit.get(k, "").strip().lower()
        if val and val not in generic_placeholders and len(val.split()) >= 2:
            populated += 1
    field_score = populated / float(len(fields))

    # 2. Specific lexical content density (content words >= 5 chars not in stopwords)
    content_words = [w.lower() for w in words if len(w) >= 5 and w.lower() not in _STOPWORDS]
    unique_content = len(set(content_words))
    specificity_score = min(1.0, unique_content / 22.0)

    # 3. Named entity / proper noun richness (mid-sentence capitalized tokens)
    raw_tokens = text.split()
    proper_nouns = {
        re.sub(r"\W+", "", t)
        for idx, t in enumerate(raw_tokens)
        if idx > 0 and t and t[0].isupper() and len(re.sub(r"\W+", "", t)) >= 2
    }
    entity_score = min(1.0, len(proper_nouns) / 5.0)

    # 4. Detail length adequacy (penalizes sparse 5-word stubs; saturates at ~45 words)
    length_score = min(1.0, len(words) / 45.0)

    score = 0.35 * field_score + 0.30 * specificity_score + 0.15 * entity_score + 0.20 * length_score
    return round(max(0.0, min(1.0, score)), 4)


def _pairwise_lexical_diversity(texts: List[str]) -> float:
    """Computes 1 - mean pairwise Jaccard similarity across all pairs of outline events."""
    if len(texts) < 2:
        return 1.0
    token_sets = [
        {w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) >= 4 and w not in _STOPWORDS}
        for t in texts
    ]
    sims: List[float] = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            a, b = token_sets[i], token_sets[j]
            if not a and not b:
                sims.append(1.0)
            elif not a or not b:
                sims.append(0.0)
            else:
                sims.append(len(a & b) / float(len(a | b)))
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    return round(max(0.0, min(1.0, 1.0 - mean_sim)), 4)


def evaluate_concreteness_and_pacing(
    blueprint: Dict[str, Any],
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Evaluates CONCOCT-inspired pacing balance, concreteness, inter-event diversity,
    and PlanCritic structural compliance for an outline blueprint.
    """
    units = _extract_outline_units(blueprint)
    if not units:
        return {
            "mean_concreteness": 0.0,
            "concreteness_std": 0.0,
            "pacing_balance_score": 0.0,
            "min_event_concreteness": 0.0,
            "sparse_event_ratio": 1.0,
            "inter_event_diversity": 0.0,
            "plan_critic_pass_rate": 0.0,
            "per_event_concreteness": [],
        }

    conc_scores = [compute_unit_concreteness(u) for u in units]
    mean_c = sum(conc_scores) / len(conc_scores)
    var_c = sum((x - mean_c) ** 2 for x in conc_scores) / len(conc_scores)
    std_c = math.sqrt(var_c)

    # Coefficient-of-variation-based pacing balance:
    # Penalizes erratic outlines where some chapters are dense and others are empty/vague.
    cv = std_c / max(mean_c, 0.15)
    balance_raw = max(0.0, 1.0 - cv)
    # Weight by mean concreteness so a uniformly empty outline does NOT score 1.0!
    pacing_balance = round(max(0.0, min(1.0, balance_raw * min(1.0, mean_c / 0.55))), 4)

    sparse_ratio = round(sum(1 for c in conc_scores if c < 0.35) / len(conc_scores), 4)

    # Inter-event semantic diversity (MiniLM embeddings if supplied, else lexical Jaccard)
    texts = [u["text"] for u in units]
    if embedder is not None and len(texts) >= 2:
        try:
            import numpy as np

            embs = embedder.encode(texts, normalize_embeddings=True)
            sim_mat = np.matmul(embs, embs.T)
            n = len(texts)
            off_diag = [float(sim_mat[i, j]) for i in range(n) for j in range(i + 1, n)]
            mean_cos = sum(off_diag) / len(off_diag) if off_diag else 0.0
            diversity = round(max(0.0, min(1.0, 1.0 - max(0.0, mean_cos))), 4)
        except Exception:
            diversity = _pairwise_lexical_diversity(texts)
    else:
        diversity = _pairwise_lexical_diversity(texts)

    # PlanCritic 5-gate structural pass rate across chapters
    chapters = blueprint.get("chapters", [])
    critic_pass_rate = 1.0
    if chapters and blueprint.get("sub_events"):
        critic = PlanCriticAgent()
        # Re-verify each chapter sequentially from a clean completed_sub_event_ids cursor
        sim_plan = dict(blueprint)
        sim_plan["completed_sub_event_ids"] = []
        passed = 0
        for ch in chapters:
            verdict = critic.review(ch, sim_plan)
            if verdict["approved"]:
                passed += 1
            for se in ch.get("sub_events", []):
                sid = se.get("sub_event_id")
                if sid and not str(sid).startswith("continuity."):
                    sim_plan["completed_sub_event_ids"].append(sid)
        critic_pass_rate = round(passed / len(chapters), 4)

    return {
        "mean_concreteness": round(mean_c, 4),
        "concreteness_std": round(std_c, 4),
        "pacing_balance_score": pacing_balance,
        "min_event_concreteness": round(min(conc_scores), 4),
        "sparse_event_ratio": sparse_ratio,
        "inter_event_diversity": diversity,
        "plan_critic_pass_rate": critic_pass_rate,
        "per_event_concreteness": conc_scores,
    }
