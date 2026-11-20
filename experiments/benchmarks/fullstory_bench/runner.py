"""
Standalone Full-Story Benchmark Runner (`FullStory-Bench`).

Reads saved story JSONLs directly (e.g., `stories_7b_outputs.jsonl` + optional
`coordinator_transitions.jsonl` from `ablation_context_gate_validator`, or
`track2_generated.jsonl` from Base vs GRPO comparisons) and computes:
  - Phase 1 (Deterministic + DeBERTa-v3 NLI + MiniLM Embeddings):
      Length Compliance (`length_score_sl`), Early Termination / Overshoot, `<think>` word budget,
      DOC-NLI Plan Adherence (`S_adh`), Inter-Chapter Continuity (`transition_cosine`),
      Anti-Context Novelty & Fallback Rate, Lexical Richness (`Distinct-3/4`, Entropy, Burstiness),
      Slop Score per 1k words, Intra-Chapter Redundancy.
  - Phase 2 (Judge LLM — `none | mock | api | local_vllm`):
      ConStory-Checker (5 Categories / 19 Subtypes + Evidence Quotes + `errors_per_1k` + `C_weighted`),
      EQ-Bench Longform (6 Axes + Degradation Penalty + 0–100 `eqbench_score`),
      Blind Pairwise Story Judge (Swapped A/B order).
"""

from __future__ import annotations

import csv
import gc
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from experiments.benchmark.evaluator import (
    NLIScorer,
    evaluate_lexical_and_burstiness,
    evaluate_plan_adherence_doc_nli,
)
from experiments.benchmarks._judge_utils import UnifiedJudgeBackend
from .constory_checker import ConStoryCheckerEvaluator
from .eqbench_evaluator import BlindPairwiseStoryJudge, EQBenchLongformEvaluator
from .metrics_adherence import (
    compute_intra_chapter_redundancy,
    compute_reasoning_and_length,
    compute_slop_score,
    compute_transition_continuity,
)


def _split_full_story_into_chapters(full_story: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"(?:^|\n)##\s+Chapter\s+\d+[^\n]*\n", full_story) if p.strip()]
    return parts if parts else ([full_story.strip()] if full_story.strip() else [])


def load_story_records(
    stories_paths: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Loads full-story records from one or more JSONL files (`stories_7b_outputs.jsonl`, `track2_generated.jsonl`)."""
    records: List[Dict[str, Any]] = []
    for idx_f, raw_p in enumerate(stories_paths):
        p = Path(raw_p)
        if not p.exists():
            raise FileNotFoundError(f"Story JSONL not found: {p}")
        default_arm = labels[idx_f] if (labels and idx_f < len(labels)) else p.stem

        with open(p, "r", encoding="utf-8-sig") as f:
            for row_idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                arm = str(row.get("arm") or row.get("condition") or row.get("group") or default_arm)
                pid = str(row.get("premise_id") or row.get("id") or f"story_{row_idx:03d}")
                chapters = row.get("chapters")
                if not chapters:
                    full_s = str(row.get("full_story") or row.get("story") or "")
                    chapters = _split_full_story_into_chapters(full_s)
                chapters = [str(c) for c in chapters]
                raw_chapters = row.get("chapters_raw_with_think")
                if raw_chapters:
                    raw_chapters = [str(c) for c in raw_chapters]
                records.append(
                    {
                        "arm": arm,
                        "premise_id": pid,
                        "genre": str(row.get("genre", "Creative Fiction")),
                        "premise": str(row.get("premise", "")),
                        "target_words": int(row.get("target_words", 8000)),
                        "chapters": chapters,
                        "raw_chapters": raw_chapters,
                        "story_plan": row.get("story_plan") or row.get("blueprint") or {},
                    }
                )
    return records


def load_transitions_index(transitions_path: Optional[str]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Indexes `coordinator_transitions.jsonl` by `(arm, premise_id)`."""
    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    if not transitions_path:
        return index
    p = Path(transitions_path)
    if not p.exists():
        raise FileNotFoundError(f"Transitions JSONL not found: {p}")
    with open(p, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            arm = str(row.get("arm") or row.get("condition") or row.get("group") or "")
            pid = str(row.get("premise_id") or row.get("id") or "")
            index[(arm, pid)].append(row)
    return index


def _mock_fullstory_judge(system_prompt: str, user_prompt: str) -> str:
    """Deterministic mock judge supporting ConStory-Checker, EQ-Bench, and Blind Pairwise."""
    if "ConStory-Checker" in system_prompt:
        # Inject one deterministic mock error if the story mentions "contradiction_trigger"
        if "contradiction_trigger" in user_prompt.lower():
            return json.dumps(
                {
                    "errors": [
                        {
                            "category": "characterization",
                            "subtype": "memory_contradiction",
                            "severity": "moderate",
                            "chapter_a": 1,
                            "chapter_b": 4,
                            "quote_a": "She knew the cipher from childhood.",
                            "quote_b": "She had never seen the cipher before.",
                            "explanation": "Protagonist forgets previously established core knowledge.",
                        }
                    ],
                    "summary": "Detected 1 moderate characterization contradiction.",
                }
            )
        return json.dumps({"errors": [], "summary": "No major contradictions detected in mock audit."})

    if "EQ-Bench Longform" in system_prompt:
        return json.dumps(
            {
                "nuanced_characters": 7.8,
                "emotionally_engaging": 7.6,
                "compelling_plot": 8.0,
                "coherence": 8.2,
                "prose_quality": 7.9,
                "meaningful_elaboration": 8.1,
                "early_chapters_score": 8.1,
                "late_chapters_score": 7.9,
                "rationale": "Mock EQ-Bench evaluation: consistent narrative progression with minor late tapering.",
            }
        )

    # Blind pairwise
    sec_a = user_prompt.split("[STORY A]:", 1)[1].split("[STORY B]:", 1)[0]
    sec_b = user_prompt.split("[STORY B]:", 1)[1]
    winner = "A" if len(sec_a) >= len(sec_b) else "B"
    return json.dumps(
        {
            "winner": winner,
            "winning_axes": ["consistency", "continuity"],
            "rationale": f"Mock pairwise judge preferred Story {winner}.",
        }
    )


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)


def _try_load_embedder(model_id: str, use_mock: bool) -> Optional[Any]:
    if use_mock or not model_id or model_id.lower() == "none":
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_id)
    except Exception as exc:
        print(f"[FullStoryBench] Embedder {model_id} unavailable ({exc}); using lexical Jaccard fallback.", flush=True)
        return None


def _release_embedder(embedder: Optional[Any]) -> None:
    if embedder is None:
        return
    del embedder
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_fullstory_benchmark(
    stories_paths: Sequence[str],
    output_dir: str,
    labels: Optional[Sequence[str]] = None,
    transitions_path: Optional[str] = None,
    use_mock: bool = False,
    nli_model_id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    embedder_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    judge_backend: str = "none",
    judge_model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
    judge_api_key: Optional[str] = None,
    judge_api_base: Optional[str] = None,
    judge_gpu_mem: float = 0.85,
    judge_max_len: int = 24576,
    pairwise_arms: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Executes the full `FullStory-Bench` pipeline and exports all tables, errors, and Markdown report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = load_story_records(stories_paths, labels=labels)
    if not records:
        raise ValueError("No story records loaded.")
    transitions_idx = load_transitions_index(transitions_path)

    # Phase 1: Deterministic + NLI + Embeddings
    nli_scorer = NLIScorer(model_id=nli_model_id, use_mock=use_mock)
    embedder = _try_load_embedder(embedder_model_id, use_mock=use_mock)

    raw_rows: List[Dict[str, Any]] = []
    for rec in records:
        chapters = rec["chapters"]
        full_text = "\n\n".join(chapters)
        len_res = compute_reasoning_and_length(chapters, rec["raw_chapters"], rec["target_words"])
        adh_res = evaluate_plan_adherence_doc_nli(
            chapters,
            rec["story_plan"],
            nli_scorer=nli_scorer,
            premise=rec["premise"],
        )
        lex_res = evaluate_lexical_and_burstiness(full_text)
        slop_res = compute_slop_score(full_text)
        red_res = compute_intra_chapter_redundancy(chapters, embedder=embedder)
        tr_list = transitions_idx.get((rec["arm"], rec["premise_id"]))
        cont_res = compute_transition_continuity(chapters, transitions=tr_list, embedder=embedder)

        raw_rows.append(
            {
                "arm": rec["arm"],
                "premise_id": rec["premise_id"],
                "genre": rec["genre"],
                "actual_words": len_res["actual_words"],
                "target_words": len_res["target_words"],
                "length_score_sl": len_res["length_score_sl"],
                "early_terminated": int(len_res["early_terminated"]),
                "overshot": int(len_res["overshot"]),
                "think_words": len_res["think_words"],
                "think_ratio_pct": len_res["think_ratio_pct"],
                "num_chapters": len_res["num_chapters"],
                "chapter_words_std": len_res["chapter_words_std"],
                "S_adh": adh_res["S_adh"],
                "mean_goal_entailment": adh_res["mean_goal_entailment"],
                "transition_cosine": cont_res["transition_cosine"],
                "anti_context_novelty": cont_res["anti_context_novelty"],
                "fallback_rate_pct": cont_res["fallback_rate_pct"],
                "distinct_3_pct": lex_res["Distinct-3 (%)"],
                "distinct_4_pct": lex_res["Distinct-4 (%)"],
                "shannon_entropy": lex_res["shannon_entropy"],
                "burstiness_std": lex_res["burstiness_std"],
                "slop_hits": slop_res["slop_hits"],
                "slop_score_per_1k": slop_res["slop_score_per_1k"],
                "intra_chapter_redundancy": red_res["intra_chapter_redundancy"],
                # Judge placeholders
                "constory_total_errors": None,
                "constory_errors_per_1k": None,
                "C_weighted": None,
                "err_characterization": None,
                "err_factual_detail": None,
                "err_narrative_style": None,
                "err_timeline_plot": None,
                "err_world_setting": None,
                "eq_nuanced_characters": None,
                "eq_emotionally_engaging": None,
                "eq_compelling_plot": None,
                "eq_coherence": None,
                "eq_prose_quality": None,
                "eq_meaningful_elaboration": None,
                "eq_degradation_penalty": None,
                "eqbench_score": None,
            }
        )

    # Free Phase 1 models before loading Judge
    nli_scorer.unload()
    _release_embedder(embedder)

    # Phase 2: Judge LLM (ConStory-Checker + EQ-Bench + Blind Pairwise)
    effective_backend = "mock" if (use_mock and judge_backend != "none") else judge_backend
    all_constory_errors: List[Dict[str, Any]] = []
    pairwise_results: List[Dict[str, Any]] = []

    if effective_backend != "none":
        judge_mgr = UnifiedJudgeBackend(
            backend=effective_backend,
            model=judge_model,
            api_key=judge_api_key,
            api_base=judge_api_base,
            gpu_memory_utilization=judge_gpu_mem,
            max_model_len=judge_max_len,
            mock_fn=_mock_fullstory_judge if effective_backend == "mock" else None,
        )
        try:
            constory_eval = ConStoryCheckerEvaluator(judge_fn=judge_mgr.judge_fn)
            eq_eval = EQBenchLongformEvaluator(judge_fn=judge_mgr.judge_fn)
            pw_judge = BlindPairwiseStoryJudge(judge_fn=judge_mgr.judge_fn)

            for row, rec in zip(raw_rows, records):
                cs_res = constory_eval.evaluate(
                    rec["premise"], rec["chapters"], actual_words=row["actual_words"]
                )
                eq_res = eq_eval.evaluate(
                    rec["premise"], rec["chapters"], slop_score_per_1k=row["slop_score_per_1k"]
                )
                row.update(
                    {
                        "constory_total_errors": cs_res["constory_total_errors"],
                        "constory_errors_per_1k": cs_res["constory_errors_per_1k"],
                        "C_weighted": cs_res["C_weighted"],
                        "err_characterization": cs_res["err_characterization"],
                        "err_factual_detail": cs_res["err_factual_detail"],
                        "err_narrative_style": cs_res["err_narrative_style"],
                        "err_timeline_plot": cs_res["err_timeline_plot"],
                        "err_world_setting": cs_res["err_world_setting"],
                        "eq_nuanced_characters": eq_res["eq_nuanced_characters"],
                        "eq_emotionally_engaging": eq_res["eq_emotionally_engaging"],
                        "eq_compelling_plot": eq_res["eq_compelling_plot"],
                        "eq_coherence": eq_res["eq_coherence"],
                        "eq_prose_quality": eq_res["eq_prose_quality"],
                        "eq_meaningful_elaboration": eq_res["eq_meaningful_elaboration"],
                        "eq_degradation_penalty": eq_res["eq_degradation_penalty"],
                        "eqbench_score": eq_res["eqbench_score"],
                    }
                )
                for err in cs_res["evidence_errors"]:
                    all_constory_errors.append(
                        {"arm": rec["arm"], "premise_id": rec["premise_id"], **err}
                    )

            if pairwise_arms:
                arm_a, arm_b = pairwise_arms
                by_key = {(r["arm"], r["premise_id"]): r for r in records}
                pids_a = {r["premise_id"] for r in records if r["arm"] == arm_a}
                pids_b = {r["premise_id"] for r in records if r["arm"] == arm_b}
                for pid in sorted(pids_a & pids_b):
                    rec_a = by_key[(arm_a, pid)]
                    rec_b = by_key[(arm_b, pid)]
                    verdict = pw_judge.compare_pair(
                        rec_a["premise"],
                        rec_a["chapters"],
                        rec_b["chapters"],
                        label_a=arm_a,
                        label_b=arm_b,
                    )
                    verdict["premise_id"] = pid
                    pairwise_results.append(verdict)
        finally:
            judge_mgr.release()

    # Export raw JSONL, errors JSONL, pairwise JSON
    raw_path = out / "fullstory_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in raw_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    errors_path = out / "constory_errors.jsonl"
    with open(errors_path, "w", encoding="utf-8") as f:
        for e in all_constory_errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    if pairwise_results:
        (out / "pairwise_verdicts.json").write_text(
            json.dumps(pairwise_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Aggregate summary by arm
    by_arm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in raw_rows:
        by_arm[r["arm"]].append(r)

    metric_cols = [
        "actual_words",
        "length_score_sl",
        "think_words",
        "S_adh",
        "transition_cosine",
        "anti_context_novelty",
        "fallback_rate_pct",
        "distinct_3_pct",
        "distinct_4_pct",
        "burstiness_std",
        "slop_score_per_1k",
        "intra_chapter_redundancy",
        "constory_errors_per_1k",
        "C_weighted",
        "eq_meaningful_elaboration",
        "eq_degradation_penalty",
        "eqbench_score",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for arm, arm_rows in by_arm.items():
        agg: Dict[str, Any] = {
            "arm": arm,
            "n_stories": len(arm_rows),
            "early_term_rate_pct": round(100.0 * sum(r["early_terminated"] for r in arm_rows) / len(arm_rows), 2),
            "overshoot_rate_pct": round(100.0 * sum(r["overshot"] for r in arm_rows) / len(arm_rows), 2),
        }
        for col in metric_cols:
            agg[col] = _safe_mean([r.get(col) for r in arm_rows])
        summary_rows.append(agg)

    csv_fields = ["arm", "n_stories", "early_term_rate_pct", "overshoot_rate_pct"] + metric_cols
    csv_path = out / "fullstory_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Render Markdown report
    fmt = lambda v: "N/A" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    md_lines = [
        "# FullStory-Bench (Suite 2 — Long-Form Story Evaluation Report)",
        "",
        f"- **Inputs**: `{', '.join(str(p) for p in stories_paths)}`",
        f"- **Total Stories Evaluated**: `{len(raw_rows)}`",
        f"- **Judge Backend**: `{effective_backend}`" + (f" (`{judge_model}`)" if effective_backend != "none" else ""),
        "",
        "## Summary Table by Arm / Condition",
        "",
        "| Arm | N | Words | Sl (%) | S_adh | Trans Cos | Anti-Ctx | D-3 (%) | Slop/1k | Redund | Err/1k | C_weighted | Degrad | EQ-Bench (0-100) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summary_rows:
        md_lines.append(
            f"| {s['arm']} | {s['n_stories']} | {fmt(s['actual_words'])} | {fmt(s['length_score_sl'])} | "
            f"{fmt(s['S_adh'])} | {fmt(s['transition_cosine'])} | {fmt(s['anti_context_novelty'])} | "
            f"{fmt(s['distinct_3_pct'])} | {fmt(s['slop_score_per_1k'])} | {fmt(s['intra_chapter_redundancy'])} | "
            f"{fmt(s['constory_errors_per_1k'])} | {fmt(s['C_weighted'])} | {fmt(s['eq_degradation_penalty'])} | "
            f"{fmt(s['eqbench_score'])} |"
        )

    if pairwise_results and pairwise_arms:
        arm_a, arm_b = pairwise_arms
        wins_a = sum(1 for v in pairwise_results if v["winner"] == arm_a)
        wins_b = sum(1 for v in pairwise_results if v["winner"] == arm_b)
        ties = sum(1 for v in pairwise_results if v["winner"] == "tie")
        cons = sum(1 for v in pairwise_results if v["position_consistent"])
        md_lines.extend(
            [
                "",
                f"## Blind Pairwise Story Comparison (`{arm_a}` vs `{arm_b}`)",
                "",
                f"- `{arm_a}` wins: **{wins_a}** | `{arm_b}` wins: **{wins_b}** | Ties: **{ties}**",
                f"- Position-swap consistency: **{cons}/{len(pairwise_results)}**",
            ]
        )

    report_path = out / "fullstory_benchmark_report.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "summary": summary_rows,
        "raw_path": str(raw_path),
        "csv_path": str(csv_path),
        "errors_path": str(errors_path),
        "report_path": str(report_path),
        "pairwise": pairwise_results,
    }
