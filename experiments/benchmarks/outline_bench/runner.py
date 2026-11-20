"""
Standalone Outline Planning Benchmark Runner (`MacroPlan-Bench`).

Evaluates outlines BEFORE prose generation across 5 axes:
  1. Premise Grounding & Relevance (S_cov, Judge outline_relevance)
  2. Causal Integrity & Progression (S_causal, dag_connectivity, S_prog, S_comp)
  3. CONCOCT-Inspired Concreteness & Pacing Balance (mean_concreteness, pacing_balance_score,
     sparse_event_ratio, inter_event_diversity, plan_critic_pass_rate)
  4. Narrative Appeal & Blind Pairwise Comparison (DOC + HANNA Judge)
  5. Predictive Utility & Cost (VR-CLI I_pi on LONGSTORY/BookSum, planning_tokens)
Plus Optional Part 3 Meta-Evaluation (Perturbation Sanity Check).
"""

from __future__ import annotations

import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from experiments.benchmark.evaluator import (
    NLIScorer,
    VRCLIPerplexityEvaluator,
    compute_causal_coherence,
    compute_conflict_completeness,
    compute_nli_progression,
    compute_premise_coverage,
    load_longstory_next_chapter_samples,
)
from experiments.benchmarks._judge_utils import UnifiedJudgeBackend
from .judge_outline import (
    OutlineJudgeEvaluator,
    format_blueprint_for_judge,
    run_outline_perturbation_sanity_check,
)
from .metrics_pacing import evaluate_concreteness_and_pacing


def _reconstruct_blueprint_from_grpo_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reconstructs a chapter-level blueprint from per-chapter GRPO dataset rows."""
    sorted_rows = sorted(rows, key=lambda r: int(r.get("chapter_index", 0)))
    chapters = []
    for r in sorted_rows:
        chapters.append(
            {
                "chapter_index": int(r.get("chapter_index", len(chapters) + 1)),
                "chapter_title": r.get("chapter_title", f"Chapter {r.get('chapter_index', len(chapters) + 1)}"),
                "chapter_goal": r.get("chapter_goal", ""),
                "required_beats": r.get("required_beats", []),
                "exit_state": r.get("exit_state", ""),
                "characters_active": r.get("characters_active", []),
                "target_words": int(r.get("target_words", 1000)),
            }
        )
    return {"chapters": chapters, "planning_metadata": {"planning_tokens": 0}}


def load_outline_records(
    input_paths: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Loads outline records from one or more files. Supports:
      1. `shared_macro_plans.json` (JSON list of {"item": {...}, "event_graph": {...}, "story_plan": {...}})
      2. JSONL with `story_plan` or `blueprint` (`stories_7b_outputs.jsonl`, `track2_generated.jsonl`)
      3. Chapter-level GRPO JSONL (`grpo_train_condition_d.jsonl`, grouped by `premise_id`)
    """
    records: List[Dict[str, Any]] = []
    for file_idx, raw_path in enumerate(input_paths):
        p = Path(raw_path)
        if not p.exists():
            raise FileNotFoundError(f"Outline benchmark input not found: {p}")
        default_group = labels[file_idx] if (labels and file_idx < len(labels)) else p.stem

        if p.suffix.lower() == ".json":
            payload = json.loads(p.read_text(encoding="utf-8-sig"))
            items = payload if isinstance(payload, list) else payload.get("plans", [payload])
            for idx, entry in enumerate(items):
                item_meta = entry.get("item", {}) if isinstance(entry.get("item"), dict) else {}
                premise_id = str(
                    entry.get("premise_id")
                    or item_meta.get("id")
                    or entry.get("id")
                    or f"plan_{idx:03d}"
                )
                premise = str(entry.get("premise") or item_meta.get("premise") or "")
                genre = str(entry.get("genre") or item_meta.get("genre") or "Creative Fiction")
                group = str(entry.get("group") or entry.get("condition") or entry.get("arm") or default_group)
                blueprint = entry.get("story_plan") or entry.get("blueprint") or {}
                if "event_graph" in entry and "event_graph" not in blueprint:
                    blueprint = dict(blueprint)
                    blueprint["event_graph"] = entry["event_graph"]
                records.append(
                    {
                        "group": group,
                        "premise_id": premise_id,
                        "genre": genre,
                        "premise": premise,
                        "blueprint": blueprint,
                    }
                )
            continue

        # JSONL handling
        rows: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        if not rows:
            continue

        # Detect chapter-level GRPO rows (no story_plan/blueprint, has chapter_index + chapter_goal)
        is_chapter_grpo = (
            "story_plan" not in rows[0]
            and "blueprint" not in rows[0]
            and "event_graph" not in rows[0]
            and "chapter_index" in rows[0]
        )
        if is_chapter_grpo:
            by_pid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in rows:
                pid = str(r.get("premise_id") or r.get("id") or "unknown")
                by_pid[pid].append(r)
            for pid, ch_rows in by_pid.items():
                first = ch_rows[0]
                records.append(
                    {
                        "group": str(first.get("group") or first.get("condition") or default_group),
                        "premise_id": pid,
                        "genre": str(first.get("genre", "Creative Fiction")),
                        "premise": str(first.get("premise", "")),
                        "blueprint": _reconstruct_blueprint_from_grpo_rows(ch_rows),
                    }
                )
        else:
            seen_keys = set()
            for idx, r in enumerate(rows):
                group = str(r.get("group") or r.get("condition") or r.get("arm") or default_group)
                pid = str(r.get("premise_id") or r.get("id") or f"plan_{idx:03d}")
                key = (group, pid)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                bp = r.get("story_plan") or r.get("blueprint") or {}
                if "event_graph" in r and "event_graph" not in bp:
                    bp = dict(bp)
                    bp["event_graph"] = r["event_graph"]
                records.append(
                    {
                        "group": group,
                        "premise_id": pid,
                        "genre": str(r.get("genre", "Creative Fiction")),
                        "premise": str(r.get("premise", "")),
                        "blueprint": bp,
                    }
                )
    return records


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)


def _mock_outline_judge(system_prompt: str, user_prompt: str) -> str:
    """Deterministic mock judge for offline unit tests and CI smoke runs."""
    if "OUTLINE A" in user_prompt and "OUTLINE B" in user_prompt:
        sec_a = user_prompt.split("[OUTLINE A]:", 1)[1].split("[OUTLINE B]:", 1)[0]
        sec_b = user_prompt.split("[OUTLINE B]:", 1)[1]
        winner = "A" if len(sec_a) >= len(sec_b) else "B"
        return json.dumps(
            {
                "winner": winner,
                "winning_dimensions": ["plot_coherence", "pacing"],
                "rationale": f"Mock pairwise judge preferred Outline {winner} for denser causal event detail.",
            }
        )
    has_events = "Event " in user_prompt
    base = 4.3 if has_events else 3.4
    return json.dumps(
        {
            "outline_relevance": base,
            "plot_coherence": base + 0.1,
            "surprise_and_twists": base - 0.2,
            "engagement_and_complexity": base,
            "weakest_chapter_index": 4,
            "evidence": [
                {"criterion": "plot_coherence", "span": "Event progression", "reason": "Mock verified causal flow."}
            ],
            "rationale": "Mock outline judge evaluation completed.",
        }
    )


def _try_load_embedder(model_id: str, use_mock: bool) -> Optional[Any]:
    if use_mock or not model_id or model_id.lower() == "none":
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_id)
    except Exception as exc:
        print(f"[OutlineBench] Embedder {model_id} unavailable ({exc}); using lexical Jaccard diversity.", flush=True)
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


def run_outline_benchmark(
    input_paths: Sequence[str],
    output_dir: str,
    labels: Optional[Sequence[str]] = None,
    use_mock: bool = False,
    nli_model_id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    embedder_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    judge_backend: str = "none",
    judge_model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
    judge_api_key: Optional[str] = None,
    judge_api_base: Optional[str] = None,
    judge_gpu_mem: float = 0.85,
    judge_max_len: int = 16384,
    pairwise_groups: Optional[Tuple[str, str]] = None,
    vrcli_data: Optional[str] = None,
    vrcli_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    vrcli_max_samples: int = 20,
    run_sanity_check: bool = False,
) -> Dict[str, Any]:
    """Executes the full `MacroPlan-Bench` evaluation pipeline and writes all report files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = load_outline_records(input_paths, labels=labels)
    if not records:
        raise ValueError("No outline records loaded from input paths.")

    # Phase 1: Deterministic + NLI + Embedding metrics (before loading heavy Judge)
    nli_scorer = NLIScorer(model_id=nli_model_id, use_mock=use_mock)
    embedder = _try_load_embedder(embedder_model_id, use_mock=use_mock)

    raw_rows: List[Dict[str, Any]] = []
    for rec in records:
        bp = rec["blueprint"]
        premise = rec["premise"]
        s_cov = compute_premise_coverage(premise, bp)
        prog_res = compute_nli_progression(bp, nli_scorer)
        causal_res = compute_causal_coherence(bp, nli_scorer=nli_scorer)
        s_comp = compute_conflict_completeness(bp)
        pacing_res = evaluate_concreteness_and_pacing(bp, embedder=embedder)

        events = bp.get("event_graph", {}).get("events", [])
        chapters = bp.get("chapters", [])
        planning_tokens = int(bp.get("planning_metadata", {}).get("planning_tokens", 0))
        if planning_tokens == 0:
            planning_tokens = int(len(format_blueprint_for_judge(bp).split()) * 1.35)

        raw_rows.append(
            {
                "group": rec["group"],
                "premise_id": rec["premise_id"],
                "genre": rec["genre"],
                "num_events": len(events),
                "num_chapters": len(chapters),
                "planning_tokens": planning_tokens,
                "S_cov": s_cov,
                "S_prog": prog_res["S_prog"],
                "S_causal": causal_res["S_causal"],
                "dag_connectivity": causal_res["dag_connectivity"],
                "S_comp": s_comp,
                "mean_concreteness": pacing_res["mean_concreteness"],
                "concreteness_std": pacing_res["concreteness_std"],
                "pacing_balance_score": pacing_res["pacing_balance_score"],
                "min_event_concreteness": pacing_res["min_event_concreteness"],
                "sparse_event_ratio": pacing_res["sparse_event_ratio"],
                "inter_event_diversity": pacing_res["inter_event_diversity"],
                "plan_critic_pass_rate": pacing_res["plan_critic_pass_rate"],
                "judge_outline_relevance": None,
                "judge_plot_coherence": None,
                "judge_surprise_and_twists": None,
                "judge_engagement_complexity": None,
                "judge_outline_overall": None,
                "judge_weakest_chapter": None,
                "judge_rationale": None,
            }
        )

    sanity_result: Optional[Dict[str, Any]] = None
    if run_sanity_check:
        clean_bps = [r["blueprint"] for r in records if len(r["blueprint"].get("event_graph", {}).get("events", [])) >= 4]
        sanity_result = run_outline_perturbation_sanity_check(clean_bps, nli_scorer=nli_scorer)
        (out / "outline_sanity_check.json").write_text(
            json.dumps(sanity_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Free Phase 1 models cleanly before VR-CLI or Local Judge
    nli_scorer.unload()
    _release_embedder(embedder)

    # Optional Phase 1b: VR-CLI Predictive Utility (I_pi)
    vrcli_summary: Optional[Dict[str, Any]] = None
    if vrcli_data:
        samples = load_longstory_next_chapter_samples(
            data_path=vrcli_data,
            max_samples=vrcli_max_samples,
            allow_synthetic_demo=use_mock,
        )
        vrcli_eval = VRCLIPerplexityEvaluator(model_id=vrcli_model_id, use_mock=use_mock)
        scores = []
        for idx, smp in enumerate(samples):
            bp = records[idx % len(records)]["blueprint"]
            plan_txt = format_blueprint_for_judge(bp)
            res = vrcli_eval.evaluate_information_gain(smp["history_text"], plan_txt, smp["gold_chapter"])
            scores.append(res["I_pi"])
        vrcli_eval.unload()
        vrcli_summary = {
            "num_samples": len(scores),
            "mean_I_pi_pct": round(sum(scores) / max(1, len(scores)), 2),
        }

    # Phase 2: Optional Judge LLM (Pointwise + Blind Pairwise)
    effective_backend = "mock" if (use_mock and judge_backend != "none") else judge_backend
    pairwise_results: List[Dict[str, Any]] = []
    if effective_backend != "none":
        judge_mgr = UnifiedJudgeBackend(
            backend=effective_backend,
            model=judge_model,
            api_key=judge_api_key,
            api_base=judge_api_base,
            gpu_memory_utilization=judge_gpu_mem,
            max_model_len=judge_max_len,
            mock_fn=_mock_outline_judge if effective_backend == "mock" else None,
        )
        try:
            outline_judge = OutlineJudgeEvaluator(judge_fn=judge_mgr.judge_fn)
            for row, rec in zip(raw_rows, records):
                j_res = outline_judge.evaluate(rec["premise"], rec["blueprint"])
                row.update(
                    {
                        "judge_outline_relevance": j_res["judge_outline_relevance"],
                        "judge_plot_coherence": j_res["judge_plot_coherence"],
                        "judge_surprise_and_twists": j_res["judge_surprise_and_twists"],
                        "judge_engagement_complexity": j_res["judge_engagement_complexity"],
                        "judge_outline_overall": j_res["judge_outline_overall"],
                        "judge_weakest_chapter": j_res["judge_weakest_chapter"],
                        "judge_rationale": j_res["judge_rationale"],
                    }
                )

            if pairwise_groups:
                grp_a, grp_b = pairwise_groups
                by_group_pid = {(r["group"], r["premise_id"]): r for r in records}
                pids_a = {r["premise_id"] for r in records if r["group"] == grp_a}
                pids_b = {r["premise_id"] for r in records if r["group"] == grp_b}
                shared_pids = sorted(pids_a & pids_b)
                for pid in shared_pids:
                    rec_a = by_group_pid[(grp_a, pid)]
                    rec_b = by_group_pid[(grp_b, pid)]
                    verdict = outline_judge.compare_pair(
                        rec_a["premise"],
                        rec_a["blueprint"],
                        rec_b["blueprint"],
                        label_a=grp_a,
                        label_b=grp_b,
                    )
                    verdict["premise_id"] = pid
                    pairwise_results.append(verdict)
                (out / "outline_pairwise.json").write_text(
                    json.dumps(pairwise_results, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        finally:
            judge_mgr.release()

    # Export raw JSONL
    raw_path = out / "outline_benchmark_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in raw_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate summary by group
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in raw_rows:
        by_group[r["group"]].append(r)

    metric_cols = [
        "S_cov",
        "S_causal",
        "dag_connectivity",
        "S_prog",
        "S_comp",
        "mean_concreteness",
        "pacing_balance_score",
        "sparse_event_ratio",
        "inter_event_diversity",
        "plan_critic_pass_rate",
        "planning_tokens",
        "judge_outline_relevance",
        "judge_plot_coherence",
        "judge_surprise_and_twists",
        "judge_engagement_complexity",
        "judge_outline_overall",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for grp, grp_rows in by_group.items():
        agg: Dict[str, Any] = {"group": grp, "n_outlines": len(grp_rows)}
        for col in metric_cols:
            agg[col] = _safe_mean([r.get(col) for r in grp_rows])
        summary_rows.append(agg)

    csv_path = out / "outline_benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "n_outlines"] + metric_cols)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Render Markdown report
    md_lines = [
        "# MacroPlan-Bench (Suite 1 — Outline Planning Benchmark Report)",
        "",
        f"- **Inputs**: `{', '.join(str(p) for p in input_paths)}`",
        f"- **Total Outlines Evaluated**: `{len(raw_rows)}`",
        f"- **Judge Backend**: `{effective_backend}`" + (f" (`{judge_model}`)" if effective_backend != "none" else ""),
        "",
        "## Summary Table by Group",
        "",
        "| Group | N | S_cov | S_causal | S_prog | S_comp | Concreteness | Pacing Balance | Sparse Ratio | Diversity | Critic Pass | Plan Tokens | Judge Overall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summary_rows:
        fmt = lambda v: "N/A" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
        md_lines.append(
            f"| {s['group']} | {s['n_outlines']} | {fmt(s['S_cov'])} | {fmt(s['S_causal'])} | "
            f"{fmt(s['S_prog'])} | {fmt(s['S_comp'])} | {fmt(s['mean_concreteness'])} | "
            f"{fmt(s['pacing_balance_score'])} | {fmt(s['sparse_event_ratio'])} | "
            f"{fmt(s['inter_event_diversity'])} | {fmt(s['plan_critic_pass_rate'])} | "
            f"{fmt(s['planning_tokens'])} | {fmt(s['judge_outline_overall'])} |"
        )

    if pairwise_results:
        grp_a, grp_b = pairwise_groups or ("A", "B")
        wins_a = sum(1 for v in pairwise_results if v["winner"] == grp_a)
        wins_b = sum(1 for v in pairwise_results if v["winner"] == grp_b)
        ties = sum(1 for v in pairwise_results if v["winner"] == "tie")
        cons = sum(1 for v in pairwise_results if v["position_consistent"])
        md_lines.extend(
            [
                "",
                f"## Blind Pairwise Comparison (`{grp_a}` vs `{grp_b}`)",
                "",
                f"- `{grp_a}` wins: **{wins_a}** | `{grp_b}` wins: **{wins_b}** | Ties: **{ties}**",
                f"- Position-swap consistency: **{cons}/{len(pairwise_results)}**",
            ]
        )

    if sanity_result is not None:
        md_lines.extend(
            [
                "",
                "## Part 3: Perturbation Sanity Check (Meta-Evaluation)",
                "",
                f"- Passed: **{sanity_result['passed']}** (tested `{sanity_result['total_tested']}` outlines)",
                f"- Causal break drop rate: `{sanity_result['causal_drop_rate']}`",
                f"- Duplicate event drop rate: `{sanity_result['duplicate_drop_rate']}`",
                f"- Sparse middle pacing drop rate: `{sanity_result['sparse_pacing_drop_rate']}`",
            ]
        )

    if vrcli_summary is not None:
        md_lines.extend(
            [
                "",
                "## VR-CLI Predictive Utility (`I_pi`)",
                "",
                f"- Samples: `{vrcli_summary['num_samples']}`, Mean `I_pi`: **{vrcli_summary['mean_I_pi_pct']}%**",
            ]
        )

    report_path = out / "outline_benchmark_report.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "summary": summary_rows,
        "raw_path": str(raw_path),
        "csv_path": str(csv_path),
        "report_path": str(report_path),
        "pairwise": pairwise_results,
        "sanity_check": sanity_result,
        "vrcli": vrcli_summary,
    }
