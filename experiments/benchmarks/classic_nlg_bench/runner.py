"""
Standalone Runner for `ClassicNLG-Bench` (Traditional Surface Metrics: BLEU-1..4, ROUGE-L,
Distinct-1..3, Entropy-2, Perplexity) + Optional Correlation Audit against `FullStory-Bench`.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from experiments.benchmarks.fullstory_bench.runner import load_story_records
from experiments.benchmarks.outline_bench.judge_outline import compute_spearman_correlation
from src.rewards import strip_think_tags
from .metrics_classic import SlidingWindowPerplexityScorer, evaluate_classic_nlg_single


def _load_reference_map(reference_file: Optional[str]) -> Dict[str, str]:
    """Loads gold reference stories indexed by id/example_id/premise_id (e.g. tell-me-a-story-test.jsonl)."""
    ref_map: Dict[str, str] = {}
    if not reference_file:
        return ref_map
    p = Path(reference_file)
    if not p.exists():
        raise FileNotFoundError(f"Reference file not found: {p}")
    with open(p, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = str(row.get("premise_id") or row.get("example_id") or row.get("id") or "")
            ref_txt = str(row.get("targets") or row.get("gold_reference") or row.get("reference") or "").strip()
            if pid and ref_txt:
                ref_map[pid] = ref_txt
    return ref_map


def _build_plan_reference(rec: Dict[str, Any]) -> str:
    """Fallback reference composed of premise + macro plan (ONLY used when explicitly requested)."""
    parts = [rec.get("premise", "")]
    bp = rec.get("story_plan") or {}
    for ev in bp.get("event_graph", {}).get("events", []):
        parts.append(
            " ".join(
                str(ev.get(k, ""))
                for k in ("title", "setting", "character", "action", "conflict", "plot_twist")
                if ev.get(k)
            )
        )
    if len(parts) == 1:
        for ch in bp.get("chapters", []):
            parts.append(f"{ch.get('chapter_title', '')} {ch.get('chapter_goal', '')} {ch.get('exit_state', '')}")
    return " ".join(p for p in parts if p).strip()


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)


def _compute_classic_vs_narrative_comparison(
    classic_rows: List[Dict[str, Any]],
    fullstory_raw_path: str,
) -> Dict[str, Any]:
    """
    Computes Spearman rank correlations between Traditional NLG metrics and
    FullStory-Bench narrative metrics (EQ-Bench, ConStory C_weighted, DOC-NLI S_adh)
    on matched `(arm, premise_id)` stories to highlight where surface metrics succeed or fail.
    """
    p = Path(fullstory_raw_path)
    if not p.exists():
        raise FileNotFoundError(f"FullStory raw JSONL not found: {p}")

    fs_by_key: Dict[tuple, Dict[str, Any]] = {}
    with open(p, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                fs_by_key[(str(r.get("arm", "")), str(r.get("premise_id", "")))] = r

    matched_pairs = []
    for c_row in classic_rows:
        key = (str(c_row["arm"]), str(c_row["premise_id"]))
        if key in fs_by_key:
            matched_pairs.append((c_row, fs_by_key[key]))

    classic_cols = ["BLEU-4", "ROUGE-L", "Distinct-3 (%)", "Entropy-2", "Perplexity"]
    narrative_cols = ["S_adh", "transition_cosine", "C_weighted", "eqbench_score"]
    correlations: Dict[str, Dict[str, Optional[float]]] = {}

    for c_col in classic_cols:
        correlations[c_col] = {}
        for n_col in narrative_cols:
            xs, ys = [], []
            for c_r, n_r in matched_pairs:
                cv, nv = c_r.get(c_col), n_r.get(n_col)
                if cv is not None and nv is not None:
                    xs.append(float(cv))
                    ys.append(float(nv))
            correlations[c_col][n_col] = compute_spearman_correlation(xs, ys) if len(xs) >= 3 else None

    return {
        "matched_stories": len(matched_pairs),
        "spearman_correlations": correlations,
    }


def run_classic_nlg_benchmark(
    stories_paths: Sequence[str],
    output_dir: str,
    labels: Optional[Sequence[str]] = None,
    reference_file: Optional[str] = None,
    reference_fallback: str = "none",
    ppl_model_id: str = "gpt2",
    ppl_stride: int = 512,
    ppl_max_length: int = 1024,
    skip_ppl: bool = False,
    use_mock: bool = False,
    compare_with_fullstory: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the standalone Classic NLG Benchmark (`BLEU-1..4`, `ROUGE-L`, `Distinct-1..3`,
    `Entropy-2`, `Perplexity`) and exports CSV, JSONL, and Markdown reports.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = load_story_records(stories_paths, labels=labels)
    if not records:
        raise ValueError("No story records loaded.")

    ref_map = _load_reference_map(reference_file)
    # Also inspect raw story files for inline `gold_reference` / `targets` fields
    inline_refs: Dict[tuple, str] = {}
    for raw_p in stories_paths:
        with open(raw_p, "r", encoding="utf-8-sig") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                arm = str(row.get("arm") or row.get("condition") or row.get("group") or Path(raw_p).stem)
                pid = str(row.get("premise_id") or row.get("id") or f"story_{idx:03d}")
                ref_val = str(row.get("gold_reference") or row.get("targets") or row.get("reference") or "").strip()
                if ref_val:
                    inline_refs[(arm, pid)] = ref_val

    ppl_scorer = (
        None
        if skip_ppl
        else SlidingWindowPerplexityScorer(
            model_id=ppl_model_id,
            stride=ppl_stride,
            max_length=ppl_max_length,
            use_mock=use_mock,
        )
    )

    raw_rows: List[Dict[str, Any]] = []
    try:
        for rec in records:
            full_story = "\n\n".join(strip_think_tags(c) for c in rec["chapters"])
            actual_words = len(full_story.split())

            ref_txt = inline_refs.get((rec["arm"], rec["premise_id"])) or ref_map.get(rec["premise_id"], "")
            ref_source = "gold_reference" if ref_txt else "none"
            if not ref_txt and reference_fallback == "premise_and_plan":
                ref_txt = _build_plan_reference(rec)
                ref_source = "premise_and_plan" if ref_txt else "none"

            metrics = evaluate_classic_nlg_single(
                story_text=full_story,
                reference_text=ref_txt,
                ppl_scorer=ppl_scorer,
            )
            raw_rows.append(
                {
                    "arm": rec["arm"],
                    "premise_id": rec["premise_id"],
                    "genre": rec["genre"],
                    "actual_words": actual_words,
                    "reference_source": ref_source,
                    **metrics,
                }
            )
    finally:
        if ppl_scorer is not None:
            ppl_scorer.unload()

    raw_path = out / "classic_nlg_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in raw_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metric_cols = [
        "actual_words",
        "BLEU-1",
        "BLEU-2",
        "BLEU-3",
        "BLEU-4",
        "ROUGE-L",
        "Distinct-1 (%)",
        "Distinct-2 (%)",
        "Distinct-3 (%)",
        "Entropy-2",
        "Perplexity",
    ]
    by_arm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in raw_rows:
        by_arm[r["arm"]].append(r)

    summary_rows: List[Dict[str, Any]] = []
    for arm, arm_rows in by_arm.items():
        agg: Dict[str, Any] = {
            "arm": arm,
            "n_stories": len(arm_rows),
            "reference_sources": ",".join(sorted({r["reference_source"] for r in arm_rows})),
        }
        for col in metric_cols:
            agg[col] = _safe_mean([r.get(col) for r in arm_rows])
        summary_rows.append(agg)

    csv_fields = ["arm", "n_stories", "reference_sources"] + metric_cols
    csv_path = out / "classic_nlg_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    comparison: Optional[Dict[str, Any]] = None
    if compare_with_fullstory:
        comparison = _compute_classic_vs_narrative_comparison(raw_rows, compare_with_fullstory)
        (out / "classic_vs_fullstory_correlation.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    fmt = lambda v: "N/A" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    md_lines = [
        "# ClassicNLG-Bench (Traditional Full-Story Surface Metrics Report)",
        "",
        f"- **Inputs**: `{', '.join(str(p) for p in stories_paths)}`",
        f"- **Total Stories Evaluated**: `{len(raw_rows)}`",
        f"- **Perplexity Model**: `{'disabled' if skip_ppl else ('mock_proxy' if use_mock else ppl_model_id)}`",
        f"- **Reference Fallback**: `{reference_fallback}`" + (f" | **Reference File**: `{reference_file}`" if reference_file else ""),
        "",
        "## Summary Table by Arm / Condition",
        "",
        "| Arm | N | Words | Ref Source | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | D-1 (%) | D-2 (%) | D-3 (%) | Entropy-2 | PPL |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summary_rows:
        md_lines.append(
            f"| {s['arm']} | {s['n_stories']} | {fmt(s['actual_words'])} | {s['reference_sources']} | "
            f"{fmt(s['BLEU-1'])} | {fmt(s['BLEU-2'])} | {fmt(s['BLEU-3'])} | {fmt(s['BLEU-4'])} | "
            f"{fmt(s['ROUGE-L'])} | {fmt(s['Distinct-1 (%)'])} | {fmt(s['Distinct-2 (%)'])} | "
            f"{fmt(s['Distinct-3 (%)'])} | {fmt(s['Entropy-2'])} | {fmt(s['Perplexity'])} |"
        )

    if comparison is not None:
        md_lines.extend(
            [
                "",
                f"## Classic NLG vs. Narrative FullStory-Bench (Spearman $\\rho$, matched N={comparison['matched_stories']})",
                "",
                "| Classic Metric | S_adh (DOC-NLI) | Trans Cos | C_weighted (ConStory) | EQ-Bench (0-100) |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for c_col, row_corr in comparison["spearman_correlations"].items():
            md_lines.append(
                f"| {c_col} | {fmt(row_corr.get('S_adh'))} | {fmt(row_corr.get('transition_cosine'))} | "
                f"{fmt(row_corr.get('C_weighted'))} | {fmt(row_corr.get('eqbench_score'))} |"
            )

    report_path = out / "classic_nlg_report.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "summary": summary_rows,
        "raw_path": str(raw_path),
        "csv_path": str(csv_path),
        "report_path": str(report_path),
        "comparison": comparison,
    }
