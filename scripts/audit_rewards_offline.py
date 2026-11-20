"""
M1 Offline Reward Calibration & Perturbation Discrimination CLI (`Phase 2 Expansion Rewards`).

Runs the 32B Local 4-bit Judge + Python/MiniLM Constraint Evaluator offline on previously
generated stories (e.g. `stories_7b_outputs.jsonl` from Drive) BEFORE spending GPU hours
on GRPO training, checking:
  1. Score distribution & saturation warnings across all 7 LLM axes + 4 Python penalties.
  2. Spearman correlation matrix between Local Expansion and Global Chapter scores (H2 check).
  3. 4-way Perturbation Discrimination (`Clean` > `Padded-Filler`, `Off-Plan`, `Contradictory`)
     + group-level reward std(R) sufficient for GRPO advantage normalization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rewards.aggregation import Phase2RewardEngine, load_reward_config
from rewards.base import EvaluationInput
from rewards.validation import (
    analyze_reward_distributions_and_correlations,
    evaluate_perturbation_discrimination,
)


def _extract_evaluation_items(stories_path: str, max_chapters: int = 16) -> List[EvaluationInput]:
    """Extracts chapter-level `EvaluationInput` items from `stories_7b_outputs.jsonl` or chapter JSONL."""
    p = Path(stories_path)
    if not p.exists():
        raise FileNotFoundError(f"Input JSONL not found: {p}")

    items: List[EvaluationInput] = []
    with open(p, "r", encoding="utf-8-sig") as f:
        for row_idx, line in enumerate(f):
            if len(items) >= max_chapters:
                break
            if not line.strip():
                continue
            row = json.loads(line)

            # Case A: Chapter-level record already has `generated_chapter` / `completion`
            if "generated_chapter" in row or "completion" in row:
                ctx_state = row.get("context_state") or {}
                items.append(
                    EvaluationInput(
                        generated_chapter=str(row.get("generated_chapter") or row.get("completion") or ""),
                        target_words=int(row.get("target_words", 1125)),
                        chapter_title=str(row.get("chapter_title", f"Chapter {row.get('chapter_index', 1)}")),
                        chapter_goal=str(row.get("chapter_goal", "")),
                        scene_outline=list(row.get("scene_outline", [])),
                        characters_active=list(row.get("characters_active", [])),
                        context_summary=str(row.get("context_summary", "")),
                        prev_chapter_tail=str(row.get("prev_chapter_tail", "")),
                        long_term_memory=dict(ctx_state.get("long_term_memory", {})),
                        chapter_index=int(row.get("chapter_index", 1)),
                        premise_id=str(row.get("premise_id", f"item_{row_idx}")),
                    )
                )
                continue

            # Case B: Full-story record (`stories_7b_outputs.jsonl`) with `chapters` + `story_plan`
            chapters = row.get("chapters_raw_with_think") or row.get("chapters") or []
            plan_chapters = (row.get("story_plan") or {}).get("chapters", [])
            pid = str(row.get("premise_id") or row.get("id") or f"story_{row_idx}")
            for ch_i, ch_text in enumerate(chapters):
                if len(items) >= max_chapters:
                    break
                ch_plan = plan_chapters[ch_i] if ch_i < len(plan_chapters) else {}
                prev_tail = " ".join(str(chapters[ch_i - 1]).split()[-180:]) if ch_i > 0 else ""
                items.append(
                    EvaluationInput(
                        generated_chapter=str(ch_text),
                        target_words=int(ch_plan.get("target_words", row.get("target_words", 8000) // max(1, len(chapters)))),
                        chapter_title=str(ch_plan.get("chapter_title", f"Chapter {ch_i + 1}")),
                        chapter_goal=str(ch_plan.get("chapter_goal", row.get("premise", ""))),
                        scene_outline=list(ch_plan.get("scene_outline", [])),
                        characters_active=list(ch_plan.get("characters_active", [])),
                        context_summary=str(plan_chapters[ch_i - 1].get("chapter_goal", "")) if ch_i > 0 and ch_i - 1 < len(plan_chapters) else "",
                        prev_chapter_tail=prev_tail,
                        long_term_memory={"major_events": [c.get("chapter_goal", "") for c in plan_chapters[:ch_i]]},
                        chapter_index=ch_i + 1,
                        premise_id=pid,
                    )
                )
    return items


def run_offline_audit(
    stories_path: str,
    reward_config_path: str,
    output_dir: str,
    max_chapters: int = 16,
    perturbation_chapters: int = 4,
    separate_passes: bool = True,
    use_mock: bool = False,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_reward_config(reward_config_path)
    cfg.setdefault("evaluator", {})
    if use_mock:
        cfg["evaluator"]["backend"] = "mock"
        cfg.setdefault("constraints", {})["use_embedder"] = False
    if separate_passes:
        cfg["evaluator"]["single_pass"] = False
        cfg["evaluator"]["include_evidence"] = True

    items = _extract_evaluation_items(stories_path, max_chapters=max_chapters)
    if not items:
        raise ValueError(f"No chapters extracted from {stories_path}")

    log_path = out / "offline_reward_logs.jsonl"
    if log_path.exists():
        log_path.unlink()

    engine = Phase2RewardEngine(config=cfg, log_jsonl_path=log_path)
    try:
        breakdowns = engine.evaluate_batch(items)
        flat_logs = [bd.to_flat_log() for bd in breakdowns]
        corr_report = analyze_reward_distributions_and_correlations(flat_logs)
        pert_report = evaluate_perturbation_discrimination(engine, items[: max(1, min(perturbation_chapters, len(items)))])
    finally:
        engine.release()

    summary = {
        "mode": engine.mode,
        "judge_model": engine.judge_model_id if not use_mock else "mock",
        "single_pass": engine.single_pass,
        "num_chapters_audited": len(items),
        "distribution_and_correlation": corr_report,
        "perturbation_discrimination": pert_report,
    }
    (out / "offline_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Render concise Markdown report
    md = [
        "# Phase 2 Expansion Reward — M1 Offline Calibration & Audit Report",
        "",
        f"- **Mode**: `{summary['mode']}` | **Judge Model**: `{summary['judge_model']}` | **Single Pass**: `{summary['single_pass']}`",
        f"- **Chapters Audited**: `{len(items)}`",
        f"- **Local vs. Global Spearman $\\rho$**: `{corr_report['local_vs_global_spearman']}` (H2 complementary OK: **{corr_report['h2_complementary_signal_ok']}**)",
        f"- **Saturated Axes Warning**: `{corr_report['saturated_axes']}`",
        "",
        "## 1. Score Distributions (LLM Judge + Python Constraints)",
        "",
        "| Component | Mean | Std | Min | Max | Frac >= 0.9 | Saturated? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for col, st in corr_report["distributions"].items():
        md.append(
            f"| `{col}` | {st['mean']:.4f} | {st['std']:.4f} | {st['min']:.4f} | {st['max']:.4f} | "
            f"{st['frac_ge_0_9']:.2f} | {'YES' if st['saturated_warning'] else 'no'} |"
        )

    md.extend(
        [
            "",
            "## 2. 4-Way Perturbation Discrimination (Reward Hacking Check)",
            "",
            f"- **Passed**: **{pert_report['passed']}** (tested `{pert_report['num_groups']}` groups)",
            f"- `Clean > Padded-Filler`: **{pert_report['clean_beats_filler_rate'] * 100:.1f}%**",
            f"- `Clean > Off-Plan`: **{pert_report['clean_beats_off_plan_rate'] * 100:.1f}%**",
            f"- `Clean > Contradictory`: **{pert_report['clean_beats_contradictory_rate'] * 100:.1f}%**",
            f"- Mean Group `std(R)`: **`{pert_report['mean_group_reward_std']}`**",
        ]
    )

    report_path = out / "offline_audit_report.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[Phase2 Offline Audit] Done! Report saved to: {report_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M1 Offline Audit for Phase 2 Expansion Rewards.")
    parser.add_argument("--stories", required=True, help="Path to stories_7b_outputs.jsonl or chapter JSONL.")
    parser.add_argument(
        "--reward-config",
        default=str(REPO_ROOT / "rewards" / "configs" / "exp02_hierarchical.yaml"),
        help="Path to Phase 2 reward YAML config.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write offline audit results.")
    parser.add_argument("--max-chapters", type=int, default=16)
    parser.add_argument("--perturbation-chapters", type=int, default=4)
    parser.add_argument("--single-pass", action="store_true", help="Use single-pass unified prompt instead of separate Local/Global passes.")
    parser.add_argument("--mock", action="store_true", help="Run CPU-safe mock mode.")
    args = parser.parse_args()

    run_offline_audit(
        stories_path=args.stories,
        reward_config_path=args.reward_config,
        output_dir=args.output_dir,
        max_chapters=args.max_chapters,
        perturbation_chapters=args.perturbation_chapters,
        separate_passes=not args.single_pass,
        use_mock=args.mock,
    )


if __name__ == "__main__":
    main()
