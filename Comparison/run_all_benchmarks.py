"""
Master Orchestrator for the 8-Model Long-Text Benchmark Suite.

Runs all 8 models sequentially (or any selected subset):
  1. AGENTS' ROOM         -> result_agents_room.csv
  2. STORYWRITER          -> result_storywriter.csv
  3. Self-Lengthen        -> result_self_lengthen.csv
  4. LongWriter-llama3.1-8b -> result_longwriter_llama.csv
  5. LongWriter-glm4-9b   -> result_longwriter_glm.csv
  6. LongWriter-Zero-32B  -> result_longwriter_zero.csv
  7. MA-GRPO-50           -> result_ma_grpo_50.csv
  8. MA-GRPO-100          -> result_ma_grpo_100.csv

Each model's CSV strictly matches the 12-column schema of `result_longwriter_llama.csv`:
  Model,Prompt_ID,Target_Len,Actual_Len,Length_Score_Sl (%),Distinct-3 (%),Distinct-4 (%),Latency (s),Peak_VRAM (GB),Speed (words/s),Prompt_Text,Raw_Output

Finally aggregates all 8 CSVs, computes Perplexity (PPL ↓) via `Qwen/Qwen2.5-0.5B` and
reference overlap (`BLEU-4`, `ROUGE-L`) against `tell-me-a-story-test.jsonl`, and outputs
the comprehensive comparative summary table (`benchmark_comparison_8models.csv`).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Allow running both as module (`python -m Comparison.run_all_benchmarks`) and directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Comparison.agents_room import run_agents_room_benchmark
from Comparison.longwriter_models import (
    run_longwriter_glm_benchmark,
    run_longwriter_llama_benchmark,
    run_longwriter_zero_benchmark,
)
from Comparison.ma_grpo_runner import run_ma_grpo_checkpoint_benchmark
from Comparison.metrics import (
    PerplexityScorer,
    compute_overlap_metrics,
    load_benchmark_prompts,
)
from Comparison.qwen_backend import QwenAgentBackend
from Comparison.self_lengthen import run_self_lengthen_benchmark
from Comparison.storywriter import run_storywriter_benchmark


ALL_MODEL_KEYS = [
    "agents_room",
    "storywriter",
    "self_lengthen",
    "longwriter_llama",
    "longwriter_glm",
    "longwriter_zero",
    "ma_grpo_50",
    "ma_grpo_100",
]

MODEL_CSV_FILENAMES = {
    "agents_room": "result_agents_room.csv",
    "storywriter": "result_storywriter.csv",
    "self_lengthen": "result_self_lengthen.csv",
    "longwriter_llama": "result_longwriter_llama.csv",
    "longwriter_glm": "result_longwriter_glm.csv",
    "longwriter_zero": "result_longwriter_zero.csv",
    "ma_grpo_50": "result_ma_grpo_50.csv",
    "ma_grpo_100": "result_ma_grpo_100.csv",
}


def build_final_comparison_table(
    output_dir: Path,
    prompts: List[Dict[str, Any]],
    summary_csv_name: str = "benchmark_comparison_8models.csv",
) -> pd.DataFrame:
    """
    Read all generated `result_*.csv` files in `output_dir`, compute PPL ↓, BLEU-4,
    and ROUGE-L, and output the final comparative benchmark summary table.
    """
    print("\n" + "=" * 95, flush=True)
    print("FINAL STAGE: AGGREGATING ALL 8 MODELS & COMPUTING PPL / BLEU-4 / ROUGE-L", flush=True)
    print("=" * 95, flush=True)

    gold_map = {p["id"]: p.get("gold_ref", "") for p in prompts}

    ordered_files = [
        ("AGENTS' ROOM", output_dir / "result_agents_room.csv"),
        ("STORYWRITER", output_dir / "result_storywriter.csv"),
        ("Self-Lengthen", output_dir / "result_self_lengthen.csv"),
        ("LongWriter-llama3.1-8b", output_dir / "result_longwriter_llama.csv"),
        ("LongWriter-glm4-9b", output_dir / "result_longwriter_glm.csv"),
        ("LongWriter-Zero-32B", output_dir / "result_longwriter_zero.csv"),
        ("MA-GRPO-50", output_dir / "result_ma_grpo_50.csv"),
        ("MA-GRPO-100", output_dir / "result_ma_grpo_100.csv"),
    ]

    dfs: List[pd.DataFrame] = []
    for display_name, csv_path in ordered_files:
        if csv_path.exists():
            df_m = pd.read_csv(csv_path)
            df_m["Model"] = display_name
            dfs.append(df_m)
            print(f" -> Loaded {len(df_m)} rows from {csv_path.name} ({display_name})", flush=True)
        else:
            print(f" -> [Notice] {csv_path.name} not found yet, skipping in aggregation.", flush=True)

    if not dfs:
        print("[Warning] No result_*.csv files found to aggregate!", flush=True)
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)

    # Compute Perplexity (PPL ↓) with Qwen/Qwen2.5-0.5B
    ppl_scorer = PerplexityScorer("Qwen/Qwen2.5-0.5B")
    ppl_scorer.load()

    ppls, bleu4s, rougels = [], [], []
    for _, row in df_all.iterrows():
        raw_out = str(row.get("Raw_Output", ""))
        p_id = str(row.get("Prompt_ID", ""))
        gold_ref = gold_map.get(p_id, "")

        ppl_val = ppl_scorer.score_text(raw_out)
        ov = compute_overlap_metrics(raw_out, gold_ref)

        ppls.append(ppl_val)
        bleu4s.append(ov["BLEU-4"])
        rougels.append(ov["ROUGE-L"])

    ppl_scorer.unload()

    df_all["PPL ↓"] = ppls
    df_all["BLEU-4"] = bleu4s
    df_all["ROUGE-L"] = rougels

    raw_combined_path = output_dir / "benchmark_results_all_raw.csv"
    df_all.to_csv(raw_combined_path, index=False, encoding="utf-8")

    # Aggregate mean per model preserving canonical 8-model order
    summary_rows = []
    for display_name, _ in ordered_files:
        sub = df_all[df_all["Model"] == display_name]
        if len(sub) == 0:
            continue
        summary_rows.append({
            "Model / Method": display_name,
            "Target_Len": round(float(sub["Target_Len"].mean()), 1),
            "Actual_Len": round(float(sub["Actual_Len"].mean()), 1),
            "Length_Score_Sl (%)": round(float(sub["Length_Score_Sl (%)"].mean()), 2),
            "Distinct-3 (%)": round(float(sub["Distinct-3 (%)"].mean()), 2),
            "Distinct-4 (%)": round(float(sub["Distinct-4 (%)"].mean()), 2),
            "PPL ↓": round(float(sub["PPL ↓"].mean()), 2),
            "BLEU-4": round(float(sub["BLEU-4"].mean()), 2),
            "ROUGE-L": round(float(sub["ROUGE-L"].mean()), 2),
            "Latency (s)": round(float(sub["Latency (s)"].mean()), 2),
            "Peak_VRAM (GB)": round(float(sub["Peak_VRAM (GB)"].max()), 2),
            "Speed (words/s)": round(float(sub["Speed (words/s)"].mean()), 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / summary_csv_name
    df_summary.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n" + "=" * 115, flush=True)
    print("BẢNG TỔNG HỢP ĐỐI SÁNH 8 MÔ HÌNH (FINAL 8-MODEL BENCHMARK COMPARISON)", flush=True)
    print("=" * 115, flush=True)
    print(df_summary.to_string(index=False), flush=True)
    print("=" * 115, flush=True)
    print(f"✅ Saved final summary table to: {summary_path}", flush=True)
    print(f"✅ Saved combined raw results to: {raw_combined_path}", flush=True)

    return df_summary


def run_all_models(
    output_dir: str,
    num_samples: int = 3,
    models_to_run: Optional[List[str]] = None,
    ckpt50_dir: Optional[str] = None,
    ckpt100_dir: Optional[str] = None,
    skip_existing: bool = True,
    clear_hf_cache_between_heavy_models: bool = False,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    selected = models_to_run if models_to_run else ALL_MODEL_KEYS
    prompts = load_benchmark_prompts(num_samples=num_samples)

    print("=" * 95, flush=True)
    print(f"STARTING 8-MODEL BENCHMARK PIPELINE ({len(prompts)} Samples per Model)", flush=True)
    print(f"Selected Models: {selected}", flush=True)
    print(f"Output Folder:   {out_path.resolve()}", flush=True)
    print("=" * 95, flush=True)

    # Group 1: Agentic / Self-Alignment baselines sharing Qwen2.5-7B-Instruct
    qwen_group = [m for m in ["agents_room", "storywriter", "self_lengthen"] if m in selected]
    need_qwen = any(
        not (skip_existing and (out_path / MODEL_CSV_FILENAMES[m]).exists())
        for m in qwen_group
    )

    shared_qwen: Optional[QwenAgentBackend] = None
    if need_qwen:
        shared_qwen = QwenAgentBackend("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True)
        shared_qwen.load()

    try:
        if "agents_room" in selected:
            csv_p = out_path / MODEL_CSV_FILENAMES["agents_room"]
            if skip_existing and csv_p.exists():
                print(f"[Skip] {csv_p.name} already exists.", flush=True)
            else:
                run_agents_room_benchmark(prompts, csv_p, backend=shared_qwen)

        if "storywriter" in selected:
            csv_p = out_path / MODEL_CSV_FILENAMES["storywriter"]
            if skip_existing and csv_p.exists():
                print(f"[Skip] {csv_p.name} already exists.", flush=True)
            else:
                run_storywriter_benchmark(prompts, csv_p, backend=shared_qwen)

        if "self_lengthen" in selected:
            csv_p = out_path / MODEL_CSV_FILENAMES["self_lengthen"]
            if skip_existing and csv_p.exists():
                print(f"[Skip] {csv_p.name} already exists.", flush=True)
            else:
                run_self_lengthen_benchmark(prompts, csv_p, backend=shared_qwen)
    finally:
        if shared_qwen is not None:
            shared_qwen.unload()
            shared_qwen = None

    # Group 2: LongWriter Family (Llama-3.1-8B, GLM-4-9B, Zero-32B)
    if "longwriter_llama" in selected:
        csv_p = out_path / MODEL_CSV_FILENAMES["longwriter_llama"]
        if skip_existing and csv_p.exists():
            print(f"[Skip] {csv_p.name} already exists.", flush=True)
        else:
            run_longwriter_llama_benchmark(
                prompts, csv_p, clear_hf_cache_after=clear_hf_cache_between_heavy_models
            )

    if "longwriter_glm" in selected:
        csv_p = out_path / MODEL_CSV_FILENAMES["longwriter_glm"]
        if skip_existing and csv_p.exists():
            print(f"[Skip] {csv_p.name} already exists.", flush=True)
        else:
            run_longwriter_glm_benchmark(
                prompts, csv_p, clear_hf_cache_after=clear_hf_cache_between_heavy_models
            )

    if "longwriter_zero" in selected:
        csv_p = out_path / MODEL_CSV_FILENAMES["longwriter_zero"]
        if skip_existing and csv_p.exists():
            print(f"[Skip] {csv_p.name} already exists.", flush=True)
        else:
            run_longwriter_zero_benchmark(
                prompts, csv_p, clear_hf_cache_after=clear_hf_cache_between_heavy_models
            )

    # Group 3: MA-GRPO Checkpoints (Checkpoint-50 and Checkpoint-100)
    if "ma_grpo_50" in selected:
        csv_p = out_path / MODEL_CSV_FILENAMES["ma_grpo_50"]
        if skip_existing and csv_p.exists():
            print(f"[Skip] {csv_p.name} already exists.", flush=True)
        else:
            run_ma_grpo_checkpoint_benchmark(
                prompts, step=50, output_csv=csv_p, checkpoint_dir=ckpt50_dir
            )

    if "ma_grpo_100" in selected:
        csv_p = out_path / MODEL_CSV_FILENAMES["ma_grpo_100"]
        if skip_existing and csv_p.exists():
            print(f"[Skip] {csv_p.name} already exists.", flush=True)
        else:
            run_ma_grpo_checkpoint_benchmark(
                prompts, step=100, output_csv=csv_p, checkpoint_dir=ckpt100_dir
            )

    # Final Aggregation & Summary Table
    return build_final_comparison_table(out_path, prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all 8 Long-Text Benchmark Models & Build Final Comparison Table.")
    parser.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/MA-GRPO-Checkpoints/benchmark_8models_output",
        help="Directory to store individual result_*.csv files and final comparison table.",
    )
    parser.add_argument("--samples", type=int, default=3, help="Number of benchmark prompts to run (3 or 5).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODEL_KEYS,
        choices=ALL_MODEL_KEYS + ["all", "eval_only"],
        help="Models to run (default: all 8 models).",
    )
    parser.add_argument("--ckpt50", default=None, help="Path to MA-GRPO checkpoint-50.")
    parser.add_argument("--ckpt100", default=None, help="Path to MA-GRPO checkpoint-100.")
    parser.add_argument("--force", action="store_true", help="Re-run even if output CSV already exists.")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear HuggingFace cache after each large LongWriter model to save disk space.",
    )
    args = parser.parse_args()

    if "eval_only" in args.models:
        prompts_list = load_benchmark_prompts(num_samples=args.samples)
        build_final_comparison_table(Path(args.output_dir), prompts_list)
    else:
        models_list = ALL_MODEL_KEYS if "all" in args.models else args.models
        run_all_models(
            output_dir=args.output_dir,
            num_samples=args.samples,
            models_to_run=models_list,
            ckpt50_dir=args.ckpt50,
            ckpt100_dir=args.ckpt100,
            skip_existing=not args.force,
            clear_hf_cache_between_heavy_models=args.clear_cache,
        )

