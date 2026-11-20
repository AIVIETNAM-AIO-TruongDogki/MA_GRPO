"""CLI entrypoint for Suite 1: MacroPlan-Bench (Standalone Outline Planning Benchmark)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarks.outline_bench.runner import run_outline_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Suite 1 (MacroPlan-Bench) on generated story outlines / macro plans."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more input files (shared_macro_plans.json, track2_generated.jsonl, grpo_train_condition_d.jsonl).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional group labels matching --inputs order.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write benchmark results.")
    parser.add_argument("--mock", action="store_true", help="Run CPU-safe mock mode (no GPU / downloads).")
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        help="HuggingFace NLI model ID.",
    )
    parser.add_argument(
        "--embedder-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model ID (or 'none').",
    )
    parser.add_argument(
        "--judge-backend",
        choices=["none", "mock", "api", "local_vllm"],
        default="none",
        help="Judge LLM backend for DOC+HANNA rubric and blind pairwise comparison.",
    )
    parser.add_argument(
        "--judge-model",
        default="Qwen/Qwen2.5-32B-Instruct-AWQ",
        help="Judge model ID (HuggingFace ID for local_vllm, or API model name).",
    )
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-gpu-mem", type=float, default=0.85)
    parser.add_argument("--judge-max-len", type=int, default=16384)
    parser.add_argument(
        "--pairwise-groups",
        nargs=2,
        default=None,
        metavar=("GROUP_A", "GROUP_B"),
        help="Optional pair of group names to compare via swapped-order blind pairwise Judge.",
    )
    parser.add_argument("--vrcli-data", default=None, help="Optional LONGSTORY/BookSum JSONL for VR-CLI I_pi.")
    parser.add_argument("--vrcli-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--vrcli-max-samples", type=int, default=20)
    parser.add_argument(
        "--run-sanity-check",
        action="store_true",
        help="Run Part 3 perturbation sanity check verifying metric sensitivity.",
    )

    args = parser.parse_args()
    res = run_outline_benchmark(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        labels=args.labels,
        use_mock=args.mock,
        nli_model_id=args.nli_model,
        embedder_model_id=args.embedder_model,
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        judge_api_key=args.judge_api_key,
        judge_api_base=args.judge_api_base,
        judge_gpu_mem=args.judge_gpu_mem,
        judge_max_len=args.judge_max_len,
        pairwise_groups=tuple(args.pairwise_groups) if args.pairwise_groups else None,
        vrcli_data=args.vrcli_data,
        vrcli_model_id=args.vrcli_model,
        vrcli_max_samples=args.vrcli_max_samples,
        run_sanity_check=args.run_sanity_check,
    )
    print(f"[MacroPlan-Bench] Done! Report saved to: {res['report_path']}", flush=True)


if __name__ == "__main__":
    main()
