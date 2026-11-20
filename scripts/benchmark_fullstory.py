"""CLI entrypoint for Suite 2: FullStory-Bench (ConStory-Bench + EQ-Bench Longform + DOC-NLI)."""

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

from experiments.benchmarks.fullstory_bench.runner import run_fullstory_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Suite 2 (FullStory-Bench) on saved multi-chapter stories JSONL."
    )
    parser.add_argument(
        "--stories",
        nargs="+",
        required=True,
        help="One or more story JSONL files (e.g. stories_7b_outputs.jsonl or track2_generated.jsonl).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional arm/condition labels matching --stories order (when rows omit arm/condition).",
    )
    parser.add_argument(
        "--transitions",
        default=None,
        help="Optional coordinator_transitions.jsonl for Anti-Context Novelty & Fallback Rate.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write benchmark results.")
    parser.add_argument("--mock", action="store_true", help="Run CPU-safe mock mode (no GPU / downloads).")
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        help="HuggingFace NLI model ID for DOC-NLI plan adherence.",
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
        help="Judge LLM backend for ConStory-Checker, EQ-Bench Longform, and Blind Pairwise.",
    )
    parser.add_argument(
        "--judge-model",
        default="Qwen/Qwen2.5-32B-Instruct-AWQ",
        help="Judge model ID (HuggingFace ID for local_vllm, or API model name).",
    )
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-gpu-mem", type=float, default=0.85)
    parser.add_argument("--judge-max-len", type=int, default=24576)
    parser.add_argument(
        "--pairwise-arms",
        nargs=2,
        default=None,
        metavar=("ARM_A", "ARM_B"),
        help="Optional pair of arm names to run swapped-order blind pairwise Judge.",
    )

    args = parser.parse_args()
    res = run_fullstory_benchmark(
        stories_paths=args.stories,
        output_dir=args.output_dir,
        labels=args.labels,
        transitions_path=args.transitions,
        use_mock=args.mock,
        nli_model_id=args.nli_model,
        embedder_model_id=args.embedder_model,
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        judge_api_key=args.judge_api_key,
        judge_api_base=args.judge_api_base,
        judge_gpu_mem=args.judge_gpu_mem,
        judge_max_len=args.judge_max_len,
        pairwise_arms=tuple(args.pairwise_arms) if args.pairwise_arms else None,
    )
    print(f"[FullStory-Bench] Done! Report saved to: {res['report_path']}", flush=True)


if __name__ == "__main__":
    main()
