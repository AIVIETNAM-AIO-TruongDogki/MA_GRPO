"""CLI entrypoint for ClassicNLG-Bench (Traditional Full-Story Surface Metrics)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarks.classic_nlg_bench.runner import run_classic_nlg_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standalone ClassicNLG-Bench (BLEU-1..4, ROUGE-L, Distinct-1..3, Entropy-2, Perplexity)."
    )
    parser.add_argument(
        "--stories",
        nargs="+",
        required=True,
        help="One or more story JSONL files (e.g. stories_7b_outputs.jsonl, track2_generated.jsonl).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional arm/condition labels matching --stories order.",
    )
    parser.add_argument(
        "--reference-file",
        default=None,
        help="Optional JSONL with gold reference stories (e.g. data/tell-me-a-story-test.jsonl).",
    )
    parser.add_argument(
        "--reference-fallback",
        choices=["none", "premise_and_plan"],
        default="none",
        help="Fallback when a story has no gold reference ('none' leaves BLEU/ROUGE as N/A; 'premise_and_plan' compares against premise+outline).",
    )
    parser.add_argument(
        "--ppl-model",
        default="gpt2",
        help="HuggingFace causal LM for strided sliding-window Perplexity (default: gpt2; e.g. Qwen/Qwen2.5-0.5B).",
    )
    parser.add_argument("--ppl-stride", type=int, default=512)
    parser.add_argument("--ppl-max-length", type=int, default=1024)
    parser.add_argument("--skip-ppl", action="store_true", help="Skip neural Perplexity computation.")
    parser.add_argument("--mock", action="store_true", help="CPU-safe mock mode for Perplexity.")
    parser.add_argument(
        "--compare-with-fullstory",
        default=None,
        help="Optional path to FullStory-Bench `fullstory_raw.jsonl` to compute Classic vs. Narrative Spearman correlations.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write benchmark results.")

    args = parser.parse_args()
    res = run_classic_nlg_benchmark(
        stories_paths=args.stories,
        output_dir=args.output_dir,
        labels=args.labels,
        reference_file=args.reference_file,
        reference_fallback=args.reference_fallback,
        ppl_model_id=args.ppl_model,
        ppl_stride=args.ppl_stride,
        ppl_max_length=args.ppl_max_length,
        skip_ppl=args.skip_ppl,
        use_mock=args.mock,
        compare_with_fullstory=args.compare_with_fullstory,
    )
    print(f"[ClassicNLG-Bench] Done! Report saved to: {res['report_path']}", flush=True)


if __name__ == "__main__":
    main()
