#!/usr/bin/env python3
"""
Generate a Stage-2 GRPO Training Dataset strictly aligned with Condition D
(Macro Event Graph ONLY, no SubEventAgent) using high-throughput Batch vLLM.

Example (Colab A100 — 80 premises -> 640 chapter prompts via 32B-AWQ Teacher Planner):
    python scripts/generate_grpo_dataset.py \
        --backend vllm \
        --model Qwen/Qwen2.5-32B-Instruct-AWQ \
        --premises-file data/mops_premises.jsonl \
        --exclude-test-file data/tell-me-a-story-test.jsonl \
        --num-premises 80 \
        --num-chapters 8 \
        --batch-size 8 \
        --output-dir /content/drive/MyDrive/MA-GRPO-Dataset \
        --resume

Example (CPU smoke test):
    python scripts/generate_grpo_dataset.py --backend mock --num-premises 4 --output-dir tmp_grpo_data
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_generator import ConditionDGRPODatasetGenerator, load_premises_pool
from src.pipeline import resolve_optimal_dtype


class VLLMBatchGenerator:
    """Thin batch-parallel vLLM wrapper for Stage-1 Outline, Prose Draft, and Coordinator calls."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
        gpu_memory_utilization: float = 0.88,
        max_model_len: int = 12288,
        tensor_parallel_size: int = 1,
    ):
        from vllm import LLM

        self.model_id = model_id
        self.max_model_len = max_model_len
        lower_m = model_id.lower()
        dtype = "float16" if ("awq" in lower_m or "gptq" in lower_m) else resolve_optimal_dtype()
        print(
            f"[VLLMBatchGenerator] Loading {model_id} (dtype={dtype}, "
            f"gpu_mem={gpu_memory_utilization}, max_model_len={max_model_len})...",
            flush=True,
        )
        self.llm = LLM(
            model=model_id,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _format_chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return f"{system_prompt}\n\n{user_prompt}\n"

    def _safe_max_tokens(self, formatted_prompts: List[str], requested_max: int) -> int:
        max_prompt_len = 0
        for text in formatted_prompts:
            try:
                n_tok = len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                n_tok = int(len(text.split()) * 1.4)
            if n_tok > max_prompt_len:
                max_prompt_len = n_tok
        available = max(64, self.max_model_len - max_prompt_len - 32)
        return max(64, min(requested_max, available))

    def generate_batch(
        self,
        requests: List[Tuple[str, str]],
        max_tokens: int = 900,
        temperature: float = 0.7,
    ) -> List[str]:
        from vllm import SamplingParams

        if not requests:
            return []

        formatted = [self._format_chat(sys_p, usr_p) for sys_p, usr_p in requests]
        safe_max = self._safe_max_tokens(formatted, max_tokens)
        params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=safe_max,
            presence_penalty=0.1,
        )
        outputs = self.llm.generate(formatted, params, use_tqdm=False)
        return [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]

    def release(self) -> None:
        if getattr(self, "llm", None) is not None:
            del self.llm
            self.llm = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Condition D (Macro-Only) GRPO training dataset via Batch vLLM."
    )
    parser.add_argument(
        "--premises-file",
        nargs="+",
        default=[str(PROJECT_ROOT / "data" / "mops_premises.jsonl")],
        help="One or more JSONL files supplying clean training premises.",
    )
    parser.add_argument(
        "--no-canonical",
        action="store_true",
        help="Disable appending CANONICAL_20_PREMISES to the pool.",
    )
    parser.add_argument(
        "--exclude-test-file",
        nargs="*",
        default=[str(PROJECT_ROOT / "data" / "tell-me-a-story-test.jsonl")],
        help="Held-out benchmark test JSONL files to strictly exclude (zero leakage).",
    )
    parser.add_argument("--num-premises", type=int, default=80, help="Number of stories to process (default: 80 -> 640 chapter prompts).")
    parser.add_argument("--num-chapters", type=int, default=8, help="Chapters per story (default: 8).")
    parser.add_argument(
        "--target-words-choices",
        nargs="+",
        type=int,
        default=[8000, 9000, 10000],
        help="Story-level target word budgets cycled across premises.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Stories processed in parallel per vLLM batch (default: 8 for 32B-AWQ on A100).")
    parser.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ", help="Teacher Planner/Coordinator model ID (default: Qwen/Qwen2.5-32B-Instruct-AWQ).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--outline-max-tokens", type=int, default=2048)
    parser.add_argument("--prose-max-tokens", type=int, default=900)
    parser.add_argument("--coordinator-max-tokens", type=int, default=768)
    parser.add_argument("--strict-mode", action="store_true", help="Quarantine outline parse failures instead of dynamic fallback.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "grpo_condition_d"))
    parser.add_argument("--output-filename", default="grpo_train_condition_d.jsonl")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSONL without duplicating completed premises.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    premises, excluded_count = load_premises_pool(
        premises_files=args.premises_file,
        include_canonical=not args.no_canonical,
        exclude_files=args.exclude_test_file,
        num_premises=args.num_premises,
        target_words_choices=args.target_words_choices,
        num_chapters=args.num_chapters,
    )
    print(
        f"[Premises Pool] Loaded {len(premises)} unique training premises "
        f"(excluded {excluded_count} test-set overlaps; expected ~{len(premises) * args.num_chapters} chapter prompts).",
        flush=True,
    )

    vllm_runner: Optional[VLLMBatchGenerator] = None
    try:
        if args.backend == "vllm":
            vllm_runner = VLLMBatchGenerator(
                model_id=args.model,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
            )
            batch_fn = vllm_runner.generate_batch
        else:
            batch_fn = None

        generator = ConditionDGRPODatasetGenerator(
            batch_generate_fn=batch_fn,
            strict_mode=args.strict_mode,
            outline_max_tokens=args.outline_max_tokens,
            prose_max_tokens=args.prose_max_tokens,
            coordinator_max_tokens=args.coordinator_max_tokens,
        )
        report = generator.generate_dataset(
            premises=premises,
            output_dir=args.output_dir,
            output_filename=args.output_filename,
            num_chapters=args.num_chapters,
            batch_size=args.batch_size,
            resume=args.resume,
            excluded_leakage_count=excluded_count,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    finally:
        if vllm_runner is not None:
            vllm_runner.release()


if __name__ == "__main__":
    main()
