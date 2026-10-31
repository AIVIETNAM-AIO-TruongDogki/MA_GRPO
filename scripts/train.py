"""Train the MA-GRPO writing policy with QLoRA and TRL's GRPOTrainer.

Each JSONL record must contain ``prompt`` and may provide the following fields:
``target_words`` (default: 1200), ``context_summary``, and
``prev_chapter_tail``.  ``prompt`` should already contain the chapter plan and
the validated coordinator state; this script deliberately trains only the
local Writing Agent.

Example:
    python scripts/train.py --dataset data/chapters.jsonl --output-dir checkpoints/ma-grpo

The reward models are deliberately separate from the trainable policy.  On a
dual-GPU machine use ``--reward-device cuda:1``; on a single GPU, ``cpu`` is
safer (but substantially slower).
"""

from __future__ import annotations

import argparse
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import torch

# Make ``src`` importable when this file is executed directly.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rewards import CompositeRewardEngine, extract_length_bounds


LOG = logging.getLogger("ma_grpo.train")


@dataclass(frozen=True)
class TrainingRecord:
    """Fields passed through the dataset to MA-GRPO reward functions."""

    prompt: str
    target_words: int = 1200
    context_summary: str = ""
    prev_chapter_tail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA + GRPO training for the MA-GRPO Writing Agent.")
    parser.add_argument("--dataset", required=True, help="JSONL training data with a `prompt` column.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID or local path.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "server_40gb.yaml"), help="MA-GRPO YAML config.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoints" / "ma-grpo"))
    parser.add_argument("--max-steps", type=int, default=None, help="Overrides training.max_steps in config.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None, help="GRPO candidates per prompt (G).")
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None, help="Overrides training.save_steps in config.")
    parser.add_argument("--reward-device", default=None, help="e.g. cuda:1, cuda:0, or cpu.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", default="tensorboard", choices=["none", "tensorboard", "wandb"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _as_text(completion: Any) -> str:
    """Normalise GRPO completion formats across TRL releases."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and completion and isinstance(completion[0], dict):
        # Conversational completions are usually [{"role": ..., "content": ...}].
        return " ".join(str(message.get("content", "")) for message in completion)
    return str(completion)


def _per_item(values: Any, size: int, default: Any) -> List[Any]:
    """Return dataset metadata as a list aligned with completion order."""
    if values is None:
        return [default] * size
    if not isinstance(values, (list, tuple)):
        return [values] * size
    if len(values) == size:
        return list(values)
    if len(values) == 1:
        return list(values) * size
    if size % len(values) == 0:
        # TRL may pass one metadata item per prompt while completions are
        # expanded to G samples per prompt.  GRPO groups are contiguous.
        repeats = size // len(values)
        return [value for value in values for _ in range(repeats)]
    return [default] * size


def extract_target_words_from_text(prompt_text: str, fallback: int = 1400) -> int:
    if not prompt_text:
        return fallback
    clean = prompt_text.replace(",", "")
    match = re.search(r"Total Target Length:\s*Approximately\s*(\d+)\s*words", clean, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"between\s+(\d+)\s*(?:and|to|–|-)\s*(\d+)\s*words", clean, re.IGNORECASE)
    if match:
        return (int(match.group(1)) + int(match.group(2))) // 2
    match = re.search(r"(\d{3,5})\s*words", clean, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return fallback


def extract_context_from_prompt(prompt_text: str) -> str:
    if not prompt_text:
        return ""
    match = re.search(r"\[CONTEXT SUMMARY\]:\s*([\s\S]*?)(?=\n\n\[|\n\[CHAPTER GOAL\])", prompt_text)
    if match:
        return match.group(1).strip()
    return ""


def extract_tail_from_prompt(prompt_text: str) -> str:
    if not prompt_text:
        return ""
    match = re.search(r"\[PRECEDING SCENE CONTINUATION.*?\]:\s*([\s\S]*?)(?=\n\n\[|\n\[CHAPTER GOAL\])", prompt_text)
    if match:
        return match.group(1).strip()
    return ""


class IntrinsicRewardFunctions:
    """Lazy, shared reward engine exposed in TRL's reward-function interface."""

    def __init__(self, config_path: str, device: str | None) -> None:
        self.config_path = config_path
        self.device = device
        self._engine: CompositeRewardEngine | None = None
        self._cache_key: tuple[Any, ...] | None = None
        self._cache_value: Dict[str, List[float]] | None = None

    @property
    def engine(self) -> CompositeRewardEngine:
        if self._engine is None:
            LOG.info("Loading frozen intrinsic reward evaluators on %s", self.device or "automatic device")
            self._engine = CompositeRewardEngine(config_path=self.config_path, device=self.device)
        return self._engine

    def _components(self, completions: Sequence[Any], **kwargs: Any) -> Dict[str, List[float]]:
        texts = [_as_text(item) for item in completions]
        count = len(texts)

        # 1. Resolve prompts
        raw_prompts = kwargs.get("prompts")
        prompt_texts = [_as_text(p) for p in _per_item(raw_prompts, count, "")] if raw_prompts is not None else [""] * count

        # 2. Resolve target_words and (l_lower, l_upper, l_max) dynamically from prompt or dataset kwargs
        kw_targets = kwargs.get("target_words")
        kw_lowers = kwargs.get("l_lower")
        kw_uppers = kwargs.get("l_upper")
        kw_maxes = kwargs.get("l_max")

        if kw_targets is not None:
            targets = [int(v) for v in _per_item(kw_targets, count, 1400)]
        else:
            targets = [extract_target_words_from_text(p, 1400) for p in prompt_texts]

        if kw_lowers is not None and kw_uppers is not None:
            lowers = [int(v) for v in _per_item(kw_lowers, count, 1330)]
            uppers = [int(v) for v in _per_item(kw_uppers, count, 1470)]
            maxes = (
                [int(v) for v in _per_item(kw_maxes, count, 1890)]
                if kw_maxes is not None
                else [int(round(u * 1.35)) for u in uppers]
            )
        else:
            bounds = [extract_length_bounds(p, t) for p, t in zip(prompt_texts, targets)]
            lowers = [b[0] for b in bounds]
            uppers = [b[1] for b in bounds]
            maxes = [b[2] for b in bounds]

        # 3. Resolve context_summary
        kw_contexts = kwargs.get("context_summary")
        if kw_contexts is not None:
            contexts = [str(v) for v in _per_item(kw_contexts, count, "")]
        else:
            contexts = [extract_context_from_prompt(p) for p in prompt_texts]

        # 4. Resolve prev_chapter_tail
        kw_tails = kwargs.get("prev_chapter_tail")
        if kw_tails is not None:
            tails = [str(v) for v in _per_item(kw_tails, count, "")]
        else:
            tails = [extract_tail_from_prompt(p) for p in prompt_texts]

        cache_key = (tuple(texts), tuple(targets), tuple(lowers), tuple(uppers), tuple(maxes), tuple(contexts), tuple(tails))
        if self._cache_key == cache_key and self._cache_value is not None:
            return self._cache_value

        # Records in a GRPO group share a prompt.  Calculate one group at a
        # time when metadata differs, keeping every reward aligned correctly.
        result: Dict[str, List[float]] = {"length": [], "coherence": [], "quality": []}
        for text, target, l_low, l_up, l_m, context, tail in zip(
            texts, targets, lowers, uppers, maxes, contexts, tails
        ):
            scores = self.engine.compute_batch_rewards(
                rollouts=[text],
                target_length=target,
                context_summary=context,
                prev_chapter_tail=tail,
                l_lower=l_low,
                l_upper=l_up,
                l_max=l_m,
            )
            # GRPOTrainer adds the rewards returned by its functions.  Apply
            # the MA-GRPO YAML weights here so this is exactly R_total.
            result["length"].append(self.engine.w_length * float(scores["r_length"][0]))
            result["coherence"].append(self.engine.w_coherence * float(scores["r_coherence"][0]))
            result["quality"].append(self.engine.w_quality * float(scores["r_quality"][0]))
        self._cache_key = cache_key
        self._cache_value = result
        return result

    def length(self, completions: Sequence[Any], **kwargs: Any) -> List[float]:
        return self._components(completions, **kwargs)["length"]

    def coherence(self, completions: Sequence[Any], **kwargs: Any) -> List[float]:
        return self._components(completions, **kwargs)["coherence"]

    def quality(self, completions: Sequence[Any], **kwargs: Any) -> List[float]:
        return self._components(completions, **kwargs)["quality"]


def load_training_dataset(path: str):
    """Load and validate JSONL while retaining all rich metadata from dataset_generator for reward functions."""
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(path), split="train")
    # Accept either 'prompt' or 'full_training_prompt' from dataset_generator standard output
    has_prompt = "prompt" in dataset.column_names or "full_training_prompt" in dataset.column_names
    if not has_prompt:
        raise ValueError("Dataset must contain either 'prompt' or 'full_training_prompt' column.")

    def normalise(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(row.get("prompt") or row.get("full_training_prompt") or "").strip()
        if not prompt:
            raise ValueError("Every training record must have a non-empty prompt.")
        target_words = int(row.get("target_words") or 1200)
        if target_words <= 0:
            raise ValueError("target_words must be a positive integer.")

        ctx_state = row.get("context_state") or {}
        short_mem = ctx_state.get("short_term_memory") or {}
        ctx_summary = str(
            row.get("context_summary")
            or short_mem.get("recent_chapter_summary")
            or ctx_state.get("plot_summary")
            or ""
        )
        prev_tail = str(
            row.get("prev_chapter_tail")
            or short_mem.get("recent_tail_excerpt")
            or ctx_state.get("recent_tail_excerpt")
            or ""
        )

        res = dict(row)
        res["prompt"] = prompt
        res["target_words"] = target_words
        res["context_summary"] = ctx_summary
        res["prev_chapter_tail"] = prev_tail
        return res

    return dataset.map(normalise, desc="Validating MA-GRPO records")


def _supported(cls: type, values: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the script usable across nearby TRL versions with renamed options."""
    parameters = inspect.signature(cls.__init__).parameters
    return {key: value for key, value in values.items() if key in parameters}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    try:
        import yaml
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise SystemExit(
            "Training dependencies are missing. Install requirements plus `datasets` and `bitsandbytes`. "
            f"Original error: {error}"
        ) from error

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    train_cfg = config.get("training", {})
    rollout_cfg = config.get("rollout", {})
    lora_cfg = config.get("lora", {})
    model_cfg = config.get("model", {})

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for QLoRA GRPO training.")
    set_seed(args.seed)
    dataset = load_training_dataset(args.dataset)
    LOG.info("Loaded %d chapter prompts from %s", len(dataset), args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2" if model_cfg.get("use_flash_attention_2", False) else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_cfg.get("r", 32)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 64)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            bias=str(lora_cfg.get("bias", "none")),
            target_modules=lora_cfg.get("target_modules"),
        ),
    )
    model.print_trainable_parameters()

    group_size = args.group_size or int(rollout_cfg.get("group_size", 4))
    max_tokens = args.max_completion_tokens or int(rollout_cfg.get("max_tokens", 1500))
    output_dir = str(args.output_dir)

    grad_accum = args.gradient_accumulation or int(train_cfg.get("gradient_accumulation_steps", 4))
    if (args.batch_size * grad_accum) % group_size != 0:
        target_accum = max(1, group_size // max(1, args.batch_size))
        LOG.warning(
            "TRL GRPO requires (batch_size * gradient_accumulation) to be divisible by num_generations. "
            "Auto-adjusting gradient_accumulation_steps from %d to %d (group_size=%d, batch_size=%d).",
            grad_accum,
            target_accum,
            group_size,
            args.batch_size,
        )
        grad_accum = target_accum

    grpo_values = {
        "output_dir": output_dir,
        "learning_rate": float(train_cfg.get("learning_rate", 5e-6)),
        "max_steps": args.max_steps if args.max_steps is not None else int(train_cfg.get("max_steps", 500)),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": grad_accum,
        "logging_steps": int(train_cfg.get("logging_steps", 2)),
        "save_steps": args.save_steps if args.save_steps is not None else int(train_cfg.get("save_steps", 10)),
        "num_generations": group_size,
        "max_completion_length": max_tokens,
        "beta": float(train_cfg.get("kl_beta", 0.04)),
        "bf16": torch.cuda.is_bf16_supported(),
        "fp16": not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "remove_unused_columns": False,
    }
    training_args = GRPOConfig(**_supported(GRPOConfig, grpo_values))

    reward_device = args.reward_device
    if reward_device is None:
        reward_device = "cuda:1" if torch.cuda.device_count() > 1 else "cpu"
    intrinsic_rewards = IntrinsicRewardFunctions(args.config, reward_device)
    trainer_values = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "reward_funcs": [intrinsic_rewards.length, intrinsic_rewards.coherence, intrinsic_rewards.quality],
        "processing_class": tokenizer,
        "tokenizer": tokenizer,  # older TRL releases use this name
    }
    trainer = GRPOTrainer(**_supported(GRPOTrainer, trainer_values))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    LOG.info("Training complete. LoRA adapter and tokenizer saved to %s", output_dir)


if __name__ == "__main__":
    main()
