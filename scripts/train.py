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

from src.rewards import CompositeRewardEngine


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
        targets = [int(v) for v in _per_item(kwargs.get("target_words"), count, 1200)]
        contexts = [str(v) for v in _per_item(kwargs.get("context_summary"), count, "")]
        tails = [str(v) for v in _per_item(kwargs.get("prev_chapter_tail"), count, "")]
        cache_key = (tuple(texts), tuple(targets), tuple(contexts), tuple(tails))
        if self._cache_key == cache_key and self._cache_value is not None:
            return self._cache_value

        # Records in a GRPO group share a prompt.  Calculate one group at a
        # time when metadata differs, keeping every reward aligned correctly.
        result: Dict[str, List[float]] = {"length": [], "coherence": [], "quality": []}
        for text, target, context, tail in zip(texts, targets, contexts, tails):
            scores = self.engine.compute_batch_rewards(
                rollouts=[text],
                target_length=target,
                context_summary=context,
                prev_chapter_tail=tail,
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
    """Load and validate JSONL while retaining metadata for reward functions."""
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(path), split="train")
    required = {"prompt"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(sorted(missing))}")

    def normalise(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(row["prompt"]).strip()
        if not prompt:
            raise ValueError("Every training record must have a non-empty prompt.")
        target_words = int(row.get("target_words") or 1200)
        if target_words <= 0:
            raise ValueError("target_words must be a positive integer.")
        return {
            "prompt": prompt,
            "target_words": target_words,
            "context_summary": str(row.get("context_summary") or ""),
            "prev_chapter_tail": str(row.get("prev_chapter_tail") or ""),
        }

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
        "logging_steps": int(train_cfg.get("logging_steps", 5)),
        "save_steps": int(train_cfg.get("save_steps", 50)),
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
