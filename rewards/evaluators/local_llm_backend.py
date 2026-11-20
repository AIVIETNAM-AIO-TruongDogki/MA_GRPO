"""
Local GPU 32B 4-bit LLM Judge Backend (Co-Resident with TRL GRPOTrainer on 1x A100 80GB, 0 API Cost)
plus CPU-Safe Deterministic Mock Backend for Unit Tests & CI.

Why HuggingFace `transformers` 4-bit instead of vLLM during GRPO training?
  - vLLM pre-allocates a static GPU KV-cache pool that collides with PyTorch GRPO backward passes.
  - Loading `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` (pre-quantized 19 GB download, pure bitsandbytes)
    or `Qwen/Qwen2.5-32B-Instruct` (NF4 4-bit, ~20 GB VRAM) with `torch.inference_mode()` co-exists
    smoothly alongside Qwen2.5-7B QLoRA (~32 GB peak) on an 80 GB A100 (~54-58 GB total).
"""

from __future__ import annotations

import gc
import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.rewards import strip_think_tags


DEFAULT_LOCAL_32B_JUDGE_MODEL = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"


class LocalHF4BitJudgeBackend:
    """
    Batched HuggingFace 4-bit NF4 / pre-quantized 32B Judge backend running in `torch.inference_mode()`.
    Supports:
      - `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` (Recommended: ~19 GB download & VRAM, pure bitsandbytes)
      - `Qwen/Qwen2.5-32B-Instruct` (on-the-fly 4-bit NF4 via bitsandbytes)
      - `Qwen/Qwen2.5-32B-Instruct-AWQ` (if AutoAWQ is installed)
      - `Qwen/Qwen2.5-14B-Instruct` / `Qwen/Qwen2.5-7B-Instruct` (lighter fallback)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_LOCAL_32B_JUDGE_MODEL,
        device: str = "cuda:0",
        eval_batch_size: int = 2,
        max_input_tokens: int = 3584,
        max_new_tokens: int = 280,
    ):
        self.model_id = model_id
        self.device = device
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        lower_id = self.model_id.lower()

        print(
            f"[Phase2 32B Judge] Loading {self.model_id} on {self.device} "
            f"(eval_batch_size={self.eval_batch_size}, compute_dtype={compute_dtype})...",
            flush=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": {"": self.device} if torch.cuda.is_available() else "auto",
            "attn_implementation": "sdpa",
        }
        if "awq" in lower_id:
            load_kwargs["torch_dtype"] = torch.float16
        elif "bnb-4bit" in lower_id or "4bit" in lower_id:
            # Pre-quantized bitsandbytes repo already ships quantization_config in config.json
            load_kwargs["torch_dtype"] = compute_dtype
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["torch_dtype"] = compute_dtype

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

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

    def generate_batch(
        self,
        requests: Sequence[Tuple[str, str]],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        if not requests:
            return []
        if self.model is None:
            self.load()

        import torch

        gen_max = int(max_new_tokens or self.max_new_tokens)
        outputs: List[str] = []
        dev = next(self.model.parameters()).device

        with torch.inference_mode():
            for start in range(0, len(requests), self.eval_batch_size):
                chunk = requests[start : start + self.eval_batch_size]
                prompts = [self._format_chat(sys_p, usr_p) for sys_p, usr_p in chunk]
                enc = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_tokens,
                ).to(dev)
                input_len = enc["input_ids"].shape[1]
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=gen_max,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
                for row in generated:
                    decoded = self.tokenizer.decode(row[input_len:], skip_special_tokens=True).strip()
                    outputs.append(decoded)
                del enc, generated
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return outputs

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: Optional[int] = None) -> str:
        res = self.generate_batch([(system_prompt, user_prompt)], max_new_tokens=max_new_tokens)
        return res[0] if res else ""

    def release(self) -> None:
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class MockDeterministicJudgeBackend:
    """
    Deterministic CPU-safe mock Judge strictly for unit tests & offline smoke verification.
    Responds sensibly to goal lexical overlap, filler repetition, and explicit contradiction markers
    so perturbation discrimination tests (`Clean > Filler / Off-Plan / Contradictory`) can run on CPU.
    """

    @staticmethod
    def _extract_section(prompt: str, header: str) -> str:
        pattern = re.escape(header) + r"\s*([\s\S]*?)(?=\n\[[A-Z][A-Z0-9 _/-]*\]:|\Z)"
        m = re.search(pattern, prompt)
        return m.group(1).strip() if m else ""

    def _score_prompt(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        goal = self._extract_section(user_prompt, "[CHAPTER GOAL]:")
        scenes = self._extract_section(user_prompt, "[SCENE OUTLINE]:")
        chapter = strip_think_tags(self._extract_section(user_prompt, "[GENERATED CHAPTER]:"))
        chars = self._extract_section(user_prompt, "[ACTIVE CHARACTERS]:")

        goal_toks = {w for w in re.findall(r"[a-z]{4,}", (goal + " " + scenes).lower())}
        chap_words = re.findall(r"[a-z]{4,}", chapter.lower())
        chap_toks = set(chap_words)

        coverage = len(goal_toks & chap_toks) / max(1, len(goal_toks)) if goal_toks else 0.75
        event_real = round(min(0.96, max(0.12, 0.20 + 0.95 * coverage)), 3)

        # Penalize repetitive filler / low distinct-3 inside chapter
        all_w = chapter.lower().split()
        d3 = (
            len(set(zip(all_w, all_w[1:], all_w[2:]))) / max(1, len(all_w) - 2)
            if len(all_w) >= 3
            else 0.5
        )
        filler_pen = 0.45 if ("empty filler" in chapter.lower() or d3 < 0.55) else 0.0
        causal_dev = round(min(0.95, max(0.15, 0.25 + 0.65 * coverage + 0.25 * d3 - filler_pen)), 3)
        meaningful = round(min(0.95, max(0.10, 0.15 + 0.55 * coverage + 0.45 * d3 - filler_pen)), 3)

        char_names = [c.strip().split()[0].lower() for c in re.split(r"[,;]", chars) if c.strip()]
        char_hit = (
            sum(1 for cn in char_names if cn and cn in chapter.lower()) / max(1, len(char_names))
            if char_names
            else 0.85
        )

        contradictions: List[Dict[str, str]] = []
        low_ch = chapter.lower()
        if "critical_contradiction" in low_ch or "resurrected without explanation" in low_ch:
            contradictions.append({"severity": "critical", "reason": "Character dead in prior chapter reappears alive."})
        if "major_contradiction" in low_ch:
            contradictions.append({"severity": "major", "reason": "Direct reversal of established motivation or location."})
        if "minor_contradiction" in low_ch:
            contradictions.append({"severity": "minor", "reason": "Minor prop detail mismatch."})

        contra_pen = min(0.55, 0.30 * sum(1 for c in contradictions if c["severity"] == "critical") + 0.15 * sum(1 for c in contradictions if c["severity"] == "major"))

        macro_adh = round(min(0.95, max(0.12, 0.18 + 0.90 * coverage - 0.5 * contra_pen)), 3)
        chap_coh = round(min(0.95, max(0.15, 0.30 + 0.55 * d3 - filler_pen * 0.5 - contra_pen * 0.5)), 3)
        char_cons = round(min(0.95, max(0.15, 0.30 + 0.60 * char_hit - contra_pen)), 3)
        narr_prog = round(min(0.95, max(0.15, 0.20 + 0.55 * coverage + 0.30 * d3 - filler_pen)), 3)
        holistic = round(0.5 * (event_real + meaningful) * 0.5 + 0.5 * (macro_adh + chap_coh) * 0.5, 3)

        if "Holistic" in system_prompt:
            return {"holistic_score": holistic, "contradictions": contradictions, "rationale": "Mock holistic audit."}
        if "Unified" in system_prompt:
            return {
                "event_realization": event_real,
                "causal_development": causal_dev,
                "meaningful_elaboration": meaningful,
                "macro_adherence": macro_adh,
                "chapter_coherence": chap_coh,
                "character_consistency": char_cons,
                "narrative_progression": narr_prog,
                "contradictions": contradictions,
            }
        if "Local Expansion" in system_prompt:
            return {
                "event_realization": event_real,
                "causal_development": causal_dev,
                "meaningful_elaboration": meaningful,
                "evidence": [{"criterion": "meaningful_elaboration", "text_span": chapter[:80], "reason": "Mock span"}],
                "missing_elements": [] if event_real >= 0.7 else ["Core chapter beat under-developed."],
            }
        return {
            "macro_adherence": macro_adh,
            "chapter_coherence": chap_coh,
            "character_consistency": char_cons,
            "narrative_progression": narr_prog,
            "contradictions": contradictions,
            "evidence": [{"criterion": "chapter_coherence", "text_span": chapter[:80], "reason": "Mock span"}],
        }

    def generate_batch(self, requests: Sequence[Tuple[str, str]], max_new_tokens: Optional[int] = None) -> List[str]:
        return [json.dumps(self._score_prompt(s, u)) for s, u in requests]

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: Optional[int] = None) -> str:
        return json.dumps(self._score_prompt(system_prompt, user_prompt))

    def release(self) -> None:
        pass


def build_judge_backend(
    backend: str = "local_hf_4bit",
    model_id: str = DEFAULT_LOCAL_32B_JUDGE_MODEL,
    device: str = "cuda:0",
    eval_batch_size: int = 2,
    max_input_tokens: int = 3584,
    max_new_tokens: int = 280,
) -> Any:
    """Factory returning a 32B Local HF 4-bit backend or CPU Mock backend."""
    if backend == "mock":
        return MockDeterministicJudgeBackend()
    if backend in {"local_hf_4bit", "local_hf", "local"}:
        return LocalHF4BitJudgeBackend(
            model_id=model_id,
            device=device,
            eval_batch_size=eval_batch_size,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unsupported Phase 2 judge backend '{backend}'. Expected 'local_hf_4bit' or 'mock'.")
