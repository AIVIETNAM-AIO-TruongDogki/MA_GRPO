"""
Shared Qwen2.5-7B-Instruct Backend for Multi-Agent and Iterative Self-Alignment Baselines:
  - AGENTS' ROOM
  - STORYWRITER
  - Self-Lengthen

Supports both standard chat generation (`generate_chat`) and raw prefix completion
(`generate_completion`) required by Self-Lengthen's step2_ext_responses.py.
"""

import gc
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def clean_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def get_peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    total_peak = 0.0
    for d in range(torch.cuda.device_count()):
        total_peak += torch.cuda.max_memory_allocated(d) / (1024**3)
    return round(total_peak, 2)


def reset_vram_stats():
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)


class QwenAgentBackend:
    """
    Unified Qwen/Qwen2.5-7B-Instruct inference engine (4-bit NF4 / BF16).
    Can be shared across AGENTS' ROOM, STORYWRITER, and Self-Lengthen without reloading.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct", load_in_4bit: bool = True):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None:
            return
        print(f"\n[QwenAgentBackend] Loading {self.model_id} (4-bit={self.load_in_4bit})...", flush=True)
        clean_gpu_memory()

        use_cuda = torch.cuda.is_available()
        compute_dtype = (
            torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)
        )

        quant_config = None
        if use_cuda and self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "device_map": "auto" if use_cuda else None,
            "torch_dtype": compute_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        print(f"[QwenAgentBackend] {self.model_id} loaded successfully!", flush=True)

    def unload(self):
        print(f"[QwenAgentBackend] Unloading {self.model_id} and freeing VRAM...", flush=True)
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        clean_gpu_memory()

    def generate_chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 1200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ) -> str:
        """Standard chat-template generation for conversational agents."""
        if self.model is None:
            self.load()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            sys_block = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
            formatted = f"{sys_block}<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(formatted, return_tensors="pt")
        dev = next(self.model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items() if k != "token_type_ids"}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0.01,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        in_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True).strip()

    def generate_completion(
        self,
        raw_prompt_prefix: str,
        max_new_tokens: int = 1500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ) -> str:
        """
        Raw prefix completion (used by Self-Lengthen's `get_completion(latter_input)`
        where the assistant turn is pre-filled with `former_output`).
        """
        if self.model is None:
            self.load()

        inputs = self.tokenizer(raw_prompt_prefix, return_tensors="pt")
        dev = next(self.model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items() if k != "token_type_ids"}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0.01,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        in_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True)
