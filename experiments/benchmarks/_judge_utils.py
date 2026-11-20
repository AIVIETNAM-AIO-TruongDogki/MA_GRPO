"""Shared JSON parsing and Judge LLM backends (API / Local vLLM / Mock) for both benchmark suites."""

from __future__ import annotations

import gc
import json
import os
import re
from typing import Any, Callable, Dict, Optional

from src.pipeline import resolve_optimal_dtype


def parse_json_object(raw_text: str) -> Dict[str, Any]:
    """Robustly extracts the first valid JSON object from an LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", raw_text or "").strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response.")
    parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object.")
    return parsed


def build_openai_compatible_judge_fn(
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Callable[[str, str], str]:
    """
    Creates an OpenAI-compatible API caller (supports OpenAI, DeepSeek, Gemini OpenAI endpoint,
    OpenRouter, Together, or local vLLM OpenAI server).
    """
    key = api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base = api_base or os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    if not key:
        raise RuntimeError(
            "API judge requested, but neither --judge-api-key nor JUDGE_API_KEY / OPENAI_API_KEY is set."
        )
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)

    def _call(system_prompt: str, user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    return _call


class UnifiedJudgeBackend:
    """Lifecycle manager for benchmark Judge LLMs (`none`, `mock`, `api`, `local_vllm`)."""

    def __init__(
        self,
        backend: str = "none",
        model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 16384,
        mock_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.backend = backend
        self.model = model
        self._vllm = None
        self.judge_fn: Optional[Callable[[str, str], str]] = None

        if backend == "none":
            return

        if backend == "mock":
            self.judge_fn = mock_fn or (lambda _s, _u: "{}")
            return

        if backend == "api":
            self.judge_fn = build_openai_compatible_judge_fn(
                model=model,
                api_key=api_key,
                api_base=api_base,
            )
            return

        if backend == "local_vllm":
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            from vllm import LLM, SamplingParams

            lower_m = model.lower()
            dtype = "float16" if ("awq" in lower_m or "gptq" in lower_m) else resolve_optimal_dtype()
            print(
                f"[Benchmark Judge vLLM] Loading {model} (dtype={dtype}, max_len={max_model_len}, mem={gpu_memory_utilization})...",
                flush=True,
            )
            self._vllm = LLM(
                model=model,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=True,
            )
            tok = self._vllm.get_tokenizer()

            def _vllm_call(sys_p: str, usr_p: str) -> str:
                formatted = tok.apply_chat_template(
                    [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                n_tok = len(tok.encode(formatted, add_special_tokens=False))
                safe_max = max(64, min(2048, max_model_len - n_tok - 32))
                params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=safe_max)
                out = self._vllm.generate([formatted], params, use_tqdm=False)
                return out[0].outputs[0].text.strip() if out[0].outputs else ""

            self.judge_fn = _vllm_call
            return

        raise ValueError(f"Unsupported judge backend: {backend}")

    def release(self) -> None:
        if self._vllm is not None:
            del self._vllm
            self._vllm = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
