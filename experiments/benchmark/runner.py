"""
End-to-End Runner for the Stage-1 Macro-Planning & Writing Utility Benchmark.

Supports 3 execution modes (or `--mode all`):
  1. Track 1 (`track1`) : VR-CLI Next-Chapter Prediction on LONGSTORY / BookSum / PG-19
                          measuring PPL_base, PPL_planned, I_pi (%), and Group A plan quality.
  2. Track 2 (`track2`) : 5-Condition Structural Ablation (Conditions A, B, C, D, E) on
                          TELL ME A STORY / MoPS premises, measuring Group A + Group B + H1-H4.
  3. Perturb (`perturb`): 5-Protocol Plan-Sensitivity & Plan-Blindness Audit (Delta Score).

Memory-safe two-phase GPU orchestration (crucial for 40GB A100/H100):
  - Phase 1 runs Planner + Writer LLM generation (caching plans/stories to `.jsonl`).
  - Phase 2 frees the Planner/Writer GPU model and loads frozen VR-CLI Qwen-7B + DeBERTa NLI.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.perturbation import PERTURBATION_TYPES
from src.pipeline import MAGRPOPipeline, VLLMWritingPolicy, resolve_optimal_dtype
from experiments.conditions import plan_next_chapter_from_history, run_condition
from experiments.benchmark.evaluator import (
    LLMJudge5DimEvaluator,
    NLIScorer,
    VRCLIPerplexityEvaluator,
    audit_hypotheses_h1_to_h4,
    compute_perturbation_delta,
    evaluate_length_and_termination,
    evaluate_lexical_and_burstiness,
    evaluate_macro_plan_group_a,
    evaluate_plan_adherence_doc_nli,
    load_longstory_next_chapter_samples,
    load_tell_me_a_story_premises,
)


def _cleanup_cuda_engine() -> None:
    """Release vLLM distributed state and empty PyTorch CUDA cache."""
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _hf_chat_completion(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_model_len: Optional[int] = None,
) -> str:
    """Shared HuggingFace Transformers chat generation helper for Planner/Writer/Judge."""
    import torch

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    safe_max_tokens = max_tokens
    if max_model_len is not None and max_model_len > 0:
        prompt_tokens = int(inputs["input_ids"].shape[1])
        safe_max_tokens = max(1, min(max_tokens, max_model_len - prompt_tokens - 32))
    do_sample = temperature > 0.05
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=safe_max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Backend Builders (Mock / HuggingFace Transformers / vLLM)
# ---------------------------------------------------------------------------
class LocalGenerationBackend:
    """Manages Planner and Writer callables plus clean GPU VRAM release."""

    def __init__(
        self,
        backend: str,
        model_name: str,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.65,
        max_model_len: int = 8192,
    ):
        self.backend = backend
        self.model_name = model_name
        self.dtype = resolve_optimal_dtype(dtype)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        self._hf_model = None
        self._hf_tokenizer = None
        self.writing_policy: Optional[VLLMWritingPolicy] = None
        self.planner_llm_fn: Optional[Callable[[str, str], str]] = None

        self._init_backend()

    def _init_backend(self) -> None:
        if self.backend == "mock":
            self.writing_policy = VLLMWritingPolicy(use_mock=True)
            self.planner_llm_fn = None
            return

        if self.backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
            print(f"[HF Backend] Loading {self.model_name} ({self.dtype})...")
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._hf_model.eval()

            def _hf_chat(system_prompt: str, user_prompt: str, max_tokens: int, temp: float, top_p: float) -> str:
                return _hf_chat_completion(
                    self._hf_model,
                    self._hf_tokenizer,
                    system_prompt,
                    user_prompt,
                    max_tokens,
                    temp,
                    top_p,
                    max_model_len=self.max_model_len,
                )

            self.planner_llm_fn = lambda sys_p, usr_p: _hf_chat(sys_p, usr_p, 4096, 0.4, 0.9)
            self.writing_policy = VLLMWritingPolicy(
                model_name=self.model_name,
                max_model_len=self.max_model_len,
                use_mock=False,
                dtype=self.dtype,
                custom_generate_fn=lambda prompt, max_tok, temp, top_p: _hf_chat("", prompt, max_tok, temp, top_p),
            )
            return

        if self.backend == "vllm":
            from vllm import SamplingParams

            self.writing_policy = VLLMWritingPolicy(
                model_name=self.model_name,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                use_mock=False,
            )
            if self.writing_policy.llm is None:
                raise RuntimeError("vLLM failed to initialize. Check CUDA/vLLM installation or pass --backend hf.")

            vllm_engine = self.writing_policy.llm

            def _vllm_planner(sys_p: str, usr_p: str) -> str:
                tokenizer = vllm_engine.get_tokenizer()
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                params = SamplingParams(
                    temperature=0.4,
                    top_p=0.9,
                    max_tokens=self.writing_policy._clamp_max_tokens(prompt_text, 4096),
                )
                outputs = vllm_engine.generate([prompt_text], params)
                return outputs[0].outputs[0].text.strip()

            self.planner_llm_fn = _vllm_planner
            return

        raise ValueError(f"Unsupported backend: {self.backend}")

    def release(self) -> None:
        """Free GPU memory before Phase 2 VR-CLI & NLI evaluation."""
        if self.backend == "mock":
            return
        if self._hf_model is not None:
            del self._hf_model
            self._hf_model = None
        if self._hf_tokenizer is not None:
            del self._hf_tokenizer
            self._hf_tokenizer = None
        if self.writing_policy is not None and self.writing_policy.llm is not None:
            del self.writing_policy.llm
            self.writing_policy.llm = None
        self.planner_llm_fn = None
        _cleanup_cuda_engine()


def resolve_judge_runtime(args: argparse.Namespace) -> Tuple[Optional[str], Optional[str]]:
    """Resolve effective (judge_backend, judge_model) from CLI args and environment."""
    if not getattr(args, "enable_judge", False):
        return None, None

    requested = getattr(args, "judge_backend", "auto")
    api_key = getattr(args, "judge_api_key", None) or os.environ.get("OPENAI_API_KEY", "").strip()
    judge_model = getattr(args, "judge_model", "gpt-4o-mini")

    if requested == "auto":
        if api_key:
            jb = "api"
        elif args.backend == "mock":
            jb = "mock"
        else:
            jb = f"local_{args.backend}"
    else:
        jb = requested

    if jb in {"local_vllm", "local_hf", "mock"} and judge_model == "gpt-4o-mini":
        judge_model = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    return jb, judge_model


class LocalJudgeBackend:
    """
    Runs LLM-as-a-Judge locally on GPU (e.g. Qwen2.5-14B-Instruct or Qwen2.5-32B-Instruct-AWQ)
    or via OpenAI API / mock without keeping generation or NLI models loaded simultaneously.
    """

    def __init__(
        self,
        judge_backend: str,
        judge_model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 16384,
    ):
        self.judge_backend = judge_backend
        self.judge_model = judge_model
        self._vllm = None
        self._hf_model = None
        self._hf_tokenizer = None
        self.judge_fn: Optional[Callable[[str, str], str]] = None

        if judge_backend == "api":
            self.judge_fn = build_openai_judge_fn(api_key, api_base, judge_model)
            if self.judge_fn is None:
                raise RuntimeError("Judge backend 'api' selected, but OPENAI_API_KEY / --judge-api-key is missing.")
            return

        if judge_backend == "mock":
            self.judge_fn = lambda _sys, _usr: json.dumps({
                "story_coherence": 4.2,
                "plan_adherence": 4.4,
                "completeness": 4.1,
                "writing_quality": 4.0,
                "global_consistency": 4.3,
                "n_fatal_inconsistencies": 0,
                "n_moderate_inconsistencies": 1,
                "n_minor_inconsistencies": 1,
                "perturbation_behavior": "Clean",
                "rationale": "Mock offline judge response.",
            })
            return

        if judge_backend == "local_vllm":
            from vllm import LLM, SamplingParams

            lower_m = judge_model.lower()
            resolved_dtype = "float16" if ("awq" in lower_m or "gptq" in lower_m) else resolve_optimal_dtype(dtype)
            print(f"[Judge vLLM] Loading local judge: {judge_model} ({resolved_dtype}, VRAM={gpu_memory_utilization})...")
            self._vllm = LLM(
                model=judge_model,
                dtype=resolved_dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=True,
            )
            tok = self._vllm.get_tokenizer()

            def _vllm_judge(sys_p: str, usr_p: str) -> str:
                prompt_text = tok.apply_chat_template(
                    [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_tokens = len(tok.encode(prompt_text, add_special_tokens=False))
                safe_max = max(64, min(2048, max_model_len - prompt_tokens - 32))
                params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=safe_max)
                outputs = self._vllm.generate([prompt_text], params)
                return outputs[0].outputs[0].text.strip()

            self.judge_fn = _vllm_judge
            return

        if judge_backend == "local_hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            resolved_dtype = resolve_optimal_dtype(dtype)
            torch_dtype = torch.bfloat16 if resolved_dtype == "bfloat16" else torch.float16
            print(f"[Judge HF] Loading local judge: {judge_model} ({resolved_dtype})...")
            self._hf_tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                judge_model,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._hf_model.eval()
            self.judge_fn = lambda sys_p, usr_p: _hf_chat_completion(
                self._hf_model,
                self._hf_tokenizer,
                sys_p,
                usr_p,
                2048,
                0.0,
                1.0,
                max_model_len=max_model_len,
            )
            return

        raise ValueError(f"Unsupported judge_backend: {judge_backend}")

    def release(self) -> None:
        if self.judge_backend in {"api", "mock"}:
            return
        if self._hf_model is not None:
            del self._hf_model
            self._hf_model = None
        if self._hf_tokenizer is not None:
            del self._hf_tokenizer
            self._hf_tokenizer = None
        if self._vllm is not None:
            del self._vllm
            self._vllm = None
        self.judge_fn = None
        _cleanup_cuda_engine()


def build_openai_judge_fn(
    api_key: Optional[str],
    api_base: Optional[str],
    model: str,
) -> Optional[Callable[[str, str], str]]:
    """Build OpenAI-compatible HTTP Judge callback using stdlib urllib."""
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if not resolved_key:
        return None
    base_url = (api_base or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    def _call_judge(system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps({
            "model": model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {resolved_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, TimeoutError) as exc:
            raise RuntimeError(f"Judge API call failed: {exc}") from exc

    return _call_judge


# ---------------------------------------------------------------------------
# Checkpoint I/O & Aggregation Helpers (Pure Stdlib)
# ---------------------------------------------------------------------------
def load_jsonl_index(path: Path, key_fields: Tuple[str, ...]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    index: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if not path.exists():
        return index
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            k = tuple(row.get(kf) for kf in key_fields)
            index[k] = row
    return index


def append_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl_records(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def mean_numeric(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and not isinstance(r[key], bool)]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def export_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# Phase 1: Generation (Track 1 Plans, Track 2 Stories, Perturbed Stories)
# ---------------------------------------------------------------------------
def run_phase1_generation(
    args: argparse.Namespace,
    gen_backend: LocalGenerationBackend,
    output_dir: Path,
) -> Dict[str, Any]:
    pipeline = MAGRPOPipeline(
        writing_policy=gen_backend.writing_policy,
        planner_llm_fn=gen_backend.planner_llm_fn,
    )
    allow_demo = args.backend == "mock" or args.allow_synthetic_demo
    plan_conditions = [c.upper() for c in args.conditions if c.upper() != "A"]
    all_conditions = [c.upper() for c in args.conditions]

    generated: Dict[str, Any] = {"track1": [], "track2": [], "perturb": []}

    # --- Track 1: Next-Chapter Plan Generation ---
    if args.mode in {"track1", "all"}:
        t1_cache_path = output_dir / "track1_plans_cache.jsonl"
        existing = load_jsonl_index(t1_cache_path, ("sample_id", "condition")) if args.resume else {}
        if not args.resume and t1_cache_path.exists():
            t1_cache_path.unlink()

        samples = load_longstory_next_chapter_samples(
            data_path=args.track1_data,
            max_samples=args.track1_samples,
            allow_synthetic_demo=allow_demo,
        )
        print(f"[Track 1 Phase 1] Generating plans for {len(samples)} samples x {plan_conditions}...")
        records: List[Dict[str, Any]] = []
        for sample in samples:
            for cond in plan_conditions:
                key = (sample["sample_id"], cond)
                if key in existing:
                    records.append(existing[key])
                    continue
                plan_res = plan_next_chapter_from_history(
                    history_text=sample["history_text"],
                    condition=cond,
                    llm_generate_fn=gen_backend.planner_llm_fn,
                    premise=sample.get("premise", ""),
                )
                rec = {
                    "sample_id": sample["sample_id"],
                    "source": sample.get("source", ""),
                    "chapter_index": sample.get("chapter_index", 2),
                    "condition": cond,
                    "premise": sample.get("premise", ""),
                    "history_text": sample["history_text"],
                    "gold_chapter": sample["gold_chapter"],
                    "plan_res": plan_res,
                }
                append_jsonl_record(t1_cache_path, rec)
                records.append(rec)
        generated["track1"] = records

    # --- Track 2: 5-Condition Ablation Generation ---
    t2_clean_cache: Dict[str, Dict[str, Any]] = {}
    if args.mode in {"track2", "perturb", "all"}:
        n_t2 = args.track2_samples if args.mode in {"track2", "all"} else args.perturb_samples
        premises = load_tell_me_a_story_premises(
            data_path=args.track2_data,
            max_samples=n_t2,
            target_words=args.target_words,
            strip_length_constraints=True,
        )

    if args.mode in {"track2", "all"}:
        t2_cache_path = output_dir / "track2_stories_cache.jsonl"
        existing = load_jsonl_index(t2_cache_path, ("sample_id", "condition")) if args.resume else {}
        if not args.resume and t2_cache_path.exists():
            t2_cache_path.unlink()

        print(f"[Track 2 Phase 1] Generating stories for {len(premises)} premises x {all_conditions}...")
        t2_records: List[Dict[str, Any]] = []
        for idx, item in enumerate(premises):
            for cond in all_conditions:
                key = (item["id"], cond)
                if key in existing:
                    rec = existing[key]
                else:
                    print(f"  -> Track 2 [{idx+1}/{len(premises)}] {item['id']} | Condition {cond}")
                    out = run_condition(
                        pipeline=pipeline,
                        condition=cond,
                        premise=item["premise"],
                        genre=item.get("genre", "Creative Fiction"),
                        total_target_words=item["target_words"],
                        num_chapters=args.num_chapters,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        enable_cot=args.enable_cot,
                    )
                    rec = {
                        "sample_id": item["id"],
                        "source": item.get("source", ""),
                        "genre": item.get("genre", "Creative Fiction"),
                        "premise": item["premise"],
                        "condition": cond,
                        "pipeline_output": out,
                    }
                    append_jsonl_record(t2_cache_path, rec)
                t2_records.append(rec)
                if cond == args.perturb_base_condition.upper():
                    t2_clean_cache[item["id"]] = rec
        generated["track2"] = t2_records

    # --- Perturbation Protocol: 5 Standardized Perturbations ---
    if args.mode in {"perturb", "all"}:
        p_cache_path = output_dir / "perturb_stories_cache.jsonl"
        existing = load_jsonl_index(p_cache_path, ("sample_id", "perturbation_type")) if args.resume else {}
        if not args.resume and p_cache_path.exists():
            p_cache_path.unlink()

        base_cond = args.perturb_base_condition.upper()
        p_premises = premises[: args.perturb_samples]
        print(f"[Perturb Phase 1] Generating {len(p_premises)} premises x {len(args.perturbations)} perturbations on Cond {base_cond}...")
        p_records: List[Dict[str, Any]] = []
        for idx, item in enumerate(p_premises):
            # Ensure clean baseline reference exists
            if item["id"] not in t2_clean_cache:
                clean_out = run_condition(
                        pipeline=pipeline,
                    condition=base_cond,
                    premise=item["premise"],
                    genre=item.get("genre", "Creative Fiction"),
                    total_target_words=item["target_words"],
                    num_chapters=args.num_chapters,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    enable_cot=args.enable_cot,
                )
                t2_clean_cache[item["id"]] = {
                    "sample_id": item["id"],
                    "premise": item["premise"],
                    "condition": base_cond,
                    "pipeline_output": clean_out,
                }
            # Pick cyclic donor plan when available for unrelated_plan_swapping
            donor_item = p_premises[(idx + 1) % len(p_premises)] if len(p_premises) > 1 else None
            donor_blueprint = (
                t2_clean_cache[donor_item["id"]]["pipeline_output"].get("blueprint")
                if donor_item and donor_item["id"] in t2_clean_cache
                else None
            )

            for ptype in args.perturbations:
                key = (item["id"], ptype)
                if key in existing:
                    p_records.append(existing[key])
                    continue
                print(f"  -> Perturb [{idx+1}/{len(p_premises)}] {item['id']} | {ptype}")
                pert_out = run_condition(
                        pipeline=pipeline,
                    condition=base_cond,
                    premise=item["premise"],
                    genre=item.get("genre", "Creative Fiction"),
                    total_target_words=item["target_words"],
                    num_chapters=args.num_chapters,
                    perturbation_type=ptype,
                    donor_plan=donor_blueprint if ptype == "unrelated_plan_swapping" else None,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    enable_cot=args.enable_cot,
                )
                rec = {
                    "sample_id": item["id"],
                    "premise": item["premise"],
                    "condition": base_cond,
                    "perturbation_type": ptype,
                    "clean_reference": t2_clean_cache[item["id"]],
                    "pipeline_output": pert_out,
                }
                append_jsonl_record(p_cache_path, rec)
                p_records.append(rec)
        generated["perturb"] = p_records

    return generated


# ---------------------------------------------------------------------------
# Phase 2: Scoring (VR-CLI Perplexity + DeBERTa NLI + Group A/B + Judge)
# ---------------------------------------------------------------------------
def _build_plan_summary(blueprint: Dict[str, Any]) -> str:
    """Build concise plan summary text passed to LLM-as-a-Judge."""
    parts = [
        f"Ch {c.get('chapter_index', i+1)}: {c.get('chapter_goal', '')}"
        for i, c in enumerate(blueprint.get("chapters", []))
    ]
    if blueprint.get("unstructured_plan"):
        parts.insert(0, blueprint["unstructured_plan"])
    return "\n".join(parts) or "No plan (Condition A direct)."


def _evaluate_track2_record(
    rec: Dict[str, Any],
    nli_scorer: NLIScorer,
    judge: Optional[LLMJudge5DimEvaluator] = None,
) -> Dict[str, Any]:
    out = rec["pipeline_output"]
    premise = rec["premise"]
    blueprint = out.get("blueprint", {})
    chapters = out.get("chapters", [])
    full_text = out.get("full_text", "")
    target_words = int(out.get("target_words", 10000))

    group_a = evaluate_macro_plan_group_a(premise, blueprint, nli_scorer=nli_scorer)
    len_m = evaluate_length_and_termination(full_text, target_words)
    adh_m = evaluate_plan_adherence_doc_nli(chapters, blueprint, nli_scorer=nli_scorer, premise=premise)
    lex_m = evaluate_lexical_and_burstiness(full_text)
    judge_m = (
        judge.evaluate(premise=premise, plan_summary=_build_plan_summary(blueprint), story_text=full_text)
        if judge is not None
        else LLMJudge5DimEvaluator._empty_scores()
    )

    row: Dict[str, Any] = {
        "sample_id": rec["sample_id"],
        "condition": rec["condition"],
        "perturbation_type": rec.get("perturbation_type", "clean"),
        "generation_latency_s": out.get("generation_latency_s", 0.0),
    }
    row.update(group_a)
    row.update(len_m)
    row.update(adh_m)
    row.update(lex_m)
    row.update(judge_m)
    return row


def _score_record_with_judge(rec: Dict[str, Any], row: Dict[str, Any], judge: LLMJudge5DimEvaluator) -> None:
    """Populate 5-dimension Judge rubric fields onto an already evaluated Track 2 / Perturb row."""
    out = rec["pipeline_output"]
    blueprint = out.get("blueprint", {})
    row.update(
        judge.evaluate(
            premise=rec["premise"],
            plan_summary=_build_plan_summary(blueprint),
            story_text=out.get("full_text", ""),
        )
    )


def run_phase2_evaluation(
    args: argparse.Namespace,
    generated: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    use_mock_eval = args.backend == "mock"
    nli_scorer = NLIScorer(model_id=args.nli_model, use_mock=use_mock_eval)
    judge_backend_name, judge_model_name = resolve_judge_runtime(args)

    summary_results: Dict[str, Any] = {
        "config": {
            "backend": args.backend,
            "model": args.model,
            "nli_model": args.nli_model,
            "judge_enabled": judge_backend_name is not None,
            "judge_backend": judge_backend_name,
            "judge_model": judge_model_name,
            "target_words": args.target_words,
            "num_chapters": args.num_chapters,
        },
        "track1_summary": {},
        "track2_summary": {},
        "perturb_summary": {},
        "hypotheses_audit": {},
    }

    # --- Phase 2a: Evaluate Track 1 (VR-CLI Perplexity + Group A) ---
    if generated.get("track1"):
        print(f"[Track 1 Phase 2] Evaluating {len(generated['track1'])} plans via VR-CLI + NLI...")
        vrcli = VRCLIPerplexityEvaluator(model_id=args.model, use_mock=use_mock_eval)
        t1_eval_rows: List[Dict[str, Any]] = []
        try:
            for item in generated["track1"]:
                plan_res = item["plan_res"]
                cond = item["condition"]
                vr_m = vrcli.evaluate_information_gain(
                    history_text=item["history_text"],
                    plan_text=plan_res["plan_text"],
                    gold_chapter=item["gold_chapter"],
                )
                temp_bp = {
                    "condition": cond,
                    "event_graph": {"events": plan_res.get("events", [])},
                    "unstructured_plan": plan_res["plan_text"] if cond == "B" else "",
                    "chapters": [{"chapter_goal": plan_res["plan_text"]}] if cond == "C" else [],
                    "planning_metadata": {
                        "planning_tokens": plan_res.get("planning_tokens", 0),
                        "planning_latency_s": plan_res.get("planning_latency_s", 0.0),
                    },
                }
                ga_m = evaluate_macro_plan_group_a(
                    premise=item.get("premise") or item["history_text"][:500],
                    blueprint=temp_bp,
                    nli_scorer=nli_scorer,
                )
                row = {
                    "sample_id": item["sample_id"],
                    "condition": cond,
                    **vr_m,
                    **ga_m,
                }
                t1_eval_rows.append(row)
        finally:
            vrcli.unload()

        write_jsonl_records(output_dir / "track1_raw.jsonl", t1_eval_rows)
        t1_summary_rows: List[Dict[str, Any]] = []
        for cond in sorted({r["condition"] for r in t1_eval_rows}):
            subset = [r for r in t1_eval_rows if r["condition"] == cond]
            agg = {
                "condition": cond,
                "n_samples": len(subset),
                "ppl_base": mean_numeric(subset, "ppl_base"),
                "ppl_planned": mean_numeric(subset, "ppl_planned"),
                "I_pi": mean_numeric(subset, "I_pi"),
                "S_cov": mean_numeric(subset, "S_cov"),
                "S_prog": mean_numeric(subset, "S_prog"),
                "S_causal": mean_numeric(subset, "S_causal"),
                "S_arc": mean_numeric(subset, "S_arc"),
                "planning_tokens": mean_numeric(subset, "planning_tokens"),
                "planning_latency_s": mean_numeric(subset, "planning_latency_s"),
            }
            t1_summary_rows.append(agg)
            summary_results["track1_summary"][cond] = agg

        export_csv(
            output_dir / "track1_vrcli_results.csv",
            t1_summary_rows,
            ["condition", "n_samples", "ppl_base", "ppl_planned", "I_pi", "S_cov", "S_prog", "S_causal", "S_arc", "planning_tokens", "planning_latency_s"],
        )

    # --- Phase 2b: Deterministic + NLI Evaluation for Track 2 & Perturbations ---
    t2_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    t2_clean_by_sample: Dict[str, Dict[str, Any]] = {}
    t2_clean_extra_recs: Dict[str, Dict[str, Any]] = {}
    p_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    try:
        if generated.get("track2"):
            print(f"[Track 2 Phase 2] Evaluating {len(generated['track2'])} stories (Group A & NLI Group B)...")
            for rec in generated["track2"]:
                ev = _evaluate_track2_record(rec, nli_scorer, judge=None)
                t2_pairs.append((rec, ev))
                if rec["condition"] == args.perturb_base_condition.upper():
                    t2_clean_by_sample[rec["sample_id"]] = ev

        if generated.get("perturb"):
            print(f"[Perturb Phase 2] Scoring {len(generated['perturb'])} perturbed runs with NLI...")
            for rec in generated["perturb"]:
                sid = rec["sample_id"]
                if sid not in t2_clean_by_sample:
                    clean_rec = rec["clean_reference"]
                    t2_clean_by_sample[sid] = _evaluate_track2_record(clean_rec, nli_scorer, judge=None)
                    t2_clean_extra_recs[sid] = clean_rec
                pert_ev = _evaluate_track2_record(rec, nli_scorer, judge=None)
                clean_bp = rec["clean_reference"]["pipeline_output"].get("blueprint", {})
                pert_chapters = rec["pipeline_output"].get("chapters", [])
                pert_ev["S_adh_on_perturbed"] = pert_ev.get("S_adh")
                # Measure adherence of perturbed story against original CLEAN goals to detect plot divergence
                pert_ev["S_adh"] = evaluate_plan_adherence_doc_nli(
                    pert_chapters, clean_bp, nli_scorer=nli_scorer, premise=rec["premise"]
                )["S_adh"]
                p_pairs.append((rec, pert_ev))
    finally:
        # Unload NLI before loading a large local Judge LLM (14B / 32B-AWQ)
        nli_scorer.unload()

    # --- Phase 2c: Optional LLM-as-a-Judge Scoring (API or Local Qwen-14B / 32B-AWQ) ---
    if judge_backend_name is not None and (t2_pairs or p_pairs):
        print(f"[Judge Phase 2] Scoring {len(t2_pairs) + len(p_pairs)} stories via {judge_backend_name} ({judge_model_name})...")
        judge_backend = LocalJudgeBackend(
            judge_backend=judge_backend_name,
            judge_model=judge_model_name or "Qwen/Qwen2.5-32B-Instruct-AWQ",
            api_key=args.judge_api_key,
            api_base=args.judge_api_base,
            dtype=args.dtype,
            gpu_memory_utilization=args.judge_gpu_memory_utilization,
            max_model_len=args.judge_max_model_len,
        )
        try:
            judge = LLMJudge5DimEvaluator(judge_fn=judge_backend.judge_fn)
            for idx, (rec, ev) in enumerate(t2_pairs, 1):
                print(f"  -> Judging Track 2 [{idx}/{len(t2_pairs)}] {rec['sample_id']} | Cond {rec['condition']}")
                _score_record_with_judge(rec, ev, judge)
            for sid, clean_rec in t2_clean_extra_recs.items():
                _score_record_with_judge(clean_rec, t2_clean_by_sample[sid], judge)
            for idx, (rec, pert_ev) in enumerate(p_pairs, 1):
                print(f"  -> Judging Perturb [{idx}/{len(p_pairs)}] {rec['sample_id']} | {rec.get('perturbation_type')}")
                _score_record_with_judge(rec, pert_ev, judge)
        finally:
            judge_backend.release()

    # --- Phase 2d: Export Track 2, Perturb & Hypothesis Audit ---
    if t2_pairs:
        t2_eval_rows = [ev for _, ev in t2_pairs]
        write_jsonl_records(output_dir / "track2_raw.jsonl", t2_eval_rows)
        t2_summary_rows: List[Dict[str, Any]] = []
        for cond in [c for c in ["A", "B", "C", "D", "E"] if any(r["condition"] == c for r in t2_eval_rows)]:
            subset = [r for r in t2_eval_rows if r["condition"] == cond]
            early_rate = round(100.0 * sum(1 for r in subset if r.get("early_terminated")) / len(subset), 2)
            agg = {
                "condition": cond,
                "n_samples": len(subset),
                "S_cov": mean_numeric(subset, "S_cov"),
                "S_prog": mean_numeric(subset, "S_prog"),
                "S_causal": mean_numeric(subset, "S_causal"),
                "S_comp": mean_numeric(subset, "S_comp"),
                "S_arc": mean_numeric(subset, "S_arc"),
                "planning_tokens": mean_numeric(subset, "planning_tokens"),
                "actual_words": mean_numeric(subset, "actual_words"),
                "target_words": mean_numeric(subset, "target_words"),
                "length_score_sl": mean_numeric(subset, "length_score_sl"),
                "early_term_rate_pct": early_rate,
                "S_adh": mean_numeric(subset, "S_adh"),
                "Distinct-3 (%)": mean_numeric(subset, "Distinct-3 (%)"),
                "Distinct-4 (%)": mean_numeric(subset, "Distinct-4 (%)"),
                "shannon_entropy": mean_numeric(subset, "shannon_entropy"),
                "burstiness_std": mean_numeric(subset, "burstiness_std"),
                "generation_latency_s": mean_numeric(subset, "generation_latency_s"),
                "S_coh": mean_numeric(subset, "S_coh"),
                "judge_plan_adherence": mean_numeric(subset, "judge_plan_adherence"),
                "judge_completeness": mean_numeric(subset, "judge_completeness"),
                "judge_writing_quality": mean_numeric(subset, "judge_writing_quality"),
                "judge_global_consistency": mean_numeric(subset, "judge_global_consistency"),
                "C_weighted": mean_numeric(subset, "C_weighted"),
            }
            t2_summary_rows.append(agg)
            summary_results["track2_summary"][cond] = agg

        export_csv(
            output_dir / "track2_ablation_5conditions.csv",
            t2_summary_rows,
            [
                "condition", "n_samples", "S_cov", "S_prog", "S_causal", "S_comp", "S_arc", "planning_tokens",
                "actual_words", "target_words", "length_score_sl", "early_term_rate_pct", "S_adh",
                "Distinct-3 (%)", "Distinct-4 (%)", "shannon_entropy", "burstiness_std",
                "generation_latency_s", "S_coh", "judge_plan_adherence", "judge_completeness",
                "judge_writing_quality", "judge_global_consistency", "C_weighted",
            ],
        )
        summary_results["hypotheses_audit"] = audit_hypotheses_h1_to_h4(summary_results["track2_summary"])

    if p_pairs:
        p_eval_rows: List[Dict[str, Any]] = []
        for rec, pert_ev in p_pairs:
            sid = rec["sample_id"]
            clean_ev = t2_clean_by_sample[sid]
            delta = compute_perturbation_delta(clean_ev, pert_ev)
            delta["S_adh_on_perturbed"] = pert_ev.get("S_adh_on_perturbed")
            p_eval_rows.append({"sample_id": sid, "condition": rec["condition"], **delta})

        write_jsonl_records(output_dir / "perturb_raw.jsonl", p_eval_rows)
        p_summary_rows: List[Dict[str, Any]] = []
        for ptype in args.perturbations:
            subset = [r for r in p_eval_rows if r.get("perturbation_type") == ptype]
            if not subset:
                continue
            diags = [r.get("diagnosis", "") for r in subset]
            dominant_diag = max(set(diags), key=diags.count) if diags else ""
            agg = {
                "perturbation_type": ptype,
                "n_samples": len(subset),
                "S_adh_on_perturbed": mean_numeric(subset, "S_adh_on_perturbed"),
                "delta_S_adh": mean_numeric(subset, "delta_S_adh"),
                "delta_S_coh": mean_numeric(subset, "delta_S_coh"),
                "delta_S_arc": mean_numeric(subset, "delta_S_arc"),
                "delta_S_causal": mean_numeric(subset, "delta_S_causal"),
                "delta_C_weighted": mean_numeric(subset, "delta_C_weighted"),
                "dominant_diagnosis": dominant_diag,
            }
            p_summary_rows.append(agg)
            summary_results["perturb_summary"][ptype] = agg

        export_csv(
            output_dir / "perturbation_delta_results.csv",
            p_summary_rows,
            [
                "perturbation_type", "n_samples", "S_adh_on_perturbed", "delta_S_adh",
                "delta_S_coh", "delta_S_arc", "delta_S_causal", "delta_C_weighted", "dominant_diagnosis",
            ],
        )

    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown_report(output_dir / "macro_planning_benchmark_report.md", summary_results)
    return summary_results


def write_markdown_report(path: Path, summary: Dict[str, Any]) -> None:
    cfg = summary["config"]
    lines: List[str] = [
        "# Stage-1 Macro-Planning & Writing Utility Benchmark Report",
        "",
        f"- **Backend**: `{cfg['backend']}` | **Model**: `{cfg['model']}`",
        f"- **NLI Scorer**: `{cfg['nli_model']}` | **Judge**: `{cfg['judge_enabled']}`"
        + (f" (`{cfg.get('judge_backend')}` / `{cfg.get('judge_model')}`)" if cfg.get("judge_enabled") else ""),
        f"- **Track 2 Config**: Target `{cfg['target_words']}` words across `{cfg['num_chapters']}` chapters",
        "",
    ]

    if summary.get("track1_summary"):
        lines.extend([
            "## 1. Track 1 — VR-CLI Next-Chapter Prediction",
            "",
            "| Cond | N | PPL_base | PPL_planned | I_pi (%) | S_cov | S_prog | S_causal | S_arc | Plan Tok |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for cond, r in summary["track1_summary"].items():
            lines.append(
                f"| {cond} | {r['n_samples']} | {r['ppl_base']} | {r['ppl_planned']} | {r['I_pi']} | "
                f"{r['S_cov']} | {r['S_prog']} | {r['S_causal']} | {r['S_arc']} | {r['planning_tokens']} |"
            )
        lines.append("")

    if summary.get("track2_summary"):
        lines.extend([
            "## 2. Track 2 — 5-Condition Structural Ablation",
            "",
            "| Cond | N | Words (L) | Sl (%) | EarlyStop (%) | S_adh | S_causal | S_arc | Distinct-3 (%) | Plan Tok | S_coh | C_weighted |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for cond, r in summary["track2_summary"].items():
            lines.append(
                f"| {cond} | {r['n_samples']} | {r['actual_words']} | {r['length_score_sl']} | {r['early_term_rate_pct']} | "
                f"{r['S_adh']} | {r['S_causal']} | {r['S_arc']} | {r['Distinct-3 (%)']} | {r['planning_tokens']} | "
                f"{r['S_coh']} | {r['C_weighted']} |"
            )
        lines.append("")

    if summary.get("perturb_summary"):
        lines.extend([
            "## 3. Plan-Sensitivity Perturbation Protocol (Delta = Clean - Perturbed)",
            "",
            "| Perturbation | N | S_adh (Assigned) | Delta S_adh (vs Clean) | Delta S_coh | Delta S_arc | Delta S_causal | Delta C_weighted | Diagnosis |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for ptype, r in summary["perturb_summary"].items():
            lines.append(
                f"| `{ptype}` | {r['n_samples']} | {r.get('S_adh_on_perturbed')} | {r['delta_S_adh']} | {r['delta_S_coh']} | "
                f"{r['delta_S_arc']} | {r['delta_S_causal']} | {r['delta_C_weighted']} | {r['dominant_diagnosis']} |"
            )
        lines.append("")

    if summary.get("hypotheses_audit"):
        lines.extend(["## 4. Hypotheses H1-H4 Audit", "", "```json", json.dumps(summary["hypotheses_audit"], indent=2), "```", ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MA-GRPO Macro Planning & Writing Utility Benchmark")
    parser.add_argument("--mode", choices=["track1", "track2", "perturb", "all"], default="all")
    parser.add_argument("--conditions", nargs="+", default=["A", "B", "C", "D", "E"])
    parser.add_argument("--perturbations", nargs="+", default=list(PERTURBATION_TYPES))
    parser.add_argument("--perturb-base-condition", default="D")
    parser.add_argument("--backend", choices=["mock", "hf", "vllm"], default="mock")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--nli-model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    parser.add_argument("--track1-data", default=None, help="Path to LONGSTORY/BookSum/PG-19 JSONL")
    parser.add_argument("--track2-data", default=None, help="Path to TELL ME A STORY JSONL")
    parser.add_argument("--allow-synthetic-demo", action="store_true")
    parser.add_argument("--track1-samples", type=int, default=50)
    parser.add_argument("--track2-samples", type=int, default=30)
    parser.add_argument("--perturb-samples", type=int, default=5)
    parser.add_argument("--target-words", type=int, default=10000)
    parser.add_argument("--num-chapters", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-cot", action="store_true")
    parser.add_argument("--enable-judge", action="store_true")
    parser.add_argument(
        "--judge-backend",
        choices=["auto", "api", "local_vllm", "local_hf", "mock"],
        default="auto",
    )
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--judge-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--judge-max-model-len", type=int, default=16384)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "benchmark_outputs"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.enable_cot = not args.no_cot
    return args


def main(argv: Optional[List[str]] = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(line_buffering=True)
        except Exception:
            pass
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    gen_backend = LocalGenerationBackend(
        backend=args.backend,
        model_name=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    try:
        generated = run_phase1_generation(args, gen_backend, output_dir)
    finally:
        gen_backend.release()

    summary = run_phase2_evaluation(args, generated, output_dir)
    elapsed = round(time.time() - t0, 2)
    print(f"\n[Done] Benchmark finished in {elapsed}s. Report saved to: {output_dir / 'macro_planning_benchmark_report.md'}")
    if summary.get("hypotheses_audit"):
        print("H1-H4 Audit:", json.dumps(summary["hypotheses_audit"], indent=2))


if __name__ == "__main__":
    main()
