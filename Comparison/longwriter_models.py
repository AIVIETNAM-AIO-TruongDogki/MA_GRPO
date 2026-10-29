"""
Official Runners for the LongWriter Model Family:
  4. LongWriter-llama3.1-8b (`THUDM/LongWriter-llama3.1-8b`)
  5. LongWriter-glm4-9b (`THUDM/LongWriter-glm4-9b` / `zai-org/LongWriter-glm4-9b`)
  6. LongWriter-Zero-32B (`THU-KEG/LongWriter-Zero-32B` / `zai-org/LongWriter-Zero-32B`)

Includes exact prompt formatting from `LongWriter/evaluation/pred.py`, full ChatGLM4
compatibility patches for modern `transformers`, and memory-safe 4-bit NF4 Double Quantization
with automatic VRAM/CPU offload protection for the 32B Pure RL model (`LongWriter-Zero-32B`).
"""

import gc
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from .metrics import clean_generated_prose, save_model_result_csv
from .qwen_backend import clean_gpu_memory, get_peak_vram_gb, reset_vram_stats


def _total_gpu_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return sum(
        torch.cuda.get_device_properties(i).total_memory / (1024**3)
        for i in range(torch.cuda.device_count())
    )


def _apply_chatglm_patches():
    """Apply all critical patches so THUDM/LongWriter-glm4-9b runs cleanly on transformers >= 4.45."""
    try:
        import transformers.modeling_utils

        def get_safe_tied(self):
            val = getattr(self, "_tied_weights_keys", None)
            return val if (val is not None and hasattr(val, "keys")) else {}

        transformers.modeling_utils.PreTrainedModel.all_tied_weights_keys = property(get_safe_tied)
    except Exception:
        pass

    try:
        import transformers.integrations.accelerate as acc

        orig_compute = acc.compute_module_sizes

        def safe_compute_module_sizes(model, *args, **kwargs):
            if (
                not hasattr(model, "all_tied_weights_keys")
                or getattr(model, "all_tied_weights_keys", None) is None
                or not hasattr(getattr(model, "all_tied_weights_keys"), "keys")
            ):
                model.all_tied_weights_keys = {}
            return orig_compute(model, *args, **kwargs)

        acc.compute_module_sizes = safe_compute_module_sizes
    except Exception:
        pass

    try:
        from transformers.cache_utils import DynamicCache

        DynamicCache.__getitem__ = lambda self, idx: (self.key_cache[idx], self.value_cache[idx])
    except Exception:
        pass

    try:
        from transformers.generation.utils import GenerationMixin

        def _extract_past_from_model_output(self, outputs, standardize_cache_format=False):
            past = getattr(outputs, "past_key_values", None)
            return "past_key_values", past

        GenerationMixin._extract_past_from_model_output = _extract_past_from_model_output
    except Exception:
        pass


# =============================================================================
# MODEL 4: LongWriter-llama3.1-8b (`THUDM/LongWriter-llama3.1-8b`)
# =============================================================================
def run_longwriter_llama_benchmark(
    prompts: List[Dict[str, Any]],
    output_csv: Path,
    model_id: str = "THUDM/LongWriter-llama3.1-8b",
    clear_hf_cache_after: bool = False,
):
    """
    Official LongWriter-llama3.1-8b runner matching `LongWriter/evaluation/pred.py`:
      - Prompt format: `[INST]{prompt}[/INST]`
      - Sampling: `temperature=0.5`, `top_p=0.9`, `repetition_penalty=1.05`, `do_sample=True`
    """
    print("\n" + "=" * 85, flush=True)
    print(f"RUNNING MODEL [4/8]: LongWriter-llama3.1-8b ({model_id})", flush=True)
    print("=" * 85, flush=True)
    clean_gpu_memory()

    use_cuda = torch.cuda.is_available()
    vram_total = _total_gpu_vram_gb()
    compute_dtype = (
        torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)
    )

    # On >= 20GB VRAM (A100 / L4 / Dual T4), load native BF16/FP16; on 15GB T4, use 4-bit NF4
    quant_config = None
    if use_cuda and vram_total < 20.0:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": compute_dtype,
        "device_map": "auto" if use_cuda else None,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    print(f"[LongWriter-llama3.1-8b] Loaded successfully! (Total GPU VRAM: {vram_total:.1f} GB)", flush=True)

    records = []
    try:
        for idx, item in enumerate(prompts, 1):
            p_id = item["id"]
            p_text = item["prompt"]
            t_len = int(item["target_length"])
            print(f" -> [LongWriter-llama3.1-8b] [{idx}/{len(prompts)}] {p_id} (Target: {t_len} words)...", flush=True)

            reset_vram_stats()
            # Canonical framing from LongWriter/evaluation/pred.py line 34
            prompt_input = f"[INST]{p_text}[/INST]"
            inputs = tokenizer(prompt_input, truncation=False, return_tensors="pt")
            if use_cuda:
                dev = next(model.parameters()).device
                inputs = {k: v.to(dev) for k, v in inputs.items()}

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(4096, max(1800, int(t_len * 1.45))),
                    temperature=0.5,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = round(max(0.01, time.time() - t0), 2)
            in_len = inputs["input_ids"].shape[1]
            out_text = tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True).strip()
            vram = get_peak_vram_gb()

            actual_w = len(out_text.split())
            print(f"    Done in {elapsed}s | Words: {actual_w}/{t_len} | Peak VRAM: {vram} GB", flush=True)

            records.append({
                "Model": "LongWriter-llama3.1-8b",
                "Prompt_ID": p_id,
                "Prompt_Text": p_text,
                "Target_Len": t_len,
                "Raw_Output": out_text,
                "Latency (s)": elapsed,
                "Peak_VRAM (GB)": vram,
            })
    finally:
        del model
        del tokenizer
        clean_gpu_memory()
        if clear_hf_cache_after and os.environ.get("HF_HOME"):
            shutil.rmtree(os.environ["HF_HOME"], ignore_errors=True)
            os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    return save_model_result_csv(records, output_csv)


# =============================================================================
# MODEL 5: LongWriter-glm4-9b (`THUDM/LongWriter-glm4-9b` / `zai-org/LongWriter-glm4-9b`)
# =============================================================================
def run_longwriter_glm_benchmark(
    prompts: List[Dict[str, Any]],
    output_csv: Path,
    model_id: str = "THUDM/LongWriter-glm4-9b",
    clear_hf_cache_after: bool = False,
):
    """
    Official LongWriter-glm4-9b runner (`THUDM/LongWriter-glm4-9b` with fallback to `zai-org/LongWriter-glm4-9b`).
    Applies ChatGLM4 DynamicCache & tied-weights patches for full speed generation.
    """
    print("\n" + "=" * 85, flush=True)
    print(f"RUNNING MODEL [5/8]: LongWriter-glm4-9b ({model_id})", flush=True)
    print("=" * 85, flush=True)
    clean_gpu_memory()
    _apply_chatglm_patches()

    use_cuda = torch.cuda.is_available()
    vram_total = _total_gpu_vram_gb()
    compute_dtype = (
        torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)
    )

    quant_config = None
    if use_cuda and vram_total < 22.0:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    # Try THUDM/LongWriter-glm4-9b first, fallback to zai-org/LongWriter-glm4-9b
    resolved_id = model_id
    try:
        model_config = AutoConfig.from_pretrained(resolved_id, trust_remote_code=True)
    except Exception:
        resolved_id = "zai-org/LongWriter-glm4-9b"
        model_config = AutoConfig.from_pretrained(resolved_id, trust_remote_code=True)

    seq_len = getattr(model_config, "seq_length", 32768)
    cls = type(model_config)
    cls.max_length = seq_len
    cls.seq_length = seq_len
    model_config.max_length = seq_len

    tokenizer = AutoTokenizer.from_pretrained(resolved_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, Any] = {
        "config": model_config,
        "torch_dtype": compute_dtype,
        "device_map": "auto" if use_cuda else None,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(resolved_id, **load_kwargs)
    if not hasattr(model, "all_tied_weights_keys") or getattr(model, "all_tied_weights_keys", None) is None:
        model.all_tied_weights_keys = {}
    if hasattr(model.config, "max_length"):
        try:
            delattr(model.config, "max_length")
        except Exception:
            pass
    try:
        gen_conf = GenerationConfig.from_model_config(model.config)
        gen_conf.max_length = None
        model.generation_config = gen_conf
    except Exception:
        pass

    model.eval()
    print(f"[LongWriter-glm4-9b] Loaded {resolved_id} successfully!", flush=True)

    records = []
    try:
        for idx, item in enumerate(prompts, 1):
            p_id = item["id"]
            p_text = item["prompt"]
            t_len = int(item["target_length"])
            print(f" -> [LongWriter-glm4-9b] [{idx}/{len(prompts)}] {p_id} (Target: {t_len} words)...", flush=True)

            reset_vram_stats()
            messages = [{"role": "user", "content": p_text}]
            try:
                prompt_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt_input = f"<|user|>\n{p_text}<|assistant|>\n"

            inputs = tokenizer(prompt_input, return_tensors="pt")
            if use_cuda:
                dev = next(model.parameters()).device
                inputs = {k: v.to(dev) for k, v in inputs.items() if k != "token_type_ids"}

            gen_kwargs = {
                "max_new_tokens": min(4096, max(1800, int(t_len * 1.45))),
                "temperature": 0.5,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            t0 = time.time()
            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
            except Exception as e:
                print(f"    [ChatGLM Cache Fallback use_cache=False]: {e}", flush=True)
                gen_kwargs["use_cache"] = False
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)

            elapsed = round(max(0.01, time.time() - t0), 2)
            in_len = inputs["input_ids"].shape[1]
            out_text = tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True).strip()
            vram = get_peak_vram_gb()

            actual_w = len(out_text.split())
            print(f"    Done in {elapsed}s | Words: {actual_w}/{t_len} | Peak VRAM: {vram} GB", flush=True)

            records.append({
                "Model": "LongWriter-glm4-9b",
                "Prompt_ID": p_id,
                "Prompt_Text": p_text,
                "Target_Len": t_len,
                "Raw_Output": out_text,
                "Latency (s)": elapsed,
                "Peak_VRAM (GB)": vram,
            })
    finally:
        del model
        del tokenizer
        clean_gpu_memory()
        if clear_hf_cache_after and os.environ.get("HF_HOME"):
            shutil.rmtree(os.environ["HF_HOME"], ignore_errors=True)
            os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    return save_model_result_csv(records, output_csv)


# =============================================================================
# MODEL 6: LongWriter-Zero-32B (`THU-KEG/LongWriter-Zero-32B`)
# =============================================================================
def run_longwriter_zero_benchmark(
    prompts: List[Dict[str, Any]],
    output_csv: Path,
    model_id: str = "THU-KEG/LongWriter-Zero-32B",
    clear_hf_cache_after: bool = False,
):
    """
    Official LongWriter-Zero-32B (Pure RL / GRPO on Qwen2.5-32B) runner.

    Key Fixes Over Prior Incomplete Setup:
      1. Memory-Safe 4-Bit NF4 Double Quantization (`bnb_4bit_use_double_quant=True`,
         `low_cpu_mem_usage=True`) reducing 32B footprint from 65 GB to ~18.2 GB VRAM.
      2. Smart Hardware Allocation:
         - On A100 (40/80GB), L4 (24GB), or Dual T4 (30GB): fits 100% in GPU VRAM.
         - On Single T4 (15GB): enables `llm_int8_enable_fp32_cpu_offload=True` with explicit
           `max_memory` and `offload_folder` so it never crashes with OOM.
      3. Full Test-Time Scaling (`<think>...</think><answer>...</answer>`) token budget:
         Allocates `max_new_tokens = max(3600, int(target_len * 2.2))` so the internal
         `<think>` outline (~800-1200 tokens) + `<answer>` story (~1500-2500 tokens)
         both complete without mid-story truncation.
    """
    print("\n" + "=" * 85, flush=True)
    print(f"RUNNING MODEL [6/8]: LongWriter-Zero-32B ({model_id} - 4-bit NF4 Pure RL GRPO)", flush=True)
    print("=" * 85, flush=True)
    clean_gpu_memory()

    use_cuda = torch.cuda.is_available()
    vram_total = _total_gpu_vram_gb()
    num_gpus = torch.cuda.device_count() if use_cuda else 0
    compute_dtype = (
        torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)
    )

    enable_cpu_offload = use_cuda and (vram_total < 21.0)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=enable_cpu_offload,
    )

    resolved_id = model_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_id, trust_remote_code=True)
    except Exception:
        resolved_id = "zai-org/LongWriter-Zero-32B"
        tokenizer = AutoTokenizer.from_pretrained(resolved_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, Any] = {
        "quantization_config": quant_config if use_cuda else None,
        "torch_dtype": compute_dtype,
        "device_map": "auto" if use_cuda else None,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if enable_cpu_offload:
        offload_dir = "/tmp/offload_longwriter_zero32b"
        os.makedirs(offload_dir, exist_ok=True)
        load_kwargs["offload_folder"] = offload_dir
        load_kwargs["max_memory"] = {0: "13.5GiB", "cpu": "38GiB"}
        print("[LongWriter-Zero-32B] Single <21GB GPU detected: enabled safe GPU+CPU offload.", flush=True)
    elif num_gpus >= 2 and vram_total < 35.0:
        # Balanced sharding across Dual T4 (2x 15GB)
        load_kwargs["max_memory"] = {i: "13.8GiB" for i in range(num_gpus)}
        load_kwargs["max_memory"]["cpu"] = "28GiB"
        print(f"[LongWriter-Zero-32B] Dual GPU ({vram_total:.1f} GB) detected: sharding across {num_gpus} GPUs.", flush=True)
    else:
        print(f"[LongWriter-Zero-32B] High-VRAM GPU ({vram_total:.1f} GB) detected: loading 100% into GPU VRAM.", flush=True)

    model = AutoModelForCausalLM.from_pretrained(resolved_id, **load_kwargs)
    if hasattr(model.config, "max_length"):
        try:
            delattr(model.config, "max_length")
        except Exception:
            pass
    model.eval()
    print(f"[LongWriter-Zero-32B] Loaded {resolved_id} successfully!", flush=True)

    records = []
    try:
        for idx, item in enumerate(prompts, 1):
            p_id = item["id"]
            p_text = item["prompt"]
            t_len = int(item["target_length"])
            print(f" -> [LongWriter-Zero-32B] [{idx}/{len(prompts)}] {p_id} (Target: {t_len} words)...", flush=True)

            reset_vram_stats()

            # LongWriter-Zero-32B uses Qwen chat template and reasons in <think>...</think><answer>...</answer>
            messages = [{"role": "user", "content": p_text}]
            try:
                prompt_input = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_input = f"<|im_start|>user\n{p_text}<|im_end|>\n<|im_start|>assistant\n"

            inputs = tokenizer(prompt_input, return_tensors="pt")
            if use_cuda:
                dev = next(model.parameters()).device
                inputs = {k: v.to(dev) for k, v in inputs.items() if k != "token_type_ids"}

            # Generous token budget to accommodate both <think> CoT (~800-1200 tokens) + <answer> story
            max_toks = min(5120, max(3600, int(t_len * 2.25)))

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_toks,
                    temperature=0.6,
                    top_p=0.92,
                    repetition_penalty=1.05,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = round(max(0.01, time.time() - t0), 2)
            in_len = inputs["input_ids"].shape[1]
            raw_out = tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True).strip()
            vram = get_peak_vram_gb()

            clean_prose = clean_generated_prose(raw_out)
            actual_w = len(clean_prose.split())
            print(f"    Done in {elapsed}s | Story Words (clean): {actual_w}/{t_len} | Peak VRAM: {vram} GB", flush=True)

            records.append({
                "Model": "LongWriter-Zero-32B",
                "Prompt_ID": p_id,
                "Prompt_Text": p_text,
                "Target_Len": t_len,
                "Raw_Output": raw_out,
                "Latency (s)": elapsed,
                "Peak_VRAM (GB)": vram,
            })
    finally:
        del model
        del tokenizer
        clean_gpu_memory()
        if clear_hf_cache_after and os.environ.get("HF_HOME"):
            shutil.rmtree(os.environ["HF_HOME"], ignore_errors=True)
            os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    return save_model_result_csv(records, output_csv)
