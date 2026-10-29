"""
MA-GRPO Checkpoint Runners:
  7. MA-GRPO-50 (`checkpoint-50`)
  8. MA-GRPO-100 (`checkpoint-100`)

Loads Base Model `Qwen/Qwen2.5-7B-Instruct` in 4-bit NF4 and attaches the trained
QLoRA GRPO adapters (`checkpoint-50` and `checkpoint-100`), formatting each benchmark
prompt with the Two-Stage MA-GRPO Scene Budget & Test-Time Reasoning (`<think>...</think>`) structure.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .metrics import clean_generated_prose, save_model_result_csv
from .qwen_backend import clean_gpu_memory, get_peak_vram_gb, reset_vram_stats


def resolve_ma_grpo_checkpoint(step: int, custom_path: Optional[str] = None) -> Path:
    """Locate checkpoint-50 or checkpoint-100 across Local Windows, Colab Drive, or Kaggle."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    ckpt_name = f"checkpoint-{step}"
    candidates = [
        Path(f"/content/drive/MyDrive/MA-GRPO-Checkpoints/ma-grpo-final/{ckpt_name}"),
        Path(f"C:/Users/ASUS/AI_LLMs/Long_text/ma-grpo-final-20261029T095313Z-1-001/ma-grpo-final/{ckpt_name}"),
        Path(f"./ma-grpo-final/{ckpt_name}"),
        Path(f"/kaggle/input/ma-grpo-checkpoints/{ckpt_name}"),
        Path(f"/kaggle/input/ma-grpo-checkpoints/ma-grpo-final/{ckpt_name}"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not find {ckpt_name}. Checked paths: {[str(c) for c in candidates]}"
    )


def format_ma_grpo_prompt(premise: str, target_len: int) -> str:
    """
    Construct the canonical MA-GRPO Stage-2 Writing Agent prompt with balanced
    Scene Breakdown budgets (25% / 45% / 30%) and <think>...</think> CoT directive.
    """
    s1 = int(round(target_len * 0.25))
    s2 = int(round(target_len * 0.45))
    s3 = max(100, target_len - s1 - s2)

    return f"""You are an elite literary novelist. Write the complete narrative prose for this chapter.
Total Target Length: Approximately {target_len} words.

[CONTEXT SUMMARY]:
{premise}

[CHAPTER GOAL]:
Deliver the complete, self-contained narrative arc fulfilling every requirement of the premise.

[SCENE BREAKDOWN & WORD BUDGETS]:
- Scene 1: Opening exposition and world details (~{s1} words)
- Scene 2: Inciting tension and primary dialogue (~{s2} words)
- Scene 3: Climax and resolution of the story goal (~{s3} words)

CRITICAL WRITING INSTRUCTIONS:
1. Test-Time Reasoning (Chain-of-Thought):
   You MUST begin your response with a <think>...</think> block containing your step-by-step narrative plan:
   - Scene-by-scene pacing: Plan word distribution to strictly match each scene's target words (~{s1}, ~{s2}, ~{s3} words) totaling ~{target_len} words.
   - Narrative transition: Hook directly into the scene continuation.
   - Tone & Sensory palette: Outline sensory imagery and avoid literal phrasing from the context summary.
2. Chapter Prose:
   Immediately after </think>, output ONLY the final narrative prose.
   - Seamlessly flow between scenes according to the budgets.
   - DO NOT include scene labels, section titles, author notes, or meta-commentary outside <think>.
Begin with <think> now:"""


def run_ma_grpo_checkpoint_benchmark(
    prompts: List[Dict[str, Any]],
    step: int,
    output_csv: Path,
    checkpoint_dir: Optional[str] = None,
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
):
    """
    Run benchmark evaluation for MA-GRPO-50 (`step=50`) or MA-GRPO-100 (`step=100`).
    """
    model_label = f"MA-GRPO-{step}"
    ckpt_path = resolve_ma_grpo_checkpoint(step, checkpoint_dir)

    print("\n" + "=" * 85, flush=True)
    print(f"RUNNING MODEL [{'7/8' if step == 50 else '8/8'}]: {model_label} (Adapter: {ckpt_path})", flush=True)
    print("=" * 85, flush=True)
    clean_gpu_memory()

    use_cuda = torch.cuda.is_available()
    compute_dtype = (
        torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)
    )

    quant_config = None
    if use_cuda:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer_src = str(ckpt_path) if (ckpt_path / "tokenizer_config.json").exists() else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"[{model_label}] Attaching LoRA adapter from {ckpt_path}...", flush=True)
    model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    model.eval()

    records = []
    try:
        for idx, item in enumerate(prompts, 1):
            p_id = item["id"]
            p_text = item["prompt"]
            t_len = int(item["target_length"])
            print(f" -> [{model_label}] [{idx}/{len(prompts)}] {p_id} (Target: {t_len} words)...", flush=True)

            reset_vram_stats()
            full_prompt = format_ma_grpo_prompt(p_text, t_len)
            inputs = tokenizer(full_prompt, return_tensors="pt")
            if use_cuda:
                dev = next(model.parameters()).device
                inputs = {k: v.to(dev) for k, v in inputs.items() if k != "token_type_ids"}

            # Budget tokens for <think> (~250-400 tokens) + story prose (~1.38 tokens/word)
            max_new_toks = min(3200, max(1900, int(t_len * 1.48) + 350))

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_toks,
                    temperature=0.75,
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
                "Model": model_label,
                "Prompt_ID": p_id,
                "Prompt_Text": p_text,
                "Target_Len": t_len,
                "Raw_Output": raw_out,
                "Latency (s)": elapsed,
                "Peak_VRAM (GB)": vram,
            })
    finally:
        del model
        del base_model
        del tokenizer
        clean_gpu_memory()

    return save_model_result_csv(records, output_csv)

