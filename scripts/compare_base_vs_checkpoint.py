"""
Compare Base Pre-trained Qwen vs MA-GRPO Fine-tuned Checkpoint (e.g. checkpoint-10).
Direct side-by-side inference and benchmark comparison table:
  - Text excerpts comparison (Reasoning + Narrative prose)
  - Length compliance Sl (%)
  - Distinct-3 & Distinct-4 (%)
  - Perplexity (PPL)
  - Comparison with SOTA baselines (StoryWriter, Agents' Room, LongWriter-Zero)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def strip_think_tags(text: str) -> str:
    clean = re.sub(r"<think>[\s\S]*?</think>", "", text)
    clean = re.sub(r"<think>[\s\S]*", "", clean)
    clean = re.sub(r"</?answer>", "", clean)
    return clean.strip()


def count_prose_words(text: str) -> int:
    clean = strip_think_tags(text)
    return len(re.findall(r"\b\w+\b", clean))


def calculate_length_score(actual_len: int, target_len: int) -> float:
    if actual_len <= 0 or target_len <= 0:
        return 0.0
    if actual_len > target_len:
        score = 100.0 * max(0.0, 1.0 - (actual_len / target_len - 1.0) / 3.0)
    else:
        score = 100.0 * max(0.0, 1.0 - (target_len / actual_len - 1.0) / 2.0)
    return round(score, 2)


def calculate_distinct_n(text: str) -> Dict[str, float]:
    clean = strip_think_tags(text)
    words = re.findall(r"\b\w+\b", clean.lower())
    total = len(words)
    res = {}
    for n in [3, 4]:
        if total < n:
            res[f"Distinct-{n} (%)"] = 0.0
            continue
        ngrams = [tuple(words[i : i + n]) for i in range(total - n + 1)]
        ratio = len(set(ngrams)) / len(ngrams) if ngrams else 0.0
        res[f"Distinct-{n} (%)"] = round(ratio * 100.0, 2)
    return res


def compute_ppl_qwen(texts: List[str], device: str = "cuda") -> List[float]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "Qwen/Qwen2.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        evaluator = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device if torch.cuda.is_available() else None,
        )
        evaluator.eval()

        scores = []
        with torch.no_grad():
            for t in texts:
                clean = strip_think_tags(t)
                if not clean:
                    scores.append(99.0)
                    continue
                enc = tokenizer(clean, return_tensors="pt", truncation=True, max_length=2048)
                input_ids = enc.input_ids.to(evaluator.device)
                out = evaluator(input_ids, labels=input_ids)
                loss = out.loss.item()
                scores.append(round(float(np.exp(loss)), 2) if loss < 15 else 99.0)
        return scores
    except Exception as e:
        print(f"PPL evaluation skipped: {e}")
        return [0.0] * len(texts)


def run_comparison(
    checkpoint_dir: str,
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    dataset_path: str = "data/chapters.jsonl",
    num_samples: int = 1,
    output_csv: str = "benchmark_base_vs_checkpoint.csv",
):
    print("=" * 90)
    print("MA-GRPO INFERENCE & BENCHMARK: BASE PRE-TRAINED QWEN vs CHECKPOINT-10")
    print(f"Base Model:       {base_model_name}")
    print(f"LoRA Checkpoint:  {checkpoint_dir}")
    print(f"Dataset Path:     {dataset_path}")
    print("=" * 90)

    # 1. Load Dataset Prompts
    prompts_data = []
    real_path = Path(dataset_path)
    if not real_path.exists():
        real_path = PROJECT_ROOT / "data" / "chapters.jsonl"
    with open(real_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= num_samples:
                break
            if line.strip():
                prompts_data.append(json.loads(line))

    print(f"Loaded {len(prompts_data)} test prompt(s) for comparison.")

    # 2. Load Base Model in 4-bit (NF4)
    print("\n[1/3] Loading Base Model in 4-bit NF4...")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    gen_params = {
        "max_new_tokens": 2000,
        "temperature": 0.80,
        "top_p": 0.92,
        "repetition_penalty": 1.05,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # 3. Generate with Base Model (Zero-Shot)
    base_outputs = []
    print("\n[2/3] Generating from Base Model (Chưa qua train)...")
    for idx, sample in enumerate(prompts_data):
        prompt = sample.get("prompt") or sample.get("full_training_prompt", "")
        target_words = int(sample.get("target_words", 1200))
        sample_id = sample.get("sample_id", f"sample_{idx+1}")

        print(f"  -> Generating Sample {idx+1}/{len(prompts_data)} [{sample_id}] (Target: {target_words} words)...")
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
        t0 = time.time()
        with torch.no_grad():
            tokens = base_model.generate(**inputs, **gen_params)
        dur = time.time() - t0
        text = tokenizer.decode(tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        w_count = count_prose_words(text)
        print(f"     Done in {dur:.1f}s | Prose Words: {w_count}")
        base_outputs.append({
            "sample_id": sample_id,
            "target_words": target_words,
            "actual_words": w_count,
            "text": text,
        })

    # 4. Attach Checkpoint-10 LoRA Adapter
    ckpt_outputs = []
    has_lora = False
    if os.path.exists(checkpoint_dir) and (
        os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin"))
    ):
        print(f"\n[3/3] Attaching LoRA Checkpoint: {checkpoint_dir}...")
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        peft_model.eval()
        has_lora = True

        for idx, sample in enumerate(prompts_data):
            prompt = sample.get("prompt") or sample.get("full_training_prompt", "")
            target_words = int(sample.get("target_words", 1200))
            sample_id = sample.get("sample_id", f"sample_{idx+1}")

            print(f"  -> Generating Sample {idx+1}/{len(prompts_data)} [{sample_id}] with Checkpoint-10...")
            inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)
            t0 = time.time()
            with torch.no_grad():
                tokens = peft_model.generate(**inputs, **gen_params)
            dur = time.time() - t0
            text = tokenizer.decode(tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            w_count = count_prose_words(text)
            print(f"     Done in {dur:.1f}s | Prose Words: {w_count}")
            ckpt_outputs.append({
                "sample_id": sample_id,
                "target_words": target_words,
                "actual_words": w_count,
                "text": text,
            })
    else:
        print(f"\n[!] Warning: Checkpoint directory '{checkpoint_dir}' does not exist or has no adapter.")
        print("    Running comparison with Base model only.")
        ckpt_outputs = base_outputs

    # 5. Display Text Excerpts Side-by-Side
    print("\n" + "=" * 90)
    print("CHI TIẾT ĐOẠN VĂN SINH THỰC TẾ (SAMPLE INFERENCE OUTPUT)")
    print("=" * 90)
    for idx, (b_item, c_item) in enumerate(zip(base_outputs, ckpt_outputs)):
        print(f"\n--- [SAMPLE {idx+1}: {b_item['sample_id']} | TARGET: {b_item['target_words']} WORDS] ---")
        print("\n🔹 [A. QWEN BASE MODEL (CHƯA TRAIN)]:")
        b_clean = strip_think_tags(b_item["text"])
        print(f"Word count: {b_item['actual_words']} words")
        print(f"Exerpt (Head):\n{b_clean[:400]}...\n")
        print(f"Exerpt (Tail):\n...{b_clean[-400:]}")

        if has_lora:
            print("\n🔸 [B. QWEN MA-GRPO (CHECKPOINT-10)]:")
            c_clean = strip_think_tags(c_item["text"])
            print(f"Word count: {c_item['actual_words']} words")
            print(f"Exerpt (Head):\n{c_clean[:400]}...\n")
            print(f"Exerpt (Tail):\n...{c_clean[-400:]}")

    # 6. Compute Quantitative Metrics
    print("\n" + "=" * 90)
    print("TÍNH TOÁN CÁC CHỈ SỐ BENCHMARK (Sl, Distinct-3, Distinct-4, PPL)")
    print("=" * 90)

    base_texts = [r["text"] for r in base_outputs]
    ckpt_texts = [r["text"] for r in ckpt_outputs]

    base_ppls = compute_ppl_qwen(base_texts)
    ckpt_ppls = compute_ppl_qwen(ckpt_texts)

    base_sl_list = [calculate_length_score(r["actual_words"], r["target_words"]) for r in base_outputs]
    ckpt_sl_list = [calculate_length_score(r["actual_words"], r["target_words"]) for r in ckpt_outputs]

    base_d3 = [calculate_distinct_n(t)["Distinct-3 (%)"] for t in base_texts]
    base_d4 = [calculate_distinct_n(t)["Distinct-4 (%)"] for t in base_texts]

    ckpt_d3 = [calculate_distinct_n(t)["Distinct-3 (%)"] for t in ckpt_texts]
    ckpt_d4 = [calculate_distinct_n(t)["Distinct-4 (%)"] for t in ckpt_texts]

    # 7. Comparison Table
    b_mean_sl = round(float(np.mean(base_sl_list)), 2)
    b_mean_d3 = round(float(np.mean(base_d3)), 2)
    b_mean_d4 = round(float(np.mean(base_d4)), 2)
    b_mean_ppl = round(float(np.mean(base_ppls)), 2)

    c_mean_sl = round(float(np.mean(ckpt_sl_list)), 2)
    c_mean_d3 = round(float(np.mean(ckpt_d3)), 2)
    c_mean_d4 = round(float(np.mean(ckpt_d4)), 2)
    c_mean_ppl = round(float(np.mean(ckpt_ppls)), 2)

    comp_rows = [
        {"Model / Method": "AGENTS' ROOM", "Length Sl (%)": 92.06, "Distinct-3 (%)": 95.48, "Distinct-4 (%)": 98.24, "PPL ↓": 5.58},
        {"Model / Method": "STORYWRITER", "Length Sl (%)": 94.96, "Distinct-3 (%)": 96.24, "Distinct-4 (%)": 98.79, "PPL ↓": 5.68},
        {"Model / Method": "Self-Lengthen", "Length Sl (%)": 87.13, "Distinct-3 (%)": 92.48, "Distinct-4 (%)": 95.95, "PPL ↓": 4.76},
        {"Model / Method": "LongWriter-Zero-32B", "Length Sl (%)": 91.50, "Distinct-3 (%)": 96.20, "Distinct-4 (%)": 98.10, "PPL ↓": 4.95},
        {"Model / Method": "Qwen2.5-7B (Base - Chưa train)", "Length Sl (%)": b_mean_sl, "Distinct-3 (%)": b_mean_d3, "Distinct-4 (%)": b_mean_d4, "PPL ↓": b_mean_ppl},
    ]
    if has_lora:
        ckpt_label = f"⭐ MA-GRPO Qwen ({os.path.basename(checkpoint_dir)})"
        comp_rows.append({"Model / Method": ckpt_label, "Length Sl (%)": c_mean_sl, "Distinct-3 (%)": c_mean_d3, "Distinct-4 (%)": c_mean_d4, "PPL ↓": c_mean_ppl})

    df_comp = pd.DataFrame(comp_rows)
    print("\n" + "=" * 90)
    if has_lora:
        print(f"BẢNG ĐỐI SÁNH TRỰC DIỆN: QWEN CHƯA TRAIN vs {os.path.basename(checkpoint_dir).upper()} vs SOTA BASELINES")
    else:
        print("BẢNG ĐÁNH GIÁ BENCHMARK: QWEN CHƯA TRAIN (ZERO-SHOT) vs SOTA BASELINES")
    print("=" * 90)
    print(df_comp.to_string(index=False))

    df_comp.to_csv(output_csv, index=False)
    print(f"\nSaved benchmark comparison table to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Side-by-side inference and benchmark comparison.")
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/MA-GRPO-Checkpoints/qwen-7b-final/checkpoint-10")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", default="data/chapters.jsonl")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--output", default="benchmark_base_vs_checkpoint.csv")
    args = parser.parse_args()

    run_comparison(
        checkpoint_dir=args.checkpoint,
        base_model_name=args.base_model,
        dataset_path=args.dataset,
        num_samples=args.samples,
        output_csv=args.output,
    )

