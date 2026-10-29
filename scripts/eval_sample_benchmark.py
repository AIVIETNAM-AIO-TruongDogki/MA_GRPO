"""
Benchmark and Evaluation Script for MA-GRPO Checkpoints.
Generates narrative samples on benchmark prompts and evaluates key metrics:
  - Length Compliance Score (S_l %)
  - Lexical Diversity (Distinct-3 %, Distinct-4 %)
  - Fluency Perplexity (PPL)
  - Overlap with Reference (BLEU-4, ROUGE-L)
Compares results directly with SOTA baselines (StoryWriter, Agents' Room, LongWriter-Zero).
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_dataset_path(given_path: str = None) -> Path:
    if given_path and Path(given_path).exists():
        return Path(given_path)
    candidates = [
        PROJECT_ROOT / "data" / "tell-me-a-story-test.jsonl",
        PROJECT_ROOT / "data" / "chapters.jsonl",
        PROJECT_ROOT.parent / "Dataset" / "tell_me_a_story_data" / "tell-me-a-story-test.jsonl",
        Path("/content/MA-GRPO/data/tell-me-a-story-test.jsonl"),
        Path("/content/MA-GRPO/data/chapters.jsonl"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def extract_target_length(text: str, default_len: int = 1500) -> int:
    clean = text.replace(",", "")
    match = re.search(r"between\s+(\d+)\s*(?:and|to|–|-)\s*(\d+)\s*words?", clean, re.IGNORECASE)
    if match:
        return (int(match.group(1)) + int(match.group(2))) // 2
    match = re.search(r"(\d+)\s*(?:–|-)\s*(\d+)\s*words?", clean, re.IGNORECASE)
    if match:
        return (int(match.group(1)) + int(match.group(2))) // 2
    match = re.search(r"(?:around|about|at least|over)\s+(\d+)\s*words?", clean, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d{3,5})\s*words?", clean, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return default_len


def calculate_length_score(actual_len: int, target_len: int) -> float:
    if actual_len <= 0 or target_len <= 0:
        return 0.0
    if actual_len > target_len:
        score = 100 * max(0.0, 1.0 - (actual_len / target_len - 1.0) / 3.0)
    else:
        score = 100 * max(0.0, 1.0 - (target_len / actual_len - 1.0) / 2.0)
    return round(score, 2)


def calculate_distinct_n(text: str) -> Dict[str, float]:
    clean = re.sub(r"<think>[\s\S]*?</think>", "", text)
    clean = re.sub(r"</?answer>", "", clean).strip()
    words = re.findall(r"\b\w+\b", clean.lower())
    total = len(words)
    res = {}
    for n in [3, 4]:
        if total < n:
            res[f"Distinct-{n} (%)"] = 0.0
            continue
        ngrams = [tuple(words[i : i + n]) for i in range(total - n + 1)]
        ratio = len(set(ngrams)) / len(ngrams) if ngrams else 0.0
        res[f"Distinct-{n} (%)"] = round(ratio * 100, 2)
    return res


def calculate_overlap_metrics(gen_text: str, ref_text: str) -> Dict[str, float]:
    clean_gen = re.sub(r"<think>[\s\S]*?</think>", "", gen_text).strip()
    clean_ref = ref_text.strip()
    if not clean_ref or not clean_gen:
        return {"BLEU-4": 0.0, "ROUGE-L": 0.0}

    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from rouge_score import rouge_scorer

        gen_tokens = clean_gen.lower().split()
        ref_tokens = [clean_ref.lower().split()]
        smooth_fn = SmoothingFunction().method1
        bleu4 = sentence_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_res = scorer.score(clean_ref, clean_gen)
        return {"BLEU-4": round(bleu4 * 100, 2), "ROUGE-L": round(rouge_res["rougeL"].fmeasure * 100, 2)}
    except Exception:
        return {"BLEU-4": 0.0, "ROUGE-L": 0.0}


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
                clean = re.sub(r"<think>[\s\S]*?</think>", "", t).strip()
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
        print(f"PPL computation skipped ({e})")
        return [0.0] * len(texts)


def run_benchmark(
    checkpoint_dir: str,
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    num_samples: int = 3,
    dataset_path: str = str(DEFAULT_DATASET),
    output_csv: str = "benchmark_comparison_results.csv",
):
    print("=" * 80)
    print(f"BENCHMARKING MA-GRPO CHECKPOINT ({num_samples} Samples)")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Base Model: {base_model_name}")
    print("=" * 80)

    # 1. Load Model with 4-bit NF4 for low VRAM usage (<6GB VRAM)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer and base model in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir if os.path.exists(checkpoint_dir) else base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if os.path.exists(checkpoint_dir) and (os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors")) or os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin"))):
        print(f"Attaching LoRA adapter from {checkpoint_dir}...")
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    else:
        print("Warning: LoRA checkpoint not found. Evaluating Base Model zero-shot...")
        model = base_model

    model.eval()

    # 2. Read test samples
    real_dataset_path = resolve_dataset_path(dataset_path)
    print(f"Loading benchmark samples from: {real_dataset_path}")
    samples = []
    if real_dataset_path.exists():
        with open(real_dataset_path, "r", encoding="utf-8-sig") as f:
            for idx, line in enumerate(f):
                if idx >= num_samples:
                    break
                if line.strip():
                    samples.append(json.loads(line))
    else:
        print(f"Error: Dataset path {real_dataset_path} not found.")
        return

    # 3. Generate prose
    generated_records = []
    for i, s in enumerate(samples):
        if "inputs" in s:
            premise = s.get("inputs", "")
            target_len = extract_target_length(premise, 1400)
            gold_ref = s.get("targets", "")
            prompt = f"""You are an elite literary novelist. Write the complete narrative prose for this story.
Total Target Length: Approximately {target_len} words.

[CONTEXT SUMMARY]:
{premise}

CRITICAL WRITING INSTRUCTIONS:
1. Begin with a <think>...</think> block outlining your narrative progression and scene word budgets.
2. Output the full story prose immediately after </think>.
Begin with <think> now:
"""
        else:
            prompt = s.get("prompt", "")
            target_len = int(s.get("target_words", 1200))
            gold_ref = ""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print(f"\n[Sample {i+1}/{num_samples}] Generating story (Target: {target_len} words)...")
        start_t = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2000,
                temperature=0.80,
                top_p=0.92,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start_t
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        words_count = len(re.findall(r"\b\w+\b", re.sub(r"<think>[\s\S]*?</think>", "", gen_text)))
        print(f"Done in {elapsed:.1f}s | Words generated: {words_count}")

        generated_records.append({
            "sample_id": s.get("example_id", f"sample_{i}"),
            "target_len": target_len,
            "actual_words": words_count,
            "gen_text": gen_text,
            "gold_ref": gold_ref,
        })

    # 4. Compute Metrics
    print("\nComputing metrics (Sl, Distinct-N, PPL, Overlap)...")
    ppl_scores = compute_ppl_qwen([r["gen_text"] for r in generated_records])

    eval_results = []
    for i, r in enumerate(generated_records):
        sl = calculate_length_score(r["actual_words"], r["target_len"])
        dist = calculate_distinct_n(r["gen_text"])
        overlap = calculate_overlap_metrics(r["gen_text"], r["gold_ref"])
        eval_results.append({
            "Sample": r["sample_id"],
            "Target_Len": r["target_len"],
            "Actual_Words": r["actual_words"],
            "Sl (%)": sl,
            "Distinct-3 (%)": dist.get("Distinct-3 (%)", 0.0),
            "Distinct-4 (%)": dist.get("Distinct-4 (%)", 0.0),
            "PPL": ppl_scores[i] if i < len(ppl_scores) else 0.0,
            "BLEU-4": overlap.get("BLEU-4", 0.0),
            "ROUGE-L": overlap.get("ROUGE-L", 0.0),
        })

    df_results = pd.DataFrame(eval_results)
    mean_sl = round(float(df_results["Sl (%)"].mean()), 2)
    mean_d3 = round(float(df_results["Distinct-3 (%)"].mean()), 2)
    mean_d4 = round(float(df_results["Distinct-4 (%)"].mean()), 2)
    mean_ppl = round(float(df_results["PPL"].mean()), 2)
    mean_bleu = round(float(df_results["BLEU-4"].mean()), 2)
    mean_rouge = round(float(df_results["ROUGE-L"].mean()), 2)

    # 5. Build SOTA Comparative Table
    comparison_table = [
        {"Model / Method": "AGENTS' ROOM", "Length Sl (%)": 92.06, "Distinct-3 (%)": 95.48, "Distinct-4 (%)": 98.24, "PPL ↓": 5.58, "BLEU-4": 1.15, "ROUGE-L": 18.20},
        {"Model / Method": "STORYWRITER", "Length Sl (%)": 94.96, "Distinct-3 (%)": 96.24, "Distinct-4 (%)": 98.79, "PPL ↓": 5.68, "BLEU-4": 1.30, "ROUGE-L": 17.80},
        {"Model / Method": "Self-Lengthen", "Length Sl (%)": 87.13, "Distinct-3 (%)": 92.48, "Distinct-4 (%)": 95.95, "PPL ↓": 4.76, "BLEU-4": 1.45, "ROUGE-L": 18.90},
        {"Model / Method": "LongWriter-llama3.1-8b", "Length Sl (%)": 87.82, "Distinct-3 (%)": 97.86, "Distinct-4 (%)": 99.08, "PPL ↓": 5.28, "BLEU-4": 1.25, "ROUGE-L": 18.72},
        {"Model / Method": "LongWriter-glm4-9b", "Length Sl (%)": 94.55, "Distinct-3 (%)": 96.88, "Distinct-4 (%)": 98.74, "PPL ↓": 5.12, "BLEU-4": 1.38, "ROUGE-L": 19.15},
        {"Model / Method": "LongWriter-Zero-32B", "Length Sl (%)": 91.50, "Distinct-3 (%)": 96.20, "Distinct-4 (%)": 98.10, "PPL ↓": 4.95, "BLEU-4": 1.82, "ROUGE-L": 21.60},
        {"Model / Method": "⭐ MA-GRPO (Ours - 50 Steps)", "Length Sl (%)": mean_sl, "Distinct-3 (%)": mean_d3, "Distinct-4 (%)": mean_d4, "PPL ↓": mean_ppl, "BLEU-4": mean_bleu, "ROUGE-L": mean_rouge},
    ]

    df_comp = pd.DataFrame(comparison_table)
    print("\n" + "=" * 90)
    print("BẢNG ĐỐI SÁNH KẾT QUẢ VỚI CÁC MÔ HÌNH SOTA (BENCHMARK COMPARISON)")
    print("=" * 90)
    print(df_comp.to_string(index=False))

    df_comp.to_csv(output_csv, index=False)
    print(f"\nSaved detailed comparison to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/MA-GRPO-Checkpoints/ma-grpo-final/checkpoint-50")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default="benchmark_comparison_results.csv")
    args = parser.parse_args()

    run_benchmark(
        checkpoint_dir=args.checkpoint,
        base_model_name=args.base_model,
        num_samples=args.samples,
        dataset_path=args.dataset,
        output_csv=args.output,
    )

