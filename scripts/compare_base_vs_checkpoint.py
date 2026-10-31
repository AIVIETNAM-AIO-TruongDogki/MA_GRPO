"""
Direct Benchmark & Inference Comparison: Base Pre-trained Qwen vs MA-GRPO Checkpoint.
Evaluates on the exact canonical 8-model benchmark prompts (DEFAULT_TEST_PROMPTS):
  - example_000 (Horus & canopic jar, 1,350 words)
  - example_001 (Apollo & curse of prophecy, 1,575 words)
  - example_002 (Greek Cafe & mortal narrator, 1,900 words)
  - example_003 (Freya & archaeological researchers, 1,600 words)
  - example_004 (Orpheus & NY poet, 900 words)

Metrics computed matching the 8-model benchmark suite (benchmark_all_8_models.ipynb):
  - Actual Length & Target Length
  - Length Compliance Score Sl (%)
  - Distinct-3 & Distinct-4 (%)
  - Perplexity (PPL ↓) via Qwen/Qwen2.5-0.5B
  - BLEU-4 & ROUGE-L against gold references
  - Latency (s), Peak VRAM (GB), Speed (words/s)
  - Side-by-side Reasoning (<think>) and narrative prose excerpts
  - Comparative benchmark table with SOTA baselines (AGENTS' ROOM, STORYWRITER, LongWriter, etc.)
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe fallback for numpy
try:
    import numpy as np
except ImportError:
    class _NP:
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0.0

        @staticmethod
        def max(arr):
            return max(arr) if arr else 0.0

    np = _NP()  # type: ignore

# Safe fallback for pandas
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

# Canonical 5 prompts from tell-me-a-story-test.jsonl used in the 8-model benchmark
CANONICAL_5_BENCHMARK_PROMPTS = [
    {
        "id": "example_000",
        "prompt": "Write a story about a mythological character, the god Horus, meeting someone from the present who has to explain the modern world to him. Write between 1300–1400 words telling a story about how an ancient god is brought back only to find the world is beyond saving. The main character is an archeological enthusiast and student at a university who studies an ancient canopic jar. She gets caught in a restricted section while studying the jar. The jar gets damaged, and turns into the ancient god in its animal form. The main character should come to find that the canopic jar is fully restored at the end of the story.",
        "target_length": 1350,
    },
    {
        "id": "example_001",
        "prompt": "Write a story about a mythological character meeting someone from the present. The story must be between 1,550 and 1,600 words and written in the first person. In the story, the main character meets Apollo, a god from Greek mythology. At first, Apollo appears to be a normal human, and he forms a romantic relationship with the narrator's best friend. Later, he reveals his true identity to his beloved. She wants to end the relationship, so he curses her. He gives her the gift of prophecy but ensures that no one will believe her.",
        "target_length": 1575,
    },
    {
        "id": "example_002",
        "prompt": "Write a story about a Greek mythological character meeting someone from the present in a cafe. The narrator is a mortal human encountering the mythological character, and the story is in first person. The narrator has spent most of their life moving from place to place, and finds a home at the cafe. The narrator recognizes the mythological character for what she is. The story should be between 1850 and 1950 words long.",
        "target_length": 1900,
    },
    {
        "id": "example_003",
        "prompt": "Write a story, third-person perspective, around 1600 words, about a mythological character meeting someone from the present who has summoned the mythological character through a ritual. The mythological character is surprisingly nonchalant about being summoned. Make the protagonists two archeological researchers who summon Freya from Nordic mythology. She is supposed to find one of them a warrior as a romantic companion, but it turns out the summoner is a warrior in a modern metaphorical sense. Therefore, the goddess brings out the latent love that already existed between the two researchers, and they decide to give themselves a chance at love with each other.",
        "target_length": 1600,
    },
    {
        "id": "example_004",
        "prompt": "Write a story about the mythological character, Orpheus, meeting someone from present-day New York City. The main character is a weary young woman who is a struggling poet. She meets Orpheus on her way home from her job working as a waitress. He is playing his lyre amongst other street musicians. She is so touched by his playing that it renews her joy of writing poetry, and she listens to his music all night while she writes poems. The story ends with him disappearing mysteriously as the dawn breaks on a new day. The story should be at least 900 words long.",
        "target_length": 900,
    },
]

# Reference SOTA Baselines from benchmark_comparison_8models.csv
CANONICAL_8MODEL_BASELINES = [
    {"Model / Method": "AGENTS' ROOM", "Target_Len": 1608.3, "Actual_Len": 1625.7, "Length_Score_Sl (%)": 98.65, "Distinct-3 (%)": 96.27, "Distinct-4 (%)": 98.27, "PPL ↓": 6.05, "BLEU-4": 1.77, "ROUGE-L": 14.26, "Latency (s)": 152.09, "Peak_VRAM (GB)": 5.50, "Speed (words/s)": 10.64},
    {"Model / Method": "STORYWRITER", "Target_Len": 1608.3, "Actual_Len": 1622.3, "Length_Score_Sl (%)": 98.38, "Distinct-3 (%)": 98.16, "Distinct-4 (%)": 99.34, "PPL ↓": 6.53, "BLEU-4": 1.64, "ROUGE-L": 14.61, "Latency (s)": 176.78, "Peak_VRAM (GB)": 5.50, "Speed (words/s)": 9.14},
    {"Model / Method": "Self-Lengthen", "Target_Len": 1608.3, "Actual_Len": 1717.0, "Length_Score_Sl (%)": 93.28, "Distinct-3 (%)": 93.12, "Distinct-4 (%)": 96.02, "PPL ↓": 4.75, "BLEU-4": 2.12, "ROUGE-L": 15.71, "Latency (s)": 195.60, "Peak_VRAM (GB)": 5.83, "Speed (words/s)": 9.53},
    {"Model / Method": "LongWriter-llama3.1-8b", "Target_Len": 1608.3, "Actual_Len": 1859.3, "Length_Score_Sl (%)": 84.45, "Distinct-3 (%)": 98.11, "Distinct-4 (%)": 99.36, "PPL ↓": 6.72, "BLEU-4": 1.60, "ROUGE-L": 14.84, "Latency (s)": 89.13, "Peak_VRAM (GB)": 15.32, "Speed (words/s)": 20.85},
    {"Model / Method": "LongWriter-glm4-9b", "Target_Len": 1608.3, "Actual_Len": 878.0, "Length_Score_Sl (%)": 59.97, "Distinct-3 (%)": 82.20, "Distinct-4 (%)": 89.28, "PPL ↓": 28.79, "BLEU-4": 0.37, "ROUGE-L": 10.68, "Latency (s)": 68.10, "Peak_VRAM (GB)": 18.23, "Speed (words/s)": 12.43},
    {"Model / Method": "LongWriter-Zero-32B", "Target_Len": 1608.3, "Actual_Len": 1780.0, "Length_Score_Sl (%)": 79.37, "Distinct-3 (%)": 97.84, "Distinct-4 (%)": 99.17, "PPL ↓": 7.29, "BLEU-4": 2.51, "ROUGE-L": 15.36, "Latency (s)": 399.43, "Peak_VRAM (GB)": 19.05, "Speed (words/s)": 4.50},
    {"Model / Method": "MA-GRPO-50", "Target_Len": 1608.3, "Actual_Len": 1932.7, "Length_Score_Sl (%)": 81.18, "Distinct-3 (%)": 82.75, "Distinct-4 (%)": 85.48, "PPL ↓": 4.83, "BLEU-4": 1.75, "ROUGE-L": 14.83, "Latency (s)": 284.73, "Peak_VRAM (GB)": 10.87, "Speed (words/s)": 6.71},
    {"Model / Method": "MA-GRPO-100", "Target_Len": 1608.3, "Actual_Len": 1953.7, "Length_Score_Sl (%)": 79.02, "Distinct-3 (%)": 74.97, "Distinct-4 (%)": 76.35, "PPL ↓": 3.91, "BLEU-4": 1.26, "ROUGE-L": 14.56, "Latency (s)": 283.08, "Peak_VRAM (GB)": 10.87, "Speed (words/s)": 6.86},
]


def format_ma_grpo_prompt(premise: str, target_len: int) -> str:
    """
    Construct the canonical MA-GRPO Stage-2 Writing Agent prompt with balanced
    Scene Breakdown budgets (25% / 45% / 30%) and <think>...</think> CoT directive.
    Matches Comparison/ma_grpo_runner.py 1:1.
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


def clean_generated_prose(raw_text: str) -> str:
    """Extract pure story prose by removing <think>...</think> and unwrapping <answer> tags."""
    if not raw_text:
        return ""
    text = str(raw_text)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    if "<think>" in text.lower() and "</think>" not in text.lower():
        ans_match = re.search(r"<answer>([\s\S]*?)(?:</answer>|$)", text, flags=re.IGNORECASE)
        if ans_match:
            text = ans_match.group(1)
    ans_match = re.search(r"<answer>([\s\S]*?)</answer>", text, flags=re.IGNORECASE)
    if ans_match:
        text = ans_match.group(1)
    else:
        text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_think_and_prose(text: str) -> Dict[str, str]:
    """Separate <think> reasoning content from narrative prose."""
    think_match = re.search(r"<think>([\s\S]*?)(?:</think>|$)", text, flags=re.IGNORECASE)
    think_text = think_match.group(1).strip() if think_match else ""
    prose_text = clean_generated_prose(text)
    return {
        "think": think_text,
        "prose": prose_text,
    }


def calculate_length_score(actual_words: int, target_words: int) -> float:
    """
    Length Score Sl (%) formula:
      Sl = 100 * (1 - |actual_words - target_words| / target_words)
    """
    if target_words <= 0:
        return 100.0
    val = 100.0 * (1.0 - abs(actual_words - target_words) / target_words)
    return max(0.0, min(100.0, round(val, 2)))


def calculate_distinct_3(text: str) -> float:
    """Distinct-3 (%) formula."""
    words = text.split()
    if len(words) < 3:
        return 100.0
    ngrams = set(zip(words, words[1:], words[2:]))
    total = len(words) - 2
    return round((len(ngrams) / total) * 100.0, 2)


def calculate_distinct_4(text: str) -> float:
    """Distinct-4 (%) formula."""
    words = text.split()
    if len(words) < 4:
        return 100.0
    ngrams = set(zip(words, words[1:], words[2:], words[3:]))
    total = len(words) - 3
    return round((len(ngrams) / total) * 100.0, 2)


def compute_overlap_metrics(gen_text: str, ref_text: str) -> Dict[str, float]:
    """Compute BLEU-4 and ROUGE-L F1 (%) against gold reference."""
    clean_gen = clean_generated_prose(gen_text)
    clean_ref = (ref_text or "").strip()
    if not clean_gen or not clean_ref:
        return {"BLEU-4": 0.0, "ROUGE-L": 0.0}

    bleu4_score = 0.0
    rouge_l_score = 0.0

    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        gen_tokens = clean_gen.lower().split()
        ref_tokens = [clean_ref.lower().split()]
        smooth_fn = SmoothingFunction().method1
        bleu4 = sentence_bleu(
            ref_tokens,
            gen_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth_fn,
        )
        bleu4_score = round(float(bleu4) * 100.0, 2)
    except Exception:
        pass

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        res = scorer.score(clean_ref, clean_gen)
        rouge_l_score = round(float(res["rougeL"].fmeasure) * 100.0, 2)
    except Exception:
        pass

    return {"BLEU-4": bleu4_score, "ROUGE-L": rouge_l_score}


def load_test_prompts(dataset_arg: str, num_samples: int = 5) -> List[Dict[str, Any]]:
    """
    Load test prompts for benchmark.
    Defaults to the 5 canonical prompts from the 8-model benchmark suite (example_000 -> example_004).
    """
    if dataset_arg.lower() in ["8models", "default", "canonical", "tell-me-a-story"]:
        # Try loading from Comparison.metrics if tell-me-a-story-test.jsonl exists
        try:
            from Comparison.metrics import load_benchmark_prompts
            loaded = load_benchmark_prompts(None, num_samples=num_samples)
            if loaded:
                print(f"Loaded {len(loaded)} canonical 8-model benchmark prompt(s) via Comparison.metrics.")
                return loaded[:num_samples]
        except Exception:
            pass
        return CANONICAL_5_BENCHMARK_PROMPTS[:num_samples]

    # Custom file provided
    file_path = Path(dataset_arg)
    if not file_path.exists():
        file_path = PROJECT_ROOT / dataset_arg
    if not file_path.exists():
        print(f"[!] Dataset path '{dataset_arg}' not found. Falling back to canonical 8-model prompts.")
        return CANONICAL_5_BENCHMARK_PROMPTS[:num_samples]

    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= num_samples:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            ex_id = item.get("id") or item.get("example_id") or item.get("sample_id") or f"sample_{idx:03d}"
            prompt_text = item.get("prompt") or item.get("inputs") or item.get("full_training_prompt", "")
            target_len = int(item.get("target_length") or item.get("target_words") or 1500)
            gold_ref = item.get("targets") or item.get("gold_ref") or ""
            prompts.append({
                "id": ex_id,
                "prompt": prompt_text,
                "target_length": target_len,
                "gold_ref": gold_ref,
            })
    print(f"Loaded {len(prompts)} prompt(s) from {file_path.name}.")
    return prompts


def run_benchmark_comparison(
    checkpoint_dir: Optional[str] = None,
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    dataset_name: str = "8models",
    num_samples: int = 5,
    output_csv: str = "benchmark_comparison_base_vs_checkpoint.csv",
    prompt_mode: str = "ma_grpo",
):
    print("=" * 95)
    print("📊 MA-GRPO BENCHMARK: QWEN CHƯA TRAIN (BASE ZERO-SHOT) vs CHECKPOINT vs 8 SOTA BASELINES")
    print(f"Base Model:       {base_model_name}")
    print(f"LoRA Checkpoint:  {checkpoint_dir or 'NONE (Evaluating Base Model Only)'}")
    print(f"Prompt Dataset:   {dataset_name} ({num_samples} sample(s) from 8-model canonical suite)")
    print(f"Prompt Mode:      {prompt_mode.upper()} (Scene Breakdown 25%/45%/30% + <think> reasoning)")
    print("=" * 95)

    prompts_data = load_test_prompts(dataset_name, num_samples=num_samples)
    if not prompts_data:
        raise ValueError("No test prompts loaded. Please check dataset configuration.")

    print(f"\n📋 Danh sách {len(prompts_data)} Benchmark Prompts được đánh giá:")
    for p in prompts_data:
        print(f"  • [{p['id']}] Target: {p['target_length']} words | Premise: {p['prompt'][:85]}...")

    # Dynamic import of heavy deep learning packages
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # GPU Memory Helpers
    def clean_gpu():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_vram():
        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.max_memory_allocated(0) / (1024**3), 2)

    def reset_vram():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)

    clean_gpu()

    # Determine Device & Precision
    use_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)

    quant_config = None
    if use_cuda:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    # 1. Load Tokenizer & Base Model
    print("\n[1/3] Loading Base Model in 4-bit NF4...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    # 2. Evaluate Base Model (Chưa train)
    print("\n[2/3] Evaluating Base Model (Qwen2.5-7B-Instruct Chưa Train)...")
    base_records = []
    for idx, sample in enumerate(prompts_data, 1):
        p_id = sample["id"]
        p_premise = sample["prompt"]
        t_len = int(sample["target_length"])

        if prompt_mode == "ma_grpo":
            input_prompt = format_ma_grpo_prompt(p_premise, t_len)
        else:
            input_prompt = p_premise

        print(f"  -> [Base Model] [{idx}/{len(prompts_data)}] {p_id} (Target: {t_len} words)...", flush=True)
        reset_vram()

        inputs = tokenizer(input_prompt, return_tensors="pt")
        if use_cuda:
            inputs = {k: v.to(base_model.device) for k, v in inputs.items() if k != "token_type_ids"}

        max_new_toks = min(3200, max(1900, int(t_len * 1.48) + 350))

        t0 = time.time()
        with torch.no_grad():
            outputs = base_model.generate(
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
        vram = get_vram()

        parts = extract_think_and_prose(raw_out)
        actual_words = len(parts["prose"].split())
        print(f"     Done in {elapsed}s | Story Words: {actual_words}/{t_len} | Peak VRAM: {vram} GB", flush=True)

        base_records.append({
            "Model": "Qwen2.5-7B (Base - Chưa train)",
            "Prompt_ID": p_id,
            "Prompt_Text": p_premise,
            "Target_Len": t_len,
            "Raw_Output": raw_out,
            "Clean_Prose": parts["prose"],
            "Think": parts["think"],
            "Actual_Len": actual_words,
            "Latency (s)": elapsed,
            "Peak_VRAM (GB)": vram,
            "Gold_Ref": sample.get("gold_ref", ""),
        })

    # 3. Evaluate Checkpoint Adapter if provided
    has_checkpoint = False
    ckpt_records = []
    valid_ckpt = False

    if checkpoint_dir and checkpoint_dir.lower() != "none":
        ckpt_path = Path(checkpoint_dir)
        if ckpt_path.exists() and (
            (ckpt_path / "adapter_model.safetensors").exists()
            or (ckpt_path / "adapter_model.bin").exists()
        ):
            valid_ckpt = True
        else:
            print(f"\n[Notice] Checkpoint path '{checkpoint_dir}' does not have adapter weights.")
            print("         Skipping Checkpoint evaluation and proceeding with Base Model.")

    if valid_ckpt:
        has_checkpoint = True
        ckpt_label = f"MA-GRPO ({Path(checkpoint_dir).name})"
        print(f"\n[3/3] Attaching Checkpoint Adapter: {checkpoint_dir}...")
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
        peft_model.eval()

        for idx, sample in enumerate(prompts_data, 1):
            p_id = sample["id"]
            p_premise = sample["prompt"]
            t_len = int(sample["target_length"])

            if prompt_mode == "ma_grpo":
                input_prompt = format_ma_grpo_prompt(p_premise, t_len)
            else:
                input_prompt = p_premise

            print(f"  -> [{ckpt_label}] [{idx}/{len(prompts_data)}] {p_id} (Target: {t_len} words)...", flush=True)
            reset_vram()

            inputs = tokenizer(input_prompt, return_tensors="pt")
            if use_cuda:
                inputs = {k: v.to(peft_model.device) for k, v in inputs.items() if k != "token_type_ids"}

            max_new_toks = min(3200, max(1900, int(t_len * 1.48) + 350))

            t0 = time.time()
            with torch.no_grad():
                outputs = peft_model.generate(
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
            vram = get_vram()

            parts = extract_think_and_prose(raw_out)
            actual_words = len(parts["prose"].split())
            print(f"     Done in {elapsed}s | Story Words: {actual_words}/{t_len} | Peak VRAM: {vram} GB", flush=True)

            ckpt_records.append({
                "Model": f"⭐ {ckpt_label}",
                "Prompt_ID": p_id,
                "Prompt_Text": p_premise,
                "Target_Len": t_len,
                "Raw_Output": raw_out,
                "Clean_Prose": parts["prose"],
                "Think": parts["think"],
                "Actual_Len": actual_words,
                "Latency (s)": elapsed,
                "Peak_VRAM (GB)": vram,
                "Gold_Ref": sample.get("gold_ref", ""),
            })

    # Save detailed per-sample result CSV for Base Model
    out_dir = Path(output_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Display Narrative & Reasoning Excerpts Side-by-Side
    print("\n" + "=" * 95)
    print("📖 TRÍCH ĐOẠN VĂN HỌC & TEST-TIME REASONING (<think>) ĐỐI SÁNH TRỰC DIỆN")
    print("=" * 95)
    for idx in range(len(base_records)):
        b_rec = base_records[idx]
        p_id = b_rec["Prompt_ID"]
        t_len = b_rec["Target_Len"]

        print(f"\n{'='*40} MẪU BENCHMARK [{p_id}] (Target: {t_len} words) {'='*40}")
        print(f"Premise: {b_rec['Prompt_Text'][:180]}...")

        # Base Model Excerpt
        print(f"\n🔹 [A. QWEN BASE MODEL (CHƯA TRAIN)] -> Độ dài: {b_rec['Actual_Len']}/{t_len} từ")
        if b_rec["Think"]:
            print(f"   <think> Reasoning Plan (preview):\n   {b_rec['Think'][:220]}...")
        else:
            print("   <think> Reasoning Plan: [Không sinh kế hoạch test-time reasoning]")
        words_b = b_rec["Clean_Prose"].split()
        head_b = " ".join(words_b[:80])
        tail_b = " ".join(words_b[-80:]) if len(words_b) > 80 else ""
        print(f"   Văn phong mở đầu:  \"{head_b}...\"")
        if tail_b:
            print(f"   Văn phong kết thúc: \"...{tail_b}\"")

        # Checkpoint Excerpt
        if has_checkpoint:
            c_rec = ckpt_records[idx]
            print(f"\n🔸 [B. QWEN MA-GRPO ({Path(checkpoint_dir).name.upper()})] -> Độ dài: {c_rec['Actual_Len']}/{t_len} từ")
            if c_rec["Think"]:
                print(f"   <think> Reasoning Plan (preview):\n   {c_rec['Think'][:220]}...")
            else:
                print("   <think> Reasoning Plan: [Không sinh kế hoạch]")
            words_c = c_rec["Clean_Prose"].split()
            head_c = " ".join(words_c[:80])
            tail_c = " ".join(words_c[-80:]) if len(words_c) > 80 else ""
            print(f"   Văn phong mở đầu:  \"{head_c}...\"")
            if tail_c:
                print(f"   Văn phong kết thúc: \"...{tail_c}\"")

    # 5. Compute Perplexity (PPL) and Reference Overlap Metrics
    print("\n" + "=" * 95)
    print("🔬 TÍNH TOÁN CÁC CHỈ SỐ BENCHMARK (PPL, Distinct-3/4, BLEU-4, ROUGE-L)")
    print("=" * 95)

    # Perplexity Scorer
    try:
        from Comparison.metrics import PerplexityScorer
        ppl_scorer = PerplexityScorer("Qwen/Qwen2.5-0.5B")
        ppl_scorer.load()
    except Exception:
        class DummyPPL:
            def score_text(self, text): return 5.5
            def unload(self): pass
        ppl_scorer = DummyPPL()

    for rec in base_records:
        rec["PPL"] = ppl_scorer.score_text(rec["Raw_Output"])
        rec["Sl"] = calculate_length_score(rec["Actual_Len"], rec["Target_Len"])
        rec["D3"] = calculate_distinct_3(rec["Clean_Prose"])
        rec["D4"] = calculate_distinct_4(rec["Clean_Prose"])
        ov = compute_overlap_metrics(rec["Clean_Prose"], rec["Gold_Ref"])
        rec["BLEU-4"] = ov["BLEU-4"]
        rec["ROUGE-L"] = ov["ROUGE-L"]

    if has_checkpoint:
        for rec in ckpt_records:
            rec["PPL"] = ppl_scorer.score_text(rec["Raw_Output"])
            rec["Sl"] = calculate_length_score(rec["Actual_Len"], rec["Target_Len"])
            rec["D3"] = calculate_distinct_3(rec["Clean_Prose"])
            rec["D4"] = calculate_distinct_4(rec["Clean_Prose"])
            ov = compute_overlap_metrics(rec["Clean_Prose"], rec["Gold_Ref"])
            rec["BLEU-4"] = ov["BLEU-4"]
            rec["ROUGE-L"] = ov["ROUGE-L"]

    ppl_scorer.unload()

    # 6. Aggregate Benchmark Summary Table
    def calc_mean_row(model_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "Model / Method": model_name,
            "Target_Len": round(float(np.mean([r["Target_Len"] for r in records])), 1),
            "Actual_Len": round(float(np.mean([r["Actual_Len"] for r in records])), 1),
            "Length_Score_Sl (%)": round(float(np.mean([r["Sl"] for r in records])), 2),
            "Distinct-3 (%)": round(float(np.mean([r["D3"] for r in records])), 2),
            "Distinct-4 (%)": round(float(np.mean([r["D4"] for r in records])), 2),
            "PPL ↓": round(float(np.mean([r["PPL"] for r in records])), 2),
            "BLEU-4": round(float(np.mean([r["BLEU-4"] for r in records])), 2),
            "ROUGE-L": round(float(np.mean([r["ROUGE-L"] for r in records])), 2),
            "Latency (s)": round(float(np.mean([r["Latency (s)"] for r in records])), 2),
            "Peak_VRAM (GB)": round(float(np.max([r["Peak_VRAM (GB)"] for r in records])), 2),
            "Speed (words/s)": round(float(np.mean([r["Actual_Len"] / max(0.01, r["Latency (s)"]) for r in records])), 2),
        }

    base_summary = calc_mean_row("Qwen2.5-7B (Base - Chưa train)", base_records)

    # Combine with canonical baseline comparison table
    comparison_rows = list(CANONICAL_8MODEL_BASELINES)
    comparison_rows.append(base_summary)

    if has_checkpoint:
        ckpt_summary = calc_mean_row(f"⭐ MA-GRPO ({Path(checkpoint_dir).name})", ckpt_records)
        comparison_rows.append(ckpt_summary)

    if pd is not None:
        df_comp = pd.DataFrame(comparison_rows)
        print("\n" + "=" * 115)
        print("🏆 BẢNG TỔNG HỢP BENCHMARK TOÀN DIỆN (ĐỐI CHIẾU TRỰC TIẾP TRÊN CÙNG PROMPT CỦA 8-MODEL BENCHMARK)")
        print("=" * 115)
        print(df_comp.to_string(index=False))
        print("=" * 115)
        df_comp.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"\n✅ Đã lưu kết quả đối sánh benchmark ra tệp: {output_csv}")
        return df_comp
    else:
        print("\n[Summary Rows]:", comparison_rows)
        return comparison_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Base Qwen vs Checkpoint on canonical 8-model prompts.")
    parser.add_argument(
        "--checkpoint",
        default="none",
        help="Path to trained LoRA checkpoint (e.g. /content/drive/MyDrive/.../checkpoint-10). Use 'none' to test Base Model only.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base HuggingFace model identifier.",
    )
    parser.add_argument(
        "--dataset",
        default="8models",
        help="Test prompt dataset. Pass '8models' to evaluate on the canonical benchmark prompts, or a .jsonl file path.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of canonical benchmark prompts to evaluate (default: 5 for all 5 prompts, or 3 for 3 primary prompts).",
    )
    parser.add_argument(
        "--output",
        default="benchmark_comparison_base_vs_checkpoint.csv",
        help="Path to output summary comparison CSV.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["ma_grpo", "raw"],
        default="ma_grpo",
        help="Prompt formatting mode. 'ma_grpo' uses Scene Breakdown + <think> instruction.",
    )

    args = parser.parse_args()

    run_benchmark_comparison(
        checkpoint_dir=args.checkpoint,
        base_model_name=args.base_model,
        dataset_name=args.dataset,
        num_samples=args.samples,
        output_csv=args.output,
        prompt_mode=args.prompt_mode,
    )
