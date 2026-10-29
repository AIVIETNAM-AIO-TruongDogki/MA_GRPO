"""
Benchmark and Evaluation Script for MA-GRPO (Checkpoint-50 / Checkpoint-100).
Generates long-form narrative stories on benchmark test prompts and outputs
an evaluation CSV strictly matching the schema and format of:
  kaggle_longwriter/output_new/result_longwriter_llama.csv

Output CSV Schema:
  Model,Prompt_ID,Target_Len,Actual_Len,Length_Score_Sl (%),Distinct-3 (%),Distinct-4 (%),Latency (s),Peak_VRAM (GB),Speed (words/s),Prompt_Text,Raw_Output
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

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Default benchmark prompts matching LongWriter / StoryWriter benchmark
DEFAULT_BENCHMARK_PROMPTS = [
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


def resolve_checkpoint_dir(checkpoint_arg: Optional[str]) -> Path:
    """Find the checkpoint directory across local and cloud environments."""
    if checkpoint_arg and Path(checkpoint_arg).exists():
        return Path(checkpoint_arg)

    candidates = [
        # Checkpoint 100 candidates
        Path("C:/Users/ASUS/AI_LLMs/Long_text/ma-grpo-final-20261029T095313Z-1-001/ma-grpo-final/checkpoint-100"),
        Path("/content/drive/MyDrive/MA-GRPO-Checkpoints/ma-grpo-final/checkpoint-100"),
        Path("./ma-grpo-final/checkpoint-100"),
        # Checkpoint 50 candidates
        Path("C:/Users/ASUS/AI_LLMs/Long_text/ma-grpo-final-20261029T095313Z-1-001/ma-grpo-final/checkpoint-50"),
        Path("/content/drive/MyDrive/MA-GRPO-Checkpoints/ma-grpo-final/checkpoint-50"),
        Path("./ma-grpo-final/checkpoint-50"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not locate checkpoint directory. Please pass --checkpoint <path>. Checked: {[str(c) for c in candidates]}"
    )


def resolve_output_path(output_arg: Optional[str], default_name: str = "result_ma_grpo_ckpt100.csv") -> Path:
    """Resolve output path, ensuring parent directory exists."""
    if output_arg:
        out_p = Path(output_arg)
    else:
        candidates = [
            Path("C:/Users/ASUS/AI_LLMs/Long_text/kaggle_longwriter/output_new") / default_name,
            Path("./output_new") / default_name,
            Path("./") / default_name,
        ]
        out_p = candidates[0] if candidates[0].parent.exists() else candidates[-1]

    out_p.parent.mkdir(parents=True, exist_ok=True)
    return out_p


def calculate_length_score(actual_words: int, target_words: int) -> float:
    """Standard LongBench-Write Length Score formula."""
    if target_words <= 0:
        return 100.0
    val = 100.0 * (1.0 - abs(actual_words - target_words) / target_words)
    return max(0.0, min(100.0, round(val, 2)))


def calculate_distinct_3(text: str) -> float:
    """Percentage of unique 3-grams in the text."""
    words = text.split()
    if len(words) < 3:
        return 100.0
    ngrams = set(zip(words, words[1:], words[2:]))
    total = len(words) - 2
    return round((len(ngrams) / total) * 100.0, 2)


def calculate_distinct_4(text: str) -> float:
    """Percentage of unique 4-grams in the text."""
    words = text.split()
    if len(words) < 4:
        return 100.0
    ngrams = set(zip(words, words[1:], words[2:], words[3:]))
    total = len(words) - 3
    return round((len(ngrams) / total) * 100.0, 2)


def clean_story_prose(raw_text: str) -> str:
    """Remove Chain-of-Thought (<think>...</think>) and answer tags to get pure prose."""
    clean = re.sub(r"<think>[\s\S]*?</think>", "", raw_text)
    clean = re.sub(r"</?answer>", "", clean).strip()
    return clean


def format_ma_grpo_prompt(premise: str, target_len: int) -> str:
    """Format input prompt according to the MA-GRPO Writing Agent specification."""
    # Scene breakdown word budget heuristics
    s1_words = int(target_len * 0.25)
    s2_words = int(target_len * 0.45)
    s3_words = target_len - s1_words - s2_words

    prompt = f"""You are an elite literary novelist. Write the complete narrative prose for this story.
Total Target Length: Approximately {target_len} words.

[CONTEXT SUMMARY]:
{premise}

[SCENE BREAKDOWN & WORD BUDGETS]:
- Scene 1: Opening exposition and world details (~{s1_words} words)
- Scene 2: Inciting tension and primary dialogue (~{s2_words} words)
- Scene 3: Climax of the chapter goal (~{s3_words} words)

CRITICAL WRITING INSTRUCTIONS:
1. Test-Time Reasoning (Chain-of-Thought):
   You MUST begin your response with a <think>...</think> block containing your step-by-step narrative plan:
   - Scene-by-scene pacing: Plan word distribution to strictly match each scene's target words (~{s1_words}, ~{s2_words}, ~{s3_words} words).
   - Tone & Sensory palette: Outline sensory imagery and avoid literal phrasing from the context summary.
2. Story Prose:
   Immediately after </think>, output ONLY the final narrative prose.
   - Seamlessly flow between scenes according to the budgets.
   - DO NOT include scene labels, section titles, author notes, or meta-commentary outside <think>.
Begin with <think> now:"""
    return prompt


def run_benchmark(
    checkpoint_dir: str,
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    num_samples: int = 3,
    output_file: str = "result_ma_grpo_ckpt100.csv",
    model_name_tag: Optional[str] = None,
    max_tokens: int = 2400,
    temperature: float = 0.75,
    top_p: float = 0.92,
):
    ckpt_path = resolve_checkpoint_dir(checkpoint_dir)
    out_path = resolve_output_path(output_file)

    if model_name_tag is None:
        ckpt_name = ckpt_path.name
        model_name_tag = f"MA-GRPO-Qwen2.5-7B-{ckpt_name}"

    print("=" * 90)
    print("BENCHMARK MA-GRPO (LONGWRITER FORMAT)")
    print(f"Model Tag:    {model_name_tag}")
    print(f"Base Model:   {base_model_name}")
    print(f"Checkpoint:   {ckpt_path}")
    print(f"Num Samples:  {num_samples}")
    print(f"Output File:  {out_path}")
    print("=" * 90)

    # 1. Device and Quantization Setup
    use_cuda = torch.cuda.is_available()
    print(f"CUDA Available: {use_cuda}")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f" -> Device: {gpu_name} | VRAM: {gpu_mem:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    # Load 4-bit NF4 if CUDA is available, else fallback
    quant_config = None
    if use_cuda:
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print("[Quantization] Using 4-bit NF4 Quantization (BitsAndBytes).")
        except Exception as e:
            print(f"[Quantization Warning] Failed to initialize BitsAndBytes: {e}. Falling back to default float16.")

    print(f"Loading Tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Base Model {base_model_name}...")
    load_kwargs: Dict[str, Any] = {
        "device_map": "auto" if use_cuda else None,
        "trust_remote_code": True,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif use_cuda:
        load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)

    # 2. Attach PEFT LoRA Adapter
    adapter_file_bin = ckpt_path / "adapter_model.bin"
    adapter_file_safetensors = ckpt_path / "adapter_model.safetensors"

    if adapter_file_safetensors.exists() or adapter_file_bin.exists():
        print(f"Attaching LoRA Adapter from {ckpt_path}...")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    else:
        print(f"[Warning] No adapter_model found in {ckpt_path}. Evaluating Base Model zero-shot.")
        model = base_model

    model.eval()
    print("[Model Ready] Model loaded successfully.")

    # 3. Process Benchmark Samples
    prompts_to_run = DEFAULT_BENCHMARK_PROMPTS[:num_samples]
    results_records: List[Dict[str, Any]] = []

    for idx, item in enumerate(prompts_to_run, 1):
        prompt_id = item["id"]
        premise = item["prompt"]
        target_len = item["target_length"]

        full_prompt = format_ma_grpo_prompt(premise, target_len)

        print("\n" + "-" * 80)
        print(f"[{idx}/{num_samples}] PROMPT: {prompt_id} | TARGET: {target_len} words")
        print(f"Prompt snippet: {premise[:100]}...")

        # Encode input
        encoded = tokenizer(full_prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device)

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        start_time = time.time()
        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed_time = round(time.time() - start_time, 2)

        # Decode newly generated tokens
        new_tokens = output_tokens[0][input_ids.shape[1] :]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Clean prose (exclude CoT <think>)
        clean_text = clean_story_prose(raw_output)
        actual_len = len(clean_text.split())

        # Measure peak VRAM
        if use_cuda:
            peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
        else:
            peak_vram = 0.0

        # Metrics calculation
        sl_score = calculate_length_score(actual_len, target_len)
        dist3 = calculate_distinct_3(clean_text)
        dist4 = calculate_distinct_4(clean_text)
        speed = round(actual_len / elapsed_time, 2) if elapsed_time > 0 else 0.0

        print(
            f"Done in {elapsed_time}s | Actual Words: {actual_len} (Target: {target_len}) | "
            f"Sl: {sl_score}% | Dist-3: {dist3}% | Dist-4: {dist4}% | VRAM: {peak_vram} GB"
        )

        record = {
            "Model": model_name_tag,
            "Prompt_ID": prompt_id,
            "Target_Len": target_len,
            "Actual_Len": actual_len,
            "Length_Score_Sl (%)": sl_score,
            "Distinct-3 (%)": dist3,
            "Distinct-4 (%)": dist4,
            "Latency (s)": elapsed_time,
            "Peak_VRAM (GB)": peak_vram,
            "Speed (words/s)": speed,
            "Prompt_Text": premise,
            "Raw_Output": raw_output,
        }
        results_records.append(record)

    # 4. Save to CSV matching result_longwriter_llama.csv
    df = pd.DataFrame(results_records)
    cols = [
        "Model",
        "Prompt_ID",
        "Target_Len",
        "Actual_Len",
        "Length_Score_Sl (%)",
        "Distinct-3 (%)",
        "Distinct-4 (%)",
        "Latency (s)",
        "Peak_VRAM (GB)",
        "Speed (words/s)",
        "Prompt_Text",
        "Raw_Output",
    ]
    df = df[cols]
    df.to_csv(out_path, index=False, encoding="utf-8")
    print("\n" + "=" * 90)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Output saved to: {out_path.resolve()}")
    print("=" * 90)

    summary_cols = ["Target_Len", "Actual_Len", "Length_Score_Sl (%)", "Distinct-3 (%)", "Distinct-4 (%)", "Latency (s)", "Speed (words/s)"]
    print("\nChi tiết kết quả từng mẫu:")
    print(df[["Prompt_ID"] + summary_cols].to_string(index=False))

    print("\nTrung bình (Mean Metrics):")
    print(df[summary_cols].mean().round(2).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MA-GRPO benchmark matching LongWriter CSV format.")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint directory (checkpoint-100 or checkpoint-50)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace base model ID")
    parser.add_argument("--samples", type=int, default=3, help="Number of benchmark samples to evaluate (1 to 5)")
    parser.add_argument("--output", default="result_ma_grpo_ckpt100.csv", help="Path to save output CSV")
    parser.add_argument("--model-tag", default=None, help="Custom name tag for Model column in CSV")
    parser.add_argument("--max-tokens", type=int, default=2400, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature")
    args = parser.parse_args()

    run_benchmark(
        checkpoint_dir=args.checkpoint,
        base_model_name=args.base_model,
        num_samples=args.samples,
        output_file=args.output,
        model_name_tag=args.model_tag,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
