"""
Unified Evaluation Metrics & CSV Exporter for the 8-Model Long-Text Benchmark.

Strictly enforces the exact 12-column CSV schema of:
  kaggle_longwriter/output_new/result_longwriter_llama.csv

Columns:
  Model,Prompt_ID,Target_Len,Actual_Len,Length_Score_Sl (%),Distinct-3 (%),Distinct-4 (%),Latency (s),Peak_VRAM (GB),Speed (words/s),Prompt_Text,Raw_Output

Also computes final comparative summary table metrics across all 8 models:
  Length Sl (%), Distinct-3 (%), Distinct-4 (%), PPL ↓, BLEU-4, ROUGE-L, Latency (s), Peak_VRAM (GB), Speed (words/s)
"""

import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Exact 12-column schema matching result_longwriter_llama.csv
RESULT_CSV_COLUMNS = [
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

# Default 5 canonical prompts from tell-me-a-story-test.jsonl
DEFAULT_TEST_PROMPTS = [
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


def extract_target_length(text: str, default_len: int = 1500) -> int:
    """Parse target word count from prompt text (handling commas and ranges)."""
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


def load_benchmark_prompts(dataset_path: Optional[str] = None, num_samples: int = 3) -> List[Dict[str, Any]]:
    """Load test prompts and gold reference stories from tell-me-a-story-test.jsonl."""
    candidates = []
    if dataset_path:
        candidates.append(Path(dataset_path))
    candidates.extend([
        PROJECT_ROOT / "data" / "tell-me-a-story-test.jsonl",
        PROJECT_ROOT.parent / "Dataset" / "tell_me_a_story_data" / "tell-me-a-story-test.jsonl",
        Path("/content/MA-GRPO/data/tell-me-a-story-test.jsonl"),
    ])

    gold_map: Dict[str, str] = {}
    loaded_prompts: List[Dict[str, Any]] = []

    for cand in candidates:
        if cand.exists():
            with open(cand, "r", encoding="utf-8-sig") as f:
                for idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    ex_id = row.get("example_id", f"example_{idx:03d}")
                    inp = row.get("inputs", "").strip().replace("\u00a0", "")
                    ref = row.get("targets", "")
                    gold_map[ex_id] = ref
                    if idx < num_samples:
                        # Match exact canonical target_length for first 5 examples
                        default_target = next(
                            (d["target_length"] for d in DEFAULT_TEST_PROMPTS if d["id"] == ex_id),
                            extract_target_length(inp, 1500),
                        )
                        loaded_prompts.append({
                            "id": ex_id,
                            "prompt": inp,
                            "target_length": default_target,
                            "gold_ref": ref,
                        })
            break

    if not loaded_prompts:
        for item in DEFAULT_TEST_PROMPTS[:num_samples]:
            loaded_prompts.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "target_length": item["target_length"],
                "gold_ref": gold_map.get(item["id"], ""),
            })

    return loaded_prompts


def clean_generated_prose(raw_text: str) -> str:
    """
    Extract pure story prose from model output by removing <think>...</think> CoT
    and unwrapping <answer>...</answer> tags if present.
    """
    if not raw_text:
        return ""
    text = str(raw_text)
    # Remove closed <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # If an unclosed <think> exists, check if <answer> is present
    if "<think>" in text.lower() and "</think>" not in text.lower():
        ans_match = re.search(r"<answer>([\s\S]*?)(?:</answer>|$)", text, flags=re.IGNORECASE)
        if ans_match:
            text = ans_match.group(1)
    # Extract inside <answer>...</answer> if present
    ans_match = re.search(r"<answer>([\s\S]*?)</answer>", text, flags=re.IGNORECASE)
    if ans_match:
        text = ans_match.group(1)
    else:
        text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE)
    return text.strip()


def calculate_length_score(actual_words: int, target_words: int) -> float:
    """
    Exact Length Score Sl (%) formula from kaggle_longwriter/run_longwriter_models.py:
      Sl = 100 * (1 - |actual_words - target_words| / target_words)
    """
    if target_words <= 0:
        return 100.0
    val = 100.0 * (1.0 - abs(actual_words - target_words) / target_words)
    return max(0.0, min(100.0, round(val, 2)))


def calculate_distinct_3(text: str) -> float:
    """Exact Distinct-3 (%) formula from kaggle_longwriter/run_longwriter_models.py."""
    words = text.split()
    if len(words) < 3:
        return 100.0
    ngrams = set(zip(words, words[1:], words[2:]))
    total = len(words) - 2
    return round((len(ngrams) / total) * 100.0, 2)


def calculate_distinct_4(text: str) -> float:
    """Exact Distinct-4 (%) formula from kaggle_longwriter/run_longwriter_models.py."""
    words = text.split()
    if len(words) < 4:
        return 100.0
    ngrams = set(zip(words, words[1:], words[2:], words[3:]))
    total = len(words) - 3
    return round((len(ngrams) / total) * 100.0, 2)


def save_model_result_csv(records: List[Dict[str, Any]], output_csv_path: Path) -> pd.DataFrame:
    """
    Save per-sample model generation records to CSV strictly matching:
      kaggle_longwriter/output_new/result_longwriter_llama.csv
    """
    evaluated_rows = []
    for r in records:
        raw_text = str(r.get("Raw_Output", ""))
        clean_text = clean_generated_prose(raw_text)
        actual_words = len(clean_text.split())
        target_len = int(r.get("Target_Len", 1500))
        latency = round(float(r.get("Latency (s)", 0.0)), 2)
        peak_vram = round(float(r.get("Peak_VRAM (GB)", 0.0)), 2)
        speed = round(actual_words / latency, 2) if latency > 0 else 0.0

        evaluated_rows.append({
            "Model": r.get("Model", "Unknown"),
            "Prompt_ID": r.get("Prompt_ID", ""),
            "Target_Len": target_len,
            "Actual_Len": actual_words,
            "Length_Score_Sl (%)": calculate_length_score(actual_words, target_len),
            "Distinct-3 (%)": calculate_distinct_3(clean_text),
            "Distinct-4 (%)": calculate_distinct_4(clean_text),
            "Latency (s)": latency,
            "Peak_VRAM (GB)": peak_vram,
            "Speed (words/s)": speed,
            "Prompt_Text": r.get("Prompt_Text", ""),
            "Raw_Output": raw_text,
        })

    df = pd.DataFrame(evaluated_rows)
    df = df[RESULT_CSV_COLUMNS]

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"\n✅ [Saved CSV] {output_csv_path} ({len(df)} samples)", flush=True)
    summary_cols = ["Target_Len", "Actual_Len", "Length_Score_Sl (%)", "Distinct-3 (%)", "Distinct-4 (%)", "Latency (s)", "Peak_VRAM (GB)", "Speed (words/s)"]
    print(df[["Prompt_ID"] + summary_cols].to_string(index=False), flush=True)
    print("\n--- Mean Metrics ---", flush=True)
    print(df[summary_cols].mean().round(2).to_string(), flush=True)
    return df


def compute_overlap_metrics(gen_text: str, ref_text: str) -> Dict[str, float]:
    """Compute BLEU-4 and ROUGE-L F1 (%) against human reference story."""
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


class PerplexityScorer:
    """Evaluates fluency Perplexity (PPL ↓) using Qwen/Qwen2.5-0.5B."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\n[PPL Scorer] Loading {self.model_id} for Perplexity evaluation...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.model.eval()

    def unload(self):
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score_text(self, raw_text: str, max_length: int = 2048) -> float:
        clean = clean_generated_prose(raw_text)
        if not clean or len(clean.split()) < 10:
            return 99.0
        if self.model is None:
            self.load()
        try:
            enc = self.tokenizer(clean, return_tensors="pt", truncation=True, max_length=max_length)
            dev = next(self.model.parameters()).device
            input_ids = enc.input_ids.to(dev)
            with torch.no_grad():
                out = self.model(input_ids, labels=input_ids)
                loss = float(out.loss.item())
            return round(float(np.exp(loss)), 2) if loss < 10 else 99.0
        except Exception as e:
            print(f"[PPL Warning]: {e}", flush=True)
            return 0.0
