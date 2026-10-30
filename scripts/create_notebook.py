import json
from pathlib import Path

nb = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "A100",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    })

def add_code(code):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True)
    })

add_md("""# 🚀 MA-GRPO: Pilot Dataset Generation (100–200 Chapter Prompts)
### High-Precision Multi-Agent Narrative Planning & Two-Tier Context Coordination
**Target Hardware:** NVIDIA A100 (40GB / 80GB) via **vLLM** Engine  
**Target Model:** `Qwen/Qwen2.5-72B-Instruct-AWQ` (or `Qwen/Qwen2.5-72B-Instruct` / `Qwen/Qwen2.5-32B-Instruct`)

---
### 📌 Research Workflow Overview
This notebook executes the **Stage 1 (Macro Planning)** and **Stage 2 (Context Coordination)** pipeline to produce a pristine pilot dataset of **100–200 chapter-level prompts** across 10–20 diverse speculative premises before scaling up to MA-Plan-10K:
1. **Stage 1 (Macro Outline & Beat Decomposition)**:
   - `OutlineAgent`: Extracts character entities and constructs an Event Graph ($E_1, \\dots, E_K$). Strict dataset mode rejects hardcoded generic fallbacks.
   - `SubEventAgent`: Decomposes macro events into DAG-sanitized narrative beats ($E_1.1, E_1.2, \\dots$) with zero cyclic dependencies.
   - `PlanCriticAgent`: Audits all 5 research criteria: **Event Coverage**, **Temporal Consistency**, **Character Consistency**, **Word Budget Exactness**, and **Event Ordering**.
2. **Stage 2 (Two-Tier Context Coordination)**:
   - `CoordinatorAgent`: Compresses story history into a **Two-Tier Memory Model** (`long_term_memory` + `short_term_memory`).
   - `ContextGateValidator`: Audits candidate context against 5 source inputs using **Negation Contradiction Detection** and **Entity Grounding**. If $V_s < 0.80$, triggers non-generic structured state fallback.
   - **Quarantine Queue**: Segregates invalid plans into `quarantine_failures.jsonl` with full diagnostic issues to preserve dataset integrity.
""")

add_code("""# Cell 1: Check GPU & Environment
!nvidia-smi

# Install high-throughput vLLM and required NLP packages
!pip install -q vllm sentence-transformers pandas tabulate matplotlib
""")

add_code("""# Cell 2: Setup MA-GRPO Codebase
import os
import sys
from pathlib import Path

# Clone repository or set local path
REPO_URL = "https://github.com/vntk2102/MA-GRPO.git"
REPO_DIR = "/content/MA-GRPO"

if not os.path.exists(REPO_DIR):
    !git clone $REPO_URL
else:
    !cd $REPO_DIR && git pull

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print(f"MA-GRPO codebase imported from: {REPO_DIR}")
""")

add_code("""# Cell 3: Initialize vLLM with Qwen2.5-72B
import torch
from vllm import LLM, SamplingParams

# For A100 80GB or 40GB, 4-bit AWQ fits in ~38GB VRAM with extreme generation speed
# You can also switch to "Qwen/Qwen2.5-32B-Instruct" or "Qwen/Qwen2.5-72B-Instruct" (bfloat16)
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-AWQ"

print(f"Loading {MODEL_NAME} on A100 GPU via vLLM PagedAttention...")
llm = LLM(
    model=MODEL_NAME,
    dtype="auto",
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048,
    repetition_penalty=1.05
)

def llm_generate_fn(system_prompt: str, user_prompt: str) -> str:
    \"\"\"Synchronous prompt generation function for Planning and Coordinator agents.\"\"\"
    full_prompt = (
        f"<|im_start|>system\\n{system_prompt}<|im_end|>\\n"
        f"<|im_start|>user\\n{user_prompt}<|im_end|>\\n"
        f"<|im_start|>assistant\\n"
    )
    outputs = llm.generate([full_prompt], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text.strip()

print("vLLM Engine Initialized Successfully!")
""")

add_code("""# Cell 4: Curated 20 High-Concept Story Premises
from src.dataset_generator import CANONICAL_20_PREMISES

print(f"Total Pre-curated Premises Available: {len(CANONICAL_20_PREMISES)}")
for idx, p in enumerate(CANONICAL_20_PREMISES[:5], 1):
    print(f"[{idx}] ({p['genre']}): {p['premise'][:100]}...")
""")

add_code("""# Cell 5: Run Pilot Generation Pipeline (10-20 Premises -> 100-200 Chapters)
from src.dataset_generator import PilotDatasetGenerator

# Choose number of premises (e.g. 15 premises * 10 chapters = 150 chapter prompts)
NUM_PREMISES_TO_RUN = 15
OUTPUT_DIR = "/content/output_pilot_dataset"

# Initialize Pilot Generator with vLLM engine and strict research validation
generator = PilotDatasetGenerator(
    llm_generate_fn=llm_generate_fn,
    strict_mode=True,            # Refuses hardcoded templates; segregates failures
    validator_threshold=0.80     # V_s threshold for Context Gate
)

print(f"Executing Pilot Generation for {NUM_PREMISES_TO_RUN} premises...")
quality_report = generator.generate_pilot_dataset(
    premises=CANONICAL_20_PREMISES,
    num_premises=NUM_PREMISES_TO_RUN,
    output_dir=OUTPUT_DIR,
    mock_simulate_prose=True
)
""")

add_code("""# Cell 6: Research Quality Report & Metric Visualizations
import json
import pandas as pd
import matplotlib.pyplot as plt

report_path = Path(OUTPUT_DIR) / "pilot_quality_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

print("=" * 60)
print("PILOT DATASET RESEARCH AUDIT SUMMARY")
print("=" * 60)
for k, v in report.items():
    if k != "artifacts":
        print(f"{k:<35}: {v}")

# Load and inspect generated samples
samples_file = report["artifacts"]["pilot_samples_file"]
samples = []
with open(samples_file, "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

df_samples = pd.DataFrame([
    {
        "sample_id": s["sample_id"],
        "premise_id": s["premise_id"],
        "genre": s["genre"],
        "chapter_index": s["chapter_index"],
        "target_words": s["target_words"],
        "scenes_count": len(s["scenes"]),
        "faithfulness_score": s["validation_metadata"].get("faithfulness_score", 1.0),
        "completeness_score": s["validation_metadata"].get("completeness_score", 1.0),
        "composite_vs": s["validation_metadata"].get("composite_score", 1.0),
        "is_valid": s["validation_metadata"].get("is_valid", True)
    }
    for s in samples
])

print("\\n--- First 10 Generated Chapter Prompts ---")
print(df_samples.head(10).to_markdown())

# Plot metric distributions
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df_samples["composite_vs"], bins=15, color="teal", edgecolor="black")
plt.title("Distribution of Context Validator Scores (V_s)")
plt.xlabel("V_s Score")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(df_samples["target_words"], bins=10, color="coral", edgecolor="black")
plt.title("Distribution of Chapter Target Words")
plt.xlabel("Target Words")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
""")

add_code("""# Cell 7: Inspect Sample Two-Tier Memory & Download Artifacts
if samples:
    print("=" * 80)
    print("DETAILED INSPECTION OF SAMPLE CHAPTER PROMPT (Two-Tier Memory Model):")
    print("=" * 80)
    sample_ch2 = samples[1] if len(samples) > 1 else samples[0]
    print(f"Sample ID: {sample_ch2['sample_id']} | Title: {sample_ch2['chapter_title']}")
    print(f"Target Words: {sample_ch2['target_words']} | Active Characters: {sample_ch2['characters_active']}")
    print("\\n[Scene Outline & Budgets]:")
    for so in sample_ch2["scene_outline"]:
        print(f"  {so}")
    
    print("\\n[Two-Tier Long-Term Memory]:")
    print(json.dumps(sample_ch2["context_state"].get("long_term_memory", {}), indent=2))
    
    print("\\n[Two-Tier Short-Term Memory]:")
    print(json.dumps(sample_ch2["context_state"].get("short_term_memory", {}), indent=2))
    
    print("\\n[Validation Metadata]:")
    print(json.dumps(sample_ch2["validation_metadata"], indent=2))

# Archive artifacts for download
!zip -q -r /content/pilot_dataset_artifacts.zip $OUTPUT_DIR
print("\\nArtifacts packaged to: /content/pilot_dataset_artifacts.zip")
""")

out_paths = [
    Path("c:/Users/ASUS/AI_LLMs/Long_text/generate_pilot_dataset_colab.ipynb"),
    Path("c:/Users/ASUS/AI_LLMs/Long_text/MA-GRPO/generate_pilot_dataset_colab.ipynb"),
]

for p in out_paths:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Wrote notebook to: {p}")
