import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NB = PROJECT_ROOT / "notebooks" / "train_qwen_grpo.ipynb"

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


# Header
add_md("""# 🚀 MA-GRPO (v5): Train Qwen2.5-7B Writing Agent via QLoRA & Group Relative Policy Optimization
### Offline RL Training Pipeline for Google Colab Pro (NVIDIA A100 40GB / 80GB)
**Research Proposal Target**: `ma-grpo-proposal-v5.md`  
**Base Policy**: `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 QLoRA, $r=32, \\alpha=64$, all linear projection layers)  
**Dataset**: Multi-Agent Pre-computed Chapter Blueprints (`data/pilot_chapters.jsonl` / `data/chapters.jsonl`)  
**Intrinsic Rewards**: Refined 3-Component Reward Architecture ($R_{\\text{length}}, R_{\\text{coherence}}, R_{\\text{quality}}$)  
**Hardware Efficiency**: Single GPU A100 handles both Policy rollouts ($G=4$) and Reward Evaluators in ~18–22 GB VRAM.

---
### 📌 Key Features in this Notebook
1. **Auto-Detect & Auto-Resume**: Automatically finds the latest saved checkpoint in Google Drive (`checkpoint-50`, `checkpoint-100`...) and continues training seamlessly without restarting from step 0.
2. **Buffer-Zone Length Reward ($R_{\\text{length}}$)**: LongWriter-Zero style $[0.85L, 1.15L]$ buffer zone with linear drop below $0.85L$ and gentle drop up to $L_{\\max} = 2.0L$.
3. **Scoped Anti-Repetition ($R_{\\text{coherence}}$)**: Transition cosine similarity + non-entity 3-gram set subtraction to prevent vocabulary collapse.
4. **Target-Centered Burstiness ($R_{\\text{quality}}$)**: Gaussian pacing centered around $B_{\\text{target}} \\approx 11.0$ with Sigmoid NLL fluency and Shannon entropy.
5. **SOTA Benchmark & Comparison**: Directly evaluates the trained checkpoint and prints the comparison table against StoryWriter, Agents' Room, and LongWriter-Zero.
""")

# Cell 1: Environment & GPU Check
add_md("## Bước 1: Kiểm tra phần cứng GPU A100 & Môi trường PyTorch")
add_code("""!nvidia-smi

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16_ok = torch.cuda.is_bf16_supported()
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print(f"BF16 Native Support: {bf16_ok}")
    if "A100" not in gpu_name:
        print("⚠️ Lưu ý: Bạn chưa chọn GPU A100. Hãy vào 'Thời gian chạy' -> 'Thay đổi loại thời gian chạy' -> chọn A100 để có tốc độ cao nhất.")
    else:
        print("✅ GPU A100 sẵn sàng cho Qwen2.5-7B GRPO training!")
""")

# Cell 2: Google Drive
add_md("""## Bước 2: Kết nối Google Drive để lưu Checkpoints
*Toàn bộ checkpoint (mỗi 50 steps) sẽ được tự động lưu vào Google Drive để tránh mất mát khi ngắt kết nối.*""")
add_code("""from google.colab import drive
import os

drive.mount('/content/drive')
CHECKPOINT_ROOT = "/content/drive/MyDrive/MA-GRPO-Checkpoints/qwen-7b-final"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
print(f"Checkpoint directory ready: {CHECKPOINT_ROOT}")
""")

# Cell 3: Clone repo
add_md("""## Bước 3: Nạp mã nguồn & Dataset `MA-GRPO`
*Clone trực tiếp từ GitHub repository.*""")
add_code("""import os
import sys
import getpass

REPO_URL = "github.com/AIVIETNAM-AIO-TruongDogki/MA_GRPO.git"
REPO_DIR = "/content/MA-GRPO"

# 1. Lấy token từ Colab Secrets (nếu có) hoặc nhập trực tiếp nếu repo là private
try:
    from google.colab import userdata
    token = userdata.get('GH_TOKEN')
except Exception:
    token = None

if not os.path.exists(REPO_DIR):
    if not token:
        try:
            # Thử clone public trước
            !git clone https://{REPO_URL} {REPO_DIR}
        except Exception:
            token = getpass.getpass("Nhập GitHub Personal Access Token (nếu repo private): ")
            !git clone https://{token}@{REPO_URL} {REPO_DIR}
    else:
        !git clone https://{token}@{REPO_URL} {REPO_DIR}
else:
    print(f"Thư mục {REPO_DIR} đã tồn tại. Đang cập nhật mã nguồn mới nhất...")
    !cd {REPO_DIR} && git pull

%cd {REPO_DIR}
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

!git log -n 1 --oneline
""")

# Cell 4: Install packages
add_md("""## Bước 4: Cài đặt các thư viện cần thiết (Tối ưu bằng PyTorch SDPA)
*PyTorch 2.x SDPA tự động kích hoạt FlashAttention-2 bản địa trên GPU A100 mà không cần biên dịch thủ công.*""")
add_code("""!pip install -q --upgrade pip
!pip install -q transformers>=4.45.0 accelerate>=0.33.0 peft>=0.12.0 trl>=0.15.0 datasets>=2.20.0 bitsandbytes>=0.43.0
!pip install -q sentence-transformers>=3.0.0 nltk rouge-score pydantic pyyaml pandas scipy tabulate
print("✅ Cài đặt thư viện hoàn tất!")
""")

# Cell 5: Inspect Config and Dataset
add_md("""## Bước 5: Kiểm tra Cấu hình `colab_a100.yaml` & Dataset""")
add_code("""import yaml
import json
from pathlib import Path

# 1. Đọc config YAML
with open("configs/colab_a100.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("=== MA-GRPO CONFIGURATION (colab_a100.yaml) ===")
print(f"Model: {config['model']['name']}")
print(f"LoRA Rank: {config['lora']['r']} | Alpha: {config['lora']['lora_alpha']}")
print(f"Learning Rate: {config['training']['learning_rate']} | KL Beta: {config['training']['kl_beta']}")
print(f"Rollout Group Size: {config['rollout']['group_size']} | Max Tokens: {config['rollout']['max_tokens']}")
print(f"Reward Weights: Length={config['rewards']['weights']['w_length']}, Coherence={config['rewards']['weights']['w_coherence']}, Quality={config['rewards']['weights']['w_quality']}")

# 2. Kiểm tra danh sách Dataset có sẵn
dataset_options = [
    ("data/pilot_chapters.jsonl", "Dataset Pilot từ Stage 1 & Stage 2 Multi-Agent (114 chapter prompts)"),
    ("data/train_combined.jsonl", "Dataset tổng hợp kết hợp (169 chapter prompts)"),
    ("data/chapters.jsonl", "Dataset chuẩn từ tell-me-a-story (55 chapter prompts)")
]

TRAIN_DATASET = "data/pilot_chapters.jsonl"
if not Path(TRAIN_DATASET).exists():
    TRAIN_DATASET = "data/chapters.jsonl"

print(f"\\n=== SELECTED DATASET: {TRAIN_DATASET} ===")
with open(TRAIN_DATASET, "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())
print(f"Target Words: {sample.get('target_words')}")
print(f"Prompt Preview:\\n{sample.get('prompt')[:350]}...")
""")

# Cell 6: Train Script
add_md("""## Bước 6: Khởi chạy / Tiếp tục Huấn luyện Qwen2.5-7B Writing Agent (Auto-Resume Checkpoint)
- **Auto-Resume**: Tự động phát hiện checkpoint cao nhất trong Drive (`checkpoint-10`, `checkpoint-20`, `checkpoint-50`...).
- **Checkpoint mỗi 10 steps**: Định kỳ lưu checkpoint sau mỗi 10 steps (`--save-steps 10`).
- **Rollout $G=4$**: Sinh 4 ứng viên song song cho mỗi prompt.
- **Hàm thưởng nội tại 3 trục**: Tự động đánh giá bằng toán học/Python thuần trên `cuda:0` mà không cần LLM judge ngoài.
- **Thời gian dự kiến**: ~1.2 – 1.8 giờ cho 500 steps trên GPU A100.""")
add_code("""import os
import glob
import re

# 1. Tự động quét checkpoint cao nhất trong Google Drive
checkpoint_dirs = glob.glob(os.path.join(CHECKPOINT_ROOT, "checkpoint-*"))
latest_checkpoint = None
latest_step = 0

for cp in checkpoint_dirs:
    match = re.search(r"checkpoint-(\\d+)", os.path.basename(cp))
    if match:
        step = int(match.group(1))
        if step > latest_step:
            latest_step = step
            latest_checkpoint = cp

resume_arg = ""
if latest_checkpoint:
    print(f"🎯 Đã tìm thấy checkpoint cao nhất: {latest_checkpoint} (Step {latest_step})")
    print(f"🚀 Sẽ tự động nối tiếp huấn luyện từ Step {latest_step}...")
    resume_arg = f"--resume-from-checkpoint {latest_checkpoint}"
else:
    print("ℹ️ Chưa có checkpoint nào trước đó. Sẽ bắt đầu huấn luyện mới từ Step 0.")

# 2. Thiết lập mục tiêu số bước huấn luyện (Thử nghiệm ban đầu: 100 hoặc 500 steps)
TARGET_MAX_STEPS = 500

cmd = f\"\"\"python scripts/train.py \\
    --config configs/colab_a100.yaml \\
    --dataset {TRAIN_DATASET} \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --reward-device cuda:0 \\
    --output-dir {CHECKPOINT_ROOT} \\
    --max-steps {TARGET_MAX_STEPS} \\
    --save-steps 10 \\
    --batch-size 1 \\
    --gradient-accumulation 4 \\
    --group-size 4 \\
    --max-completion-tokens 2200 {resume_arg}\"\"\"

print(f"\\n[Command]\\n{cmd}\\n")
!{cmd}
""")

# Cell 7: Benchmark Evaluation
add_md("""## Bước 7: Đánh giá Benchmark & So sánh với các mô hình SOTA
- Tự động nhận diện checkpoint mới nhất vừa hoàn thành.
- Đánh giá trên các tiêu chí cốt lõi của **Proposal v5**:
  - $S_l$ (Length Compliance %)
  - Distinct-3 & Distinct-4 (%)
  - Perplexity (PPL)
  - BLEU-4 & ROUGE-L
- Xuất bảng so sánh đối đầu trực tiếp với **StoryWriter, Agents' Room, LongWriter-Zero**.""")
add_code("""import os
import glob
import re
import pandas as pd

# 1. Tìm checkpoint cao nhất vừa train xong
checkpoint_dirs = glob.glob(os.path.join(CHECKPOINT_ROOT, "checkpoint-*"))
latest_cp = CHECKPOINT_ROOT
if checkpoint_dirs:
    latest_cp = max(
        checkpoint_dirs,
        key=lambda x: int(re.search(r"checkpoint-(\\d+)", x).group(1)) if re.search(r"checkpoint-(\\d+)", x) else 0
    )

print(f"🎯 Đang đánh giá Benchmark trên Checkpoint: {latest_cp}")

BENCHMARK_OUTPUT_CSV = "/content/drive/MyDrive/MA-GRPO-Checkpoints/benchmark_comparison_qwen.csv"

# 2. Khởi chạy script đánh giá benchmark
!python scripts/eval_sample_benchmark.py \\
    --checkpoint {latest_cp} \\
    --samples 3 \\
    --output {BENCHMARK_OUTPUT_CSV}

# 3. Hiển thị kết quả đối sánh nếu file đã được ghi
if os.path.exists(BENCHMARK_OUTPUT_CSV):
    print("\\n=== BẢNG ĐỐI SÁNH KẾT QUẢ VỚI CÁC MÔ HÌNH SOTA (Proposal v5) ===")
    df = pd.read_csv(BENCHMARK_OUTPUT_CSV)
    print(df.to_string(index=False))
""")

OUTPUT_NB.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully created: {OUTPUT_NB}")

