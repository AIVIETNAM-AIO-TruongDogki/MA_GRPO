import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NB = PROJECT_ROOT / "notebooks" / "benchmark_qwen.ipynb"

nb = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
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
add_md("""# 📊 MA-GRPO: Benchmark & Inference Evaluation (Base Qwen vs Checkpoint)
### Đánh giá định lượng & Trực quan hóa văn phong của Qwen2.5-7B
**Mục tiêu**:
1. **Kiểm tra Qwen Chưa Train (Base Zero-Shot)**: Đánh giá xem khi chưa có RL, mô hình đạt bao nhiêu từ, độ tuân thủ độ dài $S_l$, độ phong phú vốn từ Distinct-3/4, và Perplexity (PPL).
2. **Đối sánh Trực diện với Checkpoint (đã train)**: So sánh trước và sau khi được tối ưu bằng 3 hàm thưởng nội tại ($R_{\\text{length}}, R_{\\text{coherence}}, R_{\\text{quality}}$).
3. **So sánh với SOTA**: Đối chiếu cùng StoryWriter, Agents' Room, LongWriter-Zero-32B.
*Lưu ý: Notebook này thuần túy phục vụ Inference & Đánh giá Benchmark, không chạy vòng lặp Training.*
""")

# Cell 1: Environment & GPU Check
add_md("## Bước 1: Kiểm tra phần cứng GPU")
add_code("""!nvidia-smi

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print(f"BF16 Native Support: {torch.cuda.is_bf16_supported()}")
""")

# Cell 2: Google Drive
add_md("""## Bước 2: Kết nối Google Drive (Để đọc Checkpoint nếu có)""")
add_code("""from google.colab import drive
import os

drive.mount('/content/drive')
CHECKPOINT_ROOT = "/content/drive/MyDrive/MA-GRPO-Checkpoints/qwen-7b-final"
print(f"Đường dẫn Checkpoint: {CHECKPOINT_ROOT}")
""")

# Cell 3: Clone repo
add_md("""## Bước 3: Nạp mã nguồn & Dataset `MA-GRPO`""")
add_code("""import os
import sys

REPO_URL = "https://github.com/AIVIETNAM-AIO-TruongDogki/MA_GRPO.git"
REPO_DIR = "/content/MA-GRPO"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    print(f"Thư mục {REPO_DIR} đã tồn tại. Đang kéo bản cập nhật mới nhất...")
    !cd {REPO_DIR} && git pull

%cd {REPO_DIR}
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

!git log -n 1 --oneline
""")

# Cell 4: Install packages
add_md("""## Bước 4: Cài đặt các thư viện cần thiết cho Inference & Benchmark""")
add_code("""!pip install -q --upgrade pip
!pip install -q transformers>=4.45.0 accelerate>=0.33.0 peft>=0.12.0 bitsandbytes>=0.43.0
!pip install -q sentence-transformers>=3.0.0 nltk rouge-score pandas scipy tabulate
print("✅ Cài đặt thư viện hoàn tất!")
""")

# Cell 5: Benchmark Qwen Chưa Train (Zero-Shot)
add_md("""## Bước 5: Benchmark & Inference QWEN CHƯA TRAIN (Base Model Zero-Shot)
- Nạp `Qwen/Qwen2.5-7B-Instruct` nguyên bản (chưa qua fine-tuning RL).
- Sinh văn bản trên mẫu từ `data/chapters.jsonl` (Target: 1.200 từ).
- Đo lường và in ra:
  - Số lượng từ thực tế và trích đoạn văn bản.
  - Điểm tuân thủ độ dài $S_l$ (%), độ đa dạng 3-gram Distinct-3 (%), độ mượt mà PPL.
  - Bảng so sánh với các baseline SOTA.""")
add_code("""# Chạy đánh giá Base Model Chưa Train
!python scripts/compare_base_vs_checkpoint.py \\
    --checkpoint none \\
    --base-model Qwen/Qwen2.5-7B-Instruct \\
    --dataset data/chapters.jsonl \\
    --samples 1 \\
    --output /content/benchmark_qwen_base_zeroshot.csv
""")

# Cell 6: Compare Base vs Checkpoint-10
add_md("""## Bước 6: Đối sánh Trực diện: Qwen Chưa Train vs Qwen Checkpoint (Khi đã có Checkpoint)
- Tự động phát hiện checkpoint (ví dụ `checkpoint-10`, `checkpoint-20`...) trong Google Drive.
- Sinh trên cùng 1 prompt mẫu để so sánh:
  - Xem văn bản của Checkpoint-10 dài ra bao nhiêu từ so với Base Model.
  - Bảng so sánh chỉ số $S_l$, Distinct-3, Distinct-4, PPL đối đầu trực tiếp.""")
add_code("""import os
import glob
import re

# 1. Tìm checkpoint-10 hoặc checkpoint mới nhất trong Google Drive
CKPT_DIR = f"{CHECKPOINT_ROOT}/checkpoint-10"

if not os.path.exists(CKPT_DIR):
    checkpoint_dirs = glob.glob(os.path.join(CHECKPOINT_ROOT, "checkpoint-*"))
    if checkpoint_dirs:
        CKPT_DIR = sorted(
            checkpoint_dirs,
            key=lambda x: int(re.search(r"checkpoint-(\\d+)", x).group(1)) if re.search(r"checkpoint-(\\d+)", x) else 0
        )[0]

print(f"🎯 Thực hiện đối sánh Base Model vs Checkpoint: {CKPT_DIR}")

# 2. Chạy đối sánh trực diện
!python scripts/compare_base_vs_checkpoint.py \\
    --checkpoint {CKPT_DIR} \\
    --base-model Qwen/Qwen2.5-7B-Instruct \\
    --dataset data/chapters.jsonl \\
    --samples 1 \\
    --output /content/drive/MyDrive/MA-GRPO-Checkpoints/compare_base_vs_checkpoint.csv
""")

OUTPUT_NB.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully created: {OUTPUT_NB}")
