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
### Đánh giá định lượng & Trực quan hóa văn phong của Qwen2.5-7B trên tập 5 Prompt Chuẩn của Benchmark 8 Models

**Mục tiêu**:
1. **Kiểm tra Qwen Chưa Train (Base Zero-Shot)**: Đánh giá xem khi chưa qua fine-tuning RL, mô hình đạt bao nhiêu từ, độ tuân thủ độ dài $S_l$, độ phong phú vốn từ Distinct-3/4, và Perplexity (PPL).
2. **Đúng 5 Prompt Benchmark Kinh Điển (tell-me-a-story-test)**:
   - `example_000`: Thần Horus & bình canopic (Target: 1.350 từ)
   - `example_001`: Thần Apollo & lời nguyền tiên tri (Target: 1.575 từ)
   - `example_002`: Nhân vật thần thoại Hy Lạp trong quán cà phê (Target: 1.900 từ)
   - `example_003`: Nghi thức triệu hồi nữ thần Freya (Target: 1.600 từ)
   - `example_004`: Nhạc sĩ Orpheus tại New York (Target: 900 từ)
3. **Đối sánh Trực diện với Checkpoint (đã train)**: So sánh trước và sau khi được tối ưu bằng 3 hàm thưởng nội tại ($R_{\\text{length}}, R_{\\text{coherence}}, R_{\\text{quality}}$).
4. **Đối chiếu Trực tiếp với Bảng 8 SOTA Baselines**: StoryWriter, Agents' Room, Self-Lengthen, LongWriter-Zero-32B, MA-GRPO-50, MA-GRPO-100.

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
add_md("""## Bước 2: Kết nối Google Drive (Để nạp Checkpoint đã train nếu có)""")
add_code("""from google.colab import drive
import os

drive.mount('/content/drive')
CHECKPOINT_ROOT = "/content/drive/MyDrive/MA-GRPO-Checkpoints/qwen-7b-final"
print(f"Thư mục Checkpoints trên Drive: {CHECKPOINT_ROOT}")
""")

# Cell 3: Clone repo
add_md("""## Bước 3: Nạp mã nguồn `MA-GRPO`""")
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

# Cell 5: Inspect Canonical 5 Benchmark Prompts
add_md("""## Bước 5: Xem trước 5 Prompt Chuẩn của Benchmark 8 Models
Kiểm tra nội dung 5 prompt kinh điển (`example_000` đến `example_004`) từ tập dữ liệu `tell-me-a-story-test.jsonl`:""")
add_code("""from scripts.compare_base_vs_checkpoint import load_test_prompts

# Nạp 5 prompt benchmark chuẩn
benchmark_prompts = load_test_prompts("8models", num_samples=5)
print(f"✅ Đã nạp thành công {len(benchmark_prompts)} prompt benchmark chuẩn:\\n")
for p in benchmark_prompts:
    print(f"📌 [{p['id']}] - Target Length: {p['target_length']} words")
    print(f"   Premise: {p['prompt'][:140]}...\\n")
""")

# Cell 6: Benchmark Qwen Chưa Train (Zero-Shot) on Canonical Prompts
add_md("""## Bước 6: Benchmark & Inference QWEN CHƯA TRAIN (Base Model Zero-Shot)
- Nạp `Qwen/Qwen2.5-7B-Instruct` nguyên bản (chưa qua fine-tuning RL).
- Sinh văn bản trên **5 prompt benchmark chuẩn** (`example_000` đến `example_004`).
  *(Có thể chọn `NUM_SAMPLES = 5` để chạy đủ 5 prompt hoặc `NUM_SAMPLES = 3` để khớp đúng 3 prompt đầu có Target trung bình 1.608,3 từ như bảng 8 models)*.
- Đo lường và in ra:
  - Độ dài thực tế và trích đoạn mở đầu/kết thúc câu chuyện.
  - Kiểm tra mô hình có sinh `<think>` test-time planning hay không.
  - Điểm tuân thủ độ dài $S_l$ (%), độ đa dạng 3-gram/4-gram Distinct-3/4 (%), độ mượt mà Perplexity (PPL ↓).
  - Bảng tổng hợp đối sánh trực tiếp với 8 mô hình SOTA.""")
add_code("""# Cấu hình số mẫu muốn benchmark (5 mẫu đầy đủ hoặc 3 mẫu chuẩn)
NUM_SAMPLES = 5

# Chạy đánh giá Base Model Chưa Train trên 5 prompt chuẩn
!python scripts/compare_base_vs_checkpoint.py \\
    --checkpoint none \\
    --base-model Qwen/Qwen2.5-7B-Instruct \\
    --dataset 8models \\
    --samples {NUM_SAMPLES} \\
    --output /content/benchmark_qwen_base_zeroshot.csv
""")

# Cell 7: Compare Base vs Checkpoint-10 on Canonical Prompts
add_md("""## Bước 7: Đối sánh Trực diện: Qwen Chưa Train vs Qwen Checkpoint (Khi đã có Checkpoint)
- Tự động phát hiện checkpoint (ví dụ `checkpoint-10`, `checkpoint-20`...) trong Google Drive.
- Sinh trên **cùng các prompt benchmark chuẩn** để so sánh trực diện:
  - Xem văn bản của Checkpoint dài ra bao nhiêu từ so với Base Model.
  - So sánh tư duy lập dàn ý `<think>` giữa 2 phiên bản.
  - Bảng đối đầu trực tiếp chỉ số $S_l$, Distinct-3, Distinct-4, PPL, BLEU-4, ROUGE-L so với 8 mô hình SOTA.""")
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
        )[-1]

print(f"🎯 Thực hiện đối sánh Base Model vs Checkpoint: {CKPT_DIR}")

# 2. Chạy đối sánh trực diện trên cùng tập 5 prompt chuẩn
!python scripts/compare_base_vs_checkpoint.py \\
    --checkpoint {CKPT_DIR} \\
    --base-model Qwen/Qwen2.5-7B-Instruct \\
    --dataset 8models \\
    --samples {NUM_SAMPLES} \\
    --output /content/drive/MyDrive/MA-GRPO-Checkpoints/compare_base_vs_checkpoint.csv
""")

OUTPUT_NB.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully generated: {OUTPUT_NB}")
