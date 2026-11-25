#!/usr/bin/env bash
# Standalone Linux Server Runner for MA-GRPO Phase 2 Minimal Orthogonal RLMR (gamma=0.05)
# Tuned for 1x 48GB GPU (NVIDIA RTX PRO 6000 Blackwell) with auto-resume checkpointing & log tee.
#
# Usage (recommended inside tmux):
#   tmux new -s magrpo
#   bash scripts/run_server_train.sh
#
# Optional overrides via environment variables:
#   MAX_STEPS=320 EXP_NAME=exp02_rlmr_v2 bash scripts/run_server_train.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXP_NAME="${EXP_NAME:-exp02_rlmr_orthogonal_32b_v2}"
DATASET="${DATASET:-data/grpo_train_condition_d.jsonl}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/server_48gb_blackwell.yaml}"
REWARD_CONFIG="${REWARD_CONFIG:-rewards/configs/exp02_rlmr_orthogonal_48gb.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${EXP_NAME}}"
MAX_STEPS="${MAX_STEPS:-160}"
GROUP_SIZE="${GROUP_SIZE:-4}"
SAVE_STEPS="${SAVE_STEPS:-10}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

# 1. Pre-flight VRAM check (prevent immediate OOM if another vLLM/training job is holding the GPU)
if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  echo "[Server Check] GPU VRAM Free: ${FREE_MIB} MiB / ${TOTAL_MIB} MiB"
  if [ "${FREE_MIB:-0}" -lt 36000 ]; then
    echo "[ERROR] Less than 36,000 MiB free VRAM (${FREE_MIB} MiB free)."
    echo "        Check running GPU processes with 'nvidia-smi' (e.g. VLLM::EngineCore) before starting GRPO training."
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"

# 2. Auto-detect latest checkpoint-* for seamless resume
RESUME_ARGS=()
LATEST_CKPT="$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -n 1 || true)"
if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
  echo "[Auto-Resume] Found latest checkpoint: ${LATEST_CKPT}"
  RESUME_ARGS=(--resume-from-checkpoint "$LATEST_CKPT")
else
  echo "[Fresh Start] No existing checkpoints in ${OUTPUT_DIR}. Starting from step 0."
fi

LOG_FILE="${OUTPUT_DIR}/train.log"
echo "======================================================================"
echo "MA-GRPO Phase 2 RLMR Training (48GB Blackwell Config)"
echo "  Dataset       : ${DATASET}"
echo "  Train Config  : ${TRAIN_CONFIG}"
echo "  Reward Config : ${REWARD_CONFIG}"
echo "  Output Dir    : ${OUTPUT_DIR}"
echo "  Max Steps     : ${MAX_STEPS} (save every ${SAVE_STEPS} steps)"
echo "  Log File      : ${LOG_FILE}"
echo "======================================================================"

"$PYTHON_BIN" -u scripts/train.py \
  --dataset "$DATASET" \
  --config "$TRAIN_CONFIG" \
  --reward-config "$REWARD_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --group-size "$GROUP_SIZE" \
  --save-steps "$SAVE_STEPS" \
  --reward-device cuda:0 \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$LOG_FILE"
