#!/usr/bin/env bash
# End-to-End Linux Server Runner:
#   Step 1: Train Qwen2.5-7B QLoRA with Phase 2 Minimal Orthogonal RLMR (gamma=0.05) + Auto-Checkpoints (every 10 steps)
#   Step 2: Automatically run Full-Pipeline Benchmark (Base 7B vs Trained Checkpoint: FullStory-Bench + ClassicNLG-Bench)
#
# Recommended 1-line background run (safe to close SSH & turn off your PC):
#   nohup bash scripts/run_server_train_and_eval.sh > nohup_train_eval.log 2>&1 &
#
# Optional overrides:
#   MAX_STEPS=160 EVAL_PROMPTS=3 EXP_NAME=exp02_rlmr_orthogonal_32b_v2 nohup bash scripts/run_server_train_and_eval.sh > nohup_train_eval.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export EXP_NAME="${EXP_NAME:-exp02_rlmr_orthogonal_32b_v2}"
export OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${EXP_NAME}}"
export MAX_STEPS="${MAX_STEPS:-160}"
export EVAL_PROMPTS="${EVAL_PROMPTS:-3}"
export EVAL_OUT_DIR="${EVAL_OUT_DIR:-eval_outputs/${EXP_NAME}_vs_base_${EVAL_PROMPTS}prompts}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "======================================================================"
echo "[STEP 1/2] STARTING GRPO RLMR TRAINING (MAX_STEPS=${MAX_STEPS})"
echo "======================================================================"
bash scripts/run_server_train.sh

echo ""
echo "======================================================================"
echo "[STEP 2/2] TRAINING COMPLETE -> STARTING FULL-PIPELINE BENCHMARK"
echo "  Checkpoint Dir : ${OUTPUT_DIR}"
echo "  Eval Prompts   : ${EVAL_PROMPTS}"
echo "  Eval Out Dir   : ${EVAL_OUT_DIR}"
echo "======================================================================"
"$PYTHON_BIN" -u scripts/eval_full_pipeline.py \
  --checkpoint-dir "$OUTPUT_DIR" \
  --num-prompts "$EVAL_PROMPTS" \
  --output-dir "$EVAL_OUT_DIR"

echo ""
echo "======================================================================"
echo "[ALL DONE] Training + FullStory-Bench + ClassicNLG-Bench Finished!"
echo "  Checkpoints : ${OUTPUT_DIR}"
echo "  Eval Report : ${EVAL_OUT_DIR}/unified_comparison_summary.json"
echo "  Eval Table  : ${EVAL_OUT_DIR}/comparison_table.csv"
echo "======================================================================"
