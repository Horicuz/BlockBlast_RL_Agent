#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${1:-actionaware_explore_g03_lr2e3_n256_b256}"
TOTAL_TIMESTEPS="${2:-1500000}"
DEVICE="${DEVICE:-cpu}"
NUM_CPU="${NUM_CPU:-8}"
TORCH_THREADS="${TORCH_THREADS:-1}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_EPOCHS="${N_EPOCHS:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.002}"
GAMMA="${GAMMA:-0.3}"
ENT_COEF="${ENT_COEF:-0.10}"

LOG_ROOT="tensorboard_sweeps/${RUN_TAG}"
MODEL_ROOT="checkpoints/cnn_actionaware/${RUN_TAG}"
MODEL_PATH="${MODEL_ROOT}/block_blast_actionaware_explore_v1"

mkdir -p "${LOG_ROOT}" "${MODEL_ROOT}"

venv/bin/python train.py \
  --device "${DEVICE}" \
  --vec-env subproc \
  --subproc-start-method fork \
  --num-cpu "${NUM_CPU}" \
  --torch-threads "${TORCH_THREADS}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --model-path "${MODEL_PATH}" \
  --checkpoint-dir "${MODEL_ROOT}" \
  --checkpoint-freq 250000 \
  --tb-log-name "actionaware_explore_v1" \
  --log-dir "${LOG_ROOT}" \
  --no-resume \
  --cnn-arch actionaware \
  --features-dim 768 \
  --net-arch 512,256 \
  --shape-pool all \
  --hand-generator adaptive_playable \
  --no-apply-hole-penalty \
  --reward-placement 0.0 \
  --reward-line-scale 28.0 \
  --reward-line-bonus 1.5 \
  --reward-stage-complete 0.0 \
  --reward-no-line 0.0 \
  --reward-game-over 90.0 \
  --reward-game-over-early-weight 0.0 \
  --reward-contact-scale 24.0 \
  --reward-contact-power 1.15 \
  --reward-contact-threshold 0.40 \
  --reward-contact-penalty-scale 8.0 \
  --complexity-simple-prob 0.78 \
  --complexity-medium-prob 0.18 \
  --complexity-hard-prob 0.04 \
  --learning-rate "${LEARNING_RATE}" \
  --lr-schedule constant \
  --n-steps "${N_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --n-epochs "${N_EPOCHS}" \
  --gamma "${GAMMA}" \
  --ent-coef "${ENT_COEF}" \
  --eval-freq 100000 \
  --eval-episodes 50 \
  --max-eval-steps 5000
