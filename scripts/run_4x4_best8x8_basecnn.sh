#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${1:-4x4_best8x8_basecnn_g03_lr2e3_$(date +%Y%m%d_%H%M%S)}"
TOTAL_TIMESTEPS="${2:-25000000}"

DEVICE="${DEVICE:-cpu}"
NUM_CPU="${NUM_CPU:-8}"
TORCH_THREADS="${TORCH_THREADS:-1}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export MPLCONFIGDIR

LOG_ROOT="tensorboard_sweeps/${RUN_TAG}"
MODEL_ROOT="checkpoints/4x4_best8x8/${RUN_TAG}"
MODEL_PATH="${MODEL_ROOT}/block_blast_4x4_best8x8_basecnn_v1"

mkdir -p "${LOG_ROOT}" "${MODEL_ROOT}" "${MPLCONFIGDIR}"

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
  --tb-log-name "block_blast_4x4_best8x8_basecnn_v1" \
  --log-dir "${LOG_ROOT}" \
  --no-resume \
  --cnn-arch base \
  --features-dim 256 \
  --net-arch 256,256 \
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
  --learning-rate 0.002 \
  --lr-schedule constant \
  --n-steps 1024 \
  --batch-size 1024 \
  --n-epochs 4 \
  --gamma 0.30 \
  --ent-coef 0.04 \
  --eval-freq 100000 \
  --eval-episodes 50 \
  --max-eval-steps 5000
