#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${1:-overnight_adaptive_contact_$(date +%Y%m%d_%H%M%S)}"

LOG_ROOT="tensorboard_sweeps/${RUN_TAG}"
MODEL_ROOT="checkpoints/cnn_adaptive_contact/${RUN_TAG}"
MODEL_PATH="${MODEL_ROOT}/block_blast_cnn_adaptive_contact_v1"

mkdir -p "${LOG_ROOT}" "${MODEL_ROOT}"

venv/bin/python train.py \
  --device cpu \
  --vec-env subproc \
  --subproc-start-method fork \
  --num-cpu 8 \
  --torch-threads 1 \
  --total-timesteps 25000000 \
  --model-path "${MODEL_PATH}" \
  --checkpoint-dir "${MODEL_ROOT}" \
  --checkpoint-freq 500000 \
  --tb-log-name "cnn_adaptive_contact_12seeds_v1" \
  --log-dir "${LOG_ROOT}" \
  --no-resume \
  --init-from-model checkpoints/cnn_simple_curriculum/block_blast_cnn_simple_playable_v1 \
  --shape-pool all \
  --hand-generator adaptive_playable \
  --fixed-game-seed 123 \
  --fixed-game-seed-count 12 \
  --no-apply-hole-penalty \
  --reward-placement 0.0 \
  --reward-line-scale 30.0 \
  --reward-line-bonus 2.0 \
  --reward-stage-complete 20.0 \
  --reward-no-line 0.0 \
  --reward-game-over 180.0 \
  --reward-game-over-early-weight 1.0 \
  --reward-contact-scale 16.0 \
  --reward-contact-power 1.4 \
  --complexity-simple-prob 0.68 \
  --complexity-medium-prob 0.24 \
  --complexity-hard-prob 0.08 \
  --learning-rate 0.0001 \
  --lr-schedule constant \
  --n-steps 256 \
  --batch-size 256 \
  --n-epochs 6 \
  --gamma 0.99 \
  --ent-coef 0.02 \
  --eval-freq 250000 \
  --eval-episodes 50 \
  --max-eval-steps 5000
