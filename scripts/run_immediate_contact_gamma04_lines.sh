#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${1:-immediate_contact_gamma04_lines_$(date +%Y%m%d_%H%M%S)}"
INIT_FROM_MODEL="${INIT_FROM_MODEL:-none}"

LOG_ROOT="tensorboard_sweeps/${RUN_TAG}"
MODEL_ROOT="checkpoints/cnn_immediate_contact/${RUN_TAG}"
MODEL_PATH="${MODEL_ROOT}/block_blast_cnn_immediate_contact_gamma04_lines_v1"

mkdir -p "${LOG_ROOT}" "${MODEL_ROOT}"

INIT_ARGS=()
if [[ "${INIT_FROM_MODEL}" != "none" && -n "${INIT_FROM_MODEL}" ]]; then
  INIT_ARGS=(--init-from-model "${INIT_FROM_MODEL}")
fi

venv/bin/python train.py \
  --device cpu \
  --vec-env subproc \
  --subproc-start-method fork \
  --num-cpu 8 \
  --torch-threads 1 \
  --total-timesteps 25000000 \
  --board-size 8 \
  --model-path "${MODEL_PATH}" \
  --checkpoint-dir "${MODEL_ROOT}" \
  --checkpoint-freq 250000 \
  --tb-log-name "cnn_immediate_contact_gamma04_lines_v1" \
  --log-dir "${LOG_ROOT}" \
  --no-resume \
  ${INIT_ARGS[@]+"${INIT_ARGS[@]}"} \
  --shape-pool all \
  --hand-generator adaptive_playable \
  --no-apply-hole-penalty \
  --reward-placement 0.0 \
  --reward-line-scale 42.0 \
  --reward-line-bonus 4.0 \
  --reward-stage-complete 0.0 \
  --reward-no-line 0.0 \
  --reward-game-over 90.0 \
  --reward-game-over-early-weight 0.0 \
  --reward-contact-scale 18.0 \
  --reward-contact-power 1.15 \
  --complexity-simple-prob 0.78 \
  --complexity-medium-prob 0.18 \
  --complexity-hard-prob 0.04 \
  --learning-rate 0.0002 \
  --lr-schedule constant \
  --n-steps 1024 \
  --batch-size 1024 \
  --n-epochs 4 \
  --gamma 0.40 \
  --ent-coef 0.04 \
  --eval-freq 100000 \
  --eval-episodes 50 \
  --max-eval-steps 5000
