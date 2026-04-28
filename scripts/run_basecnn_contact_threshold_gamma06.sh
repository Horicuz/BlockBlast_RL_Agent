#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${1:-basecnn_contact_threshold_gamma06_$(date +%Y%m%d_%H%M%S)}"
INIT_FROM_MODEL="${INIT_FROM_MODEL:-none}"

LOG_ROOT="tensorboard_sweeps/${RUN_TAG}"
MODEL_ROOT="checkpoints/cnn_contact_threshold/${RUN_TAG}"
MODEL_PATH="${MODEL_ROOT}/block_blast_basecnn_contact_threshold_gamma06_v1"

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
  --model-path "${MODEL_PATH}" \
  --checkpoint-dir "${MODEL_ROOT}" \
  --checkpoint-freq 250000 \
  --tb-log-name "basecnn_contact_threshold_gamma06_v1" \
  --log-dir "${LOG_ROOT}" \
  --no-resume \
  ${INIT_ARGS[@]+"${INIT_ARGS[@]}"} \
  --cnn-arch base \
  --features-dim 256 \
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
  --learning-rate 0.0008 \
  --lr-schedule constant \
  --n-steps 1024 \
  --batch-size 1024 \
  --n-epochs 4 \
  --gamma 0.60 \
  --ent-coef 0.04 \
  --eval-freq 100000 \
  --eval-episodes 50 \
  --max-eval-steps 5000
