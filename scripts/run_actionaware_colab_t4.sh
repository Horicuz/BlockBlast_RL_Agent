#!/usr/bin/env bash
set -euo pipefail

export DEVICE="${DEVICE:-cuda}"
export NUM_CPU="${NUM_CPU:-4}"
export TORCH_THREADS="${TORCH_THREADS:-1}"
export N_STEPS="${N_STEPS:-256}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export N_EPOCHS="${N_EPOCHS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-0.002}"
export GAMMA="${GAMMA:-0.3}"
export ENT_COEF="${ENT_COEF:-0.10}"

RUN_TAG="${1:-actionaware_colab_t4_g03_lr2e3_n256_b256}"
TOTAL_TIMESTEPS="${2:-1500000}"

exec ./scripts/run_actionaware_exploratory.sh "${RUN_TAG}" "${TOTAL_TIMESTEPS}"
