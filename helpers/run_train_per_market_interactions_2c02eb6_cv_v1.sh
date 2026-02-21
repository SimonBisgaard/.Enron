#!/usr/bin/env bash
set -euo pipefail

# Pinned reproduce command for the 2c02eb6 per-market-interactions script.
# Original style command:
# uv run python train_per_market_interactions.py --name per_market_interactions_cv_v1 --exclude-2023 --exclude-2023-keep-from-month 10 --cv --cv-folds 5 --cv-val-days 14 --cv-step-days 14 --cv-min-train-days 90

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

uv run python train_per_market_interactions_2c02eb6.py \
  --name per_market_interactions_cv_v1 \
  --exclude-2023 \
  --exclude-2023-keep-from-month 10 \
  --cv \
  --cv-folds 5 \
  --cv-val-days 14 \
  --cv-step-days 14 \
  --cv-min-train-days 90
