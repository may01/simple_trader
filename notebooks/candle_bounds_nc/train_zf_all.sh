#!/usr/bin/env bash
# Sequential zone-filtered NN trainings (6 models: tf x side), single-shot mode.
# Run from the worktree root. Logs to $ZF_LOG_DIR (default ./zf_logs).
set -u
cd "$(dirname "$0")/../.."
LOG_DIR="${ZF_LOG_DIR:-$(pwd)/notebooks/candle_bounds_nc/zf_logs}"
mkdir -p "$LOG_DIR"

for spec in zf15_n2_long zf15_n2_short zf60_n2_long zf60_n2_short zf240_n2_long zf240_n2_short; do
  echo "=== START $spec $(date -u +%FT%TZ)"
  NN_TRAIN_ENV=configs/nn_train_dataset_2y.env \
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm \
    -e NN_SPEC_PATH="configs/nn_specs/zone_filtered/${spec}.yaml" \
    -e NN_STUDY="${spec}" \
    -e NN_TRAIN_MODE=single \
    nn-train > "$LOG_DIR/${spec}.log" 2>&1
  rc=$?
  echo "=== DONE $spec rc=$rc $(date -u +%FT%TZ)"
  tail -3 "$LOG_DIR/${spec}.log"
done
echo "=== ALL DONE"
