#!/usr/bin/env bash
set -euo pipefail
cd /home/ben/DL/EVENT_DECODER
LOG_DIR="outputs/recurrent_event_decoder/automation_logs"
MAIN_LOG="$LOG_DIR/run_after_current_training.log"
CURRENT_PID=925752
{
  echo "$(date '+%F %T') supervisor started"
  if ps -p "$CURRENT_PID" >/dev/null 2>&1; then
    echo "$(date '+%F %T') waiting for existing training pid $CURRENT_PID"
    while ps -p "$CURRENT_PID" >/dev/null 2>&1; do
      sleep 60
    done
    echo "$(date '+%F %T') existing training pid $CURRENT_PID finished"
  else
    echo "$(date '+%F %T') existing training pid $CURRENT_PID is not running; continuing"
  fi

  echo "$(date '+%F %T') evaluating and preserving current best checkpoint"
  conda run --no-capture-output -n DL python -m recurrent_event_decoder.evaluate_checkpoint \
    --checkpoint outputs/recurrent_event_decoder/checkpoints/best.pt \
    --manifest data/recurrent_event_processed_stride10/manifest.json \
    --output-json outputs/recurrent_event_decoder/current_run_test_accuracy.json \
    --renamed-checkpoint outputs/recurrent_event_decoder/checkpoints/current_run_best.pt \
    --batch-size 1 \
    --device cuda \
    --amp

  echo "$(date '+%F %T') starting hyperparameter sweep"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run --no-capture-output -n DL python -m recurrent_event_decoder.hyperparameter_sweep \
    --manifest data/recurrent_event_processed_stride10/manifest.json \
    --epoch-values 50,60,70,80,90,100 \
    --batch-size 1 \
    --device cuda \
    --amp \
    --activation-checkpointing \
    --learning-rates 0.0003,0.0001 \
    --weight-decays 0.0,0.0001 \
    --dropouts 0.1,0.2 \
    --grad-clips 1.0 \
    --encoder-strides 4

  echo "$(date '+%F %T') automation completed"
} >> "$MAIN_LOG" 2>&1
