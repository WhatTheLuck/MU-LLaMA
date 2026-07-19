#!/usr/bin/env bash
set -euo pipefail
JOB_ID="${1:?usage: tail_log.sh JOB_ID [LOG_DIR]}"
LOG_DIR="${2:-slurm/logs}"
LOG_FILE="$(find "$LOG_DIR" -maxdepth 1 -type f -name "*-${JOB_ID}.out" -print -quit)"
if [[ -z "$LOG_FILE" ]]; then
  echo "No log found for job $JOB_ID in $LOG_DIR" >&2
  exit 1
fi
tail -n 100 -f "$LOG_FILE"

