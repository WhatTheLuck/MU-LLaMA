#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
TASK_KIND="${1:?task kind is required}"
shift
cd "${WORKDIR:?WORKDIR must be exported by submit_sequence.py}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

PYTHON_BIN="${PYTHON_EXECUTABLE:-python}"
if [[ "$TASK_KIND" == "train" ]]; then
  "$PYTHON_BIN" train.py --config "${1:?experiment config path is required}"
elif [[ "$TASK_KIND" == "analyze" ]]; then
  "$PYTHON_BIN" scripts/analyze_results.py "$@"
else
  echo "Unknown task kind: $TASK_KIND" >&2
  exit 2
fi

