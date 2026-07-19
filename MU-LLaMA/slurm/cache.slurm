#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
CONFIG_PATH="${1:?experiment config path is required}"
cd "${WORKDIR:?WORKDIR must be exported by submit_sequence.py}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

"${PYTHON_EXECUTABLE:-python}" tools/cache_dissonance.py --config "$CONFIG_PATH"

