#!/usr/bin/env bash
set -euo pipefail
squeue -u "${USER}" -o "%.18i %.28j %.10T %.10M %.10l %.6D %R" "$@"

