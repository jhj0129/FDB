#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
test -x .venv/bin/python || { echo 'Run scripts/setup_jetson.sh first.' >&2; exit 1; }
if [[ $# -eq 0 ]]; then
    set -- --observation-mode camera --output-directory "artifacts/jetson_phase3_run_$(date +%Y%m%dT%H%M%S)_$$"
fi
exec .venv/bin/python -m fdb.core.physical_cli "$@"
