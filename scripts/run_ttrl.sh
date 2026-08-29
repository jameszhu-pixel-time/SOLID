#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOLID_ENABLE=false
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-ttrl}"

exec bash "${SCRIPT_DIR}/run_solid.sh" "$@"

