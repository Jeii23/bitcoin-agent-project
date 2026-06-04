#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARROW_HOME="${SCRIPT_DIR}/wallets/sparrow_baseline"
SPARROW_BIN="/home/jaume/feina/tools/wallet-baselines/sparrow-2.4.2/bin/Sparrow"
LOG_FILE="${SPARROW_HOME}/sparrow_gui.log"

mkdir -p "${SPARROW_HOME}"

echo "Launching Sparrow baseline GUI"
echo "  SPARROW_HOME=${SPARROW_HOME}"
echo "  LOG_FILE=${LOG_FILE}"

export SPARROW_USER_DIR="${SPARROW_HOME}"

exec "${SPARROW_BIN}" 2>&1 | tee "${LOG_FILE}"
