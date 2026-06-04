#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_HOME="${SCRIPT_DIR}/wallets/wasabi_backend"
BACKEND_BIN="/home/jaume/feina/tools/wallet-baselines/wasabi-2.7.2/wbackend"
LOG_FILE="${BACKEND_HOME}/wasabi_backend.log"

if [[ ! -x "${BACKEND_BIN}" ]]; then
  echo "Wasabi backend binary not found or not executable: ${BACKEND_BIN}" >&2
  exit 1
fi

"${SCRIPT_DIR}/configure_wasabi_private_backend.py"

mkdir -p "${BACKEND_HOME}"

echo "Starting Wasabi private backend"
echo "  PROJECT_DIR=${PROJECT_DIR}"
echo "  HOME=${BACKEND_HOME}"
echo "  URL=http://localhost:37127/"
echo "  LOG_FILE=${LOG_FILE}"

exec env \
  HOME="${BACKEND_HOME}" \
  ASPNETCORE_URLS="http://localhost:37127" \
  "${BACKEND_BIN}" \
  2>&1 | tee "${LOG_FILE}"
