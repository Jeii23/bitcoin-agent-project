#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WASABI_BIN="/home/jaume/feina/tools/wallet-baselines/wasabi-2.7.2/wassabee"
WASABI_HOME="${SCRIPT_DIR}/wallets/wasabi_baseline"
LOG_FILE="${WASABI_HOME}/wasabi_gui.log"

if [[ ! -x "${WASABI_BIN}" ]]; then
  echo "Wasabi binary not found or not executable: ${WASABI_BIN}" >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is empty. Open a graphical terminal or export the real DISPLAY before launching Wasabi." >&2
  echo "Current WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<empty>}" >&2
  exit 2
fi

mkdir -p "${WASABI_HOME}"

echo "Launching Wasabi baseline GUI"
echo "  PROJECT_DIR=${PROJECT_DIR}"
echo "  HOME=${WASABI_HOME}"
echo "  DISPLAY=${DISPLAY}"
echo "  WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<empty>}"
echo "  LOG_FILE=${LOG_FILE}"

exec env \
  HOME="${WASABI_HOME}" \
  "${WASABI_BIN}" --network=main --usetor=false --loglevel=critical \
  2>&1 | tee "${LOG_FILE}"
