#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
RUNNER="${SCRIPT_DIR}/wallet_baseline_runner.py"
CSV="${SCRIPT_DIR}/wallet_baseline.csv"
IMPORT_DIR="${SCRIPT_DIR}/wallet_imports/wasabi"
LOG_FILE="${IMPORT_DIR}/wasabi_score_watch.log"

expected=(
  "wasabi_pct10.psbt"
  "wasabi_pct30.psbt"
  "wasabi_pct50.psbt"
  "wasabi_pct80.psbt"
  "wasabi_pct95.psbt"
)

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python venv not found or not executable: ${PYTHON}" >&2
  exit 1
fi

mkdir -p "${IMPORT_DIR}"

echo "Watching for Wasabi unsigned PSBT exports..."
echo "  Import dir: ${IMPORT_DIR}"
echo "  Log file: ${LOG_FILE}"
echo "Expected files:"
printf '  - %s\n' "${expected[@]}"
echo

while true; do
  missing=()
  for filename in "${expected[@]}"; do
    if [[ ! -s "${IMPORT_DIR}/${filename}" ]]; then
      missing+=("${filename}")
    fi
  done

  if [[ "${#missing[@]}" -eq 0 ]]; then
    echo "All Wasabi PSBT files are present. Running scorer..."
    break
  fi

  printf 'Waiting for %d file(s): %s\n' "${#missing[@]}" "${missing[*]}"
  sleep 5
done

cd "${PROJECT_DIR}"
exec "${PYTHON}" "${RUNNER}" "${CSV}" \
  --filter wallet:wasabi \
  --include-disabled \
  --compare-agents \
  2>&1 | tee "${LOG_FILE}"
