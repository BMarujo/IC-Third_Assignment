#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE=${1:-model.safetensors}
LEVEL=${2:-5}
THREADS=${3:-}

SERIAL_BIN="./compressor"
OMP_BIN="./bf16_omp"

usage() {
    cat <<EOF
Usage: $0 [input_file] [compression_level] [omp_threads]

Defaults:
    input_file          model.safetensors
    compression_level   5
    omp_threads         (leave unset to use OpenMP default)

Examples:
    $0
    $0 model.safetensors 5
    $0 model.safetensors 8 8
EOF
}

if [[ "${INPUT_FILE}" == "-h" || "${INPUT_FILE}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: input file not found: ${INPUT_FILE}" >&2
    usage >&2
    exit 1
fi

echo "Building both implementations..."
make -s all

if [[ ! -x "${SERIAL_BIN}" || ! -x "${OMP_BIN}" ]]; then
    echo "Error: expected binaries not found/executable: ${SERIAL_BIN}, ${OMP_BIN}" >&2
    exit 1
fi

run_one() {
    local name="$1"; shift
    local bin="$1"; shift
    local input="$1"; shift
    local level="$1"; shift

    local compressed="${input}.${name}.zst"
    local restored="model_restored.${name}.safetensors"

    echo
    echo "=== ${name} ==="
    echo "Compressing (${level}) -> ${compressed}"
    "${bin}" compress "${input}" "${compressed}" "${level}"

    echo "Decompressing -> ${restored}"
    "${bin}" decompress "${compressed}" "${restored}"

    echo "Comparing restored file..."
    if diff "${input}" "${restored}" >/dev/null; then
        echo "OK: ${name} output matches input"
    else
        echo "FAIL: ${name} output differs from input" >&2
        exit 1
    fi
}

echo "Input: ${INPUT_FILE}"
echo "Compression level: ${LEVEL}"
if [[ -n "${THREADS}" ]]; then
    export OMP_NUM_THREADS="${THREADS}"
    echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
fi

run_one "serial" "${SERIAL_BIN}" "${INPUT_FILE}" "${LEVEL}"
run_one "omp" "${OMP_BIN}" "${INPUT_FILE}" "${LEVEL}"

echo
echo "Done."
