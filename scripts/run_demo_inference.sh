#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="${1:-$ROOT_DIR/outputs/airfrans_demo_compat/checkpoints/best.pt}"
INPUT_PATH="${2:-$ROOT_DIR/examples/demo_inference_generic_surface.json}"
EXPORT_DIR="${3:-$ROOT_DIR/outputs/airfrans_demo_compat/demo_inference}"
DEVICE="${4:-cpu}"
OUTPUT_PATH="$EXPORT_DIR/predictions.json"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

mkdir -p "$EXPORT_DIR"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$ROOT_DIR/scripts/infer.py" \
  --checkpoint "$CHECKPOINT" \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --export-dir "$EXPORT_DIR" \
  --device "$DEVICE"

echo "Inference bundle written to: $EXPORT_DIR"
