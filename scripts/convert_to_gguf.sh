#!/usr/bin/env bash
# convert_to_gguf.sh — Convert the merged Spark model to GGUF for llama.cpp / Ollama / LM Studio.
#
# Prerequisites:
#   - The merged model must exist at ./spark-merged (produced by train_spark.py,
#     or copied from spark_full_package/spark-merged).
#   - A clone of llama.cpp with convert_hf_to_gguf.py available.
#
# Usage:
#   LLAMACPP=/path/to/llama.cpp ./scripts/convert_to_gguf.sh
#
# Output (gitignored — *.gguf is excluded from the repo):
#   spark.Q4_K_M.gguf
set -euo pipefail

LLAMACPP="${LLAMACPP:-../llama.cpp}"
MERGED_DIR="${MERGED_DIR:-spark-merged}"
OUT="${OUT:-spark.Q4_K_M.gguf}"
QUANT="${QUANT:-Q4_K_M}"

if [ ! -f "$LLAMACPP/convert_hf_to_gguf.py" ]; then
  echo "ERROR: convert_hf_to_gguf.py not found in $LLAMACPP" >&2
  echo "Clone llama.cpp: git clone https://github.com/ggml-org/llama.cpp.git \"$LLAMACPP\"" >&2
  exit 1
fi

echo "Converting $MERGED_DIR -> f16 GGUF ..."
python "$LLAMACPP/convert_hf_to_gguf.py" "$MERGED_DIR" --outfile "${OUT%.gguf}.f16.gguf"

echo "Quantizing -> $QUANT ..."
python "$LLAMACPP/llama-quantize" "${OUT%.gguf}.f16.gguf" "$OUT" "$QUANT"

echo "Done: $OUT"
