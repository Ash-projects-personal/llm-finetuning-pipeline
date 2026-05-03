#!/bin/bash
# Export fine-tuned model to GGUF format for llama.cpp CPU inference
# Usage: bash export_gguf.sh <model_dir> <output_gguf_path>
MODEL_DIR=${1:-"outputs/finetuned"}
OUTPUT=${2:-"outputs/model.gguf"}

echo "[INFO] Converting $MODEL_DIR to GGUF..."

# Requires llama.cpp to be cloned alongside this repo
if [ ! -d "llama.cpp" ]; then
  echo "[INFO] Cloning llama.cpp..."
  git clone https://github.com/ggerganov/llama.cpp --depth=1
  cd llama.cpp && make -j$(nproc) && cd ..
fi

python llama.cpp/convert_hf_to_gguf.py "$MODEL_DIR" --outfile "$OUTPUT" --outtype q4_k_m

echo "[INFO] GGUF model saved to $OUTPUT"
echo "[INFO] Run inference: ./llama.cpp/main -m $OUTPUT -p 'Your prompt here' -n 256"
