#!/bin/sh
# Download the base model into models/SmolLM2-135M (about 270 MB total).
# Rerunning is safe: curl resumes partial files.
set -e
cd "$(dirname "$0")/../models/SmolLM2-135M"
BASE="https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main"
for f in .gitattributes README.md config.json generation_config.json \
         merges.txt special_tokens_map.json tokenizer.json \
         tokenizer_config.json vocab.json model.safetensors; do
    curl -L -C - -o "$f" "$BASE/$f"
done
echo "model files in place:"
ls -la
