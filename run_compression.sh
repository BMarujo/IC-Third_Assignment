#!/bin/bash
set -e

echo "Building compressor..."
make

echo "Compressing model.safetensors..."
./compressor compress model.safetensors model.safetensors.zst 5

echo "Decompressing to verify..."
./compressor decompress model.safetensors.zst model_restored.safetensors

echo "Comparing files..."
if diff model.safetensors model_restored.safetensors; then
    echo "Success! Files match."
else
    echo "Error: Files do not match."
    exit 1
fi

echo "Done."
