#!/usr/bin/env bash
# Build the WebAssembly bundle and place it where the Python package expects it.
set -euo pipefail

cd "$(dirname "$0")"

TARGET_DIR=${CARGO_TARGET_DIR:-target}
OUT_DIR=python/rustyplot/static

cargo build -p rustyplot-wasm --target wasm32-unknown-unknown --release

wasm-bindgen "${TARGET_DIR}/wasm32-unknown-unknown/release/rustyplot_wasm.wasm" \
  --out-dir "${OUT_DIR}" \
  --target web \
  --no-typescript

# Optional, and worth it: usually halves the binary.
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -Oz -o "${OUT_DIR}/rustyplot_wasm_bg.wasm" "${OUT_DIR}/rustyplot_wasm_bg.wasm"
else
  echo "note: wasm-opt not found, skipping size optimisation" >&2
fi

ls -lh "${OUT_DIR}"
