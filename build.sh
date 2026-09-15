#!/usr/bin/env bash
# Build the WebAssembly bundle and place it where the Python package expects it.
set -euo pipefail

cd "$(dirname "$0")"

TARGET_DIR=${CARGO_TARGET_DIR:-target}
OUT_DIR=python/rustyplot/static

# `strip = true` in the workspace release profile would remove the
# `target_features` custom section along with the symbols. wasm-opt reads that
# section to decide which WebAssembly features are in play; without it, it
# falls back to the MVP feature set and rejects the bulk-memory instructions
# LLVM emits ("Bulk memory operation (bulk memory is disabled)"). Keeping the
# section costs ~7 kB in the final bundle, because wasm-opt -Oz strips the
# symbols anyway.
CARGO_PROFILE_RELEASE_STRIP=false \
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
