#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/linear_gaussian_validate
./build/burnside_validate
./build/nk_compare

python3 experiments/burnside_plots.py
python3 experiments/nk_plots.py

echo "All outputs are under: $ROOT_DIR/output/"
