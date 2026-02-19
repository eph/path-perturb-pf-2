#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 experiments/run_pipeline.py --config experiments/configs/paper_refresh.toml

python3 experiments/burnside_plots.py
python3 experiments/nk_plots.py

echo "All outputs are under: $ROOT_DIR/output/"
