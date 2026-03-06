# Experiments

This directory contains small, reproducible scripts for generating the CSV artifacts and LaTeX tables used by the paper.

## Config-driven runner

Use `experiments/run_pipeline.py` with a TOML config to rebuild and refresh artifacts:

```bash
cd pppf-mixture-rbpf
python3 experiments/run_pipeline.py --config experiments/configs/paper_refresh.toml
```

The default `paper_refresh.toml` runs:

- `burnside_validate` and copies `irf_*.csv` into `../paper/data/burnside/`
- `burnside_filter_compare` and copies its summary/CSV outputs into `../paper/data/burnside_filter/`, while refreshing `../paper/tab_burnside_filter.tex`
- `nk_compare --mode baseline` and copies `loglik_repeats.csv`, `ess.csv`, and `irf.csv` into `../paper/data/nk/`
- `nk_compare --mode ablations` and refreshes `../paper/tab_nk_ablations.tex`

Enable `nk.sweeps` in the config to run multiple baseline scenarios and emit:

- `output/nk_sweeps/summary.csv`
- `../paper/tab_nk_sweeps.tex`
