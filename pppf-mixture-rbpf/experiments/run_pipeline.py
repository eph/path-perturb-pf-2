#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(root: Path, p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


def run_checked(args: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(args, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> int:
    ensure_dir(dst_dir)
    n = 0
    for p in sorted(src_dir.glob(pattern)):
        if not p.is_file():
            continue
        shutil.copy2(p, dst_dir / p.name)
        n += 1
    return n


def build_project(cfg: dict[str, Any], *, root: Path) -> None:
    if not cfg.get("enabled", True):
        return
    build_type = str(cfg.get("type", "Release"))
    jobs = cfg.get("jobs", None)
    args_config = ["cmake", "-S", ".", "-B", "build", f"-DCMAKE_BUILD_TYPE={build_type}"]
    run_checked(args_config, cwd=root)
    args_build = ["cmake", "--build", "build"]
    if isinstance(jobs, int) and jobs > 0:
        args_build += ["-j", str(jobs)]
    run_checked(args_build, cwd=root)


def run_burnside(cfg: dict[str, Any], *, root: Path) -> None:
    if not cfg.get("enabled", True):
        return
    out_dir = _resolve(root, cfg.get("out_dir", "output/burnside"))
    args = [
        str(root / "build" / "burnside_validate"),
        "--out_dir",
        str(out_dir),
        "--T",
        str(int(cfg.get("T", 40))),
    ]
    shock_sizes = cfg.get("shock_sizes", None)
    if isinstance(shock_sizes, list) and shock_sizes:
        args += ["--shock_sizes", ",".join(str(float(x)) for x in shock_sizes)]
    run_checked(args, cwd=root)

    paper_data_dir = cfg.get("paper_data_dir", None)
    if paper_data_dir is not None:
        dst = _resolve(root, paper_data_dir)
        copied = copy_glob(out_dir, "irf_*.csv", dst)
        if copied == 0:
            raise RuntimeError(f"burnside_validate produced no irf_*.csv under {out_dir}")


def write_burnside_filter_table_tex(path: Path, summary: dict[str, Any]) -> None:
    methods = summary.get("methods", {})
    ll_exact = float(methods["grid_exact"]["mean_loglik"])
    labels = {
        "grid_exact": "Grid exact",
        "grid_ce": "Grid CE",
        "grid_ut": "Grid UT",
        "pf_exact": "Bootstrap PF, exact",
        "pf_ce": "Bootstrap PF, CE",
        "pf_ut": "Bootstrap PF, UT",
        "copf_exact": "COPF, exact",
        "copf_ce": "COPF, CE",
        "copf_ut": "COPF, UT",
    }
    order = [
        "grid_exact",
        "grid_ce",
        "grid_ut",
        "pf_exact",
        "pf_ce",
        "pf_ut",
        "copf_exact",
        "copf_ce",
        "copf_ut",
    ]

    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("Method & Mean loglik & Gap & SD loglik & Mean time (ms) & Mean ESS \\\\")
    lines.append("\\midrule")
    for key in order:
        row = methods.get(key, None)
        if row is None:
            continue
        gap = float(row["mean_loglik"]) - ll_exact
        sd = float(row["sd_loglik"])
        ess = float(row.get("mean_ess", -1.0))
        ess_str = "--" if ess < 0.0 else f"{ess:.1f}"
        lines.append(
            f"{labels[key]} & {float(row['mean_loglik']):.1f} & {gap:.1f} & "
            f"{sd:.1f} & {float(row['mean_runtime_ms']):.1f} & {ess_str} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_burnside_filter(cfg: dict[str, Any], *, root: Path) -> None:
    if not cfg.get("enabled", False):
        return
    out_dir = _resolve(root, cfg.get("out_dir", "output/burnside_filter"))
    args = [
        str(root / "build" / "burnside_filter_compare"),
        "--out_dir",
        str(out_dir),
        "--T",
        str(int(cfg.get("T", 150))),
        "--N",
        str(int(cfg.get("N", 512))),
        "--R",
        str(int(cfg.get("R", 20))),
        "--grid_size",
        str(int(cfg.get("grid_size", 401))),
        "--grid_width_sd",
        str(float(cfg.get("grid_width_sd", 6.0))),
        "--meas_sd",
        str(float(cfg.get("meas_sd", 10.0))),
        "--seed_data",
        str(int(cfg.get("seed_data", 20260306))),
    ]
    run_checked(args, cwd=root)

    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"burnside_filter_compare produced no summary.json under {out_dir}")
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    paper_data_dir = cfg.get("paper_data_dir", None)
    if paper_data_dir is not None:
        dst = _resolve(root, paper_data_dir)
        ensure_dir(dst)
        for name in ["summary.json", "sim_data.csv", "loglik_repeats.csv", "ess.csv"]:
            src = out_dir / name
            if not src.is_file():
                raise RuntimeError(f"missing Burnside filter artifact: {src}")
            shutil.copy2(src, dst / name)

    paper_table = cfg.get("paper_table", None)
    if paper_table is not None:
        write_burnside_filter_table_tex(_resolve(root, paper_table), summary)


def run_nk_baseline(cfg: dict[str, Any], *, root: Path) -> None:
    if not cfg.get("enabled", True):
        return
    out_dir = _resolve(root, cfg.get("out_dir", "output/nk"))
    args = [
        str(root / "build" / "nk_compare"),
        "--mode",
        "baseline",
        "--out_dir",
        str(out_dir),
        "--T",
        str(int(cfg.get("T", 250))),
        "--N",
        str(int(cfg.get("N", 256))),
        "--R",
        str(int(cfg.get("R", 30))),
        "--horizon",
        str(int(cfg.get("horizon", 20))),
        "--omega_horizon",
        str(int(cfg.get("omega_horizon", 1))),
        "--seed_data",
        str(int(cfg.get("seed_data", 20260113))),
    ]
    if not bool(cfg.get("run_sanity", True)):
        args.append("--no_sanity")
    else:
        out_dir_sanity = _resolve(root, cfg.get("out_dir_sanity", "output/nk_sanity_no_elb"))
        args += ["--out_dir_sanity", str(out_dir_sanity), "--seed_sanity", str(int(cfg.get("seed_sanity", 20260218)))]
    run_checked(args, cwd=root)

    paper_data_dir = cfg.get("paper_data_dir", None)
    if paper_data_dir is not None:
        dst = _resolve(root, paper_data_dir)
        copied = 0
        for name in ["loglik_repeats.csv", "ess.csv", "irf.csv"]:
            src = out_dir / name
            if not src.is_file():
                raise RuntimeError(f"missing NK baseline artifact: {src}")
            ensure_dir(dst)
            shutil.copy2(src, dst / name)
            copied += 1
        if copied != 3:
            raise RuntimeError("unexpected NK baseline copy failure")


def run_nk_ablations(cfg: dict[str, Any], *, root: Path) -> None:
    if not cfg.get("enabled", True):
        return
    out_dir = _resolve(root, cfg.get("out_dir", "output/nk_ablation"))
    args = [
        str(root / "build" / "nk_compare"),
        "--mode",
        "ablations",
        "--out_dir_abl",
        str(out_dir),
        "--T",
        str(int(cfg.get("T", 250))),
        "--N",
        str(int(cfg.get("N", 256))),
        "--R",
        str(int(cfg.get("R", 30))),
        "--horizon",
        str(int(cfg.get("horizon", 20))),
        "--seed_data",
        str(int(cfg.get("seed_data", 20260113))),
    ]
    paper_table = cfg.get("paper_table", None)
    if paper_table is not None:
        args += ["--paper_table", str(_resolve(root, paper_table))]
    run_checked(args, cwd=root)


@dataclass(frozen=True)
class NkSummaryRow:
    case: str
    T: int
    N: int
    R: int
    horizon: int
    omega_horizon: int
    method: str
    mean_loglik: float
    sd_loglik: float
    mean_runtime_ms: float
    mean_ess: float | None
    gap_vs_exact: float


def read_nk_summary(out_dir: Path, *, case: str, T: int, N: int, R: int, horizon: int, omega_horizon: int) -> list[NkSummaryRow]:
    ll_path = out_dir / "loglik_repeats.csv"
    ess_path = out_dir / "ess.csv"
    if not ll_path.is_file():
        raise RuntimeError(f"missing {ll_path}")

    ll: dict[str, list[float]] = {}
    rt: dict[str, list[float]] = {}
    exact_vals: list[float] = []
    with ll_path.open("r", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            m = row["method"]
            v = float(row["loglik"])
            t = float(row["runtime_ms"])
            ll.setdefault(m, []).append(v)
            rt.setdefault(m, []).append(t)
            if m == "global_discrete_exact":
                exact_vals.append(v)
    if not exact_vals:
        raise RuntimeError("missing global_discrete_exact in loglik_repeats.csv")
    ll_exact = statistics.fmean(exact_vals)

    mean_ess: dict[str, float] = {}
    if ess_path.is_file():
        ess_acc: dict[str, list[float]] = {}
        with ess_path.open("r", newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                ess_acc.setdefault(row["method"], []).append(float(row["ess"]))
        mean_ess = {m: statistics.fmean(v) for (m, v) in ess_acc.items() if v}

    rows: list[NkSummaryRow] = []
    for m, vals in sorted(ll.items()):
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        mrt = statistics.fmean(rt.get(m, [0.0]))
        rows.append(
            NkSummaryRow(
                case=case,
                T=T,
                N=N,
                R=R,
                horizon=horizon,
                omega_horizon=omega_horizon,
                method=m,
                mean_loglik=mean,
                sd_loglik=sd,
                mean_runtime_ms=mrt,
                mean_ess=mean_ess.get(m, None),
                gap_vs_exact=mean - ll_exact,
            )
        )
    return rows


def write_nk_sweep_csv(path: Path, rows: list[NkSummaryRow]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "case",
                "T",
                "N",
                "R",
                "horizon",
                "omega_horizon",
                "method",
                "mean_loglik",
                "sd_loglik",
                "mean_runtime_ms",
                "mean_ess",
                "gap_vs_exact",
            ]
        )
        for r in rows:
            wr.writerow(
                [
                    r.case,
                    r.T,
                    r.N,
                    r.R,
                    r.horizon,
                    r.omega_horizon,
                    r.method,
                    f"{r.mean_loglik:.6f}",
                    f"{r.sd_loglik:.6f}",
                    f"{r.mean_runtime_ms:.6f}",
                    "" if r.mean_ess is None else f"{r.mean_ess:.6f}",
                    f"{r.gap_vs_exact:.6f}",
                ]
            )


def write_nk_sweep_table_tex(path: Path, rows: list[NkSummaryRow], *, methods: list[str]) -> None:
    by_case: dict[str, dict[str, NkSummaryRow]] = {}
    for r in rows:
        by_case.setdefault(r.case, {})[r.method] = r

    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append("Case & $H$ & $\\omega$ horizon & Method & Gap & SD & Mean ESS \\\\")
    lines.append("\\midrule")
    for case, mrows in sorted(by_case.items()):
        any_row = next(iter(mrows.values()))
        for mi, m in enumerate(methods):
            r = mrows.get(m, None)
            if r is None:
                continue
            name = case if mi == 0 else ""
            gap = f"{r.gap_vs_exact:.1f}"
            sd = f"{r.sd_loglik:.1f}"
            ess = "--" if r.mean_ess is None else f"{r.mean_ess:.1f}"
            lines.append(f"{name} & {any_row.horizon} & {any_row.omega_horizon} & {m} & {gap} & {sd} & {ess} \\\\")
        lines.append("\\addlinespace")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_nk_sweeps(cfg: dict[str, Any], *, root: Path) -> None:
    if not cfg.get("enabled", False):
        return

    out_dir_root = _resolve(root, cfg.get("out_dir_root", "output/nk_sweeps"))
    ensure_dir(out_dir_root)

    all_rows: list[NkSummaryRow] = []
    cases = cfg.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("nk.sweeps.enabled=true but nk.sweeps.cases is empty")

    for c in cases:
        if not isinstance(c, dict):
            raise RuntimeError("nk.sweeps.cases entries must be tables")
        name = str(c["name"])
        out_dir = out_dir_root / name
        run_cfg = dict(c)
        run_cfg["enabled"] = True
        run_cfg.setdefault("out_dir", str(out_dir))
        run_cfg.setdefault("run_sanity", False)
        run_nk_baseline(run_cfg, root=root)
        all_rows.extend(
            read_nk_summary(
                out_dir,
                case=name,
                T=int(run_cfg.get("T", 250)),
                N=int(run_cfg.get("N", 256)),
                R=int(run_cfg.get("R", 30)),
                horizon=int(run_cfg.get("horizon", 20)),
                omega_horizon=int(run_cfg.get("omega_horizon", 1)),
            )
        )

    summary_csv = cfg.get("summary_csv", None)
    if summary_csv is not None:
        write_nk_sweep_csv(_resolve(root, summary_csv), all_rows)

    table_tex = cfg.get("table_tex", None)
    if table_tex is not None:
        methods = cfg.get("methods", ["occbin_bootstrap_pf", "pppf_mixture_rbpf", "plc_copf_pf"])
        if not isinstance(methods, list) or not all(isinstance(m, str) for m in methods):
            raise RuntimeError("nk.sweeps.methods must be a list of strings")
        write_nk_sweep_table_tex(_resolve(root, table_tex), all_rows, methods=methods)


def main() -> int:
    parser = argparse.ArgumentParser(description="Config-driven experiment runner for PPPF paper artifacts.")
    parser.add_argument("--config", required=True, help="Path to a TOML config file.")
    args = parser.parse_args()

    root = repo_root()
    cfg_path = _resolve(root, args.config)
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)

    build_project(cfg.get("build", {}), root=root)

    if cfg.get("linear_gaussian_validate", {}).get("enabled", False):
        run_checked([str(root / "build" / "linear_gaussian_validate")], cwd=root)

    run_burnside(cfg.get("burnside", {}), root=root)
    run_burnside_filter(cfg.get("burnside_filter", {}), root=root)

    nk = cfg.get("nk", {})
    if not isinstance(nk, dict):
        raise RuntimeError("nk must be a table")
    run_nk_baseline(nk.get("baseline", {}), root=root)
    run_nk_ablations(nk.get("ablations", {}), root=root)
    run_nk_sweeps(nk.get("sweeps", {}), root=root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
