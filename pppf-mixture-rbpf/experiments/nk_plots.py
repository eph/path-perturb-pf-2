#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

from pretty_png_plot import plot_hist_png, plot_lines_png, plot_panels_lines_png


def read_loglik(path: Path):
    by_method = defaultdict(list)
    rt_by_method = defaultdict(list)
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            m = row["method"]
            by_method[m].append(float(row["loglik"]))
            rt_by_method[m].append(float(row["runtime_ms"]))
    return by_method, rt_by_method


def read_ess(path: Path):
    by_method = defaultdict(list)
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            m = row["method"]
            t = int(row["t"])
            ess = float(row["ess"])
            by_method[m].append((t, ess))
    # sort by t
    out = {}
    for m, pairs in by_method.items():
        pairs.sort(key=lambda x: x[0])
        out[m] = ( [p[0] for p in pairs], [p[1] for p in pairs] )
    return out


def read_irf(path: Path):
    t = []
    out = defaultdict(list)
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(int(row["t"]))
            for k, v in row.items():
                if k == "t":
                    continue
                out[k].append(float(v))
    return t, out


def main():
    out_dir = Path("output/nk")
    out_dir.mkdir(parents=True, exist_ok=True)

    ll_path = out_dir / "loglik_repeats.csv"
    ess_path = out_dir / "ess.csv"
    irf_path = out_dir / "irf.csv"

    ll_by_method, rt_by_method = read_loglik(ll_path)

    # Histogram of log-likelihoods across repeats.
    png = out_dir / "loglik_hist.png"
    if plt is not None:
        plt.figure(figsize=(7, 4))
        bins = 15
        for m, vals in ll_by_method.items():
            plt.hist(vals, bins=bins, alpha=0.5, label=m)
        plt.title("NK: log-likelihood across repeats")
        plt.xlabel("loglik")
        plt.ylabel("count")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png, dpi=150)
        plt.close()
    else:
        plot_hist_png(png, ll_by_method, title="NK LOGLIK HIST")
    print(f"Wrote {png}")

    # ESS over time.
    ess = read_ess(ess_path)
    png = out_dir / "ess.png"
    if plt is not None:
        plt.figure(figsize=(7, 4))
        for m, (tt, vv) in ess.items():
            plt.plot(tt, vv, label=m)
        plt.title("NK: ESS over time (rep=0)")
        plt.xlabel("t")
        plt.ylabel("ESS")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png, dpi=150)
        plt.close()
    else:
        # Build a single-line plot per method with shared x.
        for m, (tt, vv) in ess.items():
            pass
        # Use the x-grid from the first method.
        first_m = next(iter(ess.keys()))
        x = ess[first_m][0]
        series = {m: vv for m, (tt, vv) in ess.items()}
        plot_lines_png(png, x, series, title="NK ESS (REP0)")
    print(f"Wrote {png}")

    # IRF comparison near ELB.
    t, irf = read_irf(irf_path)
    png = out_dir / "irf.png"
    if plt is not None:
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        axes[0].plot(t, irf["x_global"], label="Global (Markov)", linewidth=2, color="black")
        axes[0].plot(t, irf["x_occbin"], label="OccBin", linewidth=2)
        axes[0].plot(t, irf["x_pppf"], label="PPPF-mixture mean", linestyle="--")
        if "x_plc" in irf:
            axes[0].plot(t, irf["x_plc"], label="PLC (interp)", linestyle=":")
        axes[0].set_ylabel("x")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, irf["pi_global"], label="Global (Markov)", linewidth=2, color="black")
        axes[1].plot(t, irf["pi_occbin"], label="OccBin", linewidth=2)
        axes[1].plot(t, irf["pi_pppf"], label="PPPF-mixture mean", linestyle="--")
        if "pi_plc" in irf:
            axes[1].plot(t, irf["pi_plc"], label="PLC (interp)", linestyle=":")
        axes[1].set_ylabel("pi")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, irf["i_global"], label="Global (Markov)", linewidth=2, color="black")
        axes[2].plot(t, irf["i_occbin"], label="OccBin", linewidth=2)
        axes[2].plot(t, irf["i_pppf"], label="PPPF-mixture mean", linestyle="--")
        if "i_plc" in irf:
            axes[2].plot(t, irf["i_plc"], label="PLC (interp)", linestyle=":")
        axes[2].set_ylabel("i")
        axes[2].set_xlabel("t")
        axes[2].grid(True, alpha=0.3)

        axes[0].legend()
        fig.suptitle("NK IRF near ELB (negative r^n shock)")
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        plt.close(fig)
    else:
        plot_panels_lines_png(
            png,
            t,
            [
                (
                    "x",
                    {
                        "Global": irf["x_global"],
                        "OccBin": irf["x_occbin"],
                        "PPPF": irf["x_pppf"],
                        "PLC": irf.get("x_plc", irf["x_global"]),
                    },
                ),
                (
                    "pi",
                    {
                        "Global": irf["pi_global"],
                        "OccBin": irf["pi_occbin"],
                        "PPPF": irf["pi_pppf"],
                        "PLC": irf.get("pi_plc", irf["pi_global"]),
                    },
                ),
                (
                    "i",
                    {
                        "Global": irf["i_global"],
                        "OccBin": irf["i_occbin"],
                        "PPPF": irf["i_pppf"],
                        "PLC": irf.get("i_plc", irf["i_global"]),
                    },
                ),
            ],
            title="NK IRF near ELB (negative r^n shock)",
        )
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
