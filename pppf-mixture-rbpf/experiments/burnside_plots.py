#!/usr/bin/env python3

import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

from pretty_png_plot import plot_lines_png


def read_csv(path: Path):
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    t = [int(row["t"]) for row in rows]
    x = [float(row["x"]) for row in rows]
    v_exact = [float(row["v_exact"]) for row in rows]
    v_ce = [float(row["v_ce"]) for row in rows]
    v_ut = [float(row["v_ut"]) for row in rows]
    return t, x, v_exact, v_ce, v_ut


def main():
    out_dir = Path("output/burnside")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(out_dir.glob("irf_*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSVs found under {out_dir}")

    for csv_path in csv_paths:
        t, x, v_exact, v_ce, v_ut = read_csv(csv_path)
        tag = csv_path.stem.replace("irf_", "")

        png_path = out_dir / f"{csv_path.stem}.png"
        if plt is not None:
            plt.figure(figsize=(7, 4))
            plt.plot(t, v_exact, label="Exact", linewidth=2)
            plt.plot(t, v_ce, label="PP-CE", linestyle="--")
            plt.plot(t, v_ut, label="PP-UT (1-step)", linestyle="-.")
            plt.title(f"Burnside IRF: shock = {tag}σ")
            plt.xlabel("t")
            plt.ylabel("v")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close()
        else:
            plot_lines_png(
                png_path,
                t,
                {"Exact": v_exact, "PP-CE": v_ce, "PP-UT": v_ut},
                title=f"Burnside IRF {tag}SIG",
            )
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
