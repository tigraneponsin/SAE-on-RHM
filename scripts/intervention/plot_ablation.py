# Usage:
#   python plot_ablation.py CSV [CSV ...] [--modes MODE ...] [--outfig PATH] [--dpi N] [--no-sharey]
#
# Positional args: one or more CSVs produced by ablate_tokens.py, one per transformer layer.
#                  Each CSV yields one subplot; subplot title is taken from the layer index in the data.
# --modes    : ablation modes to show; any subset of {zero, mean, resample}; default: zero
# --outfig   : path to save the figure (pdf/png/...); if omitted, opens an interactive window
# --dpi      : resolution for saved output (default: 150)
# --no-sharey: give each subplot its own y-axis scale instead of sharing one
#
# Examples:
#   python plot_ablation.py L0.csv L1.csv L2.csv
#   python plot_ablation.py L0.csv L1.csv L2.csv --modes zero mean resample --outfig ablation.pdf
#   python plot_ablation.py L1.csv --modes mean --no-sharey

import argparse
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

MODES = ["zero", "mean", "resample"]
MODE_COLORS = {m: plt.cm.tab10(i / 10) for i, m in enumerate(MODES)}
ROW_RE = re.compile(r"^L(\d+)_pos(\d+)_(zero|mean|resample)$")


def parse_csv(path):
    df = pd.read_csv(path)
    baseline_mask = df["name"].isin(["baseline", "__baseline__"])
    baseline_rows = df[baseline_mask]
    if baseline_rows.empty:
        sys.exit(f"No baseline row found in {path}")
    baseline_err = baseline_rows.iloc[0]["err_over_random"]

    layer_idx = None
    positions = {}
    for _, row in df[~baseline_mask].iterrows():
        m = ROW_RE.match(str(row["name"]))
        if m is None:
            continue
        if layer_idx is None:
            layer_idx = int(m.group(1))
        pos = int(m.group(2))
        mode = m.group(3)
        positions.setdefault(mode, {})[pos] = row["err_over_random"]

    return layer_idx, baseline_err, positions


def main():
    parser = argparse.ArgumentParser(
        description="Plot token-ablation results across transformer layers."
    )
    parser.add_argument("csvs", nargs="+", metavar="CSV",
                        help="One CSV per layer, produced by ablate_tokens.py")
    parser.add_argument("--modes", nargs="+", default=["zero"],
                        choices=MODES, metavar="MODE",
                        help="Ablation modes to plot (default: zero)")
    parser.add_argument("--outfig", default=None, metavar="PATH",
                        help="Save figure to path; show interactively if omitted")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for saved figure (default: 150)")
    parser.add_argument("--no-sharey", action="store_true",
                        help="Disable shared y-axis across subplots")
    args = parser.parse_args()

    layer_data = []
    for path in args.csvs:
        layer_idx, baseline_err, positions = parse_csv(path)
        layer_data.append((layer_idx, baseline_err, positions))

    n = len(layer_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=not args.no_sharey,
                             squeeze=False)
    axes = axes[0]
    fig.suptitle("Token ablation: normalized classification error", fontsize=12)

    for ax, (layer_idx, baseline_err, positions) in zip(axes, layer_data):
        for mode in args.modes:
            if mode not in positions:
                continue
            pos_dict = positions[mode]
            xs = sorted(pos_dict.keys())
            ys = [pos_dict[x] for x in xs]
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5,
                    color=MODE_COLORS[mode], label=mode)

        n_pos = max(
            (max(d.keys()) for d in positions.values() if d),
            default=0
        )
        ax.axhline(baseline_err, color="grey", linestyle="--",
                   linewidth=1.0, label="baseline")
        title = f"Layer {layer_idx}" if layer_idx is not None else "Layer ?"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Token position", fontsize=9)
        ax.set_ylabel("err_over_random", fontsize=9)
        ax.set_xticks(range(n_pos + 1))
        ax.tick_params(labelsize=8)
        ax.grid(linestyle="--", linewidth=0.4, alpha=0.6)
        ax.legend(fontsize=8, frameon=False, loc="upper left")

    plt.tight_layout()

    if args.outfig:
        fig.savefig(args.outfig, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {args.outfig}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
