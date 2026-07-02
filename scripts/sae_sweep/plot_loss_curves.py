"""Plot SAE training loss curves from a sweep directory.

Usage (CLI — recommended):

    python sae_sweep/plot_loss_curves.py \/
        --sweep_dir /work/pcsl/ponsin/Mean_Transformer/SAE/sweep_round1a_bs \\
        --color_by batch_size

Or edit the DEFAULT_* constants below and run without arguments:

    python sae_sweep/plot_loss_curves.py
"""

import argparse
import sys
from pathlib import Path

# =============================================================================
# DEFAULTS: used when the corresponding CLI argument is not provided.
# =============================================================================

DEFAULT_SWEEP_DIR = '/work/pcsl/ponsin/Mean_Transformer/SAE/sweep_onetok_lr_2ndversion'
DEFAULT_OUTFILE   = None        # None → <SWEEP_DIR>/loss_curves.png
DEFAULT_LOSS      = 'all'       # 'total', 'recon', 'sparse', or 'all'
DEFAULT_MAX_STEPS = None        # None → show all steps
DEFAULT_COLOR_BY  = 'lr'        # 'lr', 'batch_size', 'steps', 'train_size', 'lambda_l1'

# =============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common.notation import sae_label, add_report_flag


# ---------------------------------------------------------------------------
# Checkpoint loading (curves only — no SAE weights needed)
# ---------------------------------------------------------------------------

def _load_curves(ckpt_path: str):
    """Return a flat dict with layer, hyperparameters, and the eval loss curve
    arrays, or None if the file is not a valid single-layer SAE checkpoint.

    Uses sae_eval_curves when available (produced by train_sae.py with the
    separate eval split). Falls back to sae_training_curves for older
    artifacts that predate eval logging."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'sae_state' not in ckpt or 'sae_layers' not in ckpt:
        return None

    layers = [int(x) for x in ckpt['sae_layers']]
    if len(layers) != 1:
        return None

    layer_id = layers[0]

    def _get(key):
        d = ckpt.get(key, {})
        return d.get(layer_id) or d.get(str(layer_id)) or {}

    eval_curves  = _get('sae_eval_curves')
    train_curves = _get('sae_training_curves')

    # Prefer eval curves; fall back to train curves with a warning
    if eval_curves and 'step' in eval_curves:
        curves = eval_curves
        source = 'eval'
    elif train_curves and 'step' in train_curves:
        curves = train_curves
        source = 'train (no eval curves found)'
    else:
        return None

    cfg = ckpt.get('config', None)
    lr         = float(getattr(cfg, 'sae_lr',        float('nan'))) if cfg else float('nan')
    lambda_l1  = float(getattr(cfg, 'sae_lambda_l1', float('nan'))) if cfg else float('nan')
    steps_done = int(getattr(cfg, 'sae_steps',       0))            if cfg else 0

    setup      = ckpt.get('sae_training_setup', {})
    mode       = setup.get('sae_activation_source', 'all_tokens')
    token_idx  = int(setup.get('sae_token_idx', 0))
    batch_size = int(setup.get('sae_sample_batch_size', 0))

    dataset_split = ckpt.get('sae_dataset_split', {})
    train_size = int(dataset_split.get('train_size',
                     getattr(cfg, 'sae_train_size', 0) if cfg else 0))

    return {
        'layer':       layer_id,
        'lr':          lr,
        'lambda_l1':   lambda_l1,
        'steps':       steps_done,
        'batch_size':  batch_size,
        'train_size':  train_size,
        'mode':        mode,
        'token_idx':   token_idx,
        'curve_source': source,
        'step':        np.array(curves['step'],   dtype=float),
        'total':       np.array(curves['total'],  dtype=float),
        'recon':       np.array(curves['recon'],  dtype=float),
        'sparse':      np.array(curves['sparse'], dtype=float),
        'ckpt':        Path(ckpt_path).name,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

LOSS_LABELS = {
    'total':  'Total loss',
    'recon':  'Reconstruction loss (MSE)',
    'sparse': 'Sparsity loss (weighted L1)',
}

COLOR_BY_LABELS = {
    'lr':         'Learning rate',
    'batch_size': 'Batch size',
    'steps':      'Steps',
    'train_size': 'Train size',
    'lambda_l1':  'λ₁',
}


def plot_curves(records, loss_types, max_steps, outfile, color_by='lr',
               yscale='log', report_notation=False):
    import matplotlib.colors as mcolors

    layers   = sorted({r['layer'] for r in records})
    all_vals = sorted({r[color_by] for r in records})

    n_rows = len(layers)
    n_cols = len(loss_types)

    # Continuous color scale over the color_by range, shown as a single colorbar
    # instead of a per-value legend. Log scale for the multiplicative axes
    # (lambda_1, lr); linear otherwise. A degenerate (single-value or
    # non-positive-for-log) range falls back to a linear norm over [min, max].
    cmap = cm.get_cmap('viridis')
    vmin, vmax = (min(all_vals), max(all_vals)) if all_vals else (0.0, 1.0)
    log_scale = color_by in ('lambda_l1', 'lr') and vmin > 0 and vmax > vmin
    if log_scale:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    elif vmax > vmin:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin - 0.5, vmax=vmin + 0.5)

    def val_color(v):
        return cmap(norm(v))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for row, layer in enumerate(layers):
        layer_records = [r for r in records if r['layer'] == layer]
        layer_records.sort(key=lambda r: r[color_by])

        for col, loss_key in enumerate(loss_types):
            ax = axes[row][col]
            all_vals_plotted = []

            for r in layer_records:
                steps = r['step']
                if max_steps is not None:
                    mask  = steps <= max_steps
                    steps = steps[mask]
                    vals  = r[loss_key][mask]
                else:
                    vals  = r[loss_key]

                mask_first = np.ones(len(steps), dtype=bool)
                mask_first[:2] = False
                steps = steps[mask_first]
                vals  = vals[mask_first]

                ax.plot(
                    steps, vals,
                    color=val_color(r[color_by]),
                    linewidth=1.5,
                )
                if len(vals):
                    all_vals_plotted.append(vals)

            # Adapt y-axis to the range of plotted data (excluding step 0)
            if all_vals_plotted:
                concat = np.concatenate(all_vals_plotted)
                ymin, ymax = concat.min(), concat.max()
                if yscale == 'log':
                    ax.set_yscale('log')
                else:
                    margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.1 * abs(ymax)
                    ax.set_ylim(ymin - margin, ymax + margin)

            ax.set_xscale('log')
            ax.set_xlabel('Steps', fontsize=10)
            ax.set_ylabel(LOSS_LABELS[loss_key], fontsize=10)
            ax.set_title(LOSS_LABELS[loss_key], fontsize=11)

            ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)

    # Single shared colorbar over the color_by range (replaces the per-value
    # legend). The ScalarMappable carries the same cmap/norm used for the lines.
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_label = COLOR_BY_LABELS.get(color_by, color_by)
    cbar = fig.colorbar(
        sm, ax=axes.ravel().tolist(),
        orientation='horizontal',
        fraction=0.05, pad=0.12, aspect=40,
    )
    cbar.set_label(cbar_label, fontsize=10)

    # Suptitle: name the SAE(s) by index. One sweep dir is usually a single SAE.
    if len(layers) == 1:
        sae_str = f' — {sae_label(layers[0], report_notation)}'
    else:
        sae_str = ' — ' + ', '.join(
            sae_label(l, report_notation) for l in layers)
    fig.suptitle(
        f'SAE Loss curves{sae_str}',
        fontsize=13, y=0.99,
    )

    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-sweep comparison (one color per sweep dir; formerly plot_multi_sweep.py)
# ---------------------------------------------------------------------------

def plot_multi_curves(all_records, loss_types, max_steps, outfile, source_labels,
                      layer_filter=None, report_notation=False):
    """Loss curves from several sweep dirs on one figure, one color per dir."""
    layers = sorted({r['layer'] for r in all_records})
    if layer_filter is not None:
        layers = [l for l in layers if l in layer_filter]

    n_rows = len(layers)
    n_cols = len(loss_types)

    cmap = cm.get_cmap('tab10', max(len(source_labels), 1))
    label_color = {lab: cmap(i) for i, lab in enumerate(source_labels)}

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for row, layer in enumerate(layers):
        layer_recs = [r for r in all_records if r['layer'] == layer]
        for col, loss_key in enumerate(loss_types):
            ax = axes[row][col]
            for r in layer_recs:
                steps = r['step']
                if max_steps is not None:
                    mask = steps <= max_steps
                    steps = steps[mask]
                    vals = r[loss_key][mask]
                else:
                    vals = r[loss_key]
                pos = steps > 0
                ax.plot(steps[pos], vals[pos],
                        color=label_color[r['source_label']],
                        linewidth=1.5, alpha=0.8, label=r['source_label'])
            ax.set_xscale('log')
            ax.set_xlabel('Steps', fontsize=10)
            ax.set_ylabel(LOSS_LABELS[loss_key], fontsize=10)
            ax.set_title(sae_label(layer, report_notation), fontsize=11)
            ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)

    handles_seen = {}
    for ax_row in axes:
        for ax in ax_row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in handles_seen:
                    handles_seen[l] = h
    fig.legend(list(handles_seen.values()), list(handles_seen.keys()),
               title='Sweep', loc='lower center',
               ncol=min(len(source_labels), 6),
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle('Cross-sweep comparison', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--sweep_dir', type=str, default=None,
                   help=f'Single sweep directory with .pt checkpoints '
                        f'(default: {DEFAULT_SWEEP_DIR})')
    p.add_argument('--sweep_dirs', nargs='+', default=None,
                   help='Two or more sweep directories to compare on one figure '
                        '(one color per dir). Overrides --sweep_dir / --color_by.')
    p.add_argument('--labels', nargs='+', default=None,
                   help='Legend labels for --sweep_dirs (default: directory names)')
    p.add_argument('--layer', type=str, default=None,
                   help='Comma-separated layers to plot in --sweep_dirs mode '
                        '(default: all found)')
    p.add_argument('--outfile', type=str, default=None,
                   help='Output figure path (default: <sweep_dir>/loss_curves.png)')
    p.add_argument('--loss', type=str, default=None,
                   choices=['total', 'recon', 'sparse', 'all'],
                   help='Which loss to plot (default: all)')
    p.add_argument('--max_steps', type=int, default=None,
                   help='Truncate x-axis at this step count')
    p.add_argument('--color_by', type=str, default=None,
                   choices=['lr', 'batch_size', 'steps', 'train_size', 'lambda_l1'],
                   help=f'Parameter to color lines by, single-sweep mode '
                        f'(default: {DEFAULT_COLOR_BY})')
    p.add_argument('--yscale', type=str, default='log', choices=['linear', 'log'],
                   help='Y-axis scale for loss plots (default: log)')
    add_report_flag(p)
    return p.parse_args()


def _loss_types(loss):
    if loss == 'all':
        return ['total', 'recon', 'sparse']
    if loss in LOSS_LABELS:
        return [loss]
    print(f'Unknown loss value "{loss}". Choose from: total, recon, sparse, all')
    sys.exit(1)


def _run_multi(args):
    sweep_dirs = [Path(d) for d in args.sweep_dirs]
    labels = args.labels or [d.name for d in sweep_dirs]
    if len(labels) != len(sweep_dirs):
        print('ERROR: number of --labels must match number of --sweep_dirs',
              file=sys.stderr)
        sys.exit(1)
    if not args.outfile:
        print('ERROR: --outfile is required with --sweep_dirs', file=sys.stderr)
        sys.exit(1)
    loss_types = _loss_types(args.loss or DEFAULT_LOSS)
    layer_filter = (set(int(x.strip()) for x in args.layer.split(','))
                    if args.layer is not None else None)

    all_records = []
    for sweep_dir, label in zip(sweep_dirs, labels):
        ckpt_files = sorted(sweep_dir.glob('*.pt'))
        if not ckpt_files:
            print(f'WARNING: No .pt files in {sweep_dir}')
            continue
        for f in ckpt_files:
            r = _load_curves(str(f))
            if r is not None:
                r['source_label'] = label
                all_records.append(r)
    if not all_records:
        print('No valid checkpoints found across all directories.')
        sys.exit(0)
    source_labels = list(dict.fromkeys(r['source_label'] for r in all_records))
    plot_multi_curves(all_records, loss_types, args.max_steps, args.outfile,
                      source_labels, layer_filter=layer_filter,
                      report_notation=args.report_notation)


def main():
    args = _parse_args()

    if args.sweep_dirs:
        _run_multi(args)
        return

    sweep_dir = Path(args.sweep_dir or DEFAULT_SWEEP_DIR)
    loss      = args.loss or DEFAULT_LOSS
    max_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS
    color_by  = args.color_by or DEFAULT_COLOR_BY
    outfile   = args.outfile or DEFAULT_OUTFILE or str(sweep_dir / 'loss_curves.png')

    ckpt_files = sorted(sweep_dir.glob('*.pt'))
    if not ckpt_files:
        print(f'No .pt files found in {sweep_dir}')
        sys.exit(0)

    if loss == 'all':
        loss_types = ['total', 'recon', 'sparse']
    elif loss in LOSS_LABELS:
        loss_types = [loss]
    else:
        print(f'Unknown loss value "{loss}". Choose from: total, recon, sparse, all')
        sys.exit(1)

    print(f'Loading {len(ckpt_files)} checkpoint(s) from {sweep_dir} ...')
    records = []
    for f in ckpt_files:
        r = _load_curves(str(f))
        if r is None:
            print(f'  Skipping (not a valid SAE checkpoint): {f.name}')
        else:
            records.append(r)

    if not records:
        print('No valid SAE checkpoints found.')
        sys.exit(0)

    layers = sorted({r['layer'] for r in records})
    vals   = sorted({r[color_by] for r in records})
    print(f'Layers: {layers}')
    print(f'{color_by} values: {vals}')
    print(f'Checkpoints plotted: {len(records)}')

    plot_curves(records, loss_types, max_steps, outfile, color_by=color_by,
                yscale=args.yscale, report_notation=args.report_notation)


if __name__ == '__main__':
    main()
