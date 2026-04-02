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


def _format_val(color_by: str, val) -> str:
    """Format a parameter value for the legend label."""
    if color_by == 'lr':
        return f'lr={val:.0e}'
    if color_by == 'lambda_l1':
        return f'λ₁={val:.3g}'
    if color_by in ('batch_size', 'steps', 'train_size'):
        return f'{color_by}={int(val)}'
    return f'{color_by}={val}'


def plot_curves(records, loss_types, max_steps, outfile, color_by='lr',
               yscale='log'):
    layers   = sorted({r['layer'] for r in records})
    all_vals = sorted({r[color_by] for r in records})

    n_rows = len(layers)
    n_cols = len(loss_types)

    cmap = cm.get_cmap('viridis', max(len(all_vals), 1))
    val_color = {v: cmap(i) for i, v in enumerate(all_vals)}

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
                    color=val_color[r[color_by]],
                    linewidth=1.5,
                    label=_format_val(color_by, r[color_by]),
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

            mode_tag = layer_records[0]['mode'] if layer_records else ''
            if mode_tag == 'one_token':
                tok = layer_records[0]['token_idx']
                mode_tag = f'one_token (tok {tok})'
            ax.set_title(f'Layer {layer} — {mode_tag}', fontsize=11)

            ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)

    # Single shared legend — deduplicate labels
    handles_seen = {}
    for ax_row in axes:
        for ax in ax_row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles_seen[l] = h

    sorted_labels = sorted(handles_seen.keys(),
                           key=lambda s: float(s.split('=')[-1]))
    sorted_handles = [handles_seen[l] for l in sorted_labels]

    legend_title = COLOR_BY_LABELS.get(color_by, color_by)
    n_legend_cols = min(len(all_vals), 9)
    n_legend_rows = max(1, -(-len(all_vals) // n_legend_cols))  # ceil division
    bottom_pad = 0.03 + 0.055 * n_legend_rows  # reserve space proportional to legend height

    plt.tight_layout(rect=[0, bottom_pad, 1, 0.96])

    fig.legend(
        sorted_handles, sorted_labels,
        title=legend_title,
        loc='lower center',
        ncol=n_legend_cols,
        bbox_to_anchor=(0.5, 0),
        fontsize=9,
    )

    lambda_vals   = sorted({r['lambda_l1'] for r in records})
    lambda_str    = ', '.join(f'{v:.3g}' for v in lambda_vals)
    curve_sources = sorted({r.get('curve_source', '?') for r in records})
    source_str    = ' / '.join(curve_sources)
    fig.suptitle(
        f'SAE loss curves [{source_str}]  (λ₁ = {lambda_str})',
        fontsize=13, y=0.99,
    )

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
                   help=f'Sweep directory with .pt checkpoints (default: {DEFAULT_SWEEP_DIR})')
    p.add_argument('--outfile', type=str, default=None,
                   help='Output figure path (default: <sweep_dir>/loss_curves.png)')
    p.add_argument('--loss', type=str, default=None,
                   choices=['total', 'recon', 'sparse', 'all'],
                   help='Which loss to plot (default: all)')
    p.add_argument('--max_steps', type=int, default=None,
                   help='Truncate x-axis at this step count')
    p.add_argument('--color_by', type=str, default=None,
                   choices=['lr', 'batch_size', 'steps', 'train_size', 'lambda_l1'],
                   help=f'Parameter to color lines by (default: {DEFAULT_COLOR_BY})')
    p.add_argument('--yscale', type=str, default='log', choices=['linear', 'log'],
                   help='Y-axis scale for loss plots (default: linear)')
    return p.parse_args()


def main():
    args = _parse_args()

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
                yscale=args.yscale)


if __name__ == '__main__':
    main()
