"""Plot Optuna tuning results from a .pt file produced by optuna_tune_sae.py.

Generates three figures saved alongside the .pt file:
  <outname>_convergence.png  -- best value so far vs trial index
  <outname>_scatter.png      -- eval total loss vs each hyperparameter (4 subplots)
  <outname>_report.txt       -- plain-text summary of all trials, sorted by loss

Usage (CLI -- recommended):

    python sae_sweep/plot_optuna.py --results /path/to/optuna_layer3.pt

Or set the DEFAULT_RESULTS constant below and run without arguments:

    python sae_sweep/plot_optuna.py
"""

import argparse
import sys
from pathlib import Path

# =============================================================================
# DEFAULT: used when --results is not provided.
# =============================================================================
DEFAULT_RESULTS = '/work/pcsl/ponsin/Mean_Transformer/SAE/optuna_layer0/optuna_layer0.pt'
# =============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load(results_path: str):
    data = torch.load(results_path, map_location='cpu', weights_only=False)
    assert 'all_trials' in data and 'best_params' in data, (
        f'{results_path} does not look like an optuna_tune_sae.py output.'
    )
    trials = [
        t for t in data['all_trials']
        if t.get('value') is not None and not (
            isinstance(t['value'], float) and (
                t['value'] != t['value']  # nan
            )
        )
    ]
    return data, trials


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def _plot_convergence(trials, outfile: str, layer_id, fixed: dict):
    values = [t['value'] for t in trials]
    best_so_far = np.minimum.accumulate(values)
    xs = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, values, 'o', color='steelblue', markersize=4, alpha=0.6,
            label='Trial loss')
    ax.plot(xs, best_so_far, '-', color='crimson', linewidth=2,
            label='Best so far')

    ax.set_xlabel('Trial', fontsize=11)
    ax.set_ylabel('Eval total loss', fontsize=11)
    ax.set_title(
        f'Optuna convergence -- layer {layer_id}  '
        f'(lambda1={fixed.get("sae_lambda_l1", "?")}, '
        f'latent_dim={fixed.get("sae_latent_dim", "?")})',
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Convergence plot saved to {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scatter plots: loss vs each hyperparameter
# ---------------------------------------------------------------------------

PARAM_LABELS = {
    'sae_lr':                 'Learning rate',
    'sae_steps':              'Steps',
    'sae_lambda_warmup_frac': 'Lambda warmup frac',
    'sae_lr_decay_frac':      'LR decay frac',
}

LOG_PARAMS = {'sae_lr', 'sae_steps'}


def _plot_scatter(trials, outfile: str, layer_id, fixed: dict):
    params = list(PARAM_LABELS.keys())
    n = len(params)

    values = np.array([t['value'] for t in trials])
    best_idx = int(np.argmin(values))

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for col, param in enumerate(params):
        ax = axes[0][col]
        xs = np.array([t['params'].get(param, float('nan')) for t in trials])

        ax.scatter(xs, values, s=20, color='steelblue', alpha=0.6, label='Trials')
        ax.scatter(xs[best_idx], values[best_idx], s=80, color='crimson',
                   zorder=5, label='Best')

        if param in LOG_PARAMS:
            ax.set_xscale('log')

        ax.set_xlabel(PARAM_LABELS[param], fontsize=10)
        ax.set_ylabel('Eval total loss', fontsize=10)
        ax.set_title(PARAM_LABELS[param], fontsize=11)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        if col == 0:
            ax.legend(fontsize=9)

    fig.suptitle(
        f'Param vs eval total loss -- layer {layer_id}  '
        f'(lambda1={fixed.get("sae_lambda_l1", "?")}, '
        f'latent_dim={fixed.get("sae_latent_dim", "?")})',
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Scatter plot saved to {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _write_report(data, trials, outfile: str):
    sorted_trials = sorted(trials, key=lambda t: t['value'])
    best = data['best_params']
    fixed = data.get('fixed', {})

    lines = []
    lines.append('=' * 70)
    lines.append('Optuna SAE tuning report')
    lines.append('=' * 70)
    lines.append(f"Layer:      {fixed.get('sae_layer', '?')}")
    lines.append(f"lambda1:    {fixed.get('sae_lambda_l1', '?')}")
    lines.append(f"latent_dim: {fixed.get('sae_latent_dim', '?')}")
    lines.append(f"Completed trials: {len(trials)}")
    lines.append('')
    lines.append('Best parameters:')
    for k, v in best.items():
        lines.append(f'  {k}: {v}')
    lines.append(f"Best eval total loss: {data['best_value']:.6f}")
    lines.append('')
    lines.append('-' * 70)
    lines.append(f"{'Rank':<6} {'Loss':>12}  {'sae_lr':>12}  {'sae_steps':>10}  "
                 f"{'warmup':>8}  {'lr_decay':>10}")
    lines.append('-' * 70)
    for rank, t in enumerate(sorted_trials, 1):
        p = t['params']
        lines.append(
            f"{rank:<6} {t['value']:>12.6f}  "
            f"{p.get('sae_lr', float('nan')):>12.2e}  "
            f"{int(p.get('sae_steps', 0)):>10d}  "
            f"{p.get('sae_lambda_warmup_frac', float('nan')):>8.3f}  "
            f"{p.get('sae_lr_decay_frac', float('nan')):>10.3f}"
        )
    lines.append('=' * 70)

    text = '\n'.join(lines)
    Path(outfile).write_text(text)
    print(f'Text report saved to {outfile}')
    print('')
    # Also print to stdout
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--results', type=str, default=None,
                   help=f'Path to optuna_tune_sae.py output .pt file '
                        f'(default: {DEFAULT_RESULTS})')
    return p.parse_args()


def main():
    args = _parse_args()
    results_path = args.results or DEFAULT_RESULTS

    data, trials = _load(results_path)
    if not trials:
        print('No completed trials found in results file.')
        sys.exit(0)

    print(f'Loaded {len(trials)} completed trial(s) from {results_path}')

    fixed = data.get('fixed', {})
    layer_id = fixed.get('sae_layer', '?')

    base = str(Path(results_path).with_suffix(''))

    _plot_convergence(trials, f'{base}_convergence.png', layer_id, fixed)
    _plot_scatter(trials, f'{base}_scatter.png', layer_id, fixed)
    _write_report(data, trials, f'{base}_report.txt')


if __name__ == '__main__':
    main()
