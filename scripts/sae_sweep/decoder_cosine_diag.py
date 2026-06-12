"""Decoder pairwise cosine similarity (c_dec) diagnostic over an SAE sweep.

Implements the c_dec metric from Chanin & Garriga-Alonso, "Sparse but Wrong:
Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders" (arXiv:2508.16560):

    c_dec = mean over distinct latent pairs (i, j) of |cos(W_dec_i, W_dec_j)|

c_dec is minimized near the correct L0. Plotting it across a sweep that varies
sparsity (here, lambda_l1) gives a guide to which lambda yields the least feature
mixing (the global / "elbow" minimum).

In this project the SAE decoder is nn.Linear(latent_dim, input_dim), so
decoder.weight has shape (d_model, latent_dim) and each latent's decoder vector
is a COLUMN. We normalize columns (dim=0) before taking pairwise cosines, which
matches the paper's per-latent normalization.

By default c_dec is computed over ALL latents. With --active-only, dead latents
(those that never fire on the eval set) are excluded, so c_dec measures mixing
only among latents the SAE actually uses. The per-latent firing info is read from
the matching analysis_files/*.sae_eval.pt file (firing_count), so nothing is
recomputed; --active-thresh controls how "active" is defined (ever-fired by
default, or a fraction of the max mean activation).

Usage:
    python scripts/sae_sweep/decoder_cosine_diag.py /path/to/sweep_dir
    python scripts/sae_sweep/decoder_cosine_diag.py /path/to/sweep_dir --x mean_active
    python scripts/sae_sweep/decoder_cosine_diag.py /path/to/sweep_dir --active-only
    python scripts/sae_sweep/decoder_cosine_diag.py /path/to/sweep_dir --lambda-min 1e-3 --lambda-max 2e-2

--lambda-min / --lambda-max restrict the sweep to a lambda_l1 range; this filter
applies regardless of which --x axis is plotted.

The sweep dir is expected to contain a sae_checkpoints/ subdir with *.pt SAE
files (the same layout produced by generate_sweep.py / eval_sweep.py). The figure
is written to <sweep_dir>/analysis_plots/decoder_cosine_diag.png by default
(decoder_cosine_diag_active.png when --active-only is set).
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from notation import add_report_flag


def pairwise_decoder_cosine_similarity(w_dec, latent_dim_axis, active_mask=None):
    """Mean absolute pairwise cosine similarity between SAE decoder latents.

    w_dec: decoder weight tensor.
    latent_dim_axis: which axis indexes the latents (the other axis is d_model).
    active_mask: optional 1-D bool tensor over latents; when given, only latents
        with active_mask=True are included in the metric.
    Returns (c_dec, n_latents_used).
    """
    # Move latents to dim 0 so each row is one latent direction of length d_model.
    if latent_dim_axis == 1:
        w_dec = w_dec.t()
    w_dec = w_dec.to(torch.float64)
    if active_mask is not None:
        active_mask = active_mask.to(torch.bool)
        if active_mask.numel() != w_dec.shape[0]:
            raise ValueError(
                f'active_mask length {active_mask.numel()} != n_latents {w_dec.shape[0]}')
        w_dec = w_dec[active_mask]
    h = w_dec.shape[0]
    if h < 2:
        return float('nan'), h
    norm_dec = torch.nn.functional.normalize(w_dec, dim=1)
    dec_sims = norm_dec @ norm_dec.t()
    triu_mask = torch.triu(torch.ones_like(dec_sims), diagonal=1).bool()
    return dec_sims[triu_mask].abs().mean().item(), h


def load_decoder_weight(ckpt, layer_id):
    """Return the (d_model, latent_dim) decoder.weight for the given layer."""
    sae_state = ckpt['sae_state']
    if layer_id not in sae_state:
        raise KeyError(f'layer {layer_id} not in sae_state (have {list(sae_state)})')
    sd = sae_state[layer_id]
    if 'decoder.weight' not in sd:
        raise KeyError(f'decoder.weight not in state dict (have {list(sd)})')
    return sd['decoder.weight']


def load_active_mask(sweep_dir, ckpt_name, thresh):
    """Bool mask (per latent) of active latents, from analysis_files/*.sae_eval.pt.

    thresh in {'ever', '1pct', '10pct'}:
      - 'ever':  latent fired at least once on the eval set (firing_count > 0),
                 which matches the project's dead-feature definition.
      - '1pct':  mean activation > 1%  of the max latent mean activation.
      - '10pct': mean activation > 10% of the max latent mean activation.

    Returns (mask_tensor, eval_path) or (None, None) if the eval file is missing
    or lacks the needed fields.
    """
    eval_path = sweep_dir / 'analysis_files' / (Path(ckpt_name).stem + '.sae_eval.pt')
    if not eval_path.exists():
        return None, None
    ev = torch.load(eval_path, map_location='cpu', weights_only=False)
    if thresh == 'ever':
        fc = ev.get('firing_count')          # (n_pos, latent_dim)
        if fc is None:
            return None, None
        mask = fc.sum(dim=0) > 0
    else:
        fma = ev.get('feature_mean_activations')  # (latent_dim,)
        if fma is None:
            return None, None
        peak = float(fma.max())
        frac = 0.01 if thresh == '1pct' else 0.10
        mask = fma > (frac * peak)
    return mask.bool(), eval_path


def parse_lambda_from_name(name):
    """Extract lambda_l1 from a checkpoint filename like ..._l10.0005_lr1e-4_..."""
    m = re.search(r'_l1([0-9eE.+-]+?)_lr', name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def load_measured_l0(sweep_dir):
    """Map checkpoint basename -> measured L0 (mean_active) from sweep_metrics.csv."""
    csv_path = sweep_dir / 'analysis_files' / 'sweep_metrics.csv'
    if not csv_path.exists():
        return {}
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ckpt = row.get('ckpt')
            ma = row.get('mean_active')
            if ckpt and ma not in (None, ''):
                try:
                    out[ckpt] = float(ma)
                except ValueError:
                    pass
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('sweep_dir',
                        help='Sweep directory containing sae_checkpoints/*.pt')
    parser.add_argument('--x', choices=['lambda', 'mean_active'], default='lambda',
                        help='x-axis: lambda_l1 (default) or measured L0 from '
                             'sweep_metrics.csv')
    parser.add_argument('--lambda-min', type=float, default=None,
                        help='Only include checkpoints with lambda_l1 >= this value')
    parser.add_argument('--lambda-max', type=float, default=None,
                        help='Only include checkpoints with lambda_l1 <= this value')
    parser.add_argument('--active-only', action='store_true',
                        help='Compute c_dec only over active (non-dead) latents, '
                             'using firing info from analysis_files/*.sae_eval.pt')
    parser.add_argument('--active-thresh', choices=['ever', '1pct', '10pct'],
                        default='ever',
                        help='Definition of active when --active-only is set: '
                             "ever-fired (default), >1%% of max mean activation, "
                             'or >10%% of max mean activation')
    parser.add_argument('--outfile', default=None,
                        help='Output figure path (default: '
                             '<sweep_dir>/analysis_plots/decoder_cosine_diag[_active].png)')
    add_report_flag(parser)  # accepted for interface uniformity; no labels to flip
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    ckpt_dir = sweep_dir / 'sae_checkpoints'
    if not ckpt_dir.is_dir():
        raise SystemExit(f'No sae_checkpoints/ subdir under {sweep_dir}')

    ckpt_files = sorted(ckpt_dir.glob('*.pt'))
    if not ckpt_files:
        raise SystemExit(f'No *.pt SAE checkpoints found in {ckpt_dir}')

    measured_l0 = load_measured_l0(sweep_dir)

    records = []  # (lambda_l1, mean_active_or_None, c_dec, name)
    for f in ckpt_files:
        lam = parse_lambda_from_name(f.name)
        if args.lambda_min is not None or args.lambda_max is not None:
            if lam is None:
                print(f'  skip {f.name}: cannot parse lambda for range filter')
                continue
            if args.lambda_min is not None and lam < args.lambda_min:
                continue
            if args.lambda_max is not None and lam > args.lambda_max:
                continue

        try:
            ckpt = torch.load(f, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f'  skip {f.name}: failed to load ({e})')
            continue
        layers = ckpt.get('sae_layers')
        if not layers:
            print(f'  skip {f.name}: no sae_layers')
            continue
        layer_id = layers[0]
        try:
            w_dec = load_decoder_weight(ckpt, layer_id)
        except KeyError as e:
            print(f'  skip {f.name}: {e}')
            continue

        active_mask = None
        n_active_str = ''
        if args.active_only:
            active_mask, eval_path = load_active_mask(
                sweep_dir, f.name, args.active_thresh)
            if active_mask is None:
                print(f'  skip {f.name}: no eval file / firing info for --active-only')
                continue
            n_active_str = f'  n_active={int(active_mask.sum())}/{active_mask.numel()}'

        # decoder.weight is (d_model, latent_dim): latents are columns -> axis 1.
        c_dec, n_used = pairwise_decoder_cosine_similarity(
            w_dec, latent_dim_axis=1, active_mask=active_mask)
        ma = measured_l0.get(f.name)
        if not np.isfinite(c_dec):
            print(f'  skip {f.name}: <2 active latents, c_dec undefined{n_active_str}')
            continue
        records.append((lam, ma, c_dec, f.name))
        lam_s = f'{lam:g}' if lam is not None else '?'
        ma_s = f'{ma:.3f}' if ma is not None else 'NA'
        print(f'  {f.name}: lambda={lam_s}  L0={ma_s}  c_dec={c_dec:.6f}{n_active_str}')

    if not records:
        raise SystemExit('No usable checkpoints; nothing to plot.')

    if args.x == 'mean_active':
        pts = [(ma, c) for (lam, ma, c, n) in records if ma is not None]
        if not pts:
            raise SystemExit(
                'No measured L0 available (sweep_metrics.csv missing or empty). '
                'Re-run with --x lambda.')
        pts.sort(key=lambda t: t[0])
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        xlabel, logx = 'Measured L0 (mean active features)', False
    else:
        pts = [(lam, c) for (lam, ma, c, n) in records if lam is not None]
        if not pts:
            raise SystemExit('Could not parse lambda_l1 from any filename.')
        pts.sort(key=lambda t: t[0])
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        xlabel, logx = 'lambda_1', True

    i_min = int(np.argmin(ys))
    x_min, y_min = xs[i_min], ys[i_min]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, '-o', markersize=4, linewidth=1.5, color='tab:blue')
    ax.scatter([x_min], [y_min], color='red', zorder=5,
               label=f'min: x={x_min:g}, c_dec={y_min:.5f}')
    ax.axvline(x_min, color='red', linestyle='--', linewidth=1, alpha=0.7)
    if logx:
        ax.set_xscale('log')
    scope = (f'active latents only ({args.active_thresh})' if args.active_only
             else 'all latents')
    if args.lambda_min is not None or args.lambda_max is not None:
        lo = 'min' if args.lambda_min is None else f'{args.lambda_min:g}'
        hi = 'max' if args.lambda_max is None else f'{args.lambda_max:g}'
        scope += f', lambda in [{lo}, {hi}]'
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(r'$c_{dec}$  (decoder pairwise |cos|)', fontsize=11)
    ax.set_title(f'Decoder pairwise cosine similarity ({scope})\n{sweep_dir.name}',
                 fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=9)
    plt.tight_layout()

    default_name = ('decoder_cosine_diag_active.png' if args.active_only
                    else 'decoder_cosine_diag.png')
    outfile = (Path(args.outfile) if args.outfile
               else sweep_dir / 'analysis_plots' / default_name)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved to {outfile}')
    print(f'c_dec minimized at {xlabel} = {x_min:g}  (c_dec = {y_min:.6f})')


if __name__ == '__main__':
    main()
