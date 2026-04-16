"""Evaluate all trained SAE artifacts produced by a sweep.

For every .pt file found in --sweep_dir that contains SAE checkpoints, this
script reports:
  - SAE training metrics (total / recon / sparse loss at end of training)
  - Activity statistics on a fresh eval split (dead features, mean active)
  - Classification accuracy degradation when the SAE reconstruction replaces
    the transformer block output (per layer and all layers stacked)

Results are printed as a table and optionally saved to a CSV file.

Usage:
    python sae_sweep/eval_sweep.py \\
        --sweep_dir /path/to/sweep/output/ \\
        [--eval_size 16384] \\
        [--eval_seed 99999] \\
        [--batch_size 256] \\
        [--device cuda] \\
        [--outcsv /path/to/results.csv]
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Add the repo root to sys.path so we can import init, models, datasets
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.common.sae_loading import load_sae as _shared_load_sae
from scripts.common.sae_loading import load_transformer as _shared_load_transformer


# ---------------------------------------------------------------------------
# Data / model helpers
# ---------------------------------------------------------------------------
# The loader helpers used to live here; they now live in
# scripts/common/sae_loading.py so scripts/sae_direct_analysis/analyze_sae.py
# can reuse them without duplication. Thin wrappers preserve this module's
# original call signatures.

def _load_transformer(train_output_path: str, eval_size: int, eval_seed: int,
                      batch_size: int, device: str):
    model, loader, cfg, _rules, rules_source = _shared_load_transformer(
        train_output_path, eval_size, eval_seed, batch_size, device
    )
    return model, loader, cfg, rules_source


def _load_sae(ckpt_path: str, input_dim: int | None, device: str,
              load_model: bool = True):
    return _shared_load_sae(ckpt_path, input_dim, device, load_model=load_model)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _activity_stats(model, loader, sae, layer_id: int, mode: str, device: str,
                    token_idx: int = 0, act_scale: float = 1.0,
                    token_subset=None, per_position: bool = True) -> dict:
    """Dead-feature and mean-active-feature stats over eval data.

    Aggregate metrics (unchanged) sum over all tokens the SAE sees. When
    per_position is True (default), returns additional keys prefixed with
    'per_position_' giving one value per effective token position:
      - all_tokens mode: T_eff = num leaf tokens, positions 0..T_eff-1
      - one_token mode:  T_eff = 1, positions = [token_idx]
      - cls_token mode:  T_eff = 1, positions = [-1] (sentinel)

    When token_subset is supplied AND mode == 'all_tokens', the returned dict
    also includes 'subset_*' aggregate metrics computed over only those leaf
    positions (exact re-aggregation, not a re-run).
    """
    has_cls = hasattr(model, 'cls_token')
    latent_dim = sae.latent_dim
    num_leaves = int(getattr(model, 'block_size'))

    if mode == 'cls_token':
        position_indices = [-1]
        T_eff = 1
    elif mode == 'one_token':
        position_indices = [int(token_idx)]
        T_eff = 1
    else:
        position_indices = list(range(num_leaves))
        T_eff = num_leaves

    ever_active = torch.zeros(latent_dim, dtype=torch.bool, device=device)
    feature_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    active_sum = 0.0
    total_tokens = 0
    mean_active_1pct_sum = 0.0
    mean_active_10pct_sum = 0.0

    # Per-position accumulators
    ever_active_pp = torch.zeros(T_eff, latent_dim, dtype=torch.bool, device=device)
    feature_sum_pp = torch.zeros(T_eff, latent_dim, dtype=torch.float64, device=device)
    active_sum_pp = torch.zeros(T_eff, dtype=torch.float64, device=device)
    count_pp = torch.zeros(T_eff, dtype=torch.float64, device=device)
    active_1pct_sum_pp = torch.zeros(T_eff, dtype=torch.float64, device=device)
    active_10pct_sum_pp = torch.zeros(T_eff, dtype=torch.float64, device=device)

    dec_norms = sae.decoder_feature_norms().to(device)
    buf = []
    hook = model.blocks[layer_id].register_forward_hook(
        lambda _m, _i, o: buf.append(o.detach())
    )

    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device))
            if not buf:
                continue
            act = buf.pop(0)
            if mode == 'cls_token':
                act = act[:, :1, :]
            elif mode == 'one_token':
                offset = 1 if has_cls else 0
                act = act[:, offset + token_idx : offset + token_idx + 1, :]
            elif has_cls:
                act = act[:, 1:, :]
            if act.numel() == 0:
                continue
            B, Tcur, D = act.shape
            if Tcur != T_eff:
                raise RuntimeError(
                    f'unexpected token slice width at layer {layer_id}: '
                    f'got {Tcur}, expected {T_eff} for mode={mode}'
                )
            act_flat = (act.reshape(-1, D) * act_scale)
            _, z = sae(act_flat)
            features_flat = z * dec_norms.unsqueeze(0)   # weighted activations
            features = features_flat.view(B, T_eff, latent_dim)

            is_active = features > 0
            token_max = features.max(dim=2, keepdim=True).values
            is_1pct = features > 0.01 * token_max
            is_10pct = features > 0.10 * token_max

            active_sum += is_active.float().sum().item()
            total_tokens += B * T_eff
            mean_active_1pct_sum += is_1pct.float().sum().item()
            mean_active_10pct_sum += is_10pct.float().sum().item()
            ever_active |= is_active.any(dim=0).any(dim=0)
            feature_sum += features.sum(dim=(0, 1)).double()

            if per_position:
                active_sum_pp += is_active.sum(dim=(0, 2)).double()
                active_1pct_sum_pp += is_1pct.sum(dim=(0, 2)).double()
                active_10pct_sum_pp += is_10pct.sum(dim=(0, 2)).double()
                count_pp += B
                ever_active_pp |= is_active.any(dim=0)  # [T_eff, latent_dim]
                feature_sum_pp += features.sum(dim=0).double()  # [T_eff, latent_dim]

    hook.remove()

    dead = int((~ever_active).sum())

    # Per-feature mean weighted activation
    feature_mean = (feature_sum / max(total_tokens, 1)).float()

    # Inverse Participation Ratio (effective number of features)
    sum_a = feature_mean.sum().item()
    sum_a2 = feature_mean.pow(2).sum().item()
    ipr = (sum_a ** 2) / sum_a2 if sum_a2 > 0 else 0.0

    # Threshold-based active counts (fraction of max mean activation)
    feat_max = feature_mean.max().item() if latent_dim > 0 else 0.0
    active_above_1pct = int((feature_mean > 0.01 * feat_max).sum().item()) if feat_max > 0 else 0
    active_above_10pct = int((feature_mean > 0.10 * feat_max).sum().item()) if feat_max > 0 else 0

    result = {
        'dead_features': dead,
        'dead_ratio': dead / max(latent_dim, 1),
        'mean_active': active_sum / max(total_tokens, 1),
        'mean_active_ratio': active_sum / max(total_tokens * latent_dim, 1),
        'ipr': ipr,
        'active_above_1pct': active_above_1pct,
        'active_above_10pct': active_above_10pct,
        'mean_active_above_1pct': mean_active_1pct_sum / max(total_tokens, 1),
        'mean_active_above_10pct': mean_active_10pct_sum / max(total_tokens, 1),
        'feature_mean_activations': feature_mean.cpu(),
    }

    if per_position:
        count_safe = count_pp.clamp(min=1)
        feature_mean_pp = (feature_sum_pp / count_safe.unsqueeze(1)).float()
        sa_pp = feature_mean_pp.sum(dim=1)
        sa2_pp = feature_mean_pp.pow(2).sum(dim=1)
        ipr_pp = torch.where(sa2_pp > 0, sa_pp.pow(2) / sa2_pp, torch.zeros_like(sa_pp))
        ever_count_pp = ever_active_pp.sum(dim=1)

        fm_max_pp = feature_mean_pp.max(dim=1).values
        gt_1pct_pp = (feature_mean_pp > 0.01 * fm_max_pp.unsqueeze(1)).sum(dim=1)
        gt_10pct_pp = (feature_mean_pp > 0.10 * fm_max_pp.unsqueeze(1)).sum(dim=1)
        zero_mask = fm_max_pp <= 0
        gt_1pct_pp = torch.where(zero_mask, torch.zeros_like(gt_1pct_pp), gt_1pct_pp)
        gt_10pct_pp = torch.where(zero_mask, torch.zeros_like(gt_10pct_pp), gt_10pct_pp)

        result['per_position_positions'] = list(position_indices)
        result['per_position_mean_active'] = (active_sum_pp / count_safe).float().cpu()
        result['per_position_mean_active_ratio'] = (
            active_sum_pp / count_safe / max(latent_dim, 1)
        ).float().cpu()
        result['per_position_ever_active'] = ever_count_pp.cpu()
        result['per_position_dead_features'] = (latent_dim - ever_count_pp).cpu()
        result['per_position_ipr'] = ipr_pp.cpu()
        result['per_position_mean_active_above_1pct'] = (
            active_1pct_sum_pp / count_safe
        ).float().cpu()
        result['per_position_mean_active_above_10pct'] = (
            active_10pct_sum_pp / count_safe
        ).float().cpu()
        result['per_position_active_above_1pct'] = gt_1pct_pp.cpu()
        result['per_position_active_above_10pct'] = gt_10pct_pp.cpu()
        result['per_position_feature_mean_activations'] = feature_mean_pp.cpu()

        if token_subset is not None and mode == 'all_tokens':
            sel = torch.tensor(list(token_subset), dtype=torch.long, device=device)
            if sel.numel() == 0:
                raise ValueError('token_subset must not be empty')
            if (sel < 0).any() or (sel >= T_eff).any():
                raise ValueError(
                    f'token_subset contains out-of-range leaf indices: {list(token_subset)} '
                    f'(valid: 0..{T_eff - 1})'
                )
            sub_active = float(active_sum_pp[sel].sum().item())
            sub_1pct = float(active_1pct_sum_pp[sel].sum().item())
            sub_10pct = float(active_10pct_sum_pp[sel].sum().item())
            sub_count = float(count_pp[sel].sum().item())
            sub_ever = ever_active_pp[sel].any(dim=0)
            sub_dead = int((~sub_ever).sum().item())
            sub_feature_sum = feature_sum_pp[sel].sum(dim=0)
            sub_feature_mean = (sub_feature_sum / max(sub_count, 1.0)).float()
            sub_sa = float(sub_feature_mean.sum().item())
            sub_sa2 = float(sub_feature_mean.pow(2).sum().item())
            sub_ipr = (sub_sa ** 2) / sub_sa2 if sub_sa2 > 0 else 0.0
            sub_fm_max = float(sub_feature_mean.max().item()) if latent_dim > 0 else 0.0
            sub_above_1pct = (int((sub_feature_mean > 0.01 * sub_fm_max).sum().item())
                              if sub_fm_max > 0 else 0)
            sub_above_10pct = (int((sub_feature_mean > 0.10 * sub_fm_max).sum().item())
                               if sub_fm_max > 0 else 0)

            result['subset_positions'] = list(token_subset)
            result['subset_dead_features'] = sub_dead
            result['subset_dead_ratio'] = sub_dead / max(latent_dim, 1)
            result['subset_mean_active'] = sub_active / max(sub_count, 1.0)
            result['subset_mean_active_ratio'] = sub_active / max(sub_count * latent_dim, 1.0)
            result['subset_ipr'] = sub_ipr
            result['subset_active_above_1pct'] = sub_above_1pct
            result['subset_active_above_10pct'] = sub_above_10pct
            result['subset_mean_active_above_1pct'] = sub_1pct / max(sub_count, 1.0)
            result['subset_mean_active_above_10pct'] = sub_10pct / max(sub_count, 1.0)

    return result


def _eval_classification(model, loader, device: str, hooks=None):
    """Return (accuracy, mean_cross_entropy) over the loader."""
    handles = []
    if hooks:
        for module, fn in hooks:
            handles.append(module.register_forward_hook(fn))

    correct = total = 0
    total_ce = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_ce += F.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)

    for h in handles:
        h.remove()

    return correct / total, total_ce / total


def _make_sae_hook(sae, mode: str, has_cls: bool, token_idx: int = 0,
                   act_scale: float = 1.0):
    def hook(_m, _i, output):
        out = output.clone()
        if mode == 'cls_token':
            sl_shape = output[:, :1, :].shape
            flat = out[:, :1, :].reshape(-1, out.size(-1)) * act_scale
            recon, _ = sae(flat)
            out[:, :1, :] = (recon / act_scale).reshape(sl_shape)
        elif mode == 'one_token':
            offset = 1 if has_cls else 0
            pos = offset + token_idx
            sl_shape = output[:, pos : pos + 1, :].shape
            flat = out[:, pos : pos + 1, :].reshape(-1, out.size(-1)) * act_scale
            recon, _ = sae(flat)
            out[:, pos : pos + 1, :] = (recon / act_scale).reshape(sl_shape)
        else:
            sl = slice(1, None) if has_cls else slice(0, None)
            sl_shape = output[:, sl, :].shape
            flat = out[:, sl, :].reshape(-1, out.size(-1)) * act_scale
            recon, _ = sae(flat)
            out[:, sl, :] = (recon / act_scale).reshape(sl_shape)
        return out
    return hook


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _scan_checkpoints(sweep_dir: Path):
    return sorted(sweep_dir.glob('*.pt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep_dir', required=True,
                        help='Directory containing SAE .pt checkpoints from the sweep')
    parser.add_argument('--eval_size', type=int, default=2**15,
                        help='Number of RHM samples for evaluation (default: 32768)')
    parser.add_argument('--eval_seed', type=int, default=99999,
                        help='Random seed for the eval split (default: 99999)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for eval forward passes (default: 256)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: cuda if available, else cpu)')
    parser.add_argument('--outcsv', type=str, default=None,
                        help='Optional path to save results as CSV')
    parser.add_argument('--subset_positions', type=str, default=None,
                        help='Comma-separated leaf positions (e.g. "0,2,4,6"). '
                             'When set, for each all_tokens SAE the CSV will '
                             'include subset_* columns computed over these '
                             'positions only. No effect on one_token / '
                             'cls_token SAEs.')
    parser.add_argument('--per_position_csv', type=str, default=None,
                        help='Optional path to save a long-format per-position '
                             'metrics CSV (one row per (ckpt, leaf_position)).')
    args = parser.parse_args()

    token_subset = None
    if args.subset_positions:
        try:
            token_subset = [int(tok) for tok in args.subset_positions.split(',')
                            if tok.strip() != '']
        except ValueError:
            print(f'ERROR: --subset_positions must be comma-separated ints, '
                  f'got: {args.subset_positions!r}')
            sys.exit(2)
        if not token_subset:
            print('ERROR: --subset_positions parsed to an empty list')
            sys.exit(2)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    sweep_dir = Path(args.sweep_dir)

    ckpt_files = _scan_checkpoints(sweep_dir)
    if not ckpt_files:
        print(f'No .pt files found in {sweep_dir}')
        sys.exit(0)

    print(f'Found {len(ckpt_files)} checkpoint file(s) in {sweep_dir}')
    print(f'Eval: size={args.eval_size}, seed={args.eval_seed}, batch={args.batch_size}, device={device}')
    print()

    # Load and parse all valid SAE checkpoints
    records = []
    for ckpt_file in ckpt_files:
        entry = _load_sae(str(ckpt_file), input_dim=None, device='cpu', load_model=False)
        if entry is None:
            print(f'  Skipping (not a valid single-layer SAE checkpoint): {ckpt_file.name}')
        else:
            records.append(entry)

    if not records:
        print('No valid SAE checkpoints found.')
        sys.exit(0)

    # Group by transformer artifact so we load each transformer only once
    by_transformer = {}
    for r in records:
        by_transformer.setdefault(r['train_output'], []).append(r)

    rows = []  # one row per SAE

    for train_output_path, group in by_transformer.items():
        print(f'Loading transformer from: {train_output_path}')
        try:
            model, eval_loader, cfg, resolved_rules_source = _load_transformer(
                train_output_path, args.eval_size, args.eval_seed, args.batch_size, device
            )
        except Exception as exc:
            print(f'  ERROR loading transformer: {exc}')
            for r in group:
                rows.append({'ckpt': r['ckpt_path'], 'error': str(exc)})
            continue

        print(f'  Rules source: {resolved_rules_source}')
        has_cls = hasattr(model, 'cls_token')
        random_err = 1.0 - 1.0 / model.num_classes

        # Baseline (no SAE)
        baseline_acc, baseline_ce = _eval_classification(model, eval_loader, device)
        baseline_err = 1.0 - baseline_acc
        print(f'  Baseline: acc={baseline_acc:.4f}  err={baseline_err:.6f}  CE={baseline_ce:.4f}')

        # Reload each SAE with the correct input_dim now that we know it
        for r in group:
            ckpt_path = r['ckpt_path']
            layer_id = r['layer_id']
            mode = r['setup'].get('sae_activation_source', 'all_tokens')
            train_m = r['train_metrics']

            token_idx = r.get('sae_token_idx', 0)
            print(f'  Evaluating: {Path(ckpt_path).name}  (layer={layer_id}, mode={mode}'
                  + (f', token={token_idx}' if mode == 'one_token' else '') + ')')

            # Verify the SAE was trained with the same RHM rules as the transformer.
            sae_rules_source = r.get('sae_rules_source', None)
            if sae_rules_source is not None and sae_rules_source != resolved_rules_source:
                msg = (
                    f"Rules source mismatch for {Path(ckpt_path).name}: "
                    f"SAE trained with rules_source='{sae_rules_source}' but "
                    f"transformer artifact resolves to rules_source='{resolved_rules_source}'. "
                    f"Skipping this checkpoint."
                )
                print(f'    ERROR: {msg}')
                rows.append({'ckpt': ckpt_path, 'layer': layer_id, 'error': msg})
                continue

            act_scale = r.get('act_scale', 1.0)
            if act_scale != 1.0:
                print(f'    act_scale={act_scale:.6f} (activations were normalized during training)')

            try:
                entry = _load_sae(ckpt_path, input_dim=model.embedding_dim, device=device)
                sae = entry['sae']
                latent_dim = entry['latent_dim']
            except Exception as exc:
                print(f'    ERROR loading SAE: {exc}')
                rows.append({'ckpt': ckpt_path, 'layer': layer_id, 'error': str(exc)})
                continue

            # Activity stats
            try:
                act_stats = _activity_stats(
                    model, eval_loader, sae, layer_id, mode, device,
                    token_idx=token_idx, act_scale=act_scale,
                    token_subset=token_subset if mode == 'all_tokens' else None,
                    per_position=True,
                )
            except Exception as exc:
                print(f'    ERROR computing activity stats: {exc}')
                act_stats = {'dead_features': -1, 'dead_ratio': -1.0,
                             'mean_active': -1.0, 'mean_active_ratio': -1.0,
                             'ipr': -1.0, 'active_above_1pct': -1,
                             'active_above_10pct': -1,
                             'mean_active_above_1pct': -1.0,
                             'mean_active_above_10pct': -1.0,
                             'feature_mean_activations': None}

            # Classification eval with this SAE applied to its layer
            try:
                hook_fn = _make_sae_hook(sae, mode, has_cls, token_idx=token_idx,
                                         act_scale=act_scale)
                sae_acc, sae_ce = _eval_classification(
                    model, eval_loader, device,
                    hooks=[(model.blocks[layer_id], hook_fn)]
                )
                sae_err = 1.0 - sae_acc
                norm_err = sae_err / random_err if random_err > 0 else float('nan')
            except Exception as exc:
                print(f'    ERROR during classification eval: {exc}')
                sae_acc = sae_err = norm_err = sae_ce = float('nan')

            # Pull hyperparams from the embedded config / setup
            cfg_s = r['config']
            sae_lr = getattr(cfg_s, 'sae_lr', None) if cfg_s else None
            sae_l1 = getattr(cfg_s, 'sae_lambda_l1', None) if cfg_s else None
            sae_steps = train_m.get('steps', None)
            sae_bs = r['setup'].get('sae_sample_batch_size', None)

            row = {
                'ckpt': Path(ckpt_path).name,
                'layer': layer_id,
                'mode': mode,
                'token_idx': token_idx if mode == 'one_token' else None,
                'latent_dim': latent_dim,
                'lambda_l1': sae_l1,
                'lr': sae_lr,
                'steps': sae_steps,
                'batch_size': sae_bs,
                # training metrics (end of training)
                'train_total_loss': train_m.get('total_loss', float('nan')),
                'train_recon_loss': train_m.get('recon_loss', float('nan')),
                'train_sparse_loss': train_m.get('sparse_loss', float('nan')),
                # activity stats on eval split
                'dead_features': act_stats['dead_features'],
                'dead_ratio': act_stats['dead_ratio'],
                'mean_active': act_stats['mean_active'],
                'mean_active_ratio': act_stats['mean_active_ratio'],
                'ipr': act_stats['ipr'],
                'active_above_1pct': act_stats['active_above_1pct'],
                'active_above_10pct': act_stats['active_above_10pct'],
                'mean_active_above_1pct': act_stats['mean_active_above_1pct'],
                'mean_active_above_10pct': act_stats['mean_active_above_10pct'],
                'feature_mean_activations': act_stats['feature_mean_activations'],
                # classification impact
                'baseline_err': baseline_err,
                'sae_err': sae_err,
                'norm_err': norm_err,
                'baseline_ce': baseline_ce,
                'sae_ce': sae_ce,
            }
            # Optional subset-aggregated metrics (only populated for
            # all_tokens SAEs when --subset_positions was given).
            if 'subset_positions' in act_stats:
                row['subset_positions'] = ','.join(str(p) for p in act_stats['subset_positions'])
                for k in ('subset_dead_features', 'subset_dead_ratio',
                          'subset_mean_active', 'subset_mean_active_ratio',
                          'subset_ipr', 'subset_active_above_1pct',
                          'subset_active_above_10pct',
                          'subset_mean_active_above_1pct',
                          'subset_mean_active_above_10pct'):
                    row[k] = act_stats[k]
            # Per-position tensors are attached to the row for later CSV output;
            # the main CSV writer strips them via extrasaction='ignore'.
            for k in ('per_position_positions',
                      'per_position_mean_active',
                      'per_position_mean_active_ratio',
                      'per_position_ever_active',
                      'per_position_dead_features',
                      'per_position_ipr',
                      'per_position_mean_active_above_1pct',
                      'per_position_mean_active_above_10pct',
                      'per_position_active_above_1pct',
                      'per_position_active_above_10pct'):
                if k in act_stats:
                    row[k] = act_stats[k]
            rows.append(row)

    # -------------------------------------------------------------------------
    # Print table
    # -------------------------------------------------------------------------
    print()
    print('=' * 140)
    print('SAE SWEEP RESULTS')
    print('=' * 140)

    col_fmt = (
        '{layer:>5} | {mode:>9} | {token_idx:>5} | {latent_dim:>9} | {lambda_l1:>9} | {lr:>8} | {steps:>6} | {batch_size:>5} | '
        '{train_recon_loss:>11} | {train_sparse_loss:>12} | '
        '{dead_features:>13} | {dead_ratio:>9} | {mean_active:>11} | '
        '{baseline_err:>12} | {sae_err:>8} | {norm_err:>8}'
    )
    header = col_fmt.format(
        layer='layer', mode='mode', token_idx='tok',
        latent_dim='latent_dim', lambda_l1='lambda_l1', lr='lr',
        steps='steps', batch_size='bsz',
        train_recon_loss='recon_loss', train_sparse_loss='sparse_loss',
        dead_features='dead_features', dead_ratio='dead_ratio', mean_active='mean_active',
        baseline_err='baseline_err', sae_err='sae_err', norm_err='norm_err',
    )
    print(header)
    print('-' * 140)

    for r in sorted(rows, key=lambda x: (x.get('layer', -1), x.get('lambda_l1', 0), x.get('lr', 0))):
        if 'error' in r:
            print(f"  ERROR in {r['ckpt']}: {r['error']}")
            continue
        tok_str = str(r['token_idx']) if r.get('token_idx') is not None else '-'
        print(col_fmt.format(
            layer=r['layer'],
            mode=r.get('mode', '-'),
            token_idx=tok_str,
            latent_dim=r['latent_dim'],
            lambda_l1=f"{r['lambda_l1']:.2e}",
            lr=f"{r['lr']:.1e}",
            steps=r['steps'],
            batch_size=r['batch_size'],
            train_recon_loss=f"{r['train_recon_loss']:.6f}",
            train_sparse_loss=f"{r['train_sparse_loss']:.6f}",
            dead_features=r['dead_features'],
            dead_ratio=f"{r['dead_ratio']:.4f}",
            mean_active=f"{r['mean_active']:.4f}",
            baseline_err=f"{r['baseline_err']:.6f}",
            sae_err=f"{r['sae_err']:.6f}",
            norm_err=f"{r['norm_err']:.4f}",
        ))

    print('=' * 140)
    print(f'norm_err = sae_classification_error / random_chance_error  '
          f'(1.0 = same as random; lower is better)')

    # -------------------------------------------------------------------------
    # Optional CSV output
    # -------------------------------------------------------------------------
    if args.outcsv:
        fieldnames = [
            'ckpt', 'layer', 'mode', 'token_idx', 'latent_dim', 'lambda_l1', 'lr', 'steps', 'batch_size',
            'train_total_loss', 'train_recon_loss', 'train_sparse_loss',
            'dead_features', 'dead_ratio', 'mean_active', 'mean_active_ratio',
            'ipr', 'active_above_1pct', 'active_above_10pct',
            'mean_active_above_1pct', 'mean_active_above_10pct',
            'baseline_err', 'sae_err', 'norm_err', 'baseline_ce', 'sae_ce',
        ]
        if token_subset is not None:
            fieldnames += [
                'subset_positions', 'subset_dead_features', 'subset_dead_ratio',
                'subset_mean_active', 'subset_mean_active_ratio', 'subset_ipr',
                'subset_active_above_1pct', 'subset_active_above_10pct',
                'subset_mean_active_above_1pct', 'subset_mean_active_above_10pct',
            ]
        with open(args.outcsv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(r for r in rows if 'error' not in r)
        print(f'\nResults saved to: {args.outcsv}')

        # Save per-feature mean activations for rank-magnitude plotting
        features_path = Path(args.outcsv).with_suffix('.features.pt')
        features_data = {}
        for r in rows:
            if 'error' in r or r.get('feature_mean_activations') is None:
                continue
            features_data[r['ckpt']] = {
                'layer': r['layer'],
                'lambda_l1': r['lambda_l1'],
                'feature_mean_activations': r['feature_mean_activations'],
            }
        torch.save(features_data, features_path)
        print(f'Per-feature data saved to: {features_path}')

    # -------------------------------------------------------------------------
    # Optional per-position CSV output (long format: one row per (ckpt, pos))
    # -------------------------------------------------------------------------
    if args.per_position_csv:
        pp_path = Path(args.per_position_csv)
        pp_path.parent.mkdir(parents=True, exist_ok=True)
        pp_fieldnames = [
            'ckpt', 'layer', 'mode', 'token_idx', 'lambda_l1', 'lr',
            'leaf_position', 'mean_active', 'mean_active_ratio',
            'ever_active', 'dead_features', 'ipr',
            'active_above_1pct', 'active_above_10pct',
            'mean_active_above_1pct', 'mean_active_above_10pct',
        ]
        with open(pp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=pp_fieldnames)
            writer.writeheader()
            for r in rows:
                if 'error' in r:
                    continue
                positions = r.get('per_position_positions')
                if positions is None:
                    continue
                for i, leaf_pos in enumerate(positions):
                    writer.writerow({
                        'ckpt': r['ckpt'],
                        'layer': r['layer'],
                        'mode': r.get('mode', '-'),
                        'token_idx': r.get('token_idx'),
                        'lambda_l1': r.get('lambda_l1'),
                        'lr': r.get('lr'),
                        'leaf_position': leaf_pos,
                        'mean_active': float(r['per_position_mean_active'][i]),
                        'mean_active_ratio': float(r['per_position_mean_active_ratio'][i]),
                        'ever_active': int(r['per_position_ever_active'][i]),
                        'dead_features': int(r['per_position_dead_features'][i]),
                        'ipr': float(r['per_position_ipr'][i]),
                        'active_above_1pct': int(r['per_position_active_above_1pct'][i]),
                        'active_above_10pct': int(r['per_position_active_above_10pct'][i]),
                        'mean_active_above_1pct': float(r['per_position_mean_active_above_1pct'][i]),
                        'mean_active_above_10pct': float(r['per_position_mean_active_above_10pct'][i]),
                    })
        print(f'Per-position metrics saved to: {pp_path}')


if __name__ == '__main__':
    main()
