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
import copy
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Add the repo root to sys.path so we can import init, models, datasets
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import init
import models
from datasets.random_hierarchy_model import sample_rules, sample_trees


# ---------------------------------------------------------------------------
# Data / model helpers
# ---------------------------------------------------------------------------

def _resolve_rules(blob: dict, source_label: str):
    """Return the RHM rules to use for data generation, matching what was used
    during transformer training and SAE training.

    Priority:
      1. blob['output']['rules'] if present and not None  →  exact same rules
         object that was saved alongside the trained transformer weights.
      2. Regenerate deterministically from config.seed_rules and the RHM
         structural parameters.  This produces identical rules to case 1 as
         long as the config has not changed, because both paths call
         sample_rules(v, n, m, s, L, seed=seed_rules).

    Either way the returned rules are the same ones used in transformer
    training AND SAE training (train_sae.py follows the same priority).
    """
    rules = blob.get('output', {}).get('rules', None)
    cfg = blob.get('config', None)

    if rules is not None:
        return rules, 'artifact'

    # Fallback: regenerate from seed_rules (mirrors train_sae.py behaviour)
    if cfg is None:
        raise ValueError(
            f'{source_label}: transformer artifact has no saved rules and no config '
            'to regenerate them from.'
        )
    missing = [a for a in ('num_features', 'num_classes', 'num_synonyms',
                            'tuple_size', 'num_layers', 'seed_rules')
               if not hasattr(cfg, a)]
    if missing:
        raise ValueError(
            f'{source_label}: cannot regenerate rules — config is missing: {missing}'
        )
    print(
        f'  WARNING: transformer artifact has no saved rules. '
        f'Regenerating from config.seed_rules={cfg.seed_rules}. '
        f'Eval data uses the same RHM function as training only if seed_rules '
        f'and structural parameters have not changed.'
    )
    rules = sample_rules(
        cfg.num_features, cfg.num_classes, cfg.num_synonyms,
        cfg.tuple_size, cfg.num_layers, seed=cfg.seed_rules,
    )
    return rules, 'seed_rules_resampled'


def _check_sae_rules_source(sae_entry: dict, resolved_rules_source: str):
    """Assert that the SAE was trained with rules from the same source."""
    split = sae_entry.get('sae_checkpoint_data', {})
    # The rules_source is stored under 'sae_dataset_split' in the SAE artifact.
    sae_rules_source = split.get('rules_source', None)
    if sae_rules_source is None:
        return  # older artifact without the field — nothing to check
    if sae_rules_source != resolved_rules_source:
        raise RuntimeError(
            f"Rules source mismatch for {sae_entry['ckpt_path']}:\n"
            f"  SAE was trained with rules_source='{sae_rules_source}'\n"
            f"  but eval is using rules_source='{resolved_rules_source}'.\n"
            f"  Eval data would not come from the same RHM function."
        )


def _load_transformer(train_output_path: str, eval_size: int, eval_seed: int,
                      batch_size: int, device: str):
    blob = torch.load(train_output_path, map_location='cpu')
    if not isinstance(blob, dict) or 'config' not in blob or 'output' not in blob:
        raise ValueError(f'Invalid train_output format: {train_output_path}')
    if 'model' not in blob['output']:
        raise ValueError(
            f'train_output missing output.model — re-run transformer training '
            f'with --save_models: {train_output_path}'
        )

    cfg = copy.deepcopy(blob['config'])
    rules, rules_source = _resolve_rules(blob, train_output_path)

    trees = sample_trees(num_data=eval_size, rules=rules, prior=None, probs=None, seed=eval_seed)

    data_cfg = copy.deepcopy(cfg)
    data_cfg.train_size = eval_size
    data_cfg.test_size = 0
    data_cfg.batch_size = max(1, min(batch_size, eval_size))
    loader, _ = init.init_data(trees[cfg.num_layers], trees[0], data_cfg)

    model = init.init_model(cfg)
    model.load_state_dict(blob['output']['model'])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, loader, cfg, rules_source


def _load_sae(ckpt_path: str, input_dim: int | None, device: str,
              load_model: bool = True):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'sae_state' not in ckpt or 'sae_layers' not in ckpt:
        return None  # not a valid SAE checkpoint

    layers = [int(x) for x in ckpt['sae_layers']]
    if len(layers) != 1:
        # multi-layer checkpoints are not produced by the sweep (one per job)
        return None

    layer_id = layers[0]
    state = ckpt['sae_state'].get(layer_id) or ckpt['sae_state'].get(str(layer_id))
    metrics = (ckpt.get('sae_metrics', {}).get(layer_id)
               or ckpt.get('sae_metrics', {}).get(str(layer_id)) or {})
    curves = (ckpt.get('sae_training_curves', {}).get(layer_id)
              or ckpt.get('sae_training_curves', {}).get(str(layer_id)) or {})
    if state is None:
        return None

    latent_dim = int(metrics.get('latent_dim') or state['encoder.weight'].shape[0])
    sae = None
    if load_model:
        if input_dim is None or input_dim <= 0:
            raise ValueError('input_dim must be > 0 when load_model=True')
        sae = models.SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        sae.load_state_dict(state)
        sae = sae.to(device).eval()
        for p in sae.parameters():
            p.requires_grad = False

    setup = ckpt.get('sae_training_setup', {})
    cfg_stored = ckpt.get('config', None)
    source = ckpt.get('source', {})
    dataset_split = ckpt.get('sae_dataset_split', {})

    # act_scale stored as {str(layer_id): float} — default 1.0 for old checkpoints
    act_scale_dict = setup.get('act_scale', {})
    act_scale = float(
        act_scale_dict.get(layer_id)
        or act_scale_dict.get(str(layer_id))
        or 1.0
    )

    return {
        'layer_id': layer_id,
        'latent_dim': latent_dim,
        'sae': sae,
        'train_metrics': metrics,
        'curves': curves,
        'setup': setup,
        'config': cfg_stored,
        'train_output': source.get('train_output', ''),
        'ckpt_path': ckpt_path,
        # rules_source recorded at SAE training time: 'artifact' or 'seed_rules_resampled'
        'sae_rules_source': dataset_split.get('rules_source', None),
        'sae_token_idx': int(setup.get('sae_token_idx', 0)),
        'act_scale': act_scale,
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _activity_stats(model, loader, sae, layer_id: int, mode: str, device: str,
                    token_idx: int = 0, act_scale: float = 1.0) -> dict:
    """Dead-feature and mean-active-feature stats over eval data."""
    has_cls = hasattr(model, 'cls_token')
    latent_dim = sae.latent_dim
    ever_active = torch.zeros(latent_dim, dtype=torch.bool, device=device)
    feature_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    active_sum = 0.0
    total_tokens = 0

    dec_norms = sae.decoder_feature_norms().to(device)
    buf = []
    hook = model.blocks[layer_id].register_forward_hook(
        lambda _m, _i, o: buf.append(o.detach())
    )

    mean_active_1pct_sum = 0.0
    mean_active_10pct_sum = 0.0

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
            act = act.reshape(-1, act.size(-1))
            if act.numel() == 0:
                continue
            act = act * act_scale
            _, z = sae(act)
            features = z * dec_norms.unsqueeze(0)   # weighted activations
            is_active = features > 0
            active_sum += is_active.float().sum().item()
            total_tokens += is_active.size(0)
            ever_active |= is_active.any(dim=0)
            feature_sum += features.sum(dim=0).double()
            # Per-token relative threshold: count features > X% of this token's max
            token_max = features.max(dim=1, keepdim=True).values
            mean_active_1pct_sum += (features > 0.01 * token_max).float().sum().item()
            mean_active_10pct_sum += (features > 0.10 * token_max).float().sum().item()

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

    return {
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
    args = parser.parse_args()

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
                act_stats = _activity_stats(model, eval_loader, sae, layer_id, mode, device,
                                            token_idx=token_idx, act_scale=act_scale)
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

            rows.append({
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
            })

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
            lambda_l1=f"{r['lambda_l1']:.0f}",
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


if __name__ == '__main__':
    main()
