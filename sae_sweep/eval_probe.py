"""Evaluate linear probes on clean and SAE-reconstructed activations.

For every SAE .pt checkpoint in --sweep_dir this script:
  1. Infers the probe target hierarchy level from the SAE's transformer layer
     (level = L - 1 - k, where k is the transformer layer index).
  2. Generates fresh RHM data (separate train / eval seeds).
  3. Trains a linear probe on clean residual stream activations.
  4. Evaluates the probe on clean eval activations (upper bound).
  5. Evaluates the same probe on SAE-reconstructed eval activations.
  6. Reports clean_acc, recon_acc, and the accuracy drop.

Usage:
    python sae_sweep/eval_probe.py \\
        --sweep_dir /path/to/sweep/output/ \\
        [--probe_train_size 8192] \\
        [--probe_eval_size 8192] \\
        [--probe_steps 2000] \\
        [--probe_lr 1e-3] \\
        [--batch_size 256] \\
        [--device cuda] \\
        [--outcsv /path/to/probe_results.csv]
"""

import argparse
import copy
import csv
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Add the repo root to sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import init
import models
from datasets.random_hierarchy_model import sample_rules, sample_trees
from linear_probe import (
    ProbeData,
    collect_probe_data,
    eval_probe,
    normalized_identification_error,
    probe_target_level,
    train_probe,
)


# ---------------------------------------------------------------------------
# Helpers (adapted from eval_sweep.py)
# ---------------------------------------------------------------------------

def _resolve_rules(blob: dict, source_label: str):
    rules = blob.get('output', {}).get('rules', None)
    cfg = blob.get('config', None)

    if rules is not None:
        return rules, 'artifact'

    if cfg is None:
        raise ValueError(
            f'{source_label}: no saved rules and no config to regenerate from.'
        )
    missing = [a for a in ('num_features', 'num_classes', 'num_synonyms',
                            'tuple_size', 'num_layers', 'seed_rules')
               if not hasattr(cfg, a)]
    if missing:
        raise ValueError(
            f'{source_label}: cannot regenerate rules -- config missing: {missing}'
        )
    rules = sample_rules(
        cfg.num_features, cfg.num_classes, cfg.num_synonyms,
        cfg.tuple_size, cfg.num_layers, seed=cfg.seed_rules,
    )
    return rules, 'seed_rules_resampled'


def _load_sae(ckpt_path: str, input_dim: int, device: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'sae_state' not in ckpt or 'sae_layers' not in ckpt:
        return None

    layers = [int(x) for x in ckpt['sae_layers']]
    if len(layers) != 1:
        return None

    layer_id = layers[0]
    state = ckpt['sae_state'].get(layer_id) or ckpt['sae_state'].get(str(layer_id))
    metrics = (ckpt.get('sae_metrics', {}).get(layer_id)
               or ckpt.get('sae_metrics', {}).get(str(layer_id)) or {})
    if state is None:
        return None

    latent_dim = int(metrics.get('latent_dim') or state['encoder.weight'].shape[0])
    sae = models.SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    sae.load_state_dict(state)
    sae = sae.to(device).eval()
    for p in sae.parameters():
        p.requires_grad = False

    setup = ckpt.get('sae_training_setup', {})
    source = ckpt.get('source', {})

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
        'train_output': source.get('train_output', ''),
        'ckpt_path': ckpt_path,
        'sae_token_idx': int(setup.get('sae_token_idx', 0)),
        'sae_activation_source': setup.get('sae_activation_source', 'all_tokens'),
        'act_scale': act_scale,
        'train_metrics': metrics,
        'config': ckpt.get('config', None),
    }


def _normalize_model_state_dict_keys(state_dict):
    """Strip torch.compile wrapper prefix from checkpoint state dict keys."""
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if keys and all(k.startswith('_orig_mod.') for k in keys):
        return {k[len('_orig_mod.'):]: v for k, v in state_dict.items()}
    return state_dict


def _prepare_inputs(trees, cfg):
    """Transform leaf tokens into model-ready inputs (same as init.transform_inputs)."""
    num_rhm_levels = cfg.num_layers
    raw_inputs = trees[num_rhm_levels]  # leaf level
    data_cfg = copy.deepcopy(cfg)
    data_cfg.train_size = raw_inputs.size(0)
    data_cfg.test_size = 0
    return init.transform_inputs(raw_inputs, data_cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--sweep_dir', required=True,
                        help='Directory containing SAE .pt checkpoints')
    parser.add_argument('--probe_train_size', type=int, default=8192,
                        help='Number of RHM samples for probe training (default: 8192)')
    parser.add_argument('--probe_eval_size', type=int, default=8192,
                        help='Number of RHM samples for probe evaluation (default: 8192)')
    parser.add_argument('--probe_train_seed', type=int, default=77777,
                        help='RHM seed for probe training data (default: 77777)')
    parser.add_argument('--probe_eval_seed', type=int, default=88888,
                        help='RHM seed for probe eval data (default: 88888)')
    parser.add_argument('--probe_steps', type=int, default=2000,
                        help='Number of Adam steps to train the probe (default: 2000)')
    parser.add_argument('--probe_lr', type=float, default=1e-3,
                        help='Probe learning rate (default: 1e-3)')
    parser.add_argument('--probe_batch_size', type=int, default=256,
                        help='Probe training batch size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for forward passes through the transformer (default: 256)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: cuda if available, else cpu)')
    parser.add_argument('--outcsv', type=str, default=None,
                        help='Optional path to save results as CSV')
    args = parser.parse_args()

    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    sweep_dir = Path(args.sweep_dir)

    ckpt_files = sorted(sweep_dir.glob('*.pt'))
    if not ckpt_files:
        print(f'No .pt files found in {sweep_dir}')
        sys.exit(0)

    print(f'Found {len(ckpt_files)} checkpoint file(s) in {sweep_dir}')
    print(f'Probe: train_size={args.probe_train_size}, eval_size={args.probe_eval_size}, '
          f'steps={args.probe_steps}, lr={args.probe_lr}, device={device_str}')
    print()

    # Parse all valid SAE checkpoints (without loading the SAE model yet)
    records = []
    for ckpt_file in ckpt_files:
        entry = _load_sae(str(ckpt_file), input_dim=1, device='cpu')  # dummy load
        if entry is None:
            print(f'  Skipping (not a valid single-layer SAE checkpoint): {ckpt_file.name}')
        else:
            records.append(entry)

    if not records:
        print('No valid SAE checkpoints found.')
        sys.exit(0)

    # Group by transformer artifact
    by_transformer = {}
    for r in records:
        by_transformer.setdefault(r['train_output'], []).append(r)

    rows = []

    for train_output_path, group in by_transformer.items():
        print(f'Loading transformer from: {train_output_path}')
        try:
            blob = torch.load(train_output_path, map_location='cpu')
            cfg = copy.deepcopy(blob['config'])
            rules, rules_source = _resolve_rules(blob, train_output_path)

            model = init.init_model(cfg)
            model_state = _normalize_model_state_dict_keys(blob['output']['model'])
            model.load_state_dict(model_state)
            model = model.to(device).eval()
            for p in model.parameters():
                p.requires_grad = False
        except Exception as exc:
            print(f'  ERROR loading transformer: {exc}')
            for r in group:
                rows.append({'ckpt': r['ckpt_path'], 'error': str(exc)})
            continue

        num_rhm_levels = cfg.num_layers
        tuple_size = cfg.tuple_size
        model_name = cfg.model
        print(f'  model={model_name}, L={num_rhm_levels}, s={tuple_size}, '
              f'v={cfg.num_features}, n={cfg.num_classes}, m={cfg.num_synonyms}')

        # Generate probe train and eval data
        train_trees = sample_trees(
            num_data=args.probe_train_size, rules=rules, seed=args.probe_train_seed,
        )
        eval_trees = sample_trees(
            num_data=args.probe_eval_size, rules=rules, seed=args.probe_eval_seed,
        )
        train_inputs = _prepare_inputs(train_trees, cfg).to('cpu')
        eval_inputs = _prepare_inputs(eval_trees, cfg).to('cpu')

        # Process each SAE checkpoint for this transformer
        for r in group:
            ckpt_path = r['ckpt_path']
            layer_id = r['layer_id']
            mode = r['sae_activation_source']
            token_idx = r['sae_token_idx']
            act_scale = r['act_scale']

            if mode != 'one_token':
                print(f'  Skipping {Path(ckpt_path).name}: probe only supports one_token mode, '
                      f'got {mode}')
                rows.append({
                    'ckpt': Path(ckpt_path).name, 'layer': layer_id,
                    'error': f'unsupported mode: {mode}',
                })
                continue

            # Determine target hierarchy level
            try:
                target_level = probe_target_level(layer_id, num_rhm_levels)
            except ValueError as exc:
                print(f'  Skipping {Path(ckpt_path).name}: {exc}')
                rows.append({
                    'ckpt': Path(ckpt_path).name, 'layer': layer_id,
                    'error': str(exc),
                })
                continue

            if target_level == 0:
                num_target_classes = cfg.num_classes
            else:
                num_target_classes = cfg.num_features
            chance_acc = 1.0 / num_target_classes

            print(f'  {Path(ckpt_path).name}: layer={layer_id}, token={token_idx}, '
                  f'target_level={target_level} ({num_target_classes} classes, '
                  f'chance={chance_acc:.4f})')

            # Reload SAE with correct input_dim
            try:
                entry = _load_sae(ckpt_path, input_dim=model.embedding_dim, device=device_str)
                sae = entry['sae']
            except Exception as exc:
                print(f'    ERROR loading SAE: {exc}')
                rows.append({
                    'ckpt': Path(ckpt_path).name, 'layer': layer_id,
                    'error': str(exc),
                })
                continue

            # Collect clean training activations
            print(f'    Collecting clean training activations ...')
            probe_train = collect_probe_data(
                model, train_inputs, train_trees,
                layer_id=layer_id, token_idx=token_idx, model_name=model_name,
                hierarchy_level=target_level, tuple_size=tuple_size,
                num_rhm_levels=num_rhm_levels, device=device,
                act_scale=1.0, sae=None, batch_size=args.batch_size,
            )

            # Collect clean eval activations
            print(f'    Collecting clean eval activations ...')
            probe_eval_clean = collect_probe_data(
                model, eval_inputs, eval_trees,
                layer_id=layer_id, token_idx=token_idx, model_name=model_name,
                hierarchy_level=target_level, tuple_size=tuple_size,
                num_rhm_levels=num_rhm_levels, device=device,
                act_scale=1.0, sae=None, batch_size=args.batch_size,
            )

            # Train probe
            print(f'    Training probe ({args.probe_steps} steps) ...')
            probe, clean_result = train_probe(
                probe_train,
                lr=args.probe_lr,
                num_steps=args.probe_steps,
                batch_size=args.probe_batch_size,
                device=device,
                eval_data=probe_eval_clean,
                verbose=False,
            )
            clean_id_error_norm = normalized_identification_error(
                clean_result.accuracy, chance_acc
            )
            print(f'    Clean eval: acc={clean_result.accuracy:.4f}  '
                  f'loss={clean_result.loss:.4f}  '
                  f'id_err_norm={clean_id_error_norm:.4f}')

            # Collect SAE-reconstructed eval activations
            print(f'    Collecting SAE-reconstructed eval activations ...')
            probe_eval_recon = collect_probe_data(
                model, eval_inputs, eval_trees,
                layer_id=layer_id, token_idx=token_idx, model_name=model_name,
                hierarchy_level=target_level, tuple_size=tuple_size,
                num_rhm_levels=num_rhm_levels, device=device,
                act_scale=act_scale, sae=sae, batch_size=args.batch_size,
            )

            # Evaluate probe on reconstructed activations
            recon_result = eval_probe(probe, probe_eval_recon, device)
            acc_drop = clean_result.accuracy - recon_result.accuracy
            recon_id_error_norm = normalized_identification_error(
                recon_result.accuracy, chance_acc
            )
            id_error_norm_delta = recon_id_error_norm - clean_id_error_norm
            print(f'    Recon eval: acc={recon_result.accuracy:.4f}  '
                f'loss={recon_result.loss:.4f}  drop={acc_drop:.4f}  '
                f'id_err_norm={recon_id_error_norm:.4f}')

            cfg_s = r.get('config', None)
            rows.append({
                'ckpt': Path(ckpt_path).name,
                'layer': layer_id,
                'token_idx': token_idx,
                'target_level': target_level,
                'num_target_classes': num_target_classes,
                'chance_acc': f'{chance_acc:.4f}',
                'clean_acc': f'{clean_result.accuracy:.4f}',
                'clean_loss': f'{clean_result.loss:.4f}',
                'clean_id_error_norm': f'{clean_id_error_norm:.4f}',
                'recon_acc': f'{recon_result.accuracy:.4f}',
                'recon_loss': f'{recon_result.loss:.4f}',
                'recon_id_error_norm': f'{recon_id_error_norm:.4f}',
                'id_error_norm_delta': f'{id_error_norm_delta:.4f}',
                'acc_drop': f'{acc_drop:.4f}',
                'latent_dim': r['latent_dim'],
                'lambda_l1': getattr(cfg_s, 'sae_lambda_l1', None) if cfg_s else None,
                'sae_lr': getattr(cfg_s, 'sae_lr', None) if cfg_s else None,
                'act_scale': f'{act_scale:.6f}',
            })

    # Print summary table
    print()
    print('=' * 100)
    if not rows:
        print('No results.')
        sys.exit(0)

    header = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r.get(h, ''))) for r in rows)) for h in header}
    header_line = '  '.join(h.ljust(col_widths[h]) for h in header)
    print(header_line)
    print('-' * len(header_line))
    for r in rows:
        print('  '.join(str(r.get(h, '')).ljust(col_widths[h]) for h in header))

    # Save CSV
    if args.outcsv:
        with open(args.outcsv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nResults saved to {args.outcsv}')


if __name__ == '__main__':
    main()
