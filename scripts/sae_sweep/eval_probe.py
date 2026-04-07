"""Evaluate pre-trained linear probes on SAE-reconstructed activations.

For every SAE .pt checkpoint in --sweep_dir this script:
  1. Finds the pre-trained probe for the SAE's (layer, token) pair in --probe_dir.
  2. Runs a sanity check: evaluates the probe on fresh clean eval activations and
     warns if the accuracy deviates more than 0.005 from the saved value.
  3. Evaluates the same probe on SAE-reconstructed eval activations.
  4. Reports clean_acc vs recon_acc and the accuracy drop.

Pre-trained probes must be produced by probe_train/run_one_probe.py or
probe_train/train_all_probes.py and follow the naming convention:
    probe_layer{layer}_tok{token_idx}__{transformer_stem}.pt

Usage:
    python sae_sweep/eval_probe.py \\
        --sweep_dir /path/to/sweep/output/ \\
        --probe_dir /path/to/transformer/checkpoint/dir/ \\
        [--probe_eval_size 4096] \\
        [--probe_eval_seed 88888] \\
        [--batch_size 256] \\
        [--device cuda] \\
        [--outcsv /path/to/probe_results.csv]
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROBE_TRAIN_DIR = REPO_ROOT / 'scripts' / 'probe_train'
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROBE_TRAIN_DIR))

import models
from datasets.random_hierarchy_model import sample_trees
from linear_probe import (
    LinearProbe,
    collect_probe_data,
    eval_probe,
    normalized_identification_error,
    probe_target_level,
)
from probe_utils import load_transformer, prepare_inputs


# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------

def _parse_sae_metadata(ckpt_path: str):
    """Read SAE checkpoint metadata without instantiating the SAE model."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
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
        'train_output': source.get('train_output', ''),
        'ckpt_path': ckpt_path,
        'sae_token_idx': int(setup.get('sae_token_idx', 0)),
        'sae_activation_source': setup.get('sae_activation_source', 'all_tokens'),
        'act_scale': act_scale,
        'train_metrics': metrics,
        'config': ckpt.get('config', None),
    }


def _load_sae(ckpt_path: str, input_dim: int, device: str):
    """Instantiate the SAE model from a checkpoint with the correct input_dim."""
    meta = _parse_sae_metadata(ckpt_path)
    if meta is None:
        return None

    layer_id = meta['layer_id']
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['sae_state'].get(layer_id) or ckpt['sae_state'].get(str(layer_id))

    sae = models.SparseAutoencoder(input_dim=input_dim, latent_dim=meta['latent_dim'])
    sae.load_state_dict(state)
    sae = sae.to(device).eval()
    for p in sae.parameters():
        p.requires_grad = False

    meta['sae'] = sae
    return meta


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def _find_probe(probe_dir: Path, layer_id: int, token_idx: int,
                transformer_stem: str) -> Path:
    """Return the path of the pre-trained probe for (layer_id, token_idx).

    Raises FileNotFoundError with a helpful message if not found.
    """
    name = f'probe_layer{layer_id}_tok{token_idx}__{transformer_stem}.pt'
    path = probe_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f'Pre-trained probe not found: {path}\n'
            f'Run probe_train/run_one_probe.py or probe_train/train_all_probes.py first.'
        )
    return path


def _load_probe(probe_path: Path, embedding_dim: int, num_classes: int,
                device: str):
    """Load a saved probe and return (LinearProbe, saved_clean_acc, chance_acc)."""
    blob = torch.load(probe_path, map_location='cpu', weights_only=False)
    probe = LinearProbe(input_dim=embedding_dim, num_classes=num_classes)
    probe.load_state_dict(blob['probe_state'])
    probe = probe.to(device).eval()
    saved_clean_acc = float(blob['probe_result']['accuracy'])
    chance_acc = float(blob['probe_result']['chance_accuracy'])
    return probe, saved_clean_acc, chance_acc


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
    parser.add_argument('--probe_dir', type=str, default=None,
                        help='Directory containing pre-trained probe .pt files '
                             '(default: same directory as the transformer checkpoint)')
    parser.add_argument('--probe_eval_size', type=int, default=4096,
                        help='Number of RHM samples for probe evaluation (default: 4096)')
    parser.add_argument('--probe_eval_seed', type=int, default=88888,
                        help='RHM seed for probe eval data (default: 88888)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for transformer forward passes (default: 256)')
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
    print(f'Eval: eval_size={args.probe_eval_size}, device={device_str}')
    print()

    # Parse all valid SAE checkpoints (metadata only, no model instantiation)
    records = []
    for ckpt_file in ckpt_files:
        meta = _parse_sae_metadata(str(ckpt_file))
        if meta is None:
            print(f'  Skipping (not a valid single-layer SAE checkpoint): {ckpt_file.name}')
        else:
            records.append(meta)

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
            model, cfg, rules = load_transformer(train_output_path, device_str)
        except Exception as exc:
            print(f'  ERROR loading transformer: {exc}')
            for r in group:
                rows.append({'ckpt': r['ckpt_path'], 'error': str(exc)})
            continue

        num_rhm_levels = cfg.num_layers
        tuple_size = cfg.tuple_size
        model_name = cfg.model
        trf_stem = Path(train_output_path).stem
        print(f'  model={model_name}, L={num_rhm_levels}, s={tuple_size}, '
              f'v={cfg.num_features}, n={cfg.num_classes}, m={cfg.num_synonyms}')

        # Determine probe directory
        probe_dir = Path(args.probe_dir) if args.probe_dir else Path(train_output_path).parent

        # Generate eval data (shared across all SAE checkpoints for this transformer)
        eval_trees = sample_trees(
            num_data=args.probe_eval_size, rules=rules, seed=args.probe_eval_seed,
        )
        eval_inputs = prepare_inputs(eval_trees, cfg)

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

            num_target_classes = cfg.num_classes if target_level == 0 else cfg.num_features
            chance_acc = 1.0 / num_target_classes

            print(f'  {Path(ckpt_path).name}: layer={layer_id}, token={token_idx}, '
                  f'target_level={target_level} ({num_target_classes} classes, '
                  f'chance={chance_acc:.4f})')

            # Load pre-trained probe
            try:
                probe_path = _find_probe(probe_dir, layer_id, token_idx, trf_stem)
                probe, saved_clean_acc, _ = _load_probe(
                    probe_path, model.embedding_dim, num_target_classes, device_str
                )
                print(f'    Probe loaded: {probe_path.name}  '
                      f'(saved clean_acc={saved_clean_acc:.4f})')
            except Exception as exc:
                print(f'    ERROR loading probe: {exc}')
                rows.append({
                    'ckpt': Path(ckpt_path).name, 'layer': layer_id,
                    'error': str(exc),
                })
                continue

            # Load SAE with correct input_dim
            try:
                sae_entry = _load_sae(ckpt_path, input_dim=model.embedding_dim,
                                      device=device_str)
                sae = sae_entry['sae']
            except Exception as exc:
                print(f'    ERROR loading SAE: {exc}')
                rows.append({
                    'ckpt': Path(ckpt_path).name, 'layer': layer_id,
                    'error': str(exc),
                })
                continue

            # Collect clean eval activations (used for sanity check)
            print(f'    Collecting clean eval activations ...')
            probe_eval_clean = collect_probe_data(
                model, eval_inputs, eval_trees,
                layer_id=layer_id, token_idx=token_idx, model_name=model_name,
                hierarchy_level=target_level, tuple_size=tuple_size,
                num_rhm_levels=num_rhm_levels, device=device,
                act_scale=1.0, sae=None, batch_size=args.batch_size,
            )

            # Sanity check: verify probe accuracy matches saved value
            clean_result = eval_probe(probe, probe_eval_clean, device)
            clean_id_error_norm = normalized_identification_error(
                clean_result.accuracy, chance_acc
            )
            delta = abs(clean_result.accuracy - saved_clean_acc)
            sanity_msg = f'    Sanity check: clean_acc={clean_result.accuracy:.4f}  ' \
                         f'saved={saved_clean_acc:.4f}  delta={delta:.4f}'
            if delta > 0.005:
                print(sanity_msg + '  WARNING: delta > 0.005 -- probe may not match this transformer')
            else:
                print(sanity_msg + '  OK')

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
