"""Train linear probes for all (layer, token) pairs of a transformer checkpoint.

For a transformer with L layers and tuple size s, this trains L * s^L probes
covering every valid (layer, token_idx) combination. Each probe predicts the
RHM ancestor at hierarchy level L-1-layer from the residual stream activations
at the given (layer, token) position.

Probes are saved in a probes/ folder next to the transformer checkpoint as:
    probes/probe_layer{layer}_tok{token_idx}__{transformer_stem}.pt

Already-existing probe files are skipped (idempotent reruns).

Usage:
    python probe_train/train_all_probes.py \\
        --train_output /path/to/transformer.pt \\
        [--probe_train_size 8192] \\
        [--probe_eval_size 4096] \\
        [--probe_steps 2000] \\
        [--probe_lr 1e-3] \\
        [--device cuda]
"""

import argparse
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
PROBE_TRAIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_TRAIN_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROBE_TRAIN_DIR))

from probe_utils import load_transformer, prepare_inputs
from datasets.random_hierarchy_model import sample_trees
from linear_probe import (
    LinearProbe,
    collect_probe_data,
    normalized_identification_error,
    probe_target_level,
    train_probe,
)


def _probe_path(out_dir: Path, layer: int, token_idx: int, trf_stem: str) -> Path:
    return out_dir / f'probe_layer{layer}_tok{token_idx}__{trf_stem}.pt'


def _save_probe(path: Path, probe: LinearProbe, result, chance_acc: float,
                id_error_norm: float, args, cfg, layer: int, token_idx: int,
                target_level: int, num_target_classes: int):
    v = cfg.num_features
    n = cfg.num_classes
    s = cfg.tuple_size
    L = cfg.num_layers
    output = {
        'probe_state': probe.state_dict(),
        'probe_result': {
            'accuracy': result.accuracy,
            'chance_accuracy': chance_acc,
            'id_error_norm': id_error_norm,
            'per_class_accuracy': result.per_class_accuracy,
            'loss': result.loss,
            'num_samples': result.num_samples,
            'num_classes': result.num_classes,
        },
        'probe_config': {
            'train_output': str(args.train_output),
            'layer': layer,
            'token_idx': token_idx,
            'target_level': target_level,
            'num_target_classes': num_target_classes,
            'probe_train_size': args.probe_train_size,
            'probe_eval_size': args.probe_eval_size,
            'probe_train_seed': args.probe_train_seed,
            'probe_eval_seed': args.probe_eval_seed,
            'probe_steps': args.probe_steps,
            'probe_lr': args.probe_lr,
            'probe_batch_size': args.probe_batch_size,
            'model_name': cfg.model,
            'embedding_dim': cfg.embedding_dim,
            'rhm_params': {'v': v, 'n': n, 'm': cfg.num_synonyms, 's': s, 'L': L},
        },
        'transformer_config': cfg,
    }
    torch.save(output, path)


def main():
    print(
        "[WARN] Probe scripts always load the LAST transformer weights "
        "(output['model']), not output['best']['model']. If you trained an SAE "
        "on the BEST weights and want probe results to be comparable, results "
        "may not align.",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--train_output', required=True,
                        help='Path to the transformer .pt checkpoint')
    parser.add_argument('--probe_train_size', type=int, default=8192)
    parser.add_argument('--probe_eval_size', type=int, default=4096)
    parser.add_argument('--probe_train_seed', type=int, default=77777)
    parser.add_argument('--probe_eval_seed', type=int, default=88888)
    parser.add_argument('--probe_steps', type=int, default=2000)
    parser.add_argument('--probe_lr', type=float, default=1e-3)
    parser.add_argument('--probe_batch_size', type=int, default=256)
    parser.add_argument('--fwd_batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f'Device: {device_str}')

    # ------------------------------------------------------------------
    # 1. Load transformer
    # ------------------------------------------------------------------
    print(f'Loading transformer from: {args.train_output}')
    model, cfg, rules = load_transformer(args.train_output, device_str)

    L = cfg.num_layers
    s = cfg.tuple_size
    v = cfg.num_features
    n = cfg.num_classes
    model_name = cfg.model
    print(f'  model={model_name}, L={L}, s={s}, v={v}, n={n}, emb={cfg.embedding_dim}')

    trf_stem = Path(args.train_output).stem
    out_dir = Path(args.train_output).parent / 'probes'
    out_dir.mkdir(parents=True, exist_ok=True)

    # All (layer, token) pairs: layer in [0, L-1], token in [0, s^L - 1]
    num_tokens = s ** L
    pairs = [(layer, tok) for layer in range(L) for tok in range(num_tokens)]
    print(f'  {len(pairs)} probe(s) to train: {L} layer(s) x {num_tokens} token(s)')
    print()

    # ------------------------------------------------------------------
    # 2. Generate data (shared across all probes)
    # ------------------------------------------------------------------
    print(f'Generating RHM data: train={args.probe_train_size}, eval={args.probe_eval_size}')
    train_trees = sample_trees(num_data=args.probe_train_size, rules=rules,
                               seed=args.probe_train_seed)
    eval_trees = sample_trees(num_data=args.probe_eval_size, rules=rules,
                              seed=args.probe_eval_seed)
    train_inputs = prepare_inputs(train_trees, cfg)
    eval_inputs = prepare_inputs(eval_trees, cfg)
    print()

    # ------------------------------------------------------------------
    # 3. Train one probe per (layer, token) pair
    # ------------------------------------------------------------------
    summary_rows = []

    for layer, token_idx in pairs:
        probe_path = _probe_path(out_dir, layer, token_idx, trf_stem)

        if probe_path.exists():
            print(f'  [skip] layer={layer} tok={token_idx} -- already exists: {probe_path.name}')
            summary_rows.append({
                'layer': layer, 'token_idx': token_idx,
                'status': 'skipped', 'acc': '-', 'id_err_norm': '-',
            })
            continue

        target_level = probe_target_level(layer, L)
        num_target_classes = n if target_level == 0 else v
        chance_acc = 1.0 / num_target_classes

        print(f'  layer={layer} tok={token_idx}  target_level={target_level} '
              f'({num_target_classes} classes, chance={chance_acc:.4f})')

        probe_train_data = collect_probe_data(
            model, train_inputs, train_trees,
            layer_id=layer, token_idx=token_idx, model_name=model_name,
            hierarchy_level=target_level, tuple_size=s, num_rhm_levels=L,
            device=device, batch_size=args.fwd_batch_size,
        )
        probe_eval_data = collect_probe_data(
            model, eval_inputs, eval_trees,
            layer_id=layer, token_idx=token_idx, model_name=model_name,
            hierarchy_level=target_level, tuple_size=s, num_rhm_levels=L,
            device=device, batch_size=args.fwd_batch_size,
        )

        probe, result = train_probe(
            probe_train_data,
            lr=args.probe_lr,
            num_steps=args.probe_steps,
            batch_size=args.probe_batch_size,
            device=device,
            eval_data=probe_eval_data,
            verbose=True,
        )

        id_error_norm = normalized_identification_error(result.accuracy, chance_acc)
        print(f'    acc={result.accuracy:.4f}  loss={result.loss:.4f}  '
              f'id_err_norm={id_error_norm:.4f}')

        _save_probe(probe_path, probe, result, chance_acc, id_error_norm,
                    args, cfg, layer, token_idx, target_level, num_target_classes)
        print(f'    Saved: {probe_path.name}')
        print()

        summary_rows.append({
            'layer': layer, 'token_idx': token_idx, 'status': 'trained',
            'acc': f'{result.accuracy:.4f}', 'id_err_norm': f'{id_error_norm:.4f}',
        })

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print('=' * 70)
    print(f'{"layer":>6}  {"token":>6}  {"status":>8}  {"acc":>8}  {"id_err_norm":>12}')
    print('-' * 70)
    for r in summary_rows:
        print(f'{r["layer"]:>6}  {r["token_idx"]:>6}  {r["status"]:>8}  '
              f'{r["acc"]:>8}  {r["id_err_norm"]:>12}')
    print('=' * 70)


if __name__ == '__main__':
    main()
