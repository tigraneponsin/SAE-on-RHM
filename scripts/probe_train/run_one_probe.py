"""Train a linear probe on a transformer checkpoint and save the result.

Usage:
    python probe_train/run_one_probe.py \\
        --train_output /path/to/transformer.pt \\
        --layer 0 \\
        --token_idx 0 \\
        [--probe_train_size 8192] \\
        [--probe_eval_size 4096] \\
        [--probe_steps 2000] \\
        [--probe_lr 1e-3] \\
        [--outname /path/to/probe_result.pt] \\
        [--device cuda]

The script:
  1. Loads the frozen transformer from --train_output.
  2. Infers the target hierarchy level from --layer (level = L - 1 - layer).
  3. Generates fresh RHM data for probe training and evaluation.
  4. Trains a linear probe on clean residual stream activations.
  5. Evaluates the probe on a held-out eval split.
  6. Saves the trained probe and results to --outname.

Probe file saved as:
    probe_layer{layer}_tok{token_idx}__{transformer_stem}.pt
next to the transformer checkpoint (or at --outname if provided).
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
    collect_probe_data,
    normalized_identification_error,
    probe_target_level,
    train_probe,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--train_output', required=True,
                        help='Path to the transformer .pt checkpoint')
    parser.add_argument('--layer', type=int, required=True,
                        help='Transformer layer index (0-based) to probe')
    parser.add_argument('--token_idx', type=int, default=0,
                        help='0-based real token index (default: 0)')
    parser.add_argument('--probe_train_size', type=int, default=8192,
                        help='Number of RHM samples for probe training (default: 8192)')
    parser.add_argument('--probe_eval_size', type=int, default=4096,
                        help='Number of RHM samples for probe evaluation (default: 4096)')
    parser.add_argument('--probe_train_seed', type=int, default=77777,
                        help='RHM seed for training data (default: 77777)')
    parser.add_argument('--probe_eval_seed', type=int, default=88888,
                        help='RHM seed for eval data (default: 88888)')
    parser.add_argument('--probe_steps', type=int, default=2000,
                        help='Number of Adam steps (default: 2000)')
    parser.add_argument('--probe_lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--probe_batch_size', type=int, default=256,
                        help='Probe training batch size (default: 256)')
    parser.add_argument('--fwd_batch_size', type=int, default=256,
                        help='Batch size for transformer forward passes (default: 256)')
    parser.add_argument('--outname', type=str, default=None,
                        help='Output path for the probe .pt file (default: auto)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: cuda if available)')
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

    # ------------------------------------------------------------------
    # 2. Determine target level
    # ------------------------------------------------------------------
    target_level = probe_target_level(args.layer, L)
    num_target_classes = n if target_level == 0 else v
    chance_acc = 1.0 / num_target_classes
    print(f'  Probe: layer={args.layer}, token={args.token_idx}, '
          f'target_level={target_level} ({num_target_classes} classes, chance={chance_acc:.4f})')

    # ------------------------------------------------------------------
    # 3. Generate data
    # ------------------------------------------------------------------
    print(f'Generating probe data: train={args.probe_train_size}, eval={args.probe_eval_size}')
    train_trees = sample_trees(num_data=args.probe_train_size, rules=rules, seed=args.probe_train_seed)
    eval_trees = sample_trees(num_data=args.probe_eval_size, rules=rules, seed=args.probe_eval_seed)
    train_inputs = prepare_inputs(train_trees, cfg)
    eval_inputs = prepare_inputs(eval_trees, cfg)

    # ------------------------------------------------------------------
    # 4. Collect activations
    # ------------------------------------------------------------------
    print('Collecting training activations ...')
    probe_train_data = collect_probe_data(
        model, train_inputs, train_trees,
        layer_id=args.layer, token_idx=args.token_idx, model_name=model_name,
        hierarchy_level=target_level, tuple_size=s, num_rhm_levels=L,
        device=device, batch_size=args.fwd_batch_size,
    )
    print(f'  shape: {probe_train_data.activations.shape}')

    print('Collecting eval activations ...')
    probe_eval_data = collect_probe_data(
        model, eval_inputs, eval_trees,
        layer_id=args.layer, token_idx=args.token_idx, model_name=model_name,
        hierarchy_level=target_level, tuple_size=s, num_rhm_levels=L,
        device=device, batch_size=args.fwd_batch_size,
    )

    # ------------------------------------------------------------------
    # 5. Train probe
    # ------------------------------------------------------------------
    print(f'Training probe ({args.probe_steps} steps, lr={args.probe_lr}) ...')
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

    print(f'Result: acc={result.accuracy:.4f}  loss={result.loss:.4f}  '
          f'(chance={chance_acc:.4f}, id_err_norm={id_error_norm:.4f})')
    print(f'  per-class accuracy: {result.per_class_accuracy.tolist()}')

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    if args.outname is None:
        stem = Path(args.train_output).stem
        out_dir = Path(args.train_output).parent / 'probes'
        args.outname = str(out_dir / f'probe_layer{args.layer}_tok{args.token_idx}__{stem}.pt')

    Path(args.outname).parent.mkdir(parents=True, exist_ok=True)

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
            'layer': args.layer,
            'token_idx': args.token_idx,
            'target_level': target_level,
            'num_target_classes': num_target_classes,
            'probe_train_size': args.probe_train_size,
            'probe_eval_size': args.probe_eval_size,
            'probe_train_seed': args.probe_train_seed,
            'probe_eval_seed': args.probe_eval_seed,
            'probe_steps': args.probe_steps,
            'probe_lr': args.probe_lr,
            'probe_batch_size': args.probe_batch_size,
            'model_name': model_name,
            'embedding_dim': cfg.embedding_dim,
            'rhm_params': {
                'v': v, 'n': n, 'm': cfg.num_synonyms, 's': s, 'L': L,
            },
        },
        'transformer_config': cfg,
    }

    torch.save(output, args.outname)
    print(f'Saved to: {args.outname}')


if __name__ == '__main__':
    main()
