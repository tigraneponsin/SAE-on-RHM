"""Direct token-residual intervention to probe "dead token" behavior.

Replaces the post-block residual stream at chosen (layer, positions) with a
non-informative surrogate (mean / zero / resample) and reports the resulting
classification accuracy without involving any SAE.

Usage:
    python scripts/intervention/ablate_tokens.py \\
        --train_output /path/to/transformer.pt \\
        --experiments /path/to/experiments.json \\
        [--eval_size 32768] [--eval_seed 99999] [--batch_size 256] \\
        [--device cuda] \\
        [--outcsv results.csv] [--norms_csv norms.csv]

experiments.json format: a list of experiments, each with a name and a list of
interventions applied simultaneously in one forward pass:

    [
      {"name": "baseline", "interventions": []},
      {"name": "L0_kill_odd_mean",
       "interventions": [{"layer": 0, "positions": [1,3,5,7], "mode": "mean"}]},
      {"name": "L0_odd_plus_L1_234567_mean",
       "interventions": [{"layer": 0, "positions": [1,3,5,7], "mode": "mean"},
                         {"layer": 1, "positions": [2,3,6,7], "mode": "mean"}]}
    ]

Modes: "mean" (replace with per-position dataset mean at that layer),
       "zero" (replace with 0),
       "resample" (replace with activation from a different random sample in
        the same batch at the same position).
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.common.sae_loading import load_transformer


ALLOWED_MODES = {'mean', 'zero', 'resample'}


# ---------------------------------------------------------------------------
# Argument parsing and experiment validation
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--train_output', required=True,
                   help='Path to trained transformer .pt (output from main.py)')
    p.add_argument('--experiments', required=True,
                   help='Path to experiments JSON file')
    p.add_argument('--eval_size', type=int, default=2**15,
                   help='Number of RHM samples for evaluation (default 32768)')
    p.add_argument('--eval_seed', type=int, default=99999,
                   help='Seed for the eval split (default 99999)')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--device', type=str, default=None,
                   help='Device (default: cuda if available, else cpu)')
    p.add_argument('--outcsv', type=str, default=None,
                   help='Optional path to save per-experiment results CSV')
    p.add_argument('--norms_csv', type=str, default=None,
                   help='Optional path to save per-(layer, position) mean L2 norm CSV')
    p.add_argument('--resample_seed', type=int, default=0,
                   help='Seed for resample ablation RNG (default 0)')
    p.add_argument('--model_variant', choices=['best', 'last'], default='last',
                   help="Which transformer weights to load: 'best' (lowest test loss) "
                        "or 'last' (final step). Default: last for back-compat. Pair with "
                        "--model_variant best when reproducing analysis aligned to an SAE "
                        "trained on the best transformer weights.")
    return p.parse_args()


def load_experiments(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f'{path}: expected a JSON list of experiments')
    for i, exp in enumerate(data):
        if 'name' not in exp or 'interventions' not in exp:
            raise ValueError(f'{path}: experiment #{i} missing "name" or "interventions"')
        if not isinstance(exp['interventions'], list):
            raise ValueError(
                f"{path}: experiment '{exp['name']}' interventions must be a list"
            )
        for j, iv in enumerate(exp['interventions']):
            for key in ('layer', 'positions', 'mode'):
                if key not in iv:
                    raise ValueError(
                        f"{path}: exp '{exp['name']}' intervention {j} missing '{key}'"
                    )
            if iv['mode'] not in ALLOWED_MODES:
                raise ValueError(
                    f"{path}: exp '{exp['name']}' intervention {j} "
                    f"has bad mode {iv['mode']!r} (allowed: {sorted(ALLOWED_MODES)})"
                )
            if not isinstance(iv['positions'], list):
                raise ValueError(
                    f"{path}: exp '{exp['name']}' intervention {j} 'positions' must be a list"
                )
    return data


def validate_against_model(experiments, num_blocks, num_leaves):
    for exp in experiments:
        for iv in exp['interventions']:
            layer = int(iv['layer'])
            if not 0 <= layer < num_blocks:
                raise ValueError(
                    f"exp '{exp['name']}': layer {layer} out of range "
                    f"[0, {num_blocks - 1}]"
                )
            for p in iv['positions']:
                pi = int(p)
                if not 0 <= pi < num_leaves:
                    raise ValueError(
                        f"exp '{exp['name']}': leaf position {pi} out of range "
                        f"[0, {num_leaves - 1}]"
                    )


# ---------------------------------------------------------------------------
# Precompute per-(layer, position) mean activation and mean L2 norm
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_means_and_norms(model, loader, device, seq_len):
    """Single forward pass over `loader`; captures the post-block residual at
    every block. Returns:
        means: {layer: Tensor[seq_len, D]} -- E[x_{layer}[:, pos, :]]
        norms: {layer: Tensor[seq_len]}    -- E[||x_{layer}[:, pos, :]||_2]
    """
    num_blocks = len(model.blocks)
    D = model.embedding_dim
    sums = {k: torch.zeros(seq_len, D, device=device, dtype=torch.float64)
            for k in range(num_blocks)}
    norm_sums = {k: torch.zeros(seq_len, device=device, dtype=torch.float64)
                 for k in range(num_blocks)}
    count = 0

    buffers = {}

    def make_cap(k):
        def hook(_m, _i, out):
            buffers[k] = out.detach()
        return hook

    handles = [model.blocks[k].register_forward_hook(make_cap(k))
               for k in range(num_blocks)]

    try:
        for x, _ in loader:
            x = x.to(device)
            buffers.clear()
            model(x)
            count += x.size(0)
            for k, buf in buffers.items():
                sums[k] += buf.sum(dim=0).double()
                norm_sums[k] += buf.norm(dim=2).sum(dim=0).double()
    finally:
        for h in handles:
            h.remove()

    means = {k: (sums[k] / max(count, 1)).float() for k in sums}
    norms = {k: (norm_sums[k] / max(count, 1)).float() for k in norm_sums}
    return means, norms


# ---------------------------------------------------------------------------
# Intervention hook and classification eval
# ---------------------------------------------------------------------------

def make_ablation_hook(interventions_at_layer, pos_offset, mean_cache, layer_id):
    """Build a forward hook that applies the given interventions to the
    post-block output at `layer_id`.

    interventions_at_layer: list of dicts with keys 'positions' (list[int],
    0-based leaf positions) and 'mode' (one of ALLOWED_MODES).
    """
    prepped = []
    for iv in interventions_at_layer:
        sp = [pos_offset + int(p) for p in iv['positions']]
        prepped.append((sp, iv['mode']))

    def hook(_m, _i, output):
        if not prepped:
            return output
        out = output.clone()
        for seq_positions, mode in prepped:
            if not seq_positions:
                continue
            if mode == 'zero':
                out[:, seq_positions, :] = 0.0
            elif mode == 'mean':
                mean_slice = mean_cache[layer_id][seq_positions].to(
                    dtype=out.dtype, device=out.device
                )
                out[:, seq_positions, :] = mean_slice.unsqueeze(0).expand(
                    out.size(0), -1, -1
                )
            elif mode == 'resample':
                B = out.size(0)
                perm = torch.randperm(B, device=out.device)
                out[:, seq_positions, :] = output[perm][:, seq_positions, :]
            else:
                raise RuntimeError(f'unexpected mode: {mode}')
        return out
    return hook


@torch.no_grad()
def eval_classification(model, loader, device, hooks=None):
    handles = []
    if hooks:
        for module, fn in hooks:
            handles.append(module.register_forward_hook(fn))

    correct = total = 0
    total_ce = 0.0
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_ce += F.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
    finally:
        for h in handles:
            h.remove()

    return correct / total, total_ce / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model, loader, _cfg, _rules, rules_source = load_transformer(
        args.train_output, args.eval_size, args.eval_seed,
        args.batch_size, device, shuffle=False,
        model_variant=args.model_variant,
    )
    has_cls = hasattr(model, 'cls_token')
    pos_offset = 1 if has_cls else 0
    num_blocks = len(model.blocks)
    num_leaves = model.block_size
    seq_len = num_leaves + (1 if has_cls else 0)
    random_err = 1.0 - 1.0 / model.num_classes

    print(f'Loaded transformer from {args.train_output} (variant={args.model_variant})')
    print(f'  rules_source={rules_source}  has_cls={has_cls}  num_blocks={num_blocks}')
    print(f'  num_leaves={num_leaves}  num_classes={model.num_classes}')

    experiments = load_experiments(args.experiments)
    validate_against_model(experiments, num_blocks, num_leaves)
    print(f'Loaded {len(experiments)} experiment(s) from {args.experiments}')

    print('Precomputing per-layer means and residual-norm trajectories...')
    mean_cache, norm_cache = compute_means_and_norms(model, loader, device, seq_len)

    baseline_acc, baseline_ce = eval_classification(model, loader, device, hooks=None)
    baseline_err = 1.0 - baseline_acc
    print(f'Baseline: acc={baseline_acc:.4f}  err={baseline_err:.6f}  CE={baseline_ce:.4f}')

    torch.manual_seed(args.resample_seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.resample_seed)

    rows = []
    for exp in experiments:
        name = exp['name']
        interventions = exp['interventions']
        by_layer = defaultdict(list)
        for iv in interventions:
            by_layer[int(iv['layer'])].append(iv)

        hooks = []
        for layer_id, ivs in by_layer.items():
            hook_fn = make_ablation_hook(ivs, pos_offset, mean_cache, layer_id)
            hooks.append((model.blocks[layer_id], hook_fn))

        acc, ce = eval_classification(model, loader, device, hooks=hooks)
        err = 1.0 - acc
        delta_err = err - baseline_err
        err_over_random = err / random_err if random_err > 0 else float('nan')
        print(f"  {name}: acc={acc:.4f}  err={err:.6f}  "
              f"delta_err={delta_err:+.6f}  err/random={err_over_random:.4f}")

        rows.append({
            'name': name,
            'interventions': json.dumps(interventions),
            'acc': acc,
            'cross_entropy': ce,
            'err': err,
            'err_over_random': err_over_random,
            'delta_err_vs_baseline': delta_err,
        })

    if args.outcsv:
        out_path = Path(args.outcsv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['name', 'interventions', 'acc', 'cross_entropy',
                      'err', 'err_over_random', 'delta_err_vs_baseline']
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow({
                'name': '__baseline__',
                'interventions': '[]',
                'acc': baseline_acc,
                'cross_entropy': baseline_ce,
                'err': baseline_err,
                'err_over_random': (baseline_err / random_err
                                    if random_err > 0 else float('nan')),
                'delta_err_vs_baseline': 0.0,
            })
            for r in rows:
                w.writerow(r)
        print(f'Wrote results CSV: {out_path}')

    if args.norms_csv:
        out_path = Path(args.norms_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'layer', 'seq_position', 'leaf_position', 'mean_norm'
            ])
            w.writeheader()
            for layer_id in sorted(norm_cache):
                norms = norm_cache[layer_id]
                for sp in range(seq_len):
                    leaf_pos = sp - pos_offset
                    w.writerow({
                        'layer': layer_id,
                        'seq_position': sp,
                        # -1 denotes the CLS position (only when has_cls)
                        'leaf_position': leaf_pos if leaf_pos >= 0 else -1,
                        'mean_norm': float(norms[sp].item()),
                    })
        print(f'Wrote norms CSV: {out_path}')


if __name__ == '__main__':
    main()
