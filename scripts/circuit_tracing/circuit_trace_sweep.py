"""Threshold sweep for per-input circuit tracing.

Runs the threshold-independent stages of `circuit_trace` ONCE (transformer
forward, SAE forwards, all attribution edges, node table, pre-prune
fidelity), then loops over every (node_threshold, edge_threshold) pair in
the Cartesian product and writes a self-contained subdir for each cell.

A top-level `sweep_summary.json` lists every cell with the metrics that
change across configs (n_features_post, n_edges_post, n_groups,
completeness, replacement, post-prune subtree alignment).

Usage:
    python -m scripts.circuit_tracing.circuit_trace_sweep \\
        --train_output /path/transformer.pt \\
        --sae_ckpts L0.pt L1.pt L2.pt \\
        --sae_eval_artifacts L0.sae_eval.pt L1.sae_eval.pt L2.sae_eval.pt \\
        --input_idx 42 --eval_seed 0 --eval_size 1024 \\
        --sink_mode softmax_logits \\
        --node_thresholds 0.7 0.8 0.9 \\
        --edge_thresholds 0.9 0.95 0.98 \\
        --output_dir runs/circuit_sweep/input42

Sweeping a single axis is the special case where one of the lists has a
single value -- the same code path handles it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Make REPO_ROOT importable for `scripts.*` / `models.*` exactly like
# circuit_trace.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.circuit_tracing.circuit_trace import (  # noqa: E402
    prepare_pipeline, finalize_one_config, _RemovedFlag,
)


def _validate_thresholds(values: list, name: str) -> list[float]:
    out = []
    for v in values:
        f = float(v)
        if not (0.0 < f <= 1.0):
            raise SystemExit(
                f'ERROR: {name} entries must be in (0, 1]; got {f}'
            )
        out.append(f)
    if not out:
        raise SystemExit(f'ERROR: {name} requires at least one value')
    return out


def _subdir_name(node_th: float, edge_th: float) -> str:
    return f'node_{node_th:.4f}__edge_{edge_th:.4f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_output', required=True,
                   help='Path to transformer checkpoint.')
    p.add_argument('--sae_ckpts', nargs='+', required=True,
                   help='K SAE checkpoint paths, bottom-to-top.')
    p.add_argument('--sae_eval_artifacts', nargs='+', required=True,
                   help='K matching .sae_eval.pt artifact paths.')
    p.add_argument('--input_idx', type=int, required=True)
    p.add_argument('--eval_seed', type=int, default=0)
    p.add_argument('--eval_size', type=int, default=1024)
    p.add_argument('--output_dir', required=True,
                   help='Sweep root. One subdir per (node_th, edge_th).')
    p.add_argument('--mat_threshold', type=int, default=8192)
    p.add_argument('--sink_mode', choices=['softmax_logits', 'true_class'],
                   default='softmax_logits')
    p.add_argument('--node_thresholds', nargs='+', required=True,
                   help='One or more node thresholds in (0, 1].')
    p.add_argument('--edge_thresholds', nargs='+', required=True,
                   help='One or more edge thresholds in (0, 1].')
    p.add_argument(
        '--prune_fraction', action=_RemovedFlag,
        message=('Use --node_thresholds and --edge_thresholds with '
                 '--sink_mode {softmax_logits|true_class} instead.'),
    )
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--model_variant', default=None, choices=['best', 'last'])
    p.add_argument('--pooled_last_layer', action='store_true', default=False,
                   help='Treat the last-layer SAE as a single mean-pooled SAE '
                        '(post-ln_f pooled space) instead of per-position. '
                        'Auto-detected from the last SAE\'s sae_activation_source.')
    p.add_argument('--label_alpha', type=float, default=0.5,
                   help='Absorption reassignment threshold for the "reassigned" '
                        'label scheme (default 0.5). Built once in '
                        'prepare_pipeline; the threshold sweep does not vary it.')
    p.add_argument('--label_primary', default='reassigned',
                   choices=['parent', 'level', 'whole_tree', 'reassigned'],
                   help='Label scheme the top-level node/group fields mirror '
                        '(default reassigned). All four are always stored.')
    args = p.parse_args()

    node_thresholds = sorted(_validate_thresholds(args.node_thresholds, '--node_thresholds'))
    edge_thresholds = sorted(_validate_thresholds(args.edge_thresholds, '--edge_thresholds'))

    sweep_dir = Path(args.output_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'Threshold sweep: {len(node_thresholds)} x {len(edge_thresholds)} '
          f'= {len(node_thresholds) * len(edge_thresholds)} cells')
    print(f'  node_thresholds = {node_thresholds}')
    print(f'  edge_thresholds = {edge_thresholds}')
    print(f'  output_dir = {sweep_dir}')
    print('=' * 70)

    # `prepare_pipeline` consumes args.sink_mode but not args.node_threshold
    # or args.edge_threshold. We stuff placeholder attributes so the namespace
    # has the same shape `prepare_pipeline` expects via attribute access (it
    # doesn't read them, but better to be explicit).
    args.node_threshold = node_thresholds[0]
    args.edge_threshold = edge_thresholds[0]

    prepared = prepare_pipeline(args)

    cells = []
    total = len(node_thresholds) * len(edge_thresholds)
    idx = 0
    for n_th in node_thresholds:
        for e_th in edge_thresholds:
            idx += 1
            subdir = sweep_dir / _subdir_name(n_th, e_th)
            print('\n' + '-' * 70)
            print(f'Cell {idx}/{total}: node_th={n_th}, edge_th={e_th}')
            print(f'  subdir = {subdir}')
            print('-' * 70)
            cell_summary = finalize_one_config(
                prepared,
                node_threshold=n_th,
                edge_threshold=e_th,
                out_dir=subdir,
            )
            cells.append(cell_summary)

    # ---- Write sweep_summary.json ----
    sweep_summary = {
        'node_thresholds': node_thresholds,
        'edge_thresholds': edge_thresholds,
        'sink_mode': prepared['sink_mode'],
        'input_idx': prepared['input_idx'],
        'eval_seed': prepared['eval_seed'],
        'rhm': {
            's': prepared['s'], 'L': prepared['L'],
            'v': int(prepared['cfg'].num_features),
            'n': int(prepared['cfg'].num_classes),
            'm': int(prepared['cfg'].num_synonyms),
        },
        'y_true': prepared['y_true'],
        'y_pred': prepared['y_pred'],
        'y_runner_up': prepared['y_runner'],
        'bit_identity_max_err': prepared['bit_identity_max_err'],
        'pooled_last_layer': prepared.get('pooled_last_layer', False),
        'cells': cells,
    }
    summary_path = sweep_dir / 'sweep_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(sweep_summary, f, indent=2)

    # ---- Print final table ----
    print('\n' + '=' * 70)
    print('Sweep complete. Cell summary:')
    header = (
        f'  {"n_th":>6}  {"e_th":>6}  {"n_feat":>6}  {"n_edge":>6}  '
        f'{"n_grp":>5}  {"n_mgrp":>6}  {"compl":>6}  {"repl":>6}  '
        f'{"align_pp":>8}'
    )
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for c in cells:
        print(
            f'  {c["node_threshold"]:>6.3f}  {c["edge_threshold"]:>6.3f}  '
            f'{c["n_features_post"]:>6d}  {c["n_edges_post"]:>6d}  '
            f'{c["n_groups"]:>5d}  {c["n_groups_multi"]:>6d}  '
            f'{c["completeness_score"]:>6.3f}  {c["replacement_score"]:>6.3f}  '
            f'{c["subtree_alignment_fraction_postprune"]:>8.3f}'
        )
    print(f'\nWrote {summary_path}')


if __name__ == '__main__':
    main()
