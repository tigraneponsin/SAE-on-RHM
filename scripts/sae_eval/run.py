"""CLI driver: streaming SAE eval on one checkpoint or a whole sweep.

Scans for SAE checkpoints, groups by transformer source so each transformer
is loaded once, and for each SAE calls stream_sae_eval (and optionally
stream_classification_impact) with the requested flags. Saves a per-SAE
artifact (*.sae_eval.pt) and optionally a sweep CSV / per-position CSV.

Per-accumulator toggles default to OFF. Use --with-all to turn on every
block, or enable only the ones you care about:

    python scripts/sae_eval/run.py --sweep_dir /path/to/sweep/ --out_dir /path/to/out/ \\
        --with-per-feature --with-conditional --with-entropy --with-classification-impact

Passing a CSV destination with --outcsv also requires the corresponding
flags to be enabled (mean_active needs --with-scalar-aggregates, entropy
columns need --with-entropy, etc.).
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.random_hierarchy_model import sample_trees
from scripts.common.sae_loading import load_sae, load_transformer
from scripts.sae_eval.streaming import (
    StreamingFlags,
    dedupe_trees,
    stream_classification_impact,
    stream_sae_eval,
)


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

_FLAG_NAMES = [
    ('scalar-aggregates',     'scalar_aggregates',
     'dead count, IPR, mean_active, threshold counts'),
    ('per-position',          'per_position',
     'per-position versions of the scalar aggregates'),
    ('per-feature',           'per_feature',
     'baseline_mean/std, firing_count, mean_cofire (per (pos, feature))'),
    ('conditional',           'conditional',
     'conditional_mean / delta_mean / z_score per target (requires per-feature)'),
    ('entropy',               'joint_fire_and_entropy',
     'joint-fire counts, H_per_feature, H_bar_* aggregates; requires per-feature'),
    ('classification-impact', 'classification_impact',
     'second streaming pass measuring SAE reconstruction error on classification'),
]


def _add_bool_flag(parser, name: str, dest: str, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f'--with-{name}', dest=dest, action='store_true',
                       help=f'Enable: {help_text}')
    group.add_argument(f'--no-{name}', dest=dest, action='store_false',
                       help=argparse.SUPPRESS)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--sweep_dir', type=str,
                     help='Directory containing SAE .pt checkpoints')
    src.add_argument('--ckpt', type=str,
                     help='Path to a single SAE checkpoint .pt')

    parser.add_argument('--out_dir', type=str, required=True,
                        help='Destination directory for *.sae_eval.pt artifacts')

    parser.add_argument('--eval_size', type=int, default=32768)
    parser.add_argument('--eval_seed', type=int, default=99999)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dedupe', action='store_true',
                        help='Drop duplicate RHM trees before the streaming pass')
    parser.add_argument('--device', type=str, default=None)

    flag_group = parser.add_argument_group('streaming flags (default OFF)')
    for name, dest, help_text in _FLAG_NAMES:
        _add_bool_flag(flag_group, name, dest, help_text)
    flag_group.add_argument('--with-all', dest='_all', action='store_true',
                            help='Shortcut: turn on every --with-* flag above')

    csv_group = parser.add_argument_group('optional CSV outputs')
    csv_group.add_argument('--outcsv', type=str, default=None,
                           help='Path for sweep-summary CSV (requires '
                                '--with-scalar-aggregates at minimum)')
    csv_group.add_argument('--per_position_csv', type=str, default=None,
                           help='Path for long-format per-position CSV '
                                '(requires --with-per-position)')
    parser.set_defaults(**{dest: False for _, dest, _ in _FLAG_NAMES},
                        _all=False)
    return parser


def _resolve_flags(args) -> StreamingFlags:
    values = {dest: bool(getattr(args, dest)) for _, dest, _ in _FLAG_NAMES}
    if args._all:
        for dest in values:
            values[dest] = True
    flags = StreamingFlags(**values)
    flags.validate()
    return flags


# ---------------------------------------------------------------------------
# Checkpoint scanning
# ---------------------------------------------------------------------------

def _scan_checkpoints(sweep_dir: Path):
    return sorted(sweep_dir.glob('*.pt'))


def _rhm_params_from_cfg(cfg) -> dict:
    return {
        'v': int(cfg.num_features),
        'n': int(cfg.num_classes),
        'm': int(cfg.num_synonyms),
        's': int(cfg.tuple_size),
        'L': int(cfg.num_layers),
    }


# ---------------------------------------------------------------------------
# CSV emission
# ---------------------------------------------------------------------------

_CORE_CSV_COLS = [
    'ckpt', 'layer', 'mode', 'token_idx', 'latent_dim',
    'lambda_l1', 'lr', 'steps', 'batch_size',
    'train_total_loss', 'train_recon_loss', 'train_sparse_loss',
]
_SCALAR_CSV_COLS = [
    'dead_features', 'dead_ratio', 'mean_active', 'mean_active_ratio',
    'ipr', 'active_above_1pct', 'active_above_10pct',
    'mean_active_above_1pct', 'mean_active_above_10pct',
]
_CLASSIF_CSV_COLS = [
    'baseline_err', 'sae_err', 'norm_err', 'baseline_ce', 'sae_ce',
]
_ENTROPY_CSV_COLS = [
    'H_bar_fire_mean', 'H_bar_raw_mean', 'H_bar_dec_mean',
    'H_bar_fire_norm_mean', 'H_bar_raw_norm_mean', 'H_bar_dec_norm_mean',
]


def _row_from_artifact(artifact: dict, flags: StreamingFlags,
                       extras: dict) -> dict:
    row = dict(extras)
    row['ckpt'] = extras.get('ckpt')
    row['layer'] = artifact.get('layer_id')
    row['mode'] = artifact.get('mode')
    row['token_idx'] = artifact.get('sae_token_idx')
    row['latent_dim'] = artifact.get('latent_dim')
    if flags.scalar_aggregates:
        for k in _SCALAR_CSV_COLS:
            row[k] = artifact.get(k)
    if flags.classification_impact:
        for k in _CLASSIF_CSV_COLS:
            row[k] = artifact.get(k)
    if flags.joint_fire_and_entropy:
        for src_key, col in [
            ('H_bar_fire',       'H_bar_fire_mean'),
            ('H_bar_raw',        'H_bar_raw_mean'),
            ('H_bar_dec',        'H_bar_dec_mean'),
            ('H_bar_fire_norm',  'H_bar_fire_norm_mean'),
            ('H_bar_raw_norm',   'H_bar_raw_norm_mean'),
            ('H_bar_dec_norm',   'H_bar_dec_norm_mean'),
        ]:
            t = artifact.get(src_key)
            if t is None:
                row[col] = None
            else:
                t_finite = t[torch.isfinite(t)]
                row[col] = float(t_finite.mean().item()) if t_finite.numel() > 0 else float('nan')
    return row


def _write_main_csv(path: Path, rows: list, flags: StreamingFlags) -> None:
    fieldnames = list(_CORE_CSV_COLS)
    if flags.scalar_aggregates:
        fieldnames += _SCALAR_CSV_COLS
    if flags.classification_impact:
        fieldnames += _CLASSIF_CSV_COLS
    if flags.joint_fire_and_entropy:
        fieldnames += _ENTROPY_CSV_COLS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(r for r in rows if 'error' not in r)


_PP_CSV_COLS = [
    'ckpt', 'layer', 'mode', 'token_idx', 'lambda_l1', 'lr',
    'leaf_position', 'mean_active', 'mean_active_ratio',
    'ever_active', 'dead_features', 'ipr',
    'active_above_1pct', 'active_above_10pct',
    'mean_active_above_1pct', 'mean_active_above_10pct',
    'H_bar_fire', 'H_bar_raw', 'H_bar_dec',
    'H_bar_fire_norm', 'H_bar_raw_norm', 'H_bar_dec_norm',
]


def _write_per_position_csv(path: Path, artifacts: list, flags: StreamingFlags) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_PP_CSV_COLS, extrasaction='ignore')
        writer.writeheader()
        for a in artifacts:
            if 'error' in a:
                continue
            positions = a['artifact'].get('token_positions')
            if positions is None:
                continue
            positions_list = positions.tolist()
            art = a['artifact']
            for i, leaf_pos in enumerate(positions_list):
                row = {
                    'ckpt': a['ckpt'],
                    'layer': art['layer_id'],
                    'mode': art['mode'],
                    'token_idx': art.get('sae_token_idx'),
                    'lambda_l1': a.get('lambda_l1'),
                    'lr': a.get('lr'),
                    'leaf_position': leaf_pos,
                }
                if flags.per_position:
                    row.update({
                        'mean_active': float(art['per_position_mean_active'][i]),
                        'mean_active_ratio': float(art['per_position_mean_active_ratio'][i]),
                        'ever_active': int(art['per_position_ever_active'][i]),
                        'dead_features': int(art['per_position_dead_features'][i]),
                        'ipr': float(art['per_position_ipr'][i]),
                        'active_above_1pct': int(art['per_position_active_above_1pct'][i]),
                        'active_above_10pct': int(art['per_position_active_above_10pct'][i]),
                        'mean_active_above_1pct': float(
                            art['per_position_mean_active_above_1pct'][i]
                        ),
                        'mean_active_above_10pct': float(
                            art['per_position_mean_active_above_10pct'][i]
                        ),
                    })
                if flags.joint_fire_and_entropy:
                    for src_key, col in [
                        ('H_bar_fire', 'H_bar_fire'),
                        ('H_bar_raw', 'H_bar_raw'),
                        ('H_bar_dec', 'H_bar_dec'),
                        ('H_bar_fire_norm', 'H_bar_fire_norm'),
                        ('H_bar_raw_norm', 'H_bar_raw_norm'),
                        ('H_bar_dec_norm', 'H_bar_dec_norm'),
                    ]:
                        val = art[src_key][i]
                        row[col] = float(val.item()) if torch.isfinite(val) else ''
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _build_parser().parse_args()
    flags = _resolve_flags(args)

    if not any([flags.scalar_aggregates, flags.per_position, flags.per_feature,
                flags.conditional, flags.joint_fire_and_entropy,
                flags.classification_impact]):
        print('ERROR: at least one --with-<block> flag must be set.')
        print('       (or pass --with-all to enable every block).')
        return 2

    if args.outcsv and not flags.scalar_aggregates:
        print('ERROR: --outcsv requires at least --with-scalar-aggregates.')
        return 2
    if args.per_position_csv and not flags.per_position:
        print('ERROR: --per_position_csv requires --with-per-position.')
        return 2

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_files = [Path(args.ckpt)] if args.ckpt else _scan_checkpoints(Path(args.sweep_dir))
    if not ckpt_files:
        print(f'No .pt files found in {args.sweep_dir}')
        return 0

    print(f'Found {len(ckpt_files)} checkpoint file(s).')
    print(f'Flags: {flags}')
    print(f'Eval: size={args.eval_size}, seed={args.eval_seed}, '
          f'batch={args.batch_size}, dedupe={args.dedupe}, device={device}')
    print()

    # Parse metadata without loading weights.
    records = []
    for ckpt_file in ckpt_files:
        entry = load_sae(str(ckpt_file), input_dim=None, device='cpu', load_model=False)
        if entry is None:
            print(f'  Skipping (not a valid single-layer SAE checkpoint): {ckpt_file.name}')
            continue
        records.append(entry)
    if not records:
        print('No valid SAE checkpoints found.')
        return 0

    # Group by (train_output, model_variant) so SAEs trained on different
    # transformer-weight variants of the same artifact don't share a forward pass.
    by_transformer: dict = {}
    for r in records:
        by_transformer.setdefault((r['train_output'], r['model_variant']), []).append(r)

    rows: list = []       # one dict per SAE, for the main CSV
    artifacts: list = []  # for per-position CSV (holds artifact dicts in-memory)

    for (train_output_path, model_variant), group in by_transformer.items():
        variants_in_group = {r['model_variant'] for r in group}
        if len(variants_in_group) > 1:
            raise RuntimeError(
                f"SAE artifacts for transformer {train_output_path!r} disagree "
                f"on model_variant: {sorted(variants_in_group)}."
            )
        print(f'Loading transformer from: {train_output_path} (variant={model_variant})')
        try:
            model, _loader, cfg, rules, rules_source = load_transformer(
                train_output_path, args.eval_size, args.eval_seed,
                args.batch_size, device, shuffle=False,
                model_variant=model_variant,
            )
        except Exception as exc:
            print(f'  ERROR loading transformer: {exc}')
            for r in group:
                rows.append({'ckpt': Path(r['ckpt_path']).name, 'error': str(exc)})
            continue

        print(f'  Rules source: {rules_source}')
        if rules_source == 'seed_rules_resampled':
            print(
                '  WARNING: rules regenerated from seed_rules. Re-run training '
                'with --save_models to eliminate this warning.'
            )

        has_cls = hasattr(model, 'cls_token')
        rhm = _rhm_params_from_cfg(cfg)

        trees = sample_trees(num_data=args.eval_size, rules=rules,
                             prior=None, probs=None, seed=args.eval_seed)
        if args.dedupe:
            before = trees[rhm['L']].size(0)
            trees = dedupe_trees(trees)
            after = trees[rhm['L']].size(0)
            print(f'  Dedupe: {before} -> {after} unique trees')

        for r in group:
            ckpt_path = r['ckpt_path']
            layer_id = r['layer_id']
            mode = r['setup'].get('sae_activation_source', 'all_tokens')
            token_idx = r.get('sae_token_idx', 0)
            act_scale = r.get('act_scale', 1.0)
            train_m = r.get('train_metrics', {}) or {}
            cfg_s = r.get('config')

            print(f'  Evaluating: {Path(ckpt_path).name}  '
                  f'(layer={layer_id}, mode={mode}'
                  + (f', token={token_idx}' if mode == 'one_token' else '')
                  + f', act_scale={act_scale:.4f})')

            sae_rules_source = r.get('sae_rules_source', None)
            if sae_rules_source is not None and sae_rules_source != rules_source:
                msg = (f"Rules source mismatch: SAE trained with "
                       f"'{sae_rules_source}', transformer resolves to "
                       f"'{rules_source}'. Skipping.")
                print(f'    ERROR: {msg}')
                rows.append({'ckpt': Path(ckpt_path).name, 'error': msg})
                continue

            try:
                entry = load_sae(ckpt_path, input_dim=model.embedding_dim, device=device)
                sae = entry['sae']
            except Exception as exc:
                print(f'    ERROR loading SAE: {exc}')
                rows.append({'ckpt': Path(ckpt_path).name, 'error': str(exc)})
                continue

            try:
                streaming_artifact = stream_sae_eval(
                    model=model, sae=sae, trees=trees,
                    layer_id=layer_id, mode=mode, has_cls=has_cls,
                    token_idx=token_idx, act_scale=act_scale,
                    batch_size=args.batch_size, device=device,
                    flags=flags,
                    rhm=rhm if flags.joint_fire_and_entropy else None,
                    rules=rules if flags.joint_fire_and_entropy else None,
                )
            except Exception as exc:
                print(f'    ERROR during streaming eval: {exc}')
                rows.append({'ckpt': Path(ckpt_path).name, 'error': str(exc)})
                continue

            if flags.classification_impact:
                try:
                    ci = stream_classification_impact(
                        model=model, sae=sae, trees=trees,
                        layer_id=layer_id, mode=mode, has_cls=has_cls,
                        token_idx=token_idx, act_scale=act_scale,
                        batch_size=args.batch_size, device=device,
                    )
                    streaming_artifact.update(ci)
                except Exception as exc:
                    print(f'    ERROR during classification impact: {exc}')

            # Assemble full artifact with identity / provenance / training metadata.
            artifact = {
                'ckpt_path': str(ckpt_path),
                'layer_id': int(layer_id),
                'mode': str(mode),
                'sae_token_idx': int(token_idx),
                'act_scale': float(act_scale),
                'embedding_dim': int(model.embedding_dim),
                'eval_size': int(args.eval_size),
                'eval_seed': int(args.eval_seed),
                'dedupe': bool(args.dedupe),
                'rhm': rhm,
                'rules_source': rules_source,
                'sae_rules_source': sae_rules_source,
                'train_total_loss': train_m.get('total_loss', float('nan')),
                'train_recon_loss': train_m.get('recon_loss', float('nan')),
                'train_sparse_loss': train_m.get('sparse_loss', float('nan')),
                'lambda_l1': (getattr(cfg_s, 'sae_lambda_l1', None)
                              if cfg_s else None),
                'lr': (getattr(cfg_s, 'sae_lr', None) if cfg_s else None),
                'steps': train_m.get('steps', None),
                'batch_size': r['setup'].get('sae_sample_batch_size', None),
                **streaming_artifact,
            }
            out_path = out_dir / (Path(ckpt_path).stem + '.sae_eval.pt')
            torch.save(artifact, out_path)
            print(f'    saved: {out_path.name}')

            extras = {
                'ckpt': Path(ckpt_path).name,
                'train_total_loss': artifact['train_total_loss'],
                'train_recon_loss': artifact['train_recon_loss'],
                'train_sparse_loss': artifact['train_sparse_loss'],
                'lambda_l1': artifact['lambda_l1'],
                'lr': artifact['lr'],
                'steps': artifact['steps'],
                'batch_size': artifact['batch_size'],
            }
            rows.append(_row_from_artifact(artifact, flags, extras))
            if args.per_position_csv:
                artifacts.append({
                    'ckpt': Path(ckpt_path).name,
                    'lambda_l1': artifact['lambda_l1'],
                    'lr': artifact['lr'],
                    'artifact': artifact,
                })
            del artifact, streaming_artifact, sae, entry
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model, trees, rules
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Emit CSVs.
    if args.outcsv:
        _write_main_csv(Path(args.outcsv), rows, flags)
        print(f'\nMain CSV: {args.outcsv}')
    if args.per_position_csv:
        _write_per_position_csv(Path(args.per_position_csv), artifacts, flags)
        print(f'Per-position CSV: {args.per_position_csv}')

    print('\ndone.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
