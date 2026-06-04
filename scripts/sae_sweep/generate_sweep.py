"""Generate a sweep_configs.json for SAE hyperparameter search.

Usage (CLI — recommended for scripted sweeps):

    python sae_sweep/generate_sweep.py \\
        --outdir /work/pcsl/ponsin/Mean_Transformer/SAE/sweep_round1a_bs \\
        --sae_sample_batch_size 32,64,128,256,512 \\
        --sae_lr 1e-3

Or edit the DEFAULT_* sections below and run without arguments:

    python sae_sweep/generate_sweep.py

This writes {OUTDIR}/sweep_configs.json (one config per job) and prints the
Slurm array range to use in run_sweep.sh.
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

# =============================================================================
# DEFAULTS: used when the corresponding CLI argument is not provided.
# =============================================================================

DEFAULT_TRAIN_OUTPUT = '/work/pcsl/ponsin/Mean_Transformer/Transformer_for_SAE/v_16_L_3_m_4/RESULT_TRFCLASS_v_16_L_3_m=4_P_12160_0_emb_512_h_8_lr_5e-3_dropout_0.1.pkl.pt'
DEFAULT_OUTDIR       = '/work/pcsl/ponsin/Mean_Transformer/SAE/sweep_onetok_lr_2ndversion'

DEFAULT_PARAM_GRID = {
    'sae_layer':               [0, 1, 2],
    'sae_latent_dim':          [20 * 512],
    'sae_lambda_l1':           [1e-2],
    'sae_lr':                  [1e-4],
    'sae_steps':               [2**17],
    'sae_sample_batch_size':   [2**7],
    'sae_batch_limit':         [0],
    'sae_train_size':          [2**14],
    'sae_lambda_warmup_frac':  [0],
    'sae_lr_decay_frac':       [0],
}

DEFAULT_ACTIVATION_SOURCE = 'one_token'
DEFAULT_TOKEN_IDX         = 0
DEFAULT_EVAL_SIZE         = 2**14
DEFAULT_LOG_POINTS        = 128   # number of log-spaced checkpoints per job
DEFAULT_NO_ACT_SCALE      = False

# =============================================================================


def _parse_list(s, cast):
    """Parse a comma-separated string into a list, applying *cast* to each."""
    return [cast(x.strip()) for x in s.split(',') if x.strip()]


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- paths ---
    p.add_argument('--train_output', type=str, default=None,
                   help='Path to the trained transformer .pt artifact.')
    p.add_argument('--outdir', type=str, default=None,
                   help='Output directory for sweep_configs.json and SAE checkpoints.')

    # --- grid parameters (comma-separated lists) ---
    p.add_argument('--sae_layer', type=str, default=None,
                   help='Comma-separated layer indices, e.g. "0,1,2"')
    p.add_argument('--sae_latent_dim', type=str, default=None,
                   help='Comma-separated latent dims, e.g. "2048,10240"')
    p.add_argument('--sae_lambda_l1', type=str, default=None,
                   help='Comma-separated L1 coefficients, e.g. "0.1,1,10"')
    p.add_argument('--sae_lr', type=str, default=None,
                   help='Comma-separated learning rates, e.g. "1e-4,1e-3,1e-2"')
    p.add_argument('--sae_steps', type=str, default=None,
                   help='Comma-separated step counts, e.g. "65536,262144"')
    p.add_argument('--sae_sample_batch_size', type=str, default=None,
                   help='Comma-separated batch sizes, e.g. "32,64,128,256"')
    p.add_argument('--sae_batch_limit', type=str, default=None,
                   help='Comma-separated batch limits (0 = all)')
    p.add_argument('--sae_train_size', type=str, default=None,
                   help='Comma-separated training sizes, e.g. "4096,16384,65536"')
    p.add_argument('--sae_lambda_warmup_frac', type=str, default=None,
                   help='Comma-separated lambda warmup fractions, e.g. "0.05"')
    p.add_argument('--sae_lr_decay_frac', type=str, default=None,
                   help='Comma-separated LR decay fractions, e.g. "0.2"')

    # --- fixed settings ---
    p.add_argument('--sae_activation_source', type=str, default=None,
                   choices=['all_tokens', 'cls_token', 'one_token', 'mean_pooled'])
    p.add_argument('--sae_token_idx', type=int, default=None)
    p.add_argument('--sae_eval_size', type=int, default=None)
    p.add_argument('--sae_log_points', type=int, default=None,
                   help=f'Number of log-spaced checkpoints per job (default: {DEFAULT_LOG_POINTS}).')
    p.add_argument('--no_act_scale', action='store_true', default=False,
                   help='Disable activation rescaling for all jobs in the sweep.')
    p.add_argument('--model_variant', choices=['best', 'last'], default='best',
                   help='Which transformer checkpoint to use: "best" (default) or "last".')
    p.add_argument('--append', action='store_true', default=False,
                   help='Append to existing sweep_configs.json instead of overwriting. '
                        'Useful for combining different per-layer settings.')

    return p.parse_args()


def _build_grid(args):
    """Merge CLI overrides into the default parameter grid."""
    grid = dict(DEFAULT_PARAM_GRID)

    if args.sae_layer is not None:
        grid['sae_layer'] = _parse_list(args.sae_layer, int)
    if args.sae_latent_dim is not None:
        grid['sae_latent_dim'] = _parse_list(args.sae_latent_dim, int)
    if args.sae_lambda_l1 is not None:
        grid['sae_lambda_l1'] = _parse_list(args.sae_lambda_l1, float)
    if args.sae_lr is not None:
        grid['sae_lr'] = _parse_list(args.sae_lr, float)
    if args.sae_steps is not None:
        grid['sae_steps'] = _parse_list(args.sae_steps, int)
    if args.sae_sample_batch_size is not None:
        grid['sae_sample_batch_size'] = _parse_list(args.sae_sample_batch_size, int)
    if args.sae_batch_limit is not None:
        grid['sae_batch_limit'] = _parse_list(args.sae_batch_limit, int)
    if args.sae_train_size is not None:
        grid['sae_train_size'] = _parse_list(args.sae_train_size, int)
    if args.sae_lambda_warmup_frac is not None:
        grid['sae_lambda_warmup_frac'] = _parse_list(args.sae_lambda_warmup_frac, float)
    if args.sae_lr_decay_frac is not None:
        grid['sae_lr_decay_frac'] = _parse_list(args.sae_lr_decay_frac, float)

    return grid


def _trsf_meta(train_output_path: str):
    """Load the transformer artifact, return (tag, cfg)."""
    try:
        import torch
    except ImportError:
        print('ERROR: torch is required to read the transformer artifact.', file=sys.stderr)
        sys.exit(1)

    blob = torch.load(train_output_path, map_location='cpu')
    if not isinstance(blob, dict) or 'config' not in blob:
        print(f'ERROR: {train_output_path} does not look like a valid transformer artifact '
              '(expected a dict with a "config" key).', file=sys.stderr)
        sys.exit(1)

    cfg = blob['config']
    missing = [a for a in ('num_features', 'num_layers', 'num_synonyms',
                            'train_size', 'embedding_dim')
               if not hasattr(cfg, a)]
    if missing:
        print(f'ERROR: transformer config is missing fields: {missing}', file=sys.stderr)
        sys.exit(1)

    tag = (
        f"v{cfg.num_features}"
        f"_L{cfg.num_layers}"
        f"_m{cfg.num_synonyms}"
        f"_P{cfg.train_size}"
        f"_emb{cfg.embedding_dim}"
    )
    return tag, cfg


def _activation_tag(c: dict) -> str:
    """Short string encoding the activation source (and token index for one_token)."""
    src = c['sae_activation_source']
    if src == 'all_tokens':
        return 'alltok'
    if src == 'cls_token':
        return 'cls'
    if src == 'one_token':
        return f"tok{c['sae_token_idx']}"
    if src == 'mean_pooled':
        return 'meanpool'
    return src


def _make_outname(outdir: Path, trsf_tag: str, c: dict) -> str:
    ld = c['sae_latent_dim'] if c['sae_latent_dim'] is not None else 'auto'
    lr_str = f"{c['sae_lr']:.0e}".replace('+', '').replace('-0', '-')
    l1_str = f"{c['sae_lambda_l1']:.3g}".replace('+', '')
    return str(
        outdir / (
            f"sae_{trsf_tag}"
            f"_layer{c['sae_layer']}"
            f"_{_activation_tag(c)}"
            f"_ldim{ld}"
            f"_l1{l1_str}"
            f"_lr{lr_str}"
            f"_steps{c['sae_steps']}"
            f"_bs{c['sae_sample_batch_size']}"
            f"_ts{c['sae_train_size']}"
            ".pt"
        )
    )


def main():
    args = _parse_args()

    train_output = args.train_output or DEFAULT_TRAIN_OUTPUT
    outdir = Path(args.outdir or DEFAULT_OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    activation_source = args.sae_activation_source or DEFAULT_ACTIVATION_SOURCE
    token_idx = args.sae_token_idx if args.sae_token_idx is not None else DEFAULT_TOKEN_IDX
    eval_size = args.sae_eval_size if args.sae_eval_size is not None else DEFAULT_EVAL_SIZE
    no_act_scale = args.no_act_scale if args.no_act_scale else DEFAULT_NO_ACT_SCALE
    model_variant = args.model_variant

    n_log_points = args.sae_log_points if args.sae_log_points is not None else DEFAULT_LOG_POINTS

    grid = _build_grid(args)

    trsf_tag, trsf_cfg = _trsf_meta(train_output)
    print(f'Transformer tag: {trsf_tag}')

    if activation_source == 'mean_pooled':
        model_name = getattr(trsf_cfg, 'model', None)
        if model_name not in {
            'transformer_meanclass', 'transformer_meanclass_nores',
            'transformer_freeclass', 'transformer_freeclass_nores',
        }:
            print(
                f'ERROR: sae_activation_source=mean_pooled requires a meanclass or '
                f'freeclass transformer, but transformer artifact has model={model_name!r}.',
                file=sys.stderr,
            )
            sys.exit(1)
        last_layer = int(trsf_cfg.num_layers) - 1
        requested = grid.get('sae_layer')
        if requested != [last_layer]:
            print(
                f'NOTE: mean_pooled mode ignores sae_layer={requested}; '
                f'forcing sae_layer=[{last_layer}] (the last block).'
            )
        grid['sae_layer'] = [last_layer]

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    configs = []
    for combo in product(*values):
        raw = dict(zip(keys, combo))
        steps = int(raw['sae_steps'])

        pf = n_log_points

        c = {
            'train_output': str(train_output),
            'sae_layer': int(raw['sae_layer']),
            'sae_latent_dim': int(raw['sae_latent_dim']) if raw['sae_latent_dim'] is not None else None,
            'sae_lambda_l1': float(raw['sae_lambda_l1']),
            'sae_lr': float(raw['sae_lr']),
            'sae_steps': steps,
            'sae_sample_batch_size': int(raw['sae_sample_batch_size']),
            'sae_batch_limit': int(raw['sae_batch_limit']),
            'sae_activation_source': activation_source,
            **({'sae_token_idx': int(token_idx)} if activation_source == 'one_token' else {}),
            'sae_train_size': int(raw['sae_train_size']),
            'sae_eval_size': int(eval_size),
            'sae_log_points': int(pf),
            'sae_lambda_warmup_frac': float(raw['sae_lambda_warmup_frac']),
            'sae_lr_decay_frac': float(raw['sae_lr_decay_frac']),
            'no_act_scale': bool(no_act_scale),
            'model_variant': model_variant,
        }
        c['outname'] = _make_outname(outdir, trsf_tag, c)
        configs.append(c)

    out_json = outdir / 'sweep_configs.json'

    # --append: load existing configs and merge
    if args.append and out_json.exists():
        with open(out_json) as f:
            existing = json.load(f)
        print(f'Appending to existing {len(existing)} config(s) in {out_json}')
        configs = existing + configs

    with open(out_json, 'w') as f:
        json.dump(configs, f, indent=2)

    n = len(configs)
    print(f'Total {n} job configurations -> {out_json}')
    print(f'Slurm array range:  0-{n - 1}')

    # Show swept parameters
    for k in keys:
        vals = grid[k]
        if len(vals) > 1:
            print(f'  {k}: {vals}')

    print()
    print('Next steps:')
    print(f'  1. Set SWEEP_CONFIGS={out_json} in slurm/sae/run_sweep.sh')
    print(f'  2. Set #SBATCH --array=0-{n - 1} in slurm/sae/run_sweep.sh')
    print('  3. sbatch slurm/sae/run_sweep.sh')


if __name__ == '__main__':
    main()
