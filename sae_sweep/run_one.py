"""Run a single SAE training job from the sweep configuration file.

Called by run_sweep.sh with:

    python sae_sweep/run_one.py \\
        --sweep_configs /path/to/sweep_configs.json \\
        --task_id $SLURM_ARRAY_TASK_ID

The script reads configs[task_id] from sweep_configs.json and invokes
train_sae.py with the corresponding arguments.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep_configs', required=True,
                        help='Path to sweep_configs.json produced by generate_sweep.py')
    parser.add_argument('--task_id', type=int, required=True,
                        help='Slurm array task index (0-based)')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (e.g. cuda, cpu). Defaults to cuda if available.')
    args = parser.parse_args()

    with open(args.sweep_configs) as f:
        configs = json.load(f)

    if args.task_id < 0 or args.task_id >= len(configs):
        print(f'ERROR: task_id={args.task_id} out of range [0, {len(configs) - 1}]',
              file=sys.stderr)
        sys.exit(1)

    c = configs[args.task_id]
    print(
        f'[task {args.task_id}/{len(configs) - 1}] '
        f'layer={c["sae_layer"]} '
        f'latent_dim={c["sae_latent_dim"]} '
        f'lambda_l1={c["sae_lambda_l1"]} '
        f'lr={c["sae_lr"]} '
        f'steps={c["sae_steps"]} '
        f'batch_size={c["sae_sample_batch_size"]}'
    )
    print(f'Output: {c["outname"]}')

    # Ensure the output directory exists
    Path(c['outname']).parent.mkdir(parents=True, exist_ok=True)

    train_sae = REPO_ROOT / 'train_sae.py'
    cmd = [
        sys.executable, str(train_sae),
        '--train_output', c['train_output'],
        '--outname', c['outname'],
        '--sae_layer', str(c['sae_layer']),
        '--sae_lambda_l1', str(c['sae_lambda_l1']),
        '--sae_lr', str(c['sae_lr']),
        '--sae_steps', str(c['sae_steps']),
        '--sae_sample_batch_size', str(c['sae_sample_batch_size']),
        '--sae_batch_limit', str(c['sae_batch_limit']),
        '--sae_activation_source', c['sae_activation_source'],
        '--sae_token_idx', str(c['sae_token_idx']),
        '--sae_train_size', str(c['sae_train_size']),
        '--sae_eval_size', str(c['sae_eval_size']),
        '--sae_log_points', str(c['sae_log_points']),
        '--sae_lambda_warmup_frac', str(c.get('sae_lambda_warmup_frac', 0.05)),
        '--sae_lr_decay_frac', str(c.get('sae_lr_decay_frac', 0.2)),
    ]
    if c['sae_latent_dim'] is not None:
        cmd += ['--sae_latent_dim', str(c['sae_latent_dim'])]
    if c.get('no_act_scale', False):
        cmd += ['--no_act_scale']
    if args.device is not None:
        cmd += ['--device', args.device]

    print('Command:', ' '.join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
