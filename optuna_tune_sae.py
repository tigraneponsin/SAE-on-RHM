#!/usr/bin/env python3
"""Optuna hyperparameter tuning for SAE training at fixed lambda1=0.01 and a fixed layer.

Search space:
  sae_lr   : log-uniform [1e-6, 1e-2]
  batch_size: categorical [64, 128, 256, 512, 1024]

Fixed:
  sae_lambda_l1          = 0.01
  sae_lambda_warmup_frac = 0.0
  sae_lr_decay_frac      = 0.0
  sae_latent_dim         = 20 * model.embedding_dim

Objective: final eval total loss (lower is better).
"""

import os
import sys
import copy
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import optuna

import datasets
import init
from datasets.random_hierarchy_model import sample_trees
from train_sae import _load_training_artifacts, train_sae_posthoc


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _sample_trees(config, fixed_rules, train_size, eval_size, train_seed, eval_seed):
    if fixed_rules is not None:
        trees_train = sample_trees(
            num_data=train_size, rules=fixed_rules,
            prior=None, probs=None, seed=train_seed,
        )
        trees_eval = sample_trees(
            num_data=eval_size, rules=fixed_rules,
            prior=None, probs=None, seed=eval_seed,
        )
    else:
        rhm_train = datasets.RHM(
            v=config.num_features, n=config.num_classes, m=config.num_synonyms,
            s=config.tuple_size, L=config.num_layers,
            seed_rules=config.seed_rules, seed_samples=train_seed,
            num_data=train_size, probs=None, transform=None,
        )
        trees_train = rhm_train.trees
        rhm_eval = datasets.RHM(
            v=config.num_features, n=config.num_classes, m=config.num_synonyms,
            s=config.tuple_size, L=config.num_layers,
            seed_rules=config.seed_rules, seed_samples=eval_seed,
            num_data=eval_size, probs=None, transform=None,
        )
        trees_eval = rhm_eval.trees
    return trees_train, trees_eval


def _make_loaders(config, trees_train, trees_eval, train_size, eval_size, batch_size):
    train_cfg = copy.deepcopy(config)
    train_cfg.train_size = train_size
    train_cfg.test_size = 0
    train_cfg.batch_size = min(batch_size, train_size)
    train_loader, _ = init.init_data(
        trees_train[config.num_layers], trees_train[0], train_cfg
    )

    eval_cfg = copy.deepcopy(config)
    eval_cfg.train_size = eval_size
    eval_cfg.test_size = 0
    eval_cfg.batch_size = min(batch_size, eval_size)
    eval_loader, _ = init.init_data(
        trees_eval[config.num_layers], trees_eval[0], eval_cfg
    )

    return train_loader, eval_loader


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _make_objective(model, trees_train, trees_eval, base_config, layer_id, latent_dim,
                    sae_train_size, sae_eval_size):
    def objective(trial):
        cfg = copy.deepcopy(base_config)

        # Fixed hyperparameters
        cfg.sae_lambda_l1          = 0.01
        cfg.sae_latent_dim         = latent_dim
        cfg.sae_layers             = str(layer_id)
        cfg.sae_lambda_warmup_frac = 0.0
        cfg.sae_lr_decay_frac      = 0.0

        # Tuned hyperparameters
        cfg.sae_lr = trial.suggest_float('sae_lr', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
        cfg.sae_sample_batch_size  = batch_size

        train_loader, eval_loader = _make_loaders(
            cfg, trees_train, trees_eval, sae_train_size, sae_eval_size, batch_size
        )

        cfg.sae_log_points = 1

        _, _, _, sae_eval_curves, _, _ = train_sae_posthoc(
            model, train_loader, cfg, eval_loader=eval_loader
        )

        total_list = sae_eval_curves.get(layer_id, {}).get('total', [])
        if not total_list:
            return float('nan')
        return float(total_list[-1])

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    # Load transformer checkpoint
    config, model_state, _, fixed_rules = _load_training_artifacts(args)

    # Apply SAE defaults for any missing config fields
    defaults = {
        'sae_layers': 'all',
        'sae_latent_dim': None,
        'sae_lambda_l1': 0.01,
        'sae_lr': 1e-4,
        'sae_steps': 2**17,
        'sae_batch_limit': 0,
        'sae_log_points': 128,
        'sae_activation_source': 'one_token',
        'sae_token_idx': 0,
        'sae_sample_batch_size': int(config.batch_size),
        'sae_lambda_warmup_frac': 0,
        'sae_lr_decay_frac': 0,
    }
    for key, val in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, val)

    # Apply CLI overrides that are not part of the search space
    if args.device is not None:
        config.device = args.device
    if args.sae_activation_source is not None:
        config.sae_activation_source = args.sae_activation_source
    if args.sae_token_idx is not None:
        config.sae_token_idx = args.sae_token_idx
    if args.sae_steps is not None:
        config.sae_steps = args.sae_steps
    if args.sae_batch_limit is not None:
        config.sae_batch_limit = args.sae_batch_limit
    if args.sae_sample_batch_size is not None:
        config.sae_sample_batch_size = args.sae_sample_batch_size

    layer_id = args.sae_layer

    sae_train_size = (
        args.sae_train_size if args.sae_train_size is not None
        else int(config.train_size)
    )
    sae_eval_size = (
        args.sae_eval_size if args.sae_eval_size is not None
        else int(config.test_size if config.test_size > 0 else config.train_size)
    )

    transformer_seed = int(config.seed_sample)
    sae_train_seed = transformer_seed + 1
    sae_eval_seed = sae_train_seed + 1

    print(f'Sampling trees: train_size={sae_train_size}, eval_size={sae_eval_size}')
    trees_train, trees_eval = _sample_trees(
        config, fixed_rules, sae_train_size, sae_eval_size, sae_train_seed, sae_eval_seed
    )

    # Build model (once, reused across all trials)
    config.disable_compile = True
    model = init.init_model(config)
    model.load_state_dict(model_state)
    model = model.to(config.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    latent_dim = 20 * model.embedding_dim
    print(
        f'Transformer loaded on {config.device}. '
        f'embedding_dim={model.embedding_dim}, latent_dim={latent_dim} (20x embedding_dim)'
    )
    print(
        f'Starting Optuna study "{args.study_name}": '
        f'{args.n_trials} trials, layer={layer_id}, '
        f'lambda1=0.01, warmup=0, decay=0'
    )

    db_path = f'{args.outname}.db' if args.outname else 'optuna_sae.db'
    storage = f'sqlite:///{db_path}'
    study = optuna.create_study(
        direction='minimize',
        storage=storage,
        study_name=args.study_name,
        load_if_exists=True,
    )

    objective = _make_objective(
        model, trees_train, trees_eval, config, layer_id, latent_dim,
        sae_train_size, sae_eval_size,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print('\n=== Best trial ===')
    print(f'  eval total loss: {study.best_value:.6f}')
    print('  params:')
    for k, v in study.best_params.items():
        print(f'    {k}: {v}')

    out = f'{args.outname}.pt' if args.outname else 'optuna_sae_results.pt'
    torch.save({
        'best_params': study.best_params,
        'best_value': study.best_value,
        'all_trials': [
            {'params': t.params, 'value': t.value, 'state': str(t.state)}
            for t in study.trials
        ],
        'fixed': {
            'sae_lambda_l1': 0.01,
            'sae_lambda_warmup_frac': 0.0,
            'sae_lr_decay_frac': 0.0,
            'sae_latent_dim': latent_dim,
            'sae_layer': layer_id,
        },
    }, out)
    print(f'Results saved to {out}')
    print(f'Study DB at {db_path}  (resume with --load_if_exists via the same study_name)')


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(
        description='Optuna tuning for SAE at fixed lambda1=0.01 and a fixed layer'
    )

    # Checkpoint (same interface as train_sae.py)
    parser.add_argument('--train_output', type=str, default=None,
                        help='path to main.py consolidated .pt artifact')
    parser.add_argument('--config_checkpoint', type=str, default=None)
    parser.add_argument('--model_checkpoint', type=str, default=None)

    # Required tuning settings
    parser.add_argument('--sae_layer', type=int, required=True,
                        help='transformer layer to train SAE on')
    parser.add_argument('--n_trials', type=int, default=25,
                        help='number of Optuna trials (default: 25)')
    parser.add_argument('--study_name', type=str, default='sae_tuning',
                        help='Optuna study name (used for resuming via the SQLite DB)')
    parser.add_argument('--outname', type=str, default=None,
                        help='base path for outputs: <outname>.pt and <outname>.db')

    # Pass-through settings (not part of the search space)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sae_train_size', type=int, default=None)
    parser.add_argument('--sae_eval_size', type=int, default=None)
    parser.add_argument('--sae_activation_source', type=str, default=None)
    parser.add_argument('--sae_token_idx', type=int, default=None)
    parser.add_argument('--sae_steps', type=int, default=None,
                        help='number of SAE training steps per trial (default: 2**17)')
    parser.add_argument('--sae_batch_limit', type=int, default=None)
    parser.add_argument('--sae_sample_batch_size', type=int, default=None)

    args = parser.parse_args()

    assert (args.train_output is not None) or (
        args.config_checkpoint is not None and args.model_checkpoint is not None
    ), 'Provide --train_output OR both --config_checkpoint and --model_checkpoint'

    run(args)
