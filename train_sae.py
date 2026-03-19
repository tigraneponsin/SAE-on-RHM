import os
import sys
import copy
import argparse

# Ensure root SAE-on-RHM init.py is imported, not from other sources
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append('~/rhm-training')

import torch
import torch.optim as optim

import datasets, models
import init
from datasets.random_hierarchy_model import sample_trees


def parse_sae_layers(sae_layers, num_layers):
    if sae_layers is None or sae_layers.lower() == 'all':
        return list(range(num_layers))

    layers = [int(layer.strip()) for layer in sae_layers.split(',') if layer.strip()]
    assert layers, 'sae_layers must contain at least one valid layer index'
    for layer in layers:
        assert 0 <= layer < num_layers, f'sae layer {layer} is out of range [0, {num_layers - 1}]'
    return sorted(set(layers))


def train_sae_posthoc(model, train_loader, config):
    assert config.model == 'transformer_class', 'post-hoc SAE is currently implemented for transformer_class only'
    assert config.input_format == 'long', 'post-hoc SAE on transformer_class requires input_format=long'

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    layer_ids = parse_sae_layers(config.sae_layers, model.num_layers)
    sae_state, sae_metrics, sae_curves = {}, {}, {}

    latent_dim = config.sae_latent_dim if config.sae_latent_dim is not None else 4 * model.embedding_dim
    batch_limit = config.sae_batch_limit if config.sae_batch_limit > 0 else None
    print_freq = max(config.sae_print_freq, 1)
    activation_source = str(config.sae_activation_source).lower()
    assert activation_source in {'all_tokens', 'cls_token'}, (
        f"sae_activation_source={config.sae_activation_source} is invalid. "
        "Use one of: all_tokens, cls_token"
    )

    for layer_id in layer_ids:
        sae = models.SparseAutoencoder(
            input_dim=model.embedding_dim,
            latent_dim=latent_dim,
        ).to(config.device)
        optimizer = optim.AdamW(sae.parameters(), lr=config.sae_lr, weight_decay=0.0)

        activation_buffer = []
        hook = model.blocks[layer_id].register_forward_hook(
            lambda _m, _i, o: activation_buffer.append(o.detach())
        )

        step = 0
        last_total = last_recon = last_sparse = 0.0
        last_z = None
        init_logged = False
        curve_steps, curve_total, curve_recon, curve_sparse = [], [], [], []

        while step < config.sae_steps:
            for inputs, _ in train_loader:
                with torch.no_grad():
                    model(inputs.to(config.device))

                if not activation_buffer:
                    continue

                # For transformer_class: position 0 is [CLS], positions 1..T are input tokens.
                act = activation_buffer.pop(0)
                act = act[:, :1, :] if activation_source == 'cls_token' else act[:, 1:, :]
                act = act.reshape(-1, act.size(-1))

                if batch_limit is not None and act.size(0) > batch_limit:
                    act = act[torch.randperm(act.size(0), device=act.device)[:batch_limit]]

                if not init_logged:
                    with torch.no_grad():
                        init_total, init_recon, init_sparse = sae.loss(act, lambda_l1=config.sae_lambda_l1)
                        _, init_z = sae(act)
                    init_stats = sae.activation_stats(init_z)
                    curve_steps.append(0)
                    curve_total.append(float(init_total))
                    curve_recon.append(float(init_recon))
                    curve_sparse.append(float(init_sparse))
                    print(
                        f'sae layer {layer_id} step 0/{config.sae_steps} '
                        f'total={float(init_total):.6f} recon={float(init_recon):.6f} sparse={float(init_sparse):.6f} '
                        f'active_fraction={init_stats["active_fraction"]:.6f} dead_features={init_stats["dead_features"]}'
                    )
                    init_logged = True

                total_loss, recon_loss, sparse_loss = sae.loss(act, lambda_l1=config.sae_lambda_l1)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                last_total, last_recon, last_sparse = float(total_loss), float(recon_loss), float(sparse_loss)
                with torch.no_grad():
                    _, last_z = sae(act)

                step += 1
                if step % print_freq == 0 or step == config.sae_steps:
                    curve_steps.append(step)
                    curve_total.append(last_total)
                    curve_recon.append(last_recon)
                    curve_sparse.append(last_sparse)
                    print(
                        f'sae layer {layer_id} step {step}/{config.sae_steps} '
                        f'total={last_total:.6f} recon={last_recon:.6f} sparse={last_sparse:.6f}'
                    )

                if step >= config.sae_steps:
                    break

        hook.remove()

        stats = sae.activation_stats(last_z) if last_z is not None else {'active_fraction': 0.0, 'dead_features': latent_dim}
        sae_metrics[layer_id] = {
            'total_loss': last_total,
            'recon_loss': last_recon,
            'sparse_loss': last_sparse,
            'active_fraction': stats['active_fraction'],
            'dead_features': stats['dead_features'],
            'steps': step,
            'latent_dim': latent_dim,
        }
        sae_state[layer_id] = copy.deepcopy(sae.state_dict())
        sae_curves[layer_id] = {
            'step': curve_steps,
            'total': curve_total,
            'recon': curve_recon,
            'sparse': curve_sparse,
        }

    return sae_state, sae_metrics, sae_curves, layer_ids


def _resolve_sae_output_name(args):
    if args.outname is not None:
        return args.outname
    source = args.train_output if args.train_output is not None else args.model_checkpoint
    base, _ = os.path.splitext(source)
    return f'{base}_sae.pt'


def _load_training_artifacts(args):
    if args.train_output is not None:
        blob = torch.load(args.train_output, map_location='cpu')
        assert isinstance(blob, dict) and 'config' in blob and 'output' in blob, (
            'train_output must be a main.py consolidated output containing config/output.'
        )
        output = blob['output']
        assert 'model' in output, (
            'No model weights found in train_output. Re-run transformer training with --save_models '
            'or train with --checkpoints and use --config_checkpoint + --model_checkpoint.'
        )
        return blob['config'], output['model'], output.get('step'), output.get('rules')

    assert args.model_checkpoint is not None and args.config_checkpoint is not None, (
        'Use either --train_output, or both --config_checkpoint and --model_checkpoint.'
    )

    config_blob = torch.load(args.config_checkpoint, map_location='cpu')
    config = config_blob['config'] if isinstance(config_blob, dict) and 'config' in config_blob else config_blob
    rules = config_blob.get('rules') if isinstance(config_blob, dict) else None

    model_blob = torch.load(args.model_checkpoint, map_location='cpu')
    if isinstance(model_blob, dict) and 'model' in model_blob:
        return config, model_blob['model'], model_blob.get('step'), rules
    return config, model_blob, None, rules


def run(args):
    config, model_state, model_step, fixed_rules = _load_training_artifacts(args)

    # Apply SAE hyperparameter defaults (only if not already set in the loaded config)
    defaults = {
        'sae_layers': 'all',
        'sae_latent_dim': None,
        'sae_lambda_l1': 3.0,
        'sae_lr': 1e-3,
        'sae_steps': 512,
        'sae_batch_limit': 0,
        'sae_print_freq': 128,
        'sae_activation_source': 'all_tokens',
        'sae_sample_batch_size': int(config.batch_size),
    }
    for key, val in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, val)

    # Override config with any CLI arguments provided
    overrides = {
        'device': args.device,
        'sae_layers': args.sae_layers,
        'sae_latent_dim': args.sae_latent_dim,
        'sae_lambda_l1': args.sae_lambda_l1,
        'sae_lr': args.sae_lr,
        'sae_steps': args.sae_steps,
        'sae_batch_limit': args.sae_batch_limit,
        'sae_print_freq': args.sae_print_freq,
        'sae_activation_source': args.sae_activation_source,
        'sae_sample_batch_size': args.sae_sample_batch_size,
    }
    for key, val in overrides.items():
        if val is not None:
            setattr(config, key, val)

    sae_train_size = args.sae_train_size if args.sae_train_size is not None else int(config.train_size)
    sae_eval_size = args.sae_eval_size if args.sae_eval_size is not None else int(config.test_size if config.test_size > 0 else config.train_size)
    assert sae_train_size > 0 and sae_eval_size > 0, 'sae_train_size and sae_eval_size must be > 0'

    transformer_seed = int(config.seed_sample)
    sae_train_seed = args.sae_train_seed_sample if args.sae_train_seed_sample is not None else transformer_seed + 1
    if sae_train_seed == transformer_seed:
        sae_train_seed = transformer_seed + 1
    sae_eval_seed = args.sae_eval_seed_sample if args.sae_eval_seed_sample is not None else sae_train_seed + 1
    if sae_eval_seed == sae_train_seed:
        sae_eval_seed = sae_train_seed + 1

    print(
        f'SAE data split: train_size={sae_train_size} (seed_sample={sae_train_seed}), '
        f'eval_size={sae_eval_size} (seed_sample={sae_eval_seed})'
    )
    print(f'Transformer training seed_sample={transformer_seed}')
    print('Using fixed RHM rules loaded from training artifact.' if fixed_rules is not None else 'No saved RHM rules found; regenerating rules from seed_rules.')

    assert int(config.sae_sample_batch_size) > 0, 'sae_sample_batch_size must be > 0'

    # Build SAE training data split
    if fixed_rules is not None:
        trees_train = sample_trees(
            num_data=sae_train_size, rules=fixed_rules, prior=None, probs=None, seed=sae_train_seed,
        )
    else:
        rhm_train = datasets.RHM(
            v=config.num_features, n=config.num_classes, m=config.num_synonyms,
            s=config.tuple_size, L=config.num_layers,
            seed_rules=config.seed_rules, seed_samples=sae_train_seed,
            num_data=sae_train_size, probs=None, transform=None,
        )
        trees_train = rhm_train.trees

    train_data_config = copy.deepcopy(config)
    train_data_config.train_size = sae_train_size
    train_data_config.test_size = 0
    train_data_config.batch_size = min(int(config.sae_sample_batch_size), sae_train_size)
    train_loader, _ = init.init_data(trees_train[config.num_layers], trees_train[0], train_data_config)

    model = init.init_model(config)
    model.load_state_dict(model_state)
    model = model.to(config.device)

    sae_state, sae_metrics, sae_curves, layer_ids = train_sae_posthoc(model, train_loader, config)

    outname = _resolve_sae_output_name(args)
    torch.save(
        {
            'config': config,
            'source': {
                'train_output': args.train_output,
                'config_checkpoint': args.config_checkpoint,
                'model_checkpoint': args.model_checkpoint,
                'model_step': model_step,
            },
            'sae_dataset_split': {
                'rules_source': 'artifact' if fixed_rules is not None else 'seed_rules_resampled',
                'rules_seed': int(config.seed_rules),
                'train_size': sae_train_size,
                'eval_size': sae_eval_size,
                'transformer_seed_sample': transformer_seed,
                'train_seed_sample': sae_train_seed,
                'eval_seed_sample': sae_eval_seed,
                'is_train_seed_different_from_transformer': sae_train_seed != transformer_seed,
                'is_disjoint_seed': sae_train_seed != sae_eval_seed,
                'num_features': int(config.num_features),
                'num_classes': int(config.num_classes),
                'num_synonyms': int(config.num_synonyms),
                'tuple_size': int(config.tuple_size),
                'num_layers': int(config.num_layers),
            },
            'sae_layers': layer_ids,
            'sae_state': sae_state,
            'sae_metrics': sae_metrics,
            'sae_training_curves': sae_curves,
            'sae_training_setup': {
                'sae_activation_source': str(config.sae_activation_source),
                'sae_sample_batch_size': int(config.sae_sample_batch_size),
                'sae_batch_limit': int(config.sae_batch_limit),
            },
        },
        outname,
    )
    print(f'Saved SAE checkpoints to {outname}')


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description='Post-hoc SAE training on trained transformer_class checkpoints')

    parser.add_argument('--train_output', type=str, default=None, help='path to main.py output .pt/.pkl produced with --save_models')
    parser.add_argument('--config_checkpoint', type=str, default=None, help='path to <outname>_config.pt from --checkpoints runs')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='path to model checkpoint (<outname>_t*.pt) or plain state_dict')
    parser.add_argument('--outname', type=str, default=None, help='output path for SAE artifact (default: source + _sae.pt)')

    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sae_layers', type=str, default=None, help='comma-separated layer ids or all')
    parser.add_argument('--sae_latent_dim', type=int, default=None, help='latent width of SAE (default: 4 * embedding_dim)')
    parser.add_argument('--sae_lambda_l1', type=float, default=None, help='L1 sparsity coefficient for SAE latent activations')
    parser.add_argument('--sae_lr', type=float, default=None, help='learning rate for SAE optimizer')
    parser.add_argument('--sae_steps', type=int, default=None, help='number of optimization steps for each SAE')
    parser.add_argument('--sae_batch_limit', type=int, default=None, help='max tokens per SAE step (0 uses all tokens in batch)')
    parser.add_argument('--sae_sample_batch_size', type=int, default=None, help='number of RHM samples per forward pass used to gather SAE activations')
    parser.add_argument('--sae_activation_source', type=str, default=None, help='which positions feed SAE: all_tokens or cls_token')
    parser.add_argument('--sae_print_freq', type=int, default=None, help='print frequency during SAE training')
    parser.add_argument('--sae_train_size', type=int, default=None, help='number of RHM samples used to train SAE')
    parser.add_argument('--sae_eval_size', type=int, default=None, help='number of RHM samples reserved for later SAE analysis')
    parser.add_argument('--sae_train_seed_sample', type=int, default=None, help='seed_samples used for SAE training split (defaults to transformer seed + 1)')
    parser.add_argument('--sae_eval_seed_sample', type=int, default=None, help='seed_samples used for analysis split (defaults to train seed + 1)')

    args = parser.parse_args()

    assert (args.train_output is not None) or (
        args.config_checkpoint is not None and args.model_checkpoint is not None
    ), 'Provide --train_output OR both --config_checkpoint and --model_checkpoint'

    run(args)
