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


def parse_sae_layers(sae_layers, num_layers):
    if sae_layers is None or sae_layers.lower() == 'all':
        return list(range(num_layers))

    layers = [int(layer.strip()) for layer in sae_layers.split(',') if layer.strip()]
    assert len(layers) > 0, 'sae_layers must contain at least one valid layer index'
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
    sae_state = {}
    sae_metrics = {}

    latent_dim = config.sae_latent_dim if config.sae_latent_dim is not None else 4 * model.embedding_dim
    batch_limit = config.sae_batch_limit if config.sae_batch_limit > 0 else None
    print_freq = max(config.sae_print_freq, 1)

    for layer_id in layer_ids:
        sae = models.SparseAutoencoder(
            input_dim=model.embedding_dim,
            latent_dim=latent_dim,
        ).to(config.device)
        optimizer = optim.AdamW(sae.parameters(), lr=config.sae_lr, weight_decay=0.0)

        activation_buffer = []

        def save_activation(_module, _inputs, output):
            activation_buffer.append(output.detach())

        hook = model.blocks[layer_id].register_forward_hook(save_activation)

        step = 0
        running_total = 0.0
        running_recon = 0.0
        running_sparse = 0.0
        last_z = None

        while step < config.sae_steps:
            for inputs, _ in train_loader:
                with torch.no_grad():
                    _ = model(inputs.to(config.device))

                if not activation_buffer:
                    continue

                activation = activation_buffer.pop(0)
                activation = activation.reshape(-1, activation.size(-1))

                if batch_limit is not None and activation.size(0) > batch_limit:
                    sample_ids = torch.randperm(activation.size(0), device=activation.device)[:batch_limit]
                    activation = activation[sample_ids]

                total_loss, recon_loss, sparse_loss = sae.loss(activation, lambda_l1=config.sae_lambda_l1)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_total += total_loss.item()
                running_recon += recon_loss.item()
                running_sparse += sparse_loss.item()
                with torch.no_grad():
                    _, last_z = sae(activation)

                step += 1
                if step % print_freq == 0 or step == config.sae_steps:
                    print(
                        f'sae layer {layer_id} step {step}/{config.sae_steps} '
                        f'total={running_total / step:.6f} recon={running_recon / step:.6f} sparse={running_sparse / step:.6f}'
                    )

                if step >= config.sae_steps:
                    break

        hook.remove()

        stats = sae.activation_stats(last_z) if last_z is not None else {'active_fraction': 0.0, 'dead_features': latent_dim}
        sae_metrics[layer_id] = {
            'total_loss': running_total / max(step, 1),
            'recon_loss': running_recon / max(step, 1),
            'sparse_loss': running_sparse / max(step, 1),
            'active_fraction': stats['active_fraction'],
            'dead_features': stats['dead_features'],
            'steps': step,
            'latent_dim': latent_dim,
        }
        sae_state[layer_id] = copy.deepcopy(sae.state_dict())

    return sae_state, sae_metrics, layer_ids


def _set_if_missing(config, key, value):
    if not hasattr(config, key):
        setattr(config, key, value)


def _load_config_from_blob(blob):
    if isinstance(blob, dict) and 'config' in blob:
        return blob['config']
    return blob


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

        config = blob['config']
        output = blob['output']
        assert 'model' in output, (
            'No model weights found in train_output. Re-run transformer training with --save_models '
            'or train with --checkpoints and use --config_checkpoint + --model_checkpoint.'
        )
        model_state = output['model']
        model_step = output.get('step', None)
        return config, model_state, model_step

    assert args.model_checkpoint is not None and args.config_checkpoint is not None, (
        'Use either --train_output, or both --config_checkpoint and --model_checkpoint.'
    )

    config_blob = torch.load(args.config_checkpoint, map_location='cpu')
    config = _load_config_from_blob(config_blob)

    model_blob = torch.load(args.model_checkpoint, map_location='cpu')
    if isinstance(model_blob, dict) and 'model' in model_blob:
        model_state = model_blob['model']
        model_step = model_blob.get('step', None)
    else:
        model_state = model_blob
        model_step = None

    return config, model_state, model_step


def run(args):
    config, model_state, model_step = _load_training_artifacts(args)

    _set_if_missing(config, 'sae_layers', 'all')
    _set_if_missing(config, 'sae_latent_dim', None)
    _set_if_missing(config, 'sae_lambda_l1', 3.0)
    _set_if_missing(config, 'sae_lr', 1e-3)
    _set_if_missing(config, 'sae_steps', 512)
    _set_if_missing(config, 'sae_batch_limit', 0)
    _set_if_missing(config, 'sae_print_freq', 128)

    if args.device is not None:
        config.device = args.device
    if args.sae_layers is not None:
        config.sae_layers = args.sae_layers
    if args.sae_latent_dim is not None:
        config.sae_latent_dim = args.sae_latent_dim
    if args.sae_lambda_l1 is not None:
        config.sae_lambda_l1 = args.sae_lambda_l1
    if args.sae_lr is not None:
        config.sae_lr = args.sae_lr
    if args.sae_steps is not None:
        config.sae_steps = args.sae_steps
    if args.sae_batch_limit is not None:
        config.sae_batch_limit = args.sae_batch_limit
    if args.sae_print_freq is not None:
        config.sae_print_freq = args.sae_print_freq

    if config.batch_size >= config.train_size:
        config.batch_size = config.train_size
    assert (config.train_size % config.batch_size) == 0, 'batch_size must divide train_size!'

    rhm = datasets.RHM(
        v=config.num_features,
        n=config.num_classes,
        m=config.num_synonyms,
        s=config.tuple_size,
        L=config.num_layers,
        seed_rules=config.seed_rules,
        seed_samples=config.seed_sample,
        num_data=config.train_size + config.test_size,
        probs=None,
        transform=None,
    )
    inputs = rhm.trees[config.num_layers]
    targets = rhm.trees[0]
    train_loader, _ = init.init_data(inputs, targets, config)

    model = init.init_model(config)
    model.load_state_dict(model_state)
    model = model.to(config.device)

    sae_state, sae_metrics, layer_ids = train_sae_posthoc(model, train_loader, config)

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
            'sae_layers': layer_ids,
            'sae_state': sae_state,
            'sae_metrics': sae_metrics,
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
    parser.add_argument('--sae_print_freq', type=int, default=None, help='print frequency during SAE training')

    args = parser.parse_args()

    assert (args.train_output is not None) or (
        args.config_checkpoint is not None and args.model_checkpoint is not None
    ), 'Provide --train_output OR both --config_checkpoint and --model_checkpoint'

    run(args)
