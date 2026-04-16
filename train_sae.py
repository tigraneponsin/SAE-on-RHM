import os
import sys
import copy
import itertools
import argparse

# Ensure root SAE-on-RHM init.py is imported, not from other sources
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append('~/rhm-training')

import numpy as np
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


def _select_tokens(act, activation_source, is_cls_model, token_idx):
    """Select tokens from a (B, T, D) activation tensor based on source mode.

    Returns a (B, T', D) tensor where T' depends on activation_source:
      - 'cls_token': T'=1, the [CLS] position
      - 'one_token': T'=1, a single real-token position (offset by 1 for CLS models)
      - 'all_tokens': all real tokens (skipping [CLS] for CLS models)
    """
    if activation_source == 'cls_token':
        return act[:, :1, :]
    elif activation_source == 'one_token':
        offset = 1 if is_cls_model else 0
        return act[:, offset + token_idx : offset + token_idx + 1, :]
    elif is_cls_model:
        return act[:, 1:, :]
    return act


def _compute_activation_scale(model, train_loader, layer_id, activation_source,
                               is_cls_model, token_idx, device):
    """Compute a scalar scale so that E[||scale * x||_2] = sqrt(embedding_dim).

    This normalizes activations before feeding them to the SAE, making lambda
    comparable across layers and datasets (Anthropic April 2024 recommendation).
    Returns the scale factor (float). Prints the pre-scaling mean norm.
    """
    buf = []
    hook = model.blocks[layer_id].register_forward_hook(
        lambda _m, _i, o: buf.append(o.detach())
    )
    sum_norm = 0.0
    n_tokens = 0
    with torch.no_grad():
        for inputs, _ in train_loader:
            model(inputs.to(device))
            if not buf:
                continue
            act = buf.pop(0)
            act = _select_tokens(act, activation_source, is_cls_model, token_idx)
            act = act.reshape(-1, act.size(-1))
            if act.numel() == 0:
                continue
            sum_norm += float(act.norm(dim=-1).sum())
            n_tokens += act.size(0)
    hook.remove()

    if n_tokens == 0:
        return 1.0
    embedding_dim = model.embedding_dim
    mean_norm = sum_norm / n_tokens                          # E[||x||_2]
    scale = (embedding_dim ** 0.5) / mean_norm              # so E[||scale*x||_2] = sqrt(n)
    print(f'  layer {layer_id}: mean ||x||_2 = {mean_norm:.4f}, '
          f'embedding_dim = {embedding_dim}, act_scale = {scale:.6f}')
    return float(scale)


def _sae_loss_chunked(sae, act, lambda_l1, chunk_tokens=None):
    """Compute SAE losses with optional token chunking to reduce peak memory."""
    n_tokens = int(act.size(0))
    if n_tokens == 0:
        return float('nan'), float('nan'), float('nan')

    if chunk_tokens is None or n_tokens <= chunk_tokens:
        total, recon, sparse = sae.loss(act, lambda_l1=lambda_l1)
        return float(total), float(recon), float(sparse)

    sum_total = sum_recon = sum_sparse = 0.0
    seen = 0
    for start in range(0, n_tokens, chunk_tokens):
        chunk = act[start:start + chunk_tokens]
        total, recon, sparse = sae.loss(chunk, lambda_l1=lambda_l1)
        w = float(chunk.size(0))
        sum_total += float(total) * w
        sum_recon += float(recon) * w
        sum_sparse += float(sparse) * w
        seen += int(w)

    return sum_total / seen, sum_recon / seen, sum_sparse / seen


def _collect_eval_loss(sae, model, eval_loader, layer_id, activation_source,
                       is_cls_model, token_idx, lambda_l1, device,
                       eval_chunk_tokens=None, act_scale=1.0):
    """Average SAE loss over the full eval_loader at the current SAE weights.
    Returns (total, recon, sparse) averaged across all eval tokens."""
    buf = []
    hook = model.blocks[layer_id].register_forward_hook(
        lambda _m, _i, o: buf.append(o.detach())
    )
    sum_total = sum_recon = sum_sparse = 0.0
    n_tokens = 0
    with torch.no_grad():
        for inputs, _ in eval_loader:
            model(inputs.to(device))
            if not buf:
                continue
            act = buf.pop(0)
            act = _select_tokens(act, activation_source, is_cls_model, token_idx)
            act = act.reshape(-1, act.size(-1))
            if act.numel() == 0:
                continue
            act = act * act_scale
            total, recon, sparse = _sae_loss_chunked(
                sae, act, lambda_l1=lambda_l1, chunk_tokens=eval_chunk_tokens,
            )
            bs = act.size(0)
            sum_total += total * bs
            sum_recon += recon * bs
            sum_sparse += sparse * bs
            n_tokens += bs
    hook.remove()
    if n_tokens == 0:
        return float('nan'), float('nan'), float('nan')
    return sum_total / n_tokens, sum_recon / n_tokens, sum_sparse / n_tokens


def train_sae_posthoc(model, train_loader, config, eval_loader=None):
    assert config.model in {
        'transformer_class',
        'transformer_meanclass',
        'transformer_meanclass_nores',
    }, (
        'post-hoc SAE is currently implemented for transformer_class, '
        'transformer_meanclass, or transformer_meanclass_nores only'
    )
    assert config.input_format == 'long', f'post-hoc SAE on {config.model} requires input_format=long'

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    layer_ids = parse_sae_layers(config.sae_layers, model.num_layers)
    sae_state, sae_metrics, sae_curves, sae_eval_curves = {}, {}, {}, {}

    latent_dim = config.sae_latent_dim if config.sae_latent_dim is not None else 4 * model.embedding_dim
    batch_limit = config.sae_batch_limit if config.sae_batch_limit > 0 else None
    eval_chunk_tokens = batch_limit if batch_limit is not None else 4096
    n_log_points = max(int(config.sae_log_points), 2)
    log_steps_set = set(
        int(round(s)) for s in np.geomspace(1, config.sae_steps, n_log_points)
    )
    log_steps_set.add(config.sae_steps)
    activation_source = str(config.sae_activation_source).lower()
    assert activation_source in {'all_tokens', 'cls_token', 'one_token'}, (
        f"sae_activation_source={config.sae_activation_source} is invalid. "
        "Use one of: all_tokens, cls_token, one_token"
    )
    if config.model in {'transformer_meanclass', 'transformer_meanclass_nores'} and activation_source == 'cls_token':
        raise ValueError(
            f'{config.model} has no [CLS] token. '
            'Use sae_activation_source=all_tokens or one_token.'
        )

    token_idx = int(getattr(config, 'sae_token_idx', 0))
    if activation_source == 'one_token':
        num_seq_tokens = config.tuple_size ** config.num_layers
        assert 0 <= token_idx < num_seq_tokens, (
            f'sae_token_idx={token_idx} is out of range [0, {num_seq_tokens - 1}] '
            f'(tuple_size={config.tuple_size}, num_layers={config.num_layers})'
        )

    is_cls_model = (config.model == 'transformer_class')
    lambda_warmup_frac = float(getattr(config, 'sae_lambda_warmup_frac', 0.05))
    lr_decay_frac = float(getattr(config, 'sae_lr_decay_frac', 0.2))
    act_scales = {}

    no_act_scale = bool(getattr(config, 'no_act_scale', False))

    for layer_id in layer_ids:
        # --- Input normalization: compute scale so E[||scale*x||_2] = sqrt(n) ---
        if no_act_scale:
            print(f'Skipping activation scale for layer {layer_id} (no_act_scale=True)')
            act_scale = 1.0
        else:
            print(f'Computing activation scale for layer {layer_id} ...')
            act_scale = _compute_activation_scale(
                model, train_loader, layer_id, activation_source,
                is_cls_model, token_idx, config.device,
            )
        act_scales[layer_id] = act_scale

        # Seed SAE initialization for reproducibility across runs.
        torch.manual_seed(int(config.seed_sample) + 1000 + layer_id)
        sae = models.SparseAutoencoder(
            input_dim=model.embedding_dim,
            latent_dim=latent_dim,
        ).to(config.device)
        optimizer = optim.AdamW(sae.parameters(), lr=config.sae_lr, weight_decay=0.0)

        activation_buffer = []
        hook = model.blocks[layer_id].register_forward_hook(
            lambda _m, _i, o: activation_buffer.append(o.detach())
        )

        warmup_steps = int(lambda_warmup_frac * config.sae_steps)
        decay_start = int((1.0 - lr_decay_frac) * config.sae_steps)

        last_total = last_recon = last_sparse = 0.0
        last_act = None
        init_logged = False
        curve_steps, curve_total, curve_recon, curve_sparse = [], [], [], []
        eval_curve_steps, eval_curve_total, eval_curve_recon, eval_curve_sparse = [], [], [], []

        loader_iter = iter(itertools.cycle(train_loader))
        for step in range(config.sae_steps):
            inputs, _ = next(loader_iter)
            with torch.no_grad():
                model(inputs.to(config.device))

            if not activation_buffer:
                continue

            act = activation_buffer.pop(0)
            act = _select_tokens(act, activation_source, is_cls_model, token_idx)
            act = act.reshape(-1, act.size(-1))
            act = act * act_scale

            if batch_limit is not None and act.size(0) > batch_limit:
                act = act[torch.randperm(act.size(0), device=act.device)[:batch_limit]]

            # lambda warmup: ramp from 0 to sae_lambda_l1 over first warmup_steps
            if warmup_steps > 0 and step < warmup_steps:
                current_lambda = config.sae_lambda_l1 * (step / warmup_steps)
            else:
                current_lambda = config.sae_lambda_l1

            if not init_logged:
                with torch.no_grad():
                    init_total, init_recon, init_sparse, init_z = sae.loss(act, lambda_l1=0.0, return_z=True)
                init_stats = sae.activation_stats(init_z)
                curve_steps.append(0)
                curve_total.append(float(init_total))
                curve_recon.append(float(init_recon))
                curve_sparse.append(float(init_sparse))
                if eval_loader is not None:
                    ev_total, ev_recon, ev_sparse = _collect_eval_loss(
                        sae, model, eval_loader, layer_id, activation_source,
                        is_cls_model, token_idx, 0.0, config.device,
                        eval_chunk_tokens=eval_chunk_tokens,
                        act_scale=act_scale,
                    )
                    eval_curve_steps.append(0)
                    eval_curve_total.append(ev_total)
                    eval_curve_recon.append(ev_recon)
                    eval_curve_sparse.append(ev_sparse)
                print(
                    f'sae layer {layer_id} step 0/{config.sae_steps} '
                    f'total={float(init_total):.6f} recon={float(init_recon):.6f} sparse={float(init_sparse):.6f} '
                    f'active_fraction={init_stats["active_fraction"]:.6f} dead_features={init_stats["dead_features"]}'
                )
                init_logged = True

            total_loss, recon_loss, sparse_loss = sae.loss(act, lambda_l1=current_lambda)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # LR linear decay: ramp down to 0 over last lr_decay_frac of steps
            if step >= decay_start:
                frac_remaining = 1.0 - (step - decay_start) / max(config.sae_steps - decay_start, 1)
                new_lr = config.sae_lr * max(frac_remaining, 0.0)
                for pg in optimizer.param_groups:
                    pg['lr'] = new_lr

            last_total, last_recon, last_sparse = float(total_loss), float(recon_loss), float(sparse_loss)
            last_act = act

            if step + 1 in log_steps_set:
                curve_steps.append(step + 1)
                curve_total.append(last_total)
                curve_recon.append(last_recon)
                curve_sparse.append(last_sparse)
                if eval_loader is not None:
                    ev_total, ev_recon, ev_sparse = _collect_eval_loss(
                        sae, model, eval_loader, layer_id, activation_source,
                        is_cls_model, token_idx, current_lambda, config.device,
                        eval_chunk_tokens=eval_chunk_tokens,
                        act_scale=act_scale,
                    )
                    eval_curve_steps.append(step + 1)
                    eval_curve_total.append(ev_total)
                    eval_curve_recon.append(ev_recon)
                    eval_curve_sparse.append(ev_sparse)
                print(
                    f'sae layer {layer_id} step {step + 1}/{config.sae_steps} '
                    f'lambda={current_lambda:.4g} '
                    f'total={last_total:.6f} recon={last_recon:.6f} sparse={last_sparse:.6f}'
                    + (f'  |  eval total={ev_total:.6f} recon={ev_recon:.6f} sparse={ev_sparse:.6f}'
                       if eval_loader is not None else '')
                )

        hook.remove()

        if last_act is not None:
            with torch.no_grad():
                _, last_z = sae(last_act)
            stats = sae.activation_stats(last_z)
        else:
            stats = {'active_fraction': 0.0, 'dead_features': latent_dim}
        sae_metrics[layer_id] = {
            'total_loss': last_total,
            'recon_loss': last_recon,
            'sparse_loss': last_sparse,
            'active_fraction': stats['active_fraction'],
            'dead_features': stats['dead_features'],
            'steps': config.sae_steps,
            'latent_dim': latent_dim,
        }
        sae_state[layer_id] = copy.deepcopy(sae.state_dict())
        sae_curves[layer_id] = {
            'step': curve_steps,
            'total': curve_total,
            'recon': curve_recon,
            'sparse': curve_sparse,
        }
        sae_eval_curves[layer_id] = {
            'step': eval_curve_steps,
            'total': eval_curve_total,
            'recon': eval_curve_recon,
            'sparse': eval_curve_sparse,
        }

        del sae, optimizer
        torch.cuda.empty_cache()

    return sae_state, sae_metrics, sae_curves, sae_eval_curves, layer_ids, act_scales


def _resolve_sae_output_name(args):
    if args.outname is not None:
        return args.outname
    source = args.train_output if args.train_output is not None else args.model_checkpoint
    base, _ = os.path.splitext(source)
    return f'{base}_sae.pt'


def _normalize_model_state_dict_keys(state_dict):
    """Normalize compiled-model state_dict keys for plain nn.Module loading.

    Checkpoints saved from torch.compile wrappers can prefix every key with
    '_orig_mod.'. SAE training loads an uncompiled transformer model, so strip
    this prefix when present.
    """
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if keys and all(k.startswith('_orig_mod.') for k in keys):
        return {k[len('_orig_mod.'):]: v for k, v in state_dict.items()}
    return state_dict


def _load_training_artifacts(args):
    if args.train_output is not None:
        blob = torch.load(args.train_output, map_location='cpu', weights_only=False)
        assert isinstance(blob, dict) and 'config' in blob and 'output' in blob, (
            'train_output must be a main.py consolidated output containing config/output.'
        )
        output = blob['output']
        assert 'model' in output, (
            'No model weights found in train_output. Re-run transformer training with --save_models '
            'or train with --checkpoints and use --config_checkpoint + --model_checkpoint.'
        )
        model_state = _normalize_model_state_dict_keys(output['model'])
        return blob['config'], model_state, output.get('step'), output.get('rules')

    assert args.model_checkpoint is not None and args.config_checkpoint is not None, (
        'Use either --train_output, or both --config_checkpoint and --model_checkpoint.'
    )

    config_blob = torch.load(args.config_checkpoint, map_location='cpu', weights_only=False)
    config = config_blob['config'] if isinstance(config_blob, dict) and 'config' in config_blob else config_blob
    rules = config_blob.get('rules') if isinstance(config_blob, dict) else None

    model_blob = torch.load(args.model_checkpoint, map_location='cpu', weights_only=False)
    if isinstance(model_blob, dict) and 'model' in model_blob:
        model_state = _normalize_model_state_dict_keys(model_blob['model'])
        return config, model_state, model_blob.get('step'), rules
    model_state = _normalize_model_state_dict_keys(model_blob)
    return config, model_state, None, rules


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
        'sae_log_points': 50,
        'sae_activation_source': 'all_tokens',
        'sae_token_idx': 0,
        'sae_sample_batch_size': int(config.batch_size),
        'sae_lambda_warmup_frac': 0.05,
        'sae_lr_decay_frac': 0.2,
        'no_act_scale': False,
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
        'sae_log_points': args.sae_log_points,
        'sae_activation_source': args.sae_activation_source,
        'sae_token_idx': args.sae_token_idx,
        'sae_sample_batch_size': args.sae_sample_batch_size,
        'sae_lambda_warmup_frac': args.sae_lambda_warmup_frac,
        'sae_lr_decay_frac': args.sae_lr_decay_frac,
    }
    for key, val in overrides.items():
        if val is not None:
            setattr(config, key, val)
    if args.no_act_scale:
        config.no_act_scale = True

    # Optional single-layer mode for running independent SAE jobs per transformer layer.
    if args.sae_layer is not None:
        config.sae_layers = str(int(args.sae_layer))

    sae_train_size = args.sae_train_size if args.sae_train_size is not None else int(config.train_size)
    sae_eval_size = args.sae_eval_size if args.sae_eval_size is not None else int(config.test_size if config.test_size > 0 else config.train_size)
    assert sae_train_size > 0 and sae_eval_size > 0, 'sae_train_size and sae_eval_size must be > 0'

    transformer_seed = int(config.seed_sample)
    sae_train_seed = args.sae_train_seed_sample if args.sae_train_seed_sample is not None else transformer_seed + 1
    if sae_train_seed == transformer_seed:
        raise ValueError(
            f'sae_train_seed_sample ({sae_train_seed}) must differ from '
            f'transformer seed_sample ({transformer_seed}) to avoid data overlap.'
        )
    sae_eval_seed = args.sae_eval_seed_sample if args.sae_eval_seed_sample is not None else sae_train_seed + 1
    if sae_eval_seed == sae_train_seed:
        raise ValueError(
            f'sae_eval_seed_sample ({sae_eval_seed}) must differ from '
            f'sae_train_seed_sample ({sae_train_seed}) to keep train/eval disjoint.'
        )

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

    # Build SAE eval data split (same rules, different seed — used for periodic eval loss logging)
    if fixed_rules is not None:
        trees_eval = sample_trees(
            num_data=sae_eval_size, rules=fixed_rules, prior=None, probs=None, seed=sae_eval_seed,
        )
    else:
        rhm_eval = datasets.RHM(
            v=config.num_features, n=config.num_classes, m=config.num_synonyms,
            s=config.tuple_size, L=config.num_layers,
            seed_rules=config.seed_rules, seed_samples=sae_eval_seed,
            num_data=sae_eval_size, probs=None, transform=None,
        )
        trees_eval = rhm_eval.trees

    eval_data_config = copy.deepcopy(config)
    eval_data_config.train_size = sae_eval_size
    eval_data_config.test_size = 0
    eval_data_config.batch_size = min(int(config.sae_sample_batch_size), sae_eval_size)
    eval_loader, _ = init.init_data(trees_eval[config.num_layers], trees_eval[0], eval_data_config)

    # SAE training relies on forward hooks; disable compile to avoid recompiles
    # and extra memory pressure from hook-mutated graphs.
    config.disable_compile = True
    model = init.init_model(config)
    model.load_state_dict(model_state)
    model = model.to(config.device)

    sae_state, sae_metrics, sae_curves, sae_eval_curves, layer_ids, act_scales = train_sae_posthoc(
        model, train_loader, config, eval_loader=eval_loader,
    )

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
            'sae_eval_curves': sae_eval_curves,
            'sae_training_setup': {
                'sae_activation_source': str(config.sae_activation_source),
                'sae_token_idx': int(getattr(config, 'sae_token_idx', 0)),
                'sae_layers': str(config.sae_layers),
                'sae_sample_batch_size': int(config.sae_sample_batch_size),
                'sae_batch_limit': int(config.sae_batch_limit),
                'sae_lambda_warmup_frac': float(getattr(config, 'sae_lambda_warmup_frac', 0.05)),
                'sae_lr_decay_frac': float(getattr(config, 'sae_lr_decay_frac', 0.2)),
                'act_scale': {str(lid): float(s) for lid, s in act_scales.items()},
            },
        },
        outname,
    )
    print(f'Saved SAE checkpoints to {outname}')


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description='Post-hoc SAE training on trained transformer_class/transformer_meanclass/transformer_meanclass_nores checkpoints')

    parser.add_argument('--train_output', type=str, default=None, help='path to main.py output .pt/.pkl produced with --save_models')
    parser.add_argument('--config_checkpoint', type=str, default=None, help='path to <outname>_config.pt from --checkpoints runs')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='path to model checkpoint (<outname>_t*.pt) or plain state_dict')
    parser.add_argument('--outname', type=str, default=None, help='output path for SAE artifact (default: source + _sae.pt)')

    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sae_layer', type=int, default=None, help='single transformer layer id to train (overrides --sae_layers)')
    parser.add_argument('--sae_layers', type=str, default=None, help='comma-separated layer ids or all')
    parser.add_argument('--sae_latent_dim', type=int, default=None, help='latent width of SAE (default: 4 * embedding_dim)')
    parser.add_argument('--sae_lambda_l1', type=float, default=None, help='L1 sparsity coefficient for SAE latent activations')
    parser.add_argument('--sae_lr', type=float, default=None, help='learning rate for SAE optimizer')
    parser.add_argument('--sae_steps', type=int, default=None, help='number of optimization steps for each SAE')
    parser.add_argument('--sae_batch_limit', type=int, default=None, help='max tokens per SAE step (0 uses all tokens in batch)')
    parser.add_argument('--sae_sample_batch_size', type=int, default=None, help='number of RHM samples per forward pass used to gather SAE activations')
    parser.add_argument('--sae_activation_source', type=str, default=None, help='which positions feed SAE: all_tokens, cls_token (transformer_class only), or one_token')
    parser.add_argument('--sae_token_idx', type=int, default=None, help='0-based index of the real token to use when sae_activation_source=one_token (counts from 0 among sequence tokens, i.e. skips [CLS] for transformer_class)')
    parser.add_argument('--sae_log_points', type=int, default=None, help='number of log-spaced checkpoints to record during SAE training (default: 50)')
    parser.add_argument('--sae_train_size', type=int, default=None, help='number of RHM samples used to train SAE')
    parser.add_argument('--sae_eval_size', type=int, default=None, help='number of RHM samples reserved for later SAE analysis')
    parser.add_argument('--sae_train_seed_sample', type=int, default=None, help='seed_samples used for SAE training split (defaults to transformer seed + 1)')
    parser.add_argument('--sae_eval_seed_sample', type=int, default=None, help='seed_samples used for analysis split (defaults to train seed + 1)')
    parser.add_argument('--sae_lambda_warmup_frac', type=float, default=None, help='fraction of steps over which lambda is linearly warmed up from 0 (default: 0.05)')
    parser.add_argument('--sae_lr_decay_frac', type=float, default=None, help='fraction of steps over which LR is linearly decayed to 0 at the end of training (default: 0.2)')
    parser.add_argument('--no_act_scale', action='store_true', default=False, help='disable activation rescaling (act_scale=1.0 for all layers)')

    args = parser.parse_args()

    assert (args.train_output is not None) or (
        args.config_checkpoint is not None and args.model_checkpoint is not None
    ), 'Provide --train_output OR both --config_checkpoint and --model_checkpoint'

    run(args)
