import os
import sys
import time
import copy
sys.path.append('~/rhm-training')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import math
import random

import functools
import argparse

import datasets, models
import init, measures


def parse_sae_layers(sae_layers, num_layers):
    if sae_layers is None or sae_layers.lower() == 'all':
        return list(range(num_layers))

    layers = [int(layer.strip()) for layer in sae_layers.split(',') if layer.strip()]
    assert len(layers) > 0, 'sae_layers must contain at least one valid layer index'
    for layer in layers:
        assert 0 <= layer < num_layers, f'sae layer {layer} is out of range [0, {num_layers-1}]'
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
            latent_dim=latent_dim
        ).to(config.device)
        optimizer = optim.AdamW(sae.parameters(), lr=config.sae_lr, weight_decay=0.)

        activation_buffer = []

        def save_activation(_module, _inputs, output):
            activation_buffer.append(output.detach())

        hook = model.blocks[layer_id].register_forward_hook(save_activation)

        step = 0
        running_total = 0.
        running_recon = 0.
        running_sparse = 0.
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
                        f'total={running_total/step:.6f} recon={running_recon/step:.6f} sparse={running_sparse/step:.6f}'
                    )

                if step >= config.sae_steps:
                    break

        hook.remove()

        stats = sae.activation_stats(last_z) if last_z is not None else {'active_fraction': 0., 'dead_features': latent_dim}
        sae_metrics[layer_id] = {
            'total_loss': running_total / max(step, 1),
            'recon_loss': running_recon / max(step, 1),
            'sparse_loss': running_sparse / max(step, 1),
            'active_fraction': stats['active_fraction'],
            'dead_features': stats['dead_features'],
            'steps': step,
            'latent_dim': latent_dim
        }
        sae_state[layer_id] = copy.deepcopy(sae.state_dict())

    return sae_state, sae_metrics

def run( config):

    # reduce batch_size when larger than train_size
    if (config.batch_size >= config.train_size):
        config.batch_size = config.train_size
    assert (config.train_size%config.batch_size)==0, 'batch_size must divide train_size!'
    config.num_batches = config.train_size//config.batch_size
    config.max_iters = config.max_epochs*config.num_batches

    config.num_data = config.num_classes*config.num_synonyms**((config.tuple_size**config.num_layers-1)//(config.tuple_size-1))
    config.input_size = config.tuple_size**config.num_layers
    print(f'{config.train_size} training data, split into {config.num_batches} batches (total {config.num_data})')
    print(f"Training for {config.max_iters} steps")

    # Initialise RHM dataset
    rhm = datasets.RHM(
        v=config.num_features,
        n=config.num_classes,
        m=config.num_synonyms,
        s=config.tuple_size,
        L=config.num_layers,
        seed_rules=config.seed_rules,
        seed_samples=config.seed_sample,
        num_data=config.train_size+config.test_size,
        probs=None,
        transform=None
    )
    inputs = rhm.trees[config.num_layers]
    targets = rhm.trees[0]
    train_loader, test_loader = init.init_data(inputs, targets, config)

    model = init.init_model(config)
    model0 = copy.deepcopy( model)
    param_count = sum([p.numel() for p in model.parameters()])
    print(f'Training {config.model}, depth {config.depth}, width {config.width}, {param_count} params.')

    criterion, optimizer, scheduler = init.init_training( model, config)

    print_ckpts, save_ckpts = init.init_loglinckpt( config.print_freq, config.max_iters, freq=config.save_freq)
    print_ckpt = next(print_ckpts)
    save_ckpt = next(save_ckpts)

    step = 0
    dynamics, best = init.init_output(model, criterion, train_loader, test_loader, config)
    if config.checkpoints:
        torch.save(
            {'config': config, 'rules': rhm.rules},
            f"{config.outname}_config.pt"
        )
        output = {
            'model': copy.deepcopy(model.state_dict()),
            'state': dynamics[-1],
            'step': step
        }
        torch.save(
            output,
            f"{config.outname}_t{step}.pt"
        )

    for epoch in range(config.max_epochs):

        model.train()
        optimizer.zero_grad()
        running_loss = 0.

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            outputs = model(inputs.to(config.device))
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.to(config.device).view(-1))
            running_loss += loss.item()
            loss /= config.accumulation
            loss.backward()

            if ((batch_idx+1)%config.accumulation==0):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if step==print_ckpt:

                    test_loss, test_acc = measures.test(model, criterion, test_loader, config.device)

                    if test_loss<best['loss']: # update best model if loss is smaller
                        best['step'] = step
                        best['loss'] = test_loss
                        best['model'] = copy.deepcopy( model.state_dict())

                    print('step : ',step, '\t running loss: {:06.4f}'.format(running_loss/(batch_idx+1)), ', test loss: {:06.4f}'.format(test_loss))
                    print_ckpt = next(print_ckpts)

                    if step>=save_ckpt:

                        print(f'Checkpoint at step {step}, saving data ...')
                        save_dict = {'t': step, 'testloss': test_loss, 'testacc': test_acc}
                        if config.measure_train:
                            train_loss, train_acc = measures.test(model, criterion, train_loader, config.device)
                            save_dict['trainloss'] = train_loss
                            save_dict['trainacc'] = train_acc
                        dynamics.append(save_dict)

                        if config.checkpoints:
                            output = {
                                'model': copy.deepcopy(model.state_dict()),
                                'state': dynamics[-1],
                                'step': step
                            }
                            torch.save(
                                output,
                                f"{config.outname}_t{step}.pt"
                            )
                        else:
                            output = {
                                'rules': rhm.rules,
                                'init': model0.state_dict(),
                                'best': best,
                                'model': copy.deepcopy(model.state_dict()),
                                'dynamics': dynamics,
                                'step': step
                            }
                            torch.save(
                                {'config': config, 'output': output},
                                f"{config.outname}.pt"
                            )
                        save_ckpt = next(save_ckpts)


        if (running_loss/(batch_idx+1)) <= config.loss_threshold:

            save_dict = {'t': step, 'testloss': test_loss, 'testacc': test_acc}
            if config.measure_train:
                train_loss, train_acc = measures.test(model, criterion, train_loader, config.device)
                save_dict['trainloss'] = train_loss
                save_dict['trainacc'] = train_acc
            dynamics.append(save_dict)

            if config.checkpoints:
                output = {
                    'model': copy.deepcopy(model.state_dict()),
                    'state': dynamics[-1],
                    'step': step
                }
                torch.save(
                    output,
                    f"{config.outname}_t{step}.pt"
                )
            else:
                output = {
                    'rules': rhm.rules,
                    'init': model0.state_dict(),
                    'best': best,
                    'model': copy.deepcopy(model.state_dict()),
                    'dynamics': dynamics,
                    'step': step
                }
                torch.save(
                    {'config': config, 'output': output},
                    f"{config.outname}.pt"
                )

            break

    if config.sae_enable:
        sae_state, sae_metrics = train_sae_posthoc(model, train_loader, config)
        torch.save(
            {
                'config': config,
                'sae_layers': parse_sae_layers(config.sae_layers, model.num_layers),
                'sae_state': sae_state,
                'sae_metrics': sae_metrics,
            },
            f"{config.outname}_sae.pt"
        )
        print(f"Saved SAE checkpoints to {config.outname}_sae.pt")

    return None

torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='Learning the Random Hierarchy Model with deep neural networks')
parser.add_argument("--device", type=str, default='cuda')
'''
	DATASET ARGS
'''
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--num_features', metavar='v', type=int, help='number of features')
parser.add_argument('--num_classes', metavar='n', type=int, help='number of classes')
parser.add_argument('--num_synonyms', metavar='m', type=int, help='multiplicity of low-level representations')
parser.add_argument('--tuple_size', metavar='s', type=int, help='size of low-level representations')
parser.add_argument('--num_layers', metavar='L', type=int, help='number of layers')
parser.add_argument('--seed_rules', type=int, help='seed for the dataset')
parser.add_argument('--num_tokens', type=int, help='number of input tokens (spatial size)')
parser.add_argument('--train_size', metavar='Ptr', type=int, help='training set size')
parser.add_argument('--batch_size', metavar='B', type=int, help='batch size')
parser.add_argument('--test_size', metavar='Pte', type=int, help='test set size')
parser.add_argument('--seed_sample', type=int, help='seed for the sampling of train and testset')
parser.add_argument('--input_format', type=str, default='onehot')
parser.add_argument('--whitening', type=int, default=0)
'''
	ARCHITECTURE ARGS
'''
parser.add_argument('--model', type=str, help='architecture (fcn, hcnn, hlcn, transformer_mla, transformer_clm, transformer_class)')
parser.add_argument('--depth', type=int, help='depth of the network')
parser.add_argument('--width', type=int, help='width of the network')
parser.add_argument('--filter_size', type=int, default=None, help='filter size (CNN, LCN only)')
parser.add_argument('--bias', default=False, action='store_true')
parser.add_argument('--embedding_dim', type=int, default=None, help='embedding dimension (transformers only)')
parser.add_argument('--num_heads', type=int, default=None, help='number of heads (transformers only)')
parser.add_argument('--ffwd_size', type=int, default=None, help='MLP width scaling (transformer only)')
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--seed_model', type=int, help='seed for model initialization')
parser.add_argument('--sae_enable', default=False, action='store_true')
parser.add_argument('--sae_layers', type=str, default='all', help='comma-separated layer ids or all')
parser.add_argument('--sae_latent_dim', type=int, default=None, help='latent width of SAE (default: 4 * embedding_dim)')
parser.add_argument('--sae_lambda_l1', type=float, default=1e-3, help='L1 sparsity coefficient for SAE latent activations')
parser.add_argument('--sae_lr', type=float, default=1e-3, help='learning rate for SAE optimizer')
parser.add_argument('--sae_steps', type=int, default=512, help='number of optimization steps for each SAE')
parser.add_argument('--sae_batch_limit', type=int, default=0, help='max tokens per SAE step (0 uses all tokens in batch)')
parser.add_argument('--sae_print_freq', type=int, default=128, help='print frequency during SAE training')
'''
       TRAINING ARGS
'''
parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--scheduler', type=str, default=None, help='options are cosine, cosine-warmup')
parser.add_argument('--warmup_time', type=int, default=None, help='required by cosine-warmup')
parser.add_argument('--decay_time', type=int, default=None, help='required by cosine, cosine-warmup')
parser.add_argument('--accumulation', type=int, default=1)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--max_epochs', type=int, default=1)
'''
	OUTPUT ARGS
'''
parser.add_argument('--print_freq', type=int, help='frequency of prints', default=16)
parser.add_argument('--save_freq', type=int, help='frequency of saves', default=2)
parser.add_argument('--measure_train', default=False, action='store_true')
parser.add_argument('--checkpoints', default=False, action='store_true')
parser.add_argument('--loss_threshold', type=float, default=1e-3)
parser.add_argument('--outname', type=str, required=True, help='path of the output file')

config = parser.parse_args()
run( config)