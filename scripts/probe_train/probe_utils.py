"""Shared utilities for probe training and evaluation scripts."""

import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import init
from datasets.random_hierarchy_model import sample_rules


def resolve_rules(blob: dict, source_label: str):
    """Return RHM rules from a transformer checkpoint blob.

    Uses saved rules if present, otherwise regenerates from config seed.
    """
    rules = blob.get('output', {}).get('rules', None)
    if rules is not None:
        return rules
    cfg = blob.get('config', None)
    if cfg is None:
        raise ValueError(f'{source_label}: no saved rules and no config')
    return sample_rules(
        cfg.num_features, cfg.num_classes, cfg.num_synonyms,
        cfg.tuple_size, cfg.num_layers, seed=cfg.seed_rules,
    )


def align_model_state_dict_keys(model, state_dict):
    """Align checkpoint key prefix style with the instantiated model.

    Handles the _orig_mod. prefix added by torch.compile.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    src_keys = list(state_dict.keys())
    dst_keys = list(model.state_dict().keys())
    if not src_keys or not dst_keys:
        return state_dict

    src_pref = all(k.startswith('_orig_mod.') for k in src_keys)
    dst_pref = all(k.startswith('_orig_mod.') for k in dst_keys)

    if src_pref and not dst_pref:
        return {k[len('_orig_mod.'):]: v for k, v in state_dict.items()}
    if dst_pref and not src_pref:
        return {f'_orig_mod.{k}': v for k, v in state_dict.items()}
    return state_dict


def prepare_inputs(trees, cfg):
    """Transform leaf tokens into model-ready inputs."""
    num_rhm_levels = cfg.num_layers
    data_cfg = copy.deepcopy(cfg)
    data_cfg.train_size = trees[num_rhm_levels].size(0)
    data_cfg.test_size = 0
    return init.transform_inputs(trees[num_rhm_levels], data_cfg)


def load_transformer(train_output: str, device_str: str):
    """Load a frozen transformer from a checkpoint path.

    Returns (model, cfg, rules).
    """
    blob = torch.load(train_output, map_location='cpu', weights_only=False)
    cfg = copy.deepcopy(blob['config'])
    cfg.device = device_str
    rules = resolve_rules(blob, train_output)

    model = init.init_model(cfg)
    model_state = align_model_state_dict_keys(model, blob['output']['model'])
    model.load_state_dict(model_state)
    device = torch.device(device_str)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, cfg, rules
