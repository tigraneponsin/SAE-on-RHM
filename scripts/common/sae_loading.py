"""Shared loader helpers for SAE evaluation / analysis scripts.

This module centralizes the three operations every SAE post-hoc script needs:

  1. resolve_rules(blob):      pick the RHM rules used to train the transformer
  2. load_transformer(...):    reload a trained transformer + its eval loader
  3. load_sae(...):            reload a trained SAE + its training setup

These were originally private helpers inside scripts/sae_sweep/eval_sweep.py
and have been extracted verbatim (modulo renaming to drop the underscore) so
that both eval_sweep.py and the new scripts/sae_direct_analysis/analyze_sae.py
can reuse them without drifting. Behavior is unchanged.

Critical invariant: the rules returned here MUST be the exact same rules the
transformer and SAE were trained on. resolve_rules() prefers the rules object
saved in the transformer checkpoint (blob['output']['rules']) and only falls
back to regenerating from cfg.seed_rules if that field is missing. The SAE
checkpoint separately records which rules_source it was trained with so the
caller can detect mismatches.
"""

import copy
from pathlib import Path

import torch

# Callers are expected to have REPO_ROOT on sys.path so that `init`, `models`,
# and `datasets` resolve to the top-level packages.
import init
import models
from datasets.random_hierarchy_model import sample_rules, sample_trees


def resolve_rules(blob: dict, source_label: str):
    """Return the RHM rules to use for data generation, matching what was used
    during transformer training and SAE training.

    Priority:
      1. blob['output']['rules'] if present and not None  ->  exact same rules
         object that was saved alongside the trained transformer weights.
      2. Regenerate deterministically from config.seed_rules and the RHM
         structural parameters. This produces identical rules to case 1 as
         long as the config has not changed, because both paths call
         sample_rules(v, n, m, s, L, seed=seed_rules).

    Either way the returned rules are the same ones used in transformer
    training AND SAE training (train_sae.py follows the same priority).

    Returns:
        (rules, source) where source is 'artifact' or 'seed_rules_resampled'.
    """
    rules = blob.get('output', {}).get('rules', None)
    cfg = blob.get('config', None)

    if rules is not None:
        return rules, 'artifact'

    if cfg is None:
        raise ValueError(
            f'{source_label}: transformer artifact has no saved rules and no config '
            'to regenerate them from.'
        )
    missing = [a for a in ('num_features', 'num_classes', 'num_synonyms',
                           'tuple_size', 'num_layers', 'seed_rules')
               if not hasattr(cfg, a)]
    if missing:
        raise ValueError(
            f'{source_label}: cannot regenerate rules - config is missing: {missing}'
        )
    print(
        f'  WARNING: transformer artifact has no saved rules. '
        f'Regenerating from config.seed_rules={cfg.seed_rules}. '
        f'Eval data uses the same RHM function as training only if seed_rules '
        f'and structural parameters have not changed.'
    )
    rules = sample_rules(
        cfg.num_features, cfg.num_classes, cfg.num_synonyms,
        cfg.tuple_size, cfg.num_layers, seed=cfg.seed_rules,
    )
    return rules, 'seed_rules_resampled'


def load_transformer(train_output_path: str, eval_size: int, eval_seed: int,
                     batch_size: int, device: str, shuffle: bool = True,
                     model_variant: str = 'last'):
    """Load a trained transformer and build an eval dataloader on fresh RHM data.

    The dataloader is built via init.init_data(), which by default creates a
    shuffled train_loader. For analysis scripts that need to line up batch rows
    with ground truth latents, pass shuffle=False.

    model_variant selects which weights to load:
      - 'last': blob['output']['model']         (default; back-compat for old SAEs)
      - 'best': blob['output']['best']['model'] (lowest test-loss checkpoint)

    Returns: (model, loader, cfg, rules, rules_source)
    """
    if model_variant not in ('best', 'last'):
        raise ValueError(f"model_variant must be 'best' or 'last', got {model_variant!r}")

    blob = torch.load(train_output_path, map_location='cpu', weights_only=False)
    if not isinstance(blob, dict) or 'config' not in blob or 'output' not in blob:
        raise ValueError(f'Invalid train_output format: {train_output_path}')
    output = blob['output']

    if model_variant == 'best':
        best = output.get('best')
        if not isinstance(best, dict) or 'model' not in best:
            raise RuntimeError(
                f"load_transformer: model_variant='best' requested but "
                f"{train_output_path!r} has no output['best']['model']. "
                f"This transformer artifact predates best-weights tracking; "
                f"re-train it, or load an SAE that was trained on 'last' weights."
            )
        state = best['model']
    else:
        if 'model' not in output:
            raise RuntimeError(
                f"load_transformer: model_variant='last' requested but "
                f"{train_output_path!r} has no output['model']."
            )
        state = output['model']

    cfg = copy.deepcopy(blob['config'])
    rules, rules_source = resolve_rules(blob, train_output_path)

    trees = sample_trees(num_data=eval_size, rules=rules, prior=None, probs=None, seed=eval_seed)

    data_cfg = copy.deepcopy(cfg)
    data_cfg.train_size = eval_size
    data_cfg.test_size = 0
    data_cfg.batch_size = max(1, min(batch_size, eval_size))
    loader, _ = init.init_data(trees[cfg.num_layers], trees[0], data_cfg)

    if not shuffle:
        # init_data always builds train_loader with shuffle=True. Rebuild with
        # shuffle=False so callers that need ordered rows (e.g. analyze_sae.py)
        # can zip the batches back to trees[level] by position.
        dataset = loader.dataset
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=data_cfg.batch_size, shuffle=False, num_workers=0
        )

    model = init.init_model(cfg)
    model.load_state_dict(state)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, loader, cfg, rules, rules_source


def load_sae(ckpt_path: str, input_dim, device: str, load_model: bool = True):
    """Parse an SAE checkpoint and (optionally) rebuild the SparseAutoencoder.

    Returns a dict with all the fields eval scripts need, or None if the file
    is not a valid single-layer SAE checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'sae_state' not in ckpt or 'sae_layers' not in ckpt:
        return None  # not a valid SAE checkpoint

    layers = [int(x) for x in ckpt['sae_layers']]
    if len(layers) != 1:
        # multi-layer checkpoints are not produced by the sweep (one per job)
        return None

    layer_id = layers[0]
    state = ckpt['sae_state'].get(layer_id) or ckpt['sae_state'].get(str(layer_id))
    metrics = (ckpt.get('sae_metrics', {}).get(layer_id)
               or ckpt.get('sae_metrics', {}).get(str(layer_id)) or {})
    curves = (ckpt.get('sae_training_curves', {}).get(layer_id)
              or ckpt.get('sae_training_curves', {}).get(str(layer_id)) or {})
    if state is None:
        return None

    latent_dim = int(metrics.get('latent_dim') or state['encoder.weight'].shape[0])
    sae = None
    if load_model:
        if input_dim is None or input_dim <= 0:
            raise ValueError('input_dim must be > 0 when load_model=True')
        sae = models.SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        sae.load_state_dict(state)
        sae = sae.to(device).eval()
        for p in sae.parameters():
            p.requires_grad = False

    setup = ckpt.get('sae_training_setup', {})
    cfg_stored = ckpt.get('config', None)
    source = ckpt.get('source', {})
    dataset_split = ckpt.get('sae_dataset_split', {})

    # act_scale stored as {str(layer_id): float} - default 1.0 for old checkpoints
    act_scale_dict = setup.get('act_scale', {})
    act_scale = float(
        act_scale_dict.get(layer_id)
        or act_scale_dict.get(str(layer_id))
        or 1.0
    )

    return {
        'layer_id': layer_id,
        'latent_dim': latent_dim,
        'sae': sae,
        'train_metrics': metrics,
        'curves': curves,
        'setup': setup,
        'config': cfg_stored,
        'train_output': source.get('train_output', ''),
        'ckpt_path': ckpt_path,
        # rules_source recorded at SAE training time: 'artifact' or 'seed_rules_resampled'
        'sae_rules_source': dataset_split.get('rules_source', None),
        'sae_token_idx': int(setup.get('sae_token_idx', 0)),
        'act_scale': act_scale,
        # which transformer weights this SAE was trained against. Old SAE artifacts
        # predate this field and were always trained on 'last' weights.
        'model_variant': source.get('model_variant', 'last'),
    }
