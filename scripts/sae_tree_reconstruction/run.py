"""Tree reconstruction from SAE features.

Given one .sae_eval.pt artifact per transformer layer (covering layers
0..L-1 of the same trained transformer), reconstruct the full RHM
latent tree for each eval input by aggregating per-feature evidence
P(Z_{l,j} = z | f_i > 0).

Recipe (see docs/project_pipelines.md section 6):

  cond_prob[i][z] = joint_fire_count / firing_count       (from artifact)
  alive(i, p)     = firing_rate[p, i] > 0                  (from artifact)

  For each input x:
    1) one transformer forward, hooked at every layer k
    2) per-layer SAE encode + decoder weighting:
         f_act[b, p, i] = relu(W_enc selected[b, p] + b_enc) * ||W_dec[:, i]||
    3) for each (l, j):
         k = L - 1 - l
         P(l, j) = { p : p // s^(1+k) == j }
         A_p     = { i : f_act[b, p, i] > 0 and alive(i, p) }
         score_p[b, z] = sum_{i in A_p} w_i * cond_prob[i][z]
                          / sum_{i in A_p} w_i
           where w_i = f_act if --weighting activation else 1
         score[l, j][b, z] = mean over non-empty p of score_p
         Z_hat[l, j][b]    = argmax_z score[l, j][b, z]    (-1 if no p)

Both weightings (activation, uniform) are produced from the same forward
pass and saved as separate output files.

Usage:
    python scripts/sae_tree_reconstruction/run.py \\
        --artifacts <a0.sae_eval.pt> <a1.sae_eval.pt> ... \\
        --out_dir <dir> \\
        [--batch_size B] [--device cuda] [--sanity]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.random_hierarchy_model import sample_trees
from scripts.common.sae_loading import load_sae, load_transformer
from scripts.sae_eval.streaming import dedupe_trees, select_activation_tokens, _hook_module


# ---------------------------------------------------------------------------
# Artifact loading and validation
# ---------------------------------------------------------------------------

def _load_artifact(path: Path) -> dict:
    return torch.load(str(path), map_location='cpu', weights_only=False)


_REQUIRED_KEYS = (
    'ckpt_path', 'layer_id', 'mode', 'sae_token_idx', 'act_scale',
    'rhm', 'eval_size', 'eval_seed', 'dedupe',
    'joint_fire_count', 'firing_count', 'firing_rate',
    'targets', 'index_layout', 'token_positions', 'latent_dim',
)


def _validate_artifact(art: dict, path: Path) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in art]
    if missing:
        raise ValueError(
            f'{path.name}: missing required keys {missing}. '
            f'Re-run scripts/sae_eval/run.py with '
            f'--with-per-feature --with-conditional --with-entropy.'
        )


def _resolve_train_output(ckpt_path: str) -> tuple[str, dict, str]:
    """Read the SAE checkpoint and return (train_output, dataset_split, model_variant).

    dataset_split records the seeds used during transformer + SAE training:
    transformer_seed_sample, train_seed_sample, eval_seed_sample. We carry
    these forward so the tree-reconstruction script can refuse to evaluate
    on data the model has already seen.

    model_variant records which transformer weights this SAE was trained on
    ('best' or 'last'). Old SAE artifacts predate this field and default to
    'last' to preserve their training behavior.
    """
    blob = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    source = blob.get('source', {}) or {}
    src = source.get('train_output', '')
    if not src:
        raise ValueError(
            f'{ckpt_path}: no source.train_output recorded in SAE checkpoint.'
        )
    split = blob.get('sae_dataset_split', {}) or {}
    model_variant = source.get('model_variant', 'last')
    return src, split, model_variant


def _group_artifacts(paths: list[Path]) -> tuple[dict, dict]:
    """Load every artifact, validate, and return (by_layer, common_meta).

    by_layer:    {layer_id -> artifact dict}
    common_meta: dict with shared rhm / eval_size / eval_seed / dedupe / mode
                 / sae_token_idx / train_output, all asserted equal.
    """
    by_layer: dict = {}
    common: dict = {}
    train_outputs: dict = {}
    dataset_splits: dict = {}
    model_variants: dict = {}
    for p in paths:
        art = _load_artifact(p)
        _validate_artifact(art, p)
        layer_id = int(art['layer_id'])
        if layer_id in by_layer:
            raise ValueError(
                f'Duplicate artifact for layer {layer_id}: '
                f'{by_layer[layer_id]["_path"].name} and {p.name}.'
            )
        art['_path'] = p
        if art['mode'] == 'cls_token':
            raise ValueError(
                f'{p.name}: cls_token mode is out of scope for tree '
                f'reconstruction.'
            )
        src, split, mv = _resolve_train_output(art['ckpt_path'])
        train_outputs[layer_id] = src
        dataset_splits[layer_id] = split
        model_variants[layer_id] = mv
        by_layer[layer_id] = art

    if not by_layer:
        raise ValueError('No artifacts provided.')

    # Equality checks across artifacts.
    ref = next(iter(by_layer.values()))
    common['rhm'] = dict(ref['rhm'])
    common['eval_size'] = int(ref['eval_size'])
    common['eval_seed'] = int(ref['eval_seed'])
    common['dedupe'] = bool(ref['dedupe'])
    common['mode'] = str(ref['mode'])
    common['sae_token_idx'] = int(ref['sae_token_idx'])
    common['train_output'] = train_outputs[int(ref['layer_id'])]
    common['model_variant'] = model_variants[int(ref['layer_id'])]

    for layer_id, art in by_layer.items():
        if dict(art['rhm']) != common['rhm']:
            raise ValueError(f'rhm mismatch at layer {layer_id}.')
        if int(art['eval_size']) != common['eval_size']:
            raise ValueError(f'eval_size mismatch at layer {layer_id}.')
        if int(art['eval_seed']) != common['eval_seed']:
            raise ValueError(f'eval_seed mismatch at layer {layer_id}.')
        if bool(art['dedupe']) != common['dedupe']:
            raise ValueError(f'dedupe mismatch at layer {layer_id}.')
        if str(art['mode']) != common['mode']:
            raise ValueError(
                f'mode mismatch at layer {layer_id}: '
                f'{art["mode"]} vs {common["mode"]}.'
            )
        if int(art['sae_token_idx']) != common['sae_token_idx']:
            raise ValueError(f'sae_token_idx mismatch at layer {layer_id}.')
        if train_outputs[layer_id] != common['train_output']:
            raise ValueError(
                f'transformer source mismatch at layer {layer_id}: '
                f'{train_outputs[layer_id]} vs {common["train_output"]}.'
            )
        if model_variants[layer_id] != common['model_variant']:
            raise ValueError(
                f'model_variant mismatch at layer {layer_id}: '
                f'{model_variants[layer_id]} vs {common["model_variant"]}. '
                f'All SAEs in a tree-reconstruction run must have been trained '
                f'against the same transformer weights.'
            )

    L = int(common['rhm']['L'])
    expected = set(range(L))
    found = set(by_layer.keys())
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(
            f'Layer coverage mismatch. Need exactly {{0..{L - 1}}}, '
            f'missing={missing}, unexpected={extra}.'
        )

    # Collect all seeds that the transformer / SAEs have already seen, plus the
    # eval seed used to estimate cond_prob. The reconstruction generalization
    # split must use a seed disjoint from this set.
    forbidden: set = {int(common['eval_seed'])}
    forbidden_detail: dict = {'artifact_eval_seed': int(common['eval_seed'])}
    for layer_id, split in dataset_splits.items():
        for key in ('transformer_seed_sample', 'train_seed_sample',
                    'eval_seed_sample'):
            if key in split and split[key] is not None:
                forbidden.add(int(split[key]))
                forbidden_detail.setdefault(key, set()).add(int(split[key]))
    common['forbidden_seeds'] = forbidden
    common['forbidden_seeds_detail'] = forbidden_detail
    common['dataset_splits'] = dataset_splits

    return by_layer, common


# ---------------------------------------------------------------------------
# Conditional probability recovery
# ---------------------------------------------------------------------------

def _build_cond_prob(art: dict) -> dict:
    """Return per-(level, j) conditional P(Z = z | f_i > 0).

    Output: dict (level, j) -> {'values': LongTensor [V_g],
                                'cond_prob': Float64Tensor [V_g, P, F]}
    Dead features (firing_count == 0) yield NaN columns. Live features
    with all-zero joint counts at this group yield zero probability
    (correct: never co-fired with any value in this group).
    """
    joint = art['joint_fire_count'].long()       # [T_total, P, F]
    firing = art['firing_count'].long()          # [P, F]
    index_layout = art['index_layout']

    fmarg = firing.double().clamp_min(1.0)
    dead_mask = (firing == 0)                    # [P, F]

    out: dict = {}
    for group in index_layout:
        lvl = int(group['level'])
        pos = int(group['position'])
        start = int(group['start'])
        end = int(group['end'])
        values = group['values'].clone().long()  # [V_g]
        joint_sub = joint[start:end].double()    # [V_g, P, F]
        cp = joint_sub / fmarg.unsqueeze(0)      # [V_g, P, F]
        if dead_mask.any():
            cp = cp.masked_fill(dead_mask.unsqueeze(0), float('nan'))
        out[(lvl, pos)] = {'values': values, 'cond_prob': cp}
    return out


def _full_cond_prob_at(group: dict, V_full: int) -> torch.Tensor:
    """Expand a (V_g, P, F) cond_prob to (V_full, P, F) by zero-filling
    unobserved values (their conditional was never observed -> 0).
    NaN is preserved at dead-feature columns."""
    cp = group['cond_prob']                       # [V_g, P, F]
    V_g, P, F = cp.shape
    if V_g == V_full:
        return cp
    out = torch.zeros(V_full, P, F, dtype=cp.dtype)
    nan_cols = torch.isnan(cp[0]) if V_g > 0 else torch.zeros(P, F, dtype=torch.bool)
    out[group['values']] = cp
    if V_g > 0 and nan_cols.any():
        out[:, nan_cols] = float('nan')
    return out


def _num_values_for_level(level: int, rhm: dict) -> int:
    if level == 0:
        return int(rhm['n'])
    return int(rhm['v'])


# ---------------------------------------------------------------------------
# Per-layer state assembly
# ---------------------------------------------------------------------------

class _LayerState:
    """Everything we need at inference time for one transformer layer."""

    def __init__(self, art: dict, sae, dec_norms: torch.Tensor, device: str):
        self.layer_id = int(art['layer_id'])
        self.mode = str(art['mode'])
        self.token_idx = int(art['sae_token_idx'])
        self.act_scale = float(art['act_scale'])
        self.latent_dim = int(art['latent_dim'])
        self.token_positions = art['token_positions'].long().clone()  # [P]
        self.sae = sae
        self.dec_norms = dec_norms                                    # [F]
        # Alive mask: firing_rate > 0, shape [P, F].
        self.alive = (art['firing_rate'] > 0).to(device)              # [P, F] bool


# ---------------------------------------------------------------------------
# Aggregation per (l, j)
# ---------------------------------------------------------------------------

def _canonical_layer(level: int, L: int) -> int:
    return L - 1 - int(level)


def _canonical_positions(level: int, j: int, layer_id: int, s: int,
                         L: int) -> list[int]:
    """All real-token positions p with p // s^(1+layer_id) == j.

    For level 0 (root) and the canonical layer L-1, s^(1+(L-1)) == s^L =
    total number of leaves, so any p maps to j=0.
    """
    span = s ** (1 + layer_id)            # number of leaves under each j
    start = j * span
    end = min((j + 1) * span, s ** L)
    return list(range(start, end))


# ---------------------------------------------------------------------------
# Main streaming reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def stream_reconstruction(
    model,
    layers: dict,                # {layer_id -> _LayerState}
    cond_full: dict,             # {(level, j) -> Float64Tensor [V_full, P_layer, F_layer]}
    trees: dict,
    rhm: dict,
    has_cls: bool,
    batch_size: int,
    device: str,
    classifier_head: bool,
):
    """Run the eval forward pass and produce per-input reconstructions.

    Returns:
        results: dict with
            'Z_hat_act', 'Z_hat_uni':       {(l, j) -> LongTensor [N]} (-1 = no contrib)
            'score_act', 'score_uni':       {(l, j) -> Float64Tensor [N, V_full]}
            'coverage':                     {(l, j) -> BoolTensor [N]}
            'Z_true':                       {(l, j) -> LongTensor [N]}
            'head_correct':                 BoolTensor [N] or None
            'pos_argmax_act', 'pos_argmax_uni':
                                            {(l, j) -> LongTensor [N, |P(l,j)|]}, -1 if empty
    """
    L = int(rhm['L'])
    s = int(rhm['s'])
    n_inputs = trees[L].size(0)

    # Ground-truth latents.
    Z_true: dict = {}
    for level in range(L):
        level_tensor = trees[level]
        if level_tensor.ndim == 1:
            Z_true[(level, 0)] = level_tensor.long().clone()
        else:
            width = level_tensor.size(1)
            for j in range(width):
                Z_true[(level, j)] = level_tensor[:, j].long().clone()

    # Pre-compute, per (l, j): canonical layer k, list of canonical p's, and
    # for each canonical p the SAE-position index inside layers[k].token_positions.
    plan: dict = {}                                    # (l, j) -> dict
    for level in range(L):
        # number of (level, j) groups
        if level == 0:
            j_range = [0]
        else:
            j_range = list(range(s ** level))
        for j in j_range:
            k = _canonical_layer(level, L)
            ps = _canonical_positions(level, j, k, s, L)
            tok_pos = layers[k].token_positions.tolist()
            pos_to_idx = {int(p): idx for idx, p in enumerate(tok_pos)}
            sae_indices = [pos_to_idx[p] for p in ps if p in pos_to_idx]
            plan[(level, j)] = {
                'k': k,
                'real_positions': ps,
                'sae_pos_idx': sae_indices,           # list of indices into [P_k]
            }

    # Allocate output buffers (CPU, lazily for score tensors once V_full known).
    V_full_for: dict = {(l, j): _num_values_for_level(l, rhm) for (l, j) in plan}
    score_act: dict = {(l, j): torch.zeros(n_inputs, V_full_for[(l, j)],
                                           dtype=torch.float64)
                       for (l, j) in plan}
    score_uni: dict = {(l, j): torch.zeros(n_inputs, V_full_for[(l, j)],
                                           dtype=torch.float64)
                       for (l, j) in plan}
    coverage: dict = {(l, j): torch.zeros(n_inputs, dtype=torch.bool)
                      for (l, j) in plan}
    pos_argmax_act: dict = {
        (l, j): torch.full((n_inputs, len(plan[(l, j)]['sae_pos_idx'])), -1,
                           dtype=torch.long)
        for (l, j) in plan
    }
    pos_argmax_uni: dict = {
        (l, j): torch.full((n_inputs, len(plan[(l, j)]['sae_pos_idx'])), -1,
                           dtype=torch.long)
        for (l, j) in plan
    }

    head_correct = torch.zeros(n_inputs, dtype=torch.bool) if classifier_head else None

    # Hooks: one per layer.
    bufs: dict = {k: [] for k in layers}
    handles = []
    for k, st in layers.items():
        def _make_hook(buf):
            def _hook(_m, _i, o):
                buf.append(o.detach())
            return _hook
        handles.append(_hook_module(model, k, st.mode).register_forward_hook(_make_hook(bufs[k])))

    # Move cond_full to device once (these are big-ish but we need them per batch).
    cond_dev: dict = {key: t.to(device) for key, t in cond_full.items()}

    inputs = trees[L].long()
    labels_root = trees[0].long()

    try:
        cursor = 0
        n_batches = (n_inputs + batch_size - 1) // batch_size
        for bi in range(n_batches):
            s_idx = bi * batch_size
            e_idx = min((bi + 1) * batch_size, n_inputs)
            batch_inputs = inputs[s_idx:e_idx].to(device)
            B = batch_inputs.size(0)

            for k in bufs:
                bufs[k].clear()
            out_logits = model(batch_inputs)

            # Optional classifier-head accuracy (root-level sanity).
            if classifier_head and out_logits is not None:
                if out_logits.dim() == 2 and out_logits.size(-1) == int(rhm['n']):
                    pred = out_logits.argmax(dim=-1).cpu()
                    head_correct[s_idx:e_idx] = (pred == labels_root[s_idx:e_idx])

            # Per-layer SAE encode + decoder weighting.
            f_acts: dict = {}                  # k -> [B, P_k, F_k] float32 on device
            for k, st in layers.items():
                if not bufs[k]:
                    raise RuntimeError(f'No activation captured at layer {k}')
                act = bufs[k].pop(0)
                selected, _ = select_activation_tokens(
                    act, mode=st.mode, has_cls=has_cls, token_idx=st.token_idx
                )
                P_k = selected.size(1)
                flat = selected.reshape(B * P_k, -1) * st.act_scale
                _, z = st.sae(flat)            # [B*P_k, F_k], post-ReLU
                f_act = (z * st.dec_norms.unsqueeze(0)).view(B, P_k, st.latent_dim)
                f_acts[k] = f_act              # device, float32

            # Aggregate per (l, j).
            for (l, j), info in plan.items():
                k = info['k']
                idxs = info['sae_pos_idx']
                if not idxs:
                    continue
                f_act_k = f_acts[k]            # [B, P_k, F_k]
                alive_k = layers[k].alive      # [P_k, F_k]
                cp = cond_dev[(l, j)]          # [V_full, P_k, F_k]
                V_full = cp.size(0)

                # Accumulators per (l, j) per-batch.
                num_act = torch.zeros(B, V_full, dtype=torch.float64, device=device)
                den_act = torch.zeros(B, dtype=torch.float64, device=device)
                num_uni = torch.zeros(B, V_full, dtype=torch.float64, device=device)
                den_uni = torch.zeros(B, dtype=torch.float64, device=device)
                covered = torch.zeros(B, dtype=torch.long, device=device)

                for slot, p_idx in enumerate(idxs):
                    f = f_act_k[:, p_idx, :].double()                # [B, F_k]
                    a_p = layers[k].alive[p_idx]                     # [F_k]
                    fire = (f > 0) & a_p.unsqueeze(0)                # [B, F_k]
                    # weights
                    w_act = torch.where(fire, f, torch.zeros_like(f))     # [B, F_k]
                    w_uni = fire.double()                                  # [B, F_k]
                    # cp at this position: [V_full, F_k] -> handle NaN dead cols.
                    cp_p = cp[:, p_idx, :]                                 # [V_full, F_k]
                    cp_p_clean = torch.where(
                        torch.isnan(cp_p), torch.zeros_like(cp_p), cp_p
                    )
                    # Per-input position score sums.
                    # num[b, z] = sum_i w[b, i] * cp_p[z, i]
                    n_act = w_act @ cp_p_clean.t()                         # [B, V_full]
                    n_uni = w_uni @ cp_p_clean.t()                         # [B, V_full]
                    d_act = w_act.sum(dim=1)                               # [B]
                    d_uni = w_uni.sum(dim=1)                               # [B]
                    nonempty = d_uni > 0                                   # [B]

                    # Per-position normalized score, used for argmax bookkeeping.
                    sp_act = torch.zeros_like(n_act)
                    sp_uni = torch.zeros_like(n_uni)
                    nonempty_act = d_act > 0
                    if nonempty_act.any():
                        sp_act[nonempty_act] = (
                            n_act[nonempty_act] /
                            d_act[nonempty_act].unsqueeze(-1)
                        )
                    if nonempty.any():
                        sp_uni[nonempty] = (
                            n_uni[nonempty] /
                            d_uni[nonempty].unsqueeze(-1)
                        )
                        # Accumulate normalized score (we average across positions).
                        num_act[nonempty] += sp_act[nonempty]
                        den_act[nonempty] += 1.0
                        num_uni[nonempty] += sp_uni[nonempty]
                        den_uni[nonempty] += 1.0
                        covered[nonempty] += 1

                        # Per-position argmax bookkeeping (write into absolute rows).
                        pa_act = sp_act.argmax(dim=-1).cpu()
                        pa_uni = sp_uni.argmax(dim=-1).cpu()
                        ne_cpu = nonempty.cpu()
                        rel_rows = torch.nonzero(ne_cpu, as_tuple=False).flatten()
                        if rel_rows.numel() > 0:
                            abs_rows = rel_rows + s_idx
                            pos_argmax_act[(l, j)][abs_rows, slot] = pa_act[rel_rows]
                            pos_argmax_uni[(l, j)][abs_rows, slot] = pa_uni[rel_rows]

                covered_mask = covered > 0
                if covered_mask.any():
                    rows_dev = torch.nonzero(covered_mask, as_tuple=False).flatten()
                    score_act_lj = num_act[rows_dev] / den_act[rows_dev].unsqueeze(-1)
                    score_uni_lj = num_uni[rows_dev] / den_uni[rows_dev].unsqueeze(-1)
                    abs_rows = (rows_dev + s_idx).cpu()
                    score_act[(l, j)][abs_rows] = score_act_lj.cpu()
                    score_uni[(l, j)][abs_rows] = score_uni_lj.cpu()
                    coverage[(l, j)][abs_rows] = True

            cursor = e_idx
            del f_acts
    finally:
        for h in handles:
            h.remove()

    # Argmax to Z_hat (-1 if not covered).
    Z_hat_act: dict = {}
    Z_hat_uni: dict = {}
    for (l, j) in plan:
        sa = score_act[(l, j)]
        su = score_uni[(l, j)]
        cov = coverage[(l, j)]
        zh_a = torch.full((n_inputs,), -1, dtype=torch.long)
        zh_u = torch.full((n_inputs,), -1, dtype=torch.long)
        if cov.any():
            zh_a[cov] = sa[cov].argmax(dim=-1)
            zh_u[cov] = su[cov].argmax(dim=-1)
        Z_hat_act[(l, j)] = zh_a
        Z_hat_uni[(l, j)] = zh_u

    return {
        'Z_hat_act': Z_hat_act,
        'Z_hat_uni': Z_hat_uni,
        'score_act': score_act,
        'score_uni': score_uni,
        'coverage': coverage,
        'Z_true': Z_true,
        'head_correct': head_correct,
        'pos_argmax_act': pos_argmax_act,
        'pos_argmax_uni': pos_argmax_uni,
        'plan': plan,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics_for_weighting(Z_hat: dict, Z_true: dict, coverage: dict, L: int):
    """Return {'per_level_acc': {l -> float}, 'whole_tree_acc': float,
              'coverage': {l -> float}}."""
    per_level: dict = {}
    cov_per_level: dict = {}
    n_inputs = next(iter(Z_true.values())).size(0)
    all_correct = torch.ones(n_inputs, dtype=torch.bool)
    all_covered = torch.ones(n_inputs, dtype=torch.bool)
    for l in range(L):
        # collect (l, j) pairs
        keys = [k for k in Z_true if k[0] == l]
        correct_count = 0
        cov_count = 0
        denom = 0
        for key in keys:
            cov = coverage[key]
            zh = Z_hat[key]
            zt = Z_true[key]
            cov_count += int(cov.sum().item())
            denom += cov.numel()
            mask = cov
            if mask.any():
                correct_count += int(((zh == zt) & mask).sum().item())
            all_correct &= ((zh == zt) | (~cov))   # treat uncovered as not wrong here
            all_covered &= cov
        per_level[l] = (correct_count / max(cov_count, 1))
        cov_per_level[l] = (cov_count / max(denom, 1))
    whole = float(((all_correct & all_covered).float().mean()).item())
    return {
        'per_level_acc': per_level,
        'whole_tree_acc': whole,
        'coverage': cov_per_level,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--artifacts', type=str, nargs='+', required=True,
                   help='Explicit list of .sae_eval.pt files, one per layer.')
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--sanity', action='store_true',
                   help='Print extra sanity diagnostics (head accuracy, '
                        'per-position consistency).')
    p.add_argument('--eval_seed_recon', type=int, default=None,
                   help='Seed for the generalization eval set used to compute '
                        'per-input f_i(x). Must be disjoint from the four '
                        'training-side seeds (transformer, sae train, sae '
                        'eval, artifact eval). Default: artifact eval_seed + 1.')
    p.add_argument('--eval_size_recon', type=int, default=None,
                   help='Number of fresh inputs to draw for reconstruction. '
                        'Default: same as artifact eval_size.')
    p.add_argument('--allow_seed_overlap', action='store_true',
                   help='[unsafe] Override the disjointness check on '
                        'eval_seed_recon. Useful only for in-distribution '
                        'sanity runs where you intentionally want to score '
                        'on the same data the conditionals were estimated on.')
    return p


def main() -> int:
    args = _build_parser().parse_args()
    paths = [Path(a) for a in args.artifacts]
    for p in paths:
        if not p.is_file():
            print(f'ERROR: not a file: {p}', file=sys.stderr)
            return 2

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    by_layer, common = _group_artifacts(paths)
    rhm = common['rhm']
    L = int(rhm['L'])
    print(f'Layers: {sorted(by_layer.keys())}  rhm={rhm}  '
          f'mode={common["mode"]}  eval_size={common["eval_size"]}  '
          f'eval_seed={common["eval_seed"]}  dedupe={common["dedupe"]}')

    # ---- Decide reconstruction (generalization) split ----
    eval_seed_recon = (args.eval_seed_recon
                       if args.eval_seed_recon is not None
                       else int(common['eval_seed']) + 1)
    eval_size_recon = (args.eval_size_recon
                       if args.eval_size_recon is not None
                       else int(common['eval_size']))
    forbidden = common['forbidden_seeds']
    print(f'Forbidden seeds (already seen by model/SAEs/cond_prob): '
          f'{sorted(forbidden)}')
    print(f'Reconstruction split: eval_size={eval_size_recon}  '
          f'eval_seed={eval_seed_recon}')
    if eval_seed_recon in forbidden:
        msg = (
            f'eval_seed_recon={eval_seed_recon} collides with a seed already '
            f'seen during training or used to estimate cond_prob. '
            f'Forbidden seeds: {sorted(forbidden)}.'
        )
        if args.allow_seed_overlap:
            print(f'WARNING: {msg} Continuing because --allow_seed_overlap '
                  f'is set.')
        else:
            print(f'ERROR: {msg} Pass --eval_seed_recon explicitly with a '
                  f'fresh value, or use --allow_seed_overlap if you '
                  f'intentionally want an in-distribution check.',
                  file=sys.stderr)
            return 2

    # ---- Load transformer + rules ----
    print(f'Loading transformer from: {common["train_output"]} '
          f'(variant={common["model_variant"]})')
    model, _loader, cfg, rules, rules_source = load_transformer(
        common['train_output'], eval_size_recon, eval_seed_recon,
        args.batch_size, device, shuffle=False,
        model_variant=common['model_variant'],
    )
    print(f'  rules source: {rules_source}')
    has_cls = hasattr(model, 'cls_token')

    # ---- Sample fresh trees for the reconstruction split ----
    # cond_prob[i][z] in the artifacts was estimated on the original eval set
    # (eval_size, eval_seed); per-input f_i(x) is computed here on a disjoint
    # split so we measure generalization, not in-distribution fit.
    trees = sample_trees(num_data=eval_size_recon, rules=rules,
                         prior=None, probs=None, seed=eval_seed_recon)
    if common['dedupe']:
        before = trees[L].size(0)
        trees = dedupe_trees(trees)
        after = trees[L].size(0)
        print(f'  Dedupe: {before} -> {after} unique trees')
    print(f'  N inputs: {trees[L].size(0)}')

    # ---- Load per-layer SAEs ----
    layer_states: dict = {}
    for k in sorted(by_layer.keys()):
        art = by_layer[k]
        entry = load_sae(art['ckpt_path'], input_dim=model.embedding_dim,
                         device=device)
        if entry is None:
            print(f'ERROR: could not load SAE checkpoint {art["ckpt_path"]}',
                  file=sys.stderr)
            return 2
        sae = entry['sae']
        dec_norms = sae.decoder_feature_norms().to(device)
        layer_states[k] = _LayerState(art, sae, dec_norms, device)
        print(f'  Layer {k}: latent_dim={art["latent_dim"]}  '
              f'P={art["token_positions"].numel()}  '
              f'act_scale={layer_states[k].act_scale:.4f}')

    # ---- Build per-(l, j) full conditional probability tensors ----
    #
    # For each layer k, expand cond_prob to [V_full_for(level), P_k, F_k] so we
    # can index by (l, j) directly.
    cond_full: dict = {}
    for k in sorted(by_layer.keys()):
        cp_groups = _build_cond_prob(by_layer[k])
        # For tree reconstruction we only need groups whose canonical layer is k,
        # i.e. level = L - 1 - k. But the artifact already only has one group
        # per (level, j) pair. We pick those matching this layer.
        target_level = L - 1 - k
        for (lvl, j), group in cp_groups.items():
            if lvl != target_level:
                continue
            V_full = _num_values_for_level(lvl, rhm)
            cond_full[(lvl, j)] = _full_cond_prob_at(group, V_full)

    # Sanity: every (l, j) in plan must have a cond_prob.
    for level in range(L):
        j_range = [0] if level == 0 else range(int(rhm['s']) ** level)
        for j in j_range:
            if (level, j) not in cond_full:
                raise RuntimeError(
                    f'Missing cond_prob for (level={level}, j={j}). '
                    f'The artifact for layer {L - 1 - level} does not enumerate '
                    f'this (level, position). Check enumerate_targets coverage.'
                )

    classifier_head = hasattr(model, 'classifier')

    # ---- Run reconstruction ----
    print('Running reconstruction forward pass...')
    results = stream_reconstruction(
        model=model, layers=layer_states, cond_full=cond_full,
        trees=trees, rhm=rhm, has_cls=has_cls,
        batch_size=args.batch_size, device=device,
        classifier_head=classifier_head,
    )

    # ---- Metrics ----
    metrics_act = _metrics_for_weighting(
        results['Z_hat_act'], results['Z_true'], results['coverage'], L
    )
    metrics_uni = _metrics_for_weighting(
        results['Z_hat_uni'], results['Z_true'], results['coverage'], L
    )

    print('\nPer-level accuracy (covered only):')
    print(f'  level | activation | uniform | coverage')
    for l in range(L):
        print(f'  {l:5d} | {metrics_act["per_level_acc"][l]:.4f}     | '
              f'{metrics_uni["per_level_acc"][l]:.4f}  | '
              f'{metrics_act["coverage"][l]:.4f}')
    print(f'Whole-tree acc: activation={metrics_act["whole_tree_acc"]:.4f}  '
          f'uniform={metrics_uni["whole_tree_acc"]:.4f}')

    if args.sanity:
        if results['head_correct'] is not None:
            head_acc = float(results['head_correct'].float().mean().item())
            print(f'\n[sanity] transformer classifier head root accuracy: {head_acc:.4f}')
            print(f'[sanity] reconstructor root accuracy:                 '
                  f'activation={metrics_act["per_level_acc"][0]:.4f}  '
                  f'uniform={metrics_uni["per_level_acc"][0]:.4f}')
        # Position consistency: for (l, j) with > 1 canonical positions, fraction of
        # inputs where every per-position argmax agrees.
        print('[sanity] per-position argmax agreement (activation weighting):')
        for (l, j), info in results['plan'].items():
            n_pos = len(info['sae_pos_idx'])
            if n_pos < 2:
                continue
            pa = results['pos_argmax_act'][(l, j)]
            valid = pa >= 0                                  # [N, n_pos]
            row_valid = valid.all(dim=1)
            if row_valid.any():
                first = pa[row_valid][:, :1]
                agree = (pa[row_valid] == first).all(dim=1).float().mean().item()
                print(f'  (l={l}, j={j}, |P|={n_pos}): agree={agree:.4f}  '
                      f'valid_rows={int(row_valid.sum().item())}')

    # ---- Save outputs ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transformer_stem = Path(common['train_output']).stem

    provenance = {
        'artifacts': [str(p) for p in paths],
        'transformer_train_output': common['train_output'],
        'token_mode': common['mode'],
        'sae_token_idx': common['sae_token_idx'],
        'cond_prob_eval_size': common['eval_size'],
        'cond_prob_eval_seed': common['eval_seed'],
        'reconstruction_eval_size': eval_size_recon,
        'reconstruction_eval_seed': eval_seed_recon,
        'allow_seed_overlap': bool(args.allow_seed_overlap),
        'forbidden_seeds': sorted(forbidden),
        'forbidden_seeds_detail': {
            k: (sorted(v) if isinstance(v, set) else v)
            for k, v in common['forbidden_seeds_detail'].items()
        },
        'dedupe': common['dedupe'],
        'rhm': rhm,
        'rules_source': rules_source,
    }

    def _save(weighting: str, Z_hat: dict, score: dict, metrics: dict):
        path = out_dir / f'{transformer_stem}.tree_recon.{weighting}.pt'
        blob = {
            'weighting': weighting,
            'Z_hat': Z_hat,
            'score': score,
            'coverage': results['coverage'],
            'Z_true': results['Z_true'],
            'metrics': metrics,
            **provenance,
        }
        torch.save(blob, path)
        print(f'  saved: {path.name}')
        return path

    print('\nSaving outputs...')
    p_act = _save('activation', results['Z_hat_act'], results['score_act'], metrics_act)
    p_uni = _save('uniform', results['Z_hat_uni'], results['score_uni'], metrics_uni)

    # CSV summary.
    csv_path = out_dir / f'{transformer_stem}.tree_recon.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['weighting', 'level', 'per_level_acc', 'coverage',
                        'whole_tree_acc'],
        )
        writer.writeheader()
        for weighting, m in [('activation', metrics_act), ('uniform', metrics_uni)]:
            for l in range(L):
                writer.writerow({
                    'weighting': weighting,
                    'level': l,
                    'per_level_acc': m['per_level_acc'][l],
                    'coverage': m['coverage'][l],
                    'whole_tree_acc': m['whole_tree_acc'],
                })
    print(f'  saved: {csv_path.name}')

    print('\ndone.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
