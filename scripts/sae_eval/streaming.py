"""Unified streaming-eval library for SAEs trained on the Random Hierarchy Model.

One forward pass over the eval set, accumulating whichever blocks are
enabled by StreamingFlags, returns a dict (the canonical sae_eval artifact).

Artifact schema (a dict saved as *.sae_eval.pt):

    Identity:
        ckpt_path, layer_id, mode ('all_tokens'|'one_token'|'cls_token'),
        sae_token_idx, act_scale, latent_dim, embedding_dim, eval_size,
        eval_seed, dedupe, rhm {v,n,m,s,L}, rules_source, sae_rules_source,
        token_positions [P], num_samples_used.

    Scalar activity aggregates (flag: scalar_aggregates):
        dead_features, dead_ratio, mean_active, mean_active_ratio, ipr,
        active_above_1pct, active_above_10pct,
        mean_active_above_1pct, mean_active_above_10pct.

    Per-position activity (flag: per_position):
        per_position_mean_active [P], per_position_ever_active [P],
        per_position_dead_features [P], per_position_ipr [P],
        per_position_feature_mean_activations [P, F], L0_mean [P].

    Per-feature (flag: per_feature):
        feature_mean_activations [F], decoder_norms [F],
        baseline_mean [P, F], baseline_std [P, F],
        firing_count [P, F], firing_rate [P, F], mean_cofire [P, F].

    Per-target conditional (flag: conditional):
        targets (list of {level, position, value, count}), index_layout,
        conditional_mean [T, P, F], conditional_count [T, P],
        delta_mean [T, P, F], z_score [T, P, F].

    Entropy (flag: joint_fire_and_entropy; requires per_feature):
        joint_fire_count [T, P, F], H_per_feature [T, P, F],
        prior_theoretical / prior_empirical: dict {(level, pos) -> [V_g]},
        H_theoretical / H_empirical: dict {(level, pos) -> float},
        H_bar_fire [P], H_bar_raw [P], H_bar_dec [P],
        H_bar_fire_norm [P], H_bar_raw_norm [P], H_bar_dec_norm [P].

    Classification impact (flag: classification_impact, second pass):
        baseline_err, sae_err, norm_err, baseline_ce, sae_ce.

    Training-side metadata (for CSV joining):
        train_total_loss, train_recon_loss, train_sparse_loss,
        lambda_l1, lr, steps, batch_size.

All entropies are in nats. Accumulators run in float64 on the device,
finalized tensors are cast to float32 (long for counts) on CPU.
Missing keys mean "not computed" (the flag for that block was False).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

@dataclass
class StreamingFlags:
    """Toggles controlling which accumulator blocks run in stream_sae_eval.

    Each flag gates its own code path. Dependencies are validated at the
    top of stream_sae_eval: joint_fire_and_entropy requires per_feature.
    Nothing is computed unless the corresponding flag is True.
    """
    scalar_aggregates: bool = False
    per_position: bool = False
    per_feature: bool = False
    conditional: bool = False
    joint_fire_and_entropy: bool = False
    classification_impact: bool = False

    def validate(self) -> None:
        if self.scalar_aggregates and not self.per_feature:
            raise ValueError(
                'scalar_aggregates requires per_feature=True (dead-feature '
                'count, IPR, and threshold counts are derived from the '
                'per-feature means).'
            )
        if self.conditional and not self.per_feature:
            raise ValueError(
                'conditional requires per_feature=True (delta_mean and '
                'z_score are computed against baseline_mean / baseline_std).'
            )
        if self.joint_fire_and_entropy and not self.per_feature:
            raise ValueError(
                'joint_fire_and_entropy requires per_feature=True '
                '(entropy uses firing_count and baseline_mean).'
            )

    @classmethod
    def all_on(cls) -> 'StreamingFlags':
        return cls(**{f.name: True for f in fields(cls)})


# ---------------------------------------------------------------------------
# Pure entropy math (testable with synthetic tensors, no model required)
# ---------------------------------------------------------------------------

def shannon_entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """H(p) = -sum p * log(p) in nats, with the 0 * log 0 = 0 convention.

    Computed in float64 for numerical stability.
    """
    p64 = p.double()
    return -torch.special.xlogy(p64, p64).sum(dim=dim)


def per_feature_entropy(joint_fire: torch.Tensor,
                        marginal_fire: torch.Tensor) -> torch.Tensor:
    """H_i = H(Z | F_i > 0) per feature from joint-fire and firing counts.

    Args:
        joint_fire: LongTensor or FloatTensor of shape (..., V, F). Count (or
            weight) of events where Z = v and F_i > 0, for each value v.
        marginal_fire: Tensor of shape (..., F). Count of events where
            F_i > 0 (must equal joint_fire.sum(dim=-2) for counts).

    Returns:
        FloatTensor of shape (..., F), entropy in nats. Features with
        marginal_fire == 0 (dead) get NaN.
    """
    if joint_fire.shape[-1] != marginal_fire.shape[-1]:
        raise ValueError(
            f'Feature-axis mismatch: joint_fire has {joint_fire.shape[-1]} '
            f'features, marginal_fire has {marginal_fire.shape[-1]}.'
        )
    joint64 = joint_fire.double()
    marginal64 = marginal_fire.double()

    denom = marginal64.clamp_min(1.0).unsqueeze(-2)  # (..., 1, F)
    p = joint64 / denom                              # (..., V, F)
    H = -torch.special.xlogy(p, p).sum(dim=-2)       # (..., F)

    dead = (marginal64 == 0)                         # (..., F)
    if dead.any():
        H = H.masked_fill(dead, float('nan'))
    return H.float()


def weighted_aggregate(H: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean of H ignoring NaN entries (dead features).

    Aggregation formula from the handoff doc:
        H_bar = sum_i w_i * H_i / sum_i w_i
    over features with H_i finite. Returns NaN if the weight sum is 0.

    Args:
        H: FloatTensor of shape (..., F). May contain NaN for dead features.
        weights: FloatTensor of shape (..., F), non-negative.

    Returns:
        FloatTensor of shape (...) with the weighted aggregate.
    """
    if H.shape != weights.shape:
        raise ValueError(f'Shape mismatch: H {H.shape} vs weights {weights.shape}.')
    H64 = H.double()
    w64 = weights.double()
    mask = torch.isfinite(H64) & (w64 > 0)
    w_masked = torch.where(mask, w64, torch.zeros_like(w64))
    h_masked = torch.where(mask, H64, torch.zeros_like(H64))
    num = (w_masked * h_masked).sum(dim=-1)
    den = w_masked.sum(dim=-1)
    out = torch.where(den > 0, num / den.clamp_min(1e-30),
                      torch.full_like(den, float('nan')))
    return out.float()


# ---------------------------------------------------------------------------
# Token-position selection (ported from scripts/sae_direct_analysis/analyze_sae.py)
# ---------------------------------------------------------------------------

def select_activation_tokens(act: torch.Tensor, mode: str, has_cls: bool,
                             token_idx: int):
    """Slice the SAE-input view from a (batch, seq_len, emb_dim) activation.

    For 'mean_pooled', the caller is expected to hook `model.ln_f` so `act` is
    already post-ln_f; this function only mean-pools over the sequence dim.

    Returns (selected, token_positions) where token_positions is a 1-D
    LongTensor of 0-based real-token indices, or tensor([-1]) for
    cls_token / mean_pooled modes (sentinel: not a real-token position).
    """
    if mode == 'cls_token':
        if not has_cls:
            raise ValueError('cls_token mode requires a model with a CLS token')
        selected = act[:, :1, :]
        token_positions = torch.tensor([-1], dtype=torch.long)
    elif mode == 'one_token':
        offset = 1 if has_cls else 0
        selected = act[:, offset + token_idx : offset + token_idx + 1, :]
        token_positions = torch.tensor([int(token_idx)], dtype=torch.long)
    elif mode == 'all_tokens':
        selected = act[:, 1:, :] if has_cls else act
        num_real = selected.size(1)
        token_positions = torch.arange(num_real, dtype=torch.long)
    elif mode == 'mean_pooled':
        selected = act.mean(dim=1, keepdim=True)
        token_positions = torch.tensor([-1], dtype=torch.long)
    else:
        raise ValueError(f'Unknown sae_activation_source mode: {mode!r}')
    return selected, token_positions


def _hook_module(model, layer_id: int, mode: str):
    """Return the module to attach the activation-capture hook to.

    For 'mean_pooled', hook `model.ln_f` so the captured tensor is post-ln_f.
    For all other modes, hook `model.blocks[layer_id]` (pre-ln_f).
    """
    if mode == 'mean_pooled':
        return model.ln_f
    return model.blocks[layer_id]


# ---------------------------------------------------------------------------
# Target enumeration over RHM trees
# (ported from scripts/sae_direct_analysis/analyze_sae.py)
# ---------------------------------------------------------------------------

def _level_values_at_position(level_tensor: torch.Tensor, pos: int) -> torch.Tensor:
    if level_tensor.ndim == 1:
        return level_tensor
    return level_tensor[:, pos]


def enumerate_targets(trees: dict):
    """Enumerate (level, position, value) triples observed in the eval trees.

    Returns:
        targets:      list of dicts {level, position, value, count}.
        index_layout: list of {level, position, values, start, end}, one per
                      (level, position) group. targets[start:end] is the
                      values slice for this group.
    """
    targets: list[dict] = []
    index_layout: list[dict] = []
    max_level = max(trees.keys())
    for level in range(max_level + 1):
        level_tensor = trees[level]
        width = 1 if level_tensor.ndim == 1 else level_tensor.size(1)
        for pos in range(width):
            col = _level_values_at_position(level_tensor, pos)
            unique, counts = torch.unique(col, return_counts=True)
            start = len(targets)
            for v, c in zip(unique.tolist(), counts.tolist()):
                targets.append({
                    'level': int(level),
                    'position': int(pos),
                    'value': int(v),
                    'count': int(c),
                })
            end = len(targets)
            index_layout.append({
                'level': int(level),
                'position': int(pos),
                'values': unique.clone().long(),
                'start': int(start),
                'end': int(end),
            })
    return targets, index_layout


def dedupe_trees(trees: dict) -> dict:
    """Drop duplicate RHM trees (keyed on trees[L]).

    Selects the first-occurrence row of each unique leaf sequence to keep
    all levels internally consistent.
    """
    L = max(trees.keys())
    leaves = trees[L].long()
    _, inverse = torch.unique(leaves, dim=0, return_inverse=True)
    n_unique = int(inverse.max().item()) + 1
    first_idx = torch.full((n_unique,), -1, dtype=torch.long)
    order = torch.arange(leaves.size(0), dtype=torch.long)
    first_idx = first_idx.scatter_reduce(
        0, inverse.long(), order, reduce='amin', include_self=False
    )
    keep_t = torch.sort(first_idx).values
    return {l: trees[l][keep_t] for l in range(L + 1)}


# ---------------------------------------------------------------------------
# Main streaming pass: stream_sae_eval
# ---------------------------------------------------------------------------

def _index_layout_to_cpu(index_layout: list[dict]) -> list[dict]:
    """Strip device-resident fields from index_layout for safe torch.save."""
    return [
        {'level': g['level'], 'position': g['position'],
         'values': g['values'].cpu(),
         'start': g['start'], 'end': g['end']}
        for g in index_layout
    ]


@torch.no_grad()
def stream_sae_eval(
    model,
    sae,
    trees: dict,
    *,
    layer_id: int,
    mode: str,
    has_cls: bool,
    token_idx: int,
    act_scale: float,
    batch_size: int,
    device: str,
    flags: StreamingFlags,
    rhm: dict | None = None,
    rules: dict | None = None,
) -> dict:
    """Single streaming forward pass producing the canonical artifact dict.

    Only the accumulators enabled by `flags` are computed; other keys are
    absent from the returned dict.

    When `flags.joint_fire_and_entropy` is set, `rhm` (dict with keys
    n, v, m, s, L) and `rules` (the RHM rules dict) must be supplied so
    the theoretical prior P(Z) / H(Z) can be propagated and used to
    normalize the aggregates.
    """
    flags.validate()

    if flags.joint_fire_and_entropy and (rhm is None or rules is None):
        raise ValueError(
            'joint_fire_and_entropy requires rhm and rules arguments '
            '(needed for theoretical P(Z) and the layer-to-level mapping).'
        )

    if not any([flags.scalar_aggregates, flags.per_position, flags.per_feature,
                flags.conditional, flags.joint_fire_and_entropy]):
        raise ValueError('stream_sae_eval called with all flags off.')

    # ---- Eval loader ----
    inputs = trees[max(trees.keys())].long()
    labels = trees[0].long()
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # ---- Target enumeration (once, up front) ----
    need_targets = flags.conditional or flags.joint_fire_and_entropy
    if need_targets:
        targets, index_layout = enumerate_targets(trees)
        num_targets = len(targets)
        for group in index_layout:
            group['values_dev'] = group['values'].to(device)
    else:
        targets = None
        index_layout = None
        num_targets = 0

    # ---- Hook the transformer block (or ln_f for mean_pooled) ----
    buf: list[torch.Tensor] = []
    hook = _hook_module(model, layer_id, mode).register_forward_hook(
        lambda _m, _i, o: buf.append(o.detach())
    )

    dec_norms = sae.decoder_feature_norms().to(device)
    latent_dim = int(sae.latent_dim)

    # ---- Accumulators (lazily allocated once P is known) ----
    num_positions = None
    saved_token_positions = None

    sum_f = None                 # [P, F] float64 -- used by per_feature and per_position
    sum_f2 = None                # [P, F] float64 -- per_feature
    sum_fire = None              # [P, F] float64 -- per_feature
    sum_cofire = None            # [P, F] float64 -- per_feature
    sum_L0 = None                # [P]    float64 -- per_feature

    ever_active_pp = None        # [P, F] bool   -- per_position
    active_sum_pp = None         # [P]    float64 -- per_position
    active_1pct_sum_pp = None    # [P]    float64 -- per_position
    active_10pct_sum_pp = None   # [P]    float64 -- per_position

    mean_active_1pct_sum = 0.0   # scalar -- scalar_aggregates (global 1pct threshold)
    mean_active_10pct_sum = 0.0  # scalar -- scalar_aggregates

    sum_f_cond = None            # [T, P, F] -- conditional
    sum_fire_cond = None         # [T, P, F] -- joint_fire_and_entropy
    count_cond = None            # [T] long  -- conditional or entropy

    count_total = 0

    try:
        sample_cursor = 0
        for batch_inputs, _ in loader:
            B = batch_inputs.size(0)
            batch_inputs = batch_inputs.to(device)
            model(batch_inputs)
            if not buf:
                raise RuntimeError(f'No activations captured at layer {layer_id}')
            act = buf.pop(0)
            selected, token_positions = select_activation_tokens(
                act, mode=mode, has_cls=has_cls, token_idx=token_idx,
            )
            P = selected.size(1)
            flat = selected.reshape(B * P, -1) * act_scale
            _, z = sae(flat)                                    # [B*P, F]
            f_act = (z * dec_norms.unsqueeze(0)).view(B, P, latent_dim).double()

            if num_positions is None:
                num_positions = P
                saved_token_positions = token_positions.clone()
                if flags.per_feature or flags.per_position:
                    sum_f = torch.zeros(P, latent_dim, dtype=torch.float64, device=device)
                if flags.per_feature:
                    sum_f2 = torch.zeros_like(sum_f)
                    sum_fire = torch.zeros_like(sum_f)
                    sum_cofire = torch.zeros_like(sum_f)
                    sum_L0 = torch.zeros(P, dtype=torch.float64, device=device)
                if flags.per_position:
                    ever_active_pp = torch.zeros(P, latent_dim, dtype=torch.bool, device=device)
                    active_sum_pp = torch.zeros(P, dtype=torch.float64, device=device)
                    active_1pct_sum_pp = torch.zeros(P, dtype=torch.float64, device=device)
                    active_10pct_sum_pp = torch.zeros(P, dtype=torch.float64, device=device)
                if flags.conditional:
                    sum_f_cond = torch.zeros(num_targets, P, latent_dim,
                                             dtype=torch.float64, device=device)
                if flags.joint_fire_and_entropy:
                    sum_fire_cond = torch.zeros(num_targets, P, latent_dim,
                                                dtype=torch.float64, device=device)
                if need_targets:
                    count_cond = torch.zeros(num_targets, dtype=torch.long, device=device)
            elif P != num_positions:
                raise RuntimeError(
                    f'Token-count changed between batches: expected {num_positions}, got {P}'
                )

            is_active = (f_act > 0)                              # [B, P, F] bool

            if flags.per_feature or flags.per_position:
                sum_f += f_act.sum(dim=0)                        # [P, F]

            if flags.per_feature:
                sum_f2 += (f_act * f_act).sum(dim=0)             # [P, F]
                fire = is_active.double()                        # [B, P, F]
                L0_per_pos = fire.sum(dim=2)                     # [B, P]
                sum_fire += fire.sum(dim=0)                      # [P, F]
                sum_cofire += torch.einsum('bpf,bp->pf', fire, L0_per_pos)
                sum_L0 += L0_per_pos.sum(dim=0)                  # [P]

            if flags.per_position or flags.scalar_aggregates:
                token_max = f_act.max(dim=2, keepdim=True).values
                is_1pct = (f_act > 0.01 * token_max)
                is_10pct = (f_act > 0.10 * token_max)
                if flags.per_position:
                    ever_active_pp |= is_active.any(dim=0)
                    active_sum_pp += is_active.sum(dim=(0, 2)).double()
                    active_1pct_sum_pp += is_1pct.sum(dim=(0, 2)).double()
                    active_10pct_sum_pp += is_10pct.sum(dim=(0, 2)).double()
                if flags.scalar_aggregates:
                    mean_active_1pct_sum += float(is_1pct.double().sum().item())
                    mean_active_10pct_sum += float(is_10pct.double().sum().item())

            count_total += B

            if need_targets:
                batch_end = sample_cursor + B
                fire_double = is_active.double() if flags.joint_fire_and_entropy else None
                for group in index_layout:
                    level = group['level']
                    pos_g = group['position']
                    values_dev = group['values_dev']
                    level_tensor = trees[level]
                    col = (level_tensor if level_tensor.ndim == 1
                           else level_tensor[:, pos_g])
                    col_batch = col[sample_cursor:batch_end].to(device).long()
                    onehot = (col_batch.unsqueeze(1) == values_dev.unsqueeze(0)).double()
                    if flags.conditional:
                        sum_f_cond[group['start']:group['end']] += (
                            torch.einsum('bv,bpf->vpf', onehot, f_act)
                        )
                    if flags.joint_fire_and_entropy:
                        sum_fire_cond[group['start']:group['end']] += (
                            torch.einsum('bv,bpf->vpf', onehot, fire_double)
                        )
                    count_cond[group['start']:group['end']] += onehot.sum(dim=0).long()
                sample_cursor = batch_end
    finally:
        hook.remove()

    if num_positions is None:
        raise RuntimeError('No batches were processed; loader was empty.')

    P = num_positions
    count_total_t = float(count_total)

    result: dict = {
        'num_samples_used': int(count_total),
        'token_positions': saved_token_positions.cpu(),
        'latent_dim': latent_dim,
    }

    # ---- Finalize per_feature ----
    baseline_mean_dev = None
    baseline_std_dev = None
    if flags.per_feature:
        baseline_mean_dev = sum_f / count_total_t                                # [P, F] float64
        baseline_var = (sum_f2 / count_total_t) - baseline_mean_dev ** 2
        baseline_std_dev = baseline_var.clamp_min(0).sqrt()
        firing_count = sum_fire.long()                                           # [P, F]
        firing_rate = (sum_fire / count_total_t).float()                         # [P, F]
        mean_cofire = sum_cofire / sum_fire.clamp_min(1.0)                       # [P, F]
        mean_cofire_cpu = mean_cofire.float()
        mean_cofire_cpu[sum_fire == 0] = float('nan')
        L0_mean = (sum_L0 / count_total_t).float()                               # [P]
        # global per-feature mean = per-position mean averaged over positions
        per_position_feature_mean = baseline_mean_dev.float()                    # [P, F]
        feature_mean_activations = per_position_feature_mean.mean(dim=0)         # [F]
        result.update({
            'feature_mean_activations': feature_mean_activations.cpu(),
            'decoder_norms': dec_norms.float().cpu(),
            'baseline_mean': baseline_mean_dev.float().cpu(),
            'baseline_std': baseline_std_dev.float().cpu(),
            'firing_count': firing_count.cpu(),
            'firing_rate': firing_rate.cpu(),
            'mean_cofire': mean_cofire_cpu.cpu(),
            'L0_mean': L0_mean.cpu(),
        })

    # ---- Finalize per_position ----
    if flags.per_position:
        ever_active_count_pp = ever_active_pp.sum(dim=1).long()                  # [P]
        per_position_dead = (latent_dim - ever_active_count_pp).long()
        per_position_mean_active = (active_sum_pp / count_total_t).float()
        # IPR at each position, from per-position feature means
        fm_pp = (sum_f / count_total_t).float()                                  # [P, F]
        sa_pp = fm_pp.sum(dim=1)
        sa2_pp = fm_pp.pow(2).sum(dim=1)
        per_position_ipr = torch.where(
            sa2_pp > 0, sa_pp.pow(2) / sa2_pp, torch.zeros_like(sa_pp)
        ).float()
        fm_max_pp = fm_pp.max(dim=1).values
        zero_mask = fm_max_pp <= 0
        gt_1pct_pp = (fm_pp > 0.01 * fm_max_pp.unsqueeze(1)).sum(dim=1).long()
        gt_10pct_pp = (fm_pp > 0.10 * fm_max_pp.unsqueeze(1)).sum(dim=1).long()
        gt_1pct_pp = torch.where(zero_mask, torch.zeros_like(gt_1pct_pp), gt_1pct_pp)
        gt_10pct_pp = torch.where(zero_mask, torch.zeros_like(gt_10pct_pp), gt_10pct_pp)
        result.update({
            'per_position_mean_active': per_position_mean_active.cpu(),
            'per_position_mean_active_ratio': (
                per_position_mean_active / max(latent_dim, 1)
            ).cpu(),
            'per_position_ever_active': ever_active_count_pp.cpu(),
            'per_position_dead_features': per_position_dead.cpu(),
            'per_position_ipr': per_position_ipr.cpu(),
            'per_position_mean_active_above_1pct': (
                active_1pct_sum_pp / count_total_t
            ).float().cpu(),
            'per_position_mean_active_above_10pct': (
                active_10pct_sum_pp / count_total_t
            ).float().cpu(),
            'per_position_active_above_1pct': gt_1pct_pp.cpu(),
            'per_position_active_above_10pct': gt_10pct_pp.cpu(),
            'per_position_feature_mean_activations': fm_pp.cpu(),
        })

    # ---- Finalize scalar_aggregates ----
    if flags.scalar_aggregates:
        # dead / mean_active use the per-feature accumulators from per_feature.
        ever_active_global = (sum_fire.sum(dim=0) > 0)                           # [F]
        dead = int((~ever_active_global).sum().item())
        global_feature_mean = (sum_f.sum(dim=0) / (count_total_t * P)).float()   # [F]
        sum_a = float(global_feature_mean.sum().item())
        sum_a2 = float(global_feature_mean.pow(2).sum().item())
        ipr = (sum_a ** 2) / sum_a2 if sum_a2 > 0 else 0.0
        feat_max = float(global_feature_mean.max().item()) if latent_dim > 0 else 0.0
        active_above_1pct = (
            int((global_feature_mean > 0.01 * feat_max).sum().item()) if feat_max > 0 else 0
        )
        active_above_10pct = (
            int((global_feature_mean > 0.10 * feat_max).sum().item()) if feat_max > 0 else 0
        )
        total_tokens_seen = count_total * P
        global_active_sum = float(sum_fire.sum().item())
        result.update({
            'dead_features': dead,
            'dead_ratio': dead / max(latent_dim, 1),
            'mean_active': global_active_sum / max(total_tokens_seen, 1),
            'mean_active_ratio': (
                global_active_sum / max(total_tokens_seen * latent_dim, 1)
            ),
            'ipr': ipr,
            'active_above_1pct': active_above_1pct,
            'active_above_10pct': active_above_10pct,
            'mean_active_above_1pct': mean_active_1pct_sum / max(total_tokens_seen, 1),
            'mean_active_above_10pct': mean_active_10pct_sum / max(total_tokens_seen, 1),
        })

    # ---- Finalize conditional ----
    if flags.conditional:
        cc = count_cond.double().clamp_min(1.0)
        conditional_mean = sum_f_cond / cc.view(num_targets, 1, 1)
        zero_mask = (count_cond == 0)
        if zero_mask.any():
            conditional_mean[zero_mask] = float('nan')
        delta_mean = conditional_mean - baseline_mean_dev.unsqueeze(0)
        eps = 1e-8
        z_score = delta_mean / (baseline_std_dev.unsqueeze(0) + eps)
        count_cond_tp = count_cond.unsqueeze(-1).expand(num_targets, P).contiguous()
        result.update({
            'targets': targets,
            'index_layout': _index_layout_to_cpu(index_layout),
            'conditional_mean': conditional_mean.float().cpu(),
            'conditional_count': count_cond_tp.cpu(),
            'delta_mean': delta_mean.float().cpu(),
            'z_score': z_score.float().cpu(),
        })

    # ---- Entropy finalization ----
    if flags.joint_fire_and_entropy:
        if 'targets' not in result:
            result['targets'] = targets
            result['index_layout'] = _index_layout_to_cpu(index_layout)

        joint_fire_count = sum_fire_cond.long()                     # [T, P, F]
        result['joint_fire_count'] = joint_fire_count.cpu()

        firing_count_dev = sum_fire.long()                          # [P, F] from per_feature

        # H_per_feature has one entropy per (group, SAE position, feature).
        # Stored as [num_groups, P, F] (NOT [T, P, F]) since entropy is a
        # function of the full value distribution within a group, not per
        # target value. Callers index by group via index_layout.
        num_groups = len(index_layout)
        H_per_feature = torch.zeros(num_groups, P, latent_dim,
                                    dtype=torch.float64, device=device)
        for g_idx, group in enumerate(index_layout):
            start, end = group['start'], group['end']
            # joint_sub shape (V_g, P, F); marginal shape (P, F).
            joint_sub = joint_fire_count[start:end].double()
            marginal = firing_count_dev.double().clamp_min(1.0)
            p_iz = joint_sub / marginal.unsqueeze(0)                # (V_g, P, F)
            H_g = -torch.special.xlogy(p_iz, p_iz).sum(dim=0)       # (P, F)
            # Dead features (marginal == 0 at this SAE position) -> NaN.
            dead_mask = (firing_count_dev == 0)                     # (P, F)
            if dead_mask.any():
                H_g = H_g.masked_fill(dead_mask, float('nan'))
            H_per_feature[g_idx] = H_g
        result['H_per_feature'] = H_per_feature.float().cpu()

        # Priors and reference entropies.
        from datasets.random_hierarchy_model import latent_prior as _latent_prior
        n_cls = int(rhm['n'])
        v_tok = int(rhm['v'])
        s_tup = int(rhm['s'])
        L_levels = int(rhm['L'])
        prior_theo_full = _latent_prior(rules, n_cls, v_tok)        # dict {level -> (s^level, V_level)}

        prior_theoretical: dict = {}
        prior_empirical: dict = {}
        H_theoretical: dict = {}
        H_empirical: dict = {}
        count_cond_cpu = count_cond.cpu()
        for group in index_layout:
            lvl = group['level']
            pos = group['position']
            vals = group['values'].cpu()
            start, end = group['start'], group['end']

            theo_row = prior_theo_full[lvl][pos].double()            # full V_level
            theo_sub = theo_row[vals].float()                        # V_g at observed values
            emp_sub = count_cond_cpu[start:end].double() / max(count_total, 1)
            emp_sub_f = emp_sub.float()

            key = (int(lvl), int(pos))
            prior_theoretical[key] = theo_sub
            prior_empirical[key] = emp_sub_f
            H_theoretical[key] = float(shannon_entropy(theo_row).item())
            H_empirical[key] = float(shannon_entropy(emp_sub).item())

        result['prior_theoretical'] = prior_theoretical
        result['prior_empirical'] = prior_empirical
        result['H_theoretical'] = H_theoretical
        result['H_empirical'] = H_empirical

        # Aggregates per SAE position, using the matched (level, j) group.
        H_bar_fire = torch.full((P,), float('nan'))
        H_bar_raw = torch.full((P,), float('nan'))
        H_bar_dec = torch.full((P,), float('nan'))
        H_bar_fire_norm = torch.full((P,), float('nan'))
        H_bar_raw_norm = torch.full((P,), float('nan'))
        H_bar_dec_norm = torch.full((P,), float('nan'))

        if mode not in {'cls_token'}:
            baseline_mean_cpu = result['baseline_mean']              # [P, F] float32
            firing_rate_cpu = result['firing_rate']                  # [P, F]
            decoder_norms_cpu = result['decoder_norms']              # [F]
            H_per_feature_cpu = result['H_per_feature']              # [num_groups, P, F]

            group_index = {(g['level'], g['position']): idx
                           for idx, g in enumerate(index_layout)}

            for p_idx in range(P):
                if mode == 'mean_pooled':
                    # Mean-pooled aggregates all leaf positions, so the
                    # leaf-position -> (level, j) mapping does not apply.
                    # Condition entropy on the root class instead.
                    matched_level = 0
                    matched_j = 0
                else:
                    matched_level = L_levels - 1 - layer_id
                    p_real = int(saved_token_positions[p_idx].item())
                    matched_j = p_real // (s_tup ** (1 + layer_id))
                g_idx = group_index.get((matched_level, matched_j))
                if g_idx is None:
                    continue  # shouldn't happen on well-formed trees
                H_slice = H_per_feature_cpu[g_idx, p_idx, :]         # [F]
                w_fire = firing_rate_cpu[p_idx, :]
                w_raw = baseline_mean_cpu[p_idx, :] / decoder_norms_cpu.clamp_min(1e-30)
                w_dec = baseline_mean_cpu[p_idx, :]
                H_bar_fire[p_idx] = weighted_aggregate(H_slice, w_fire)
                H_bar_raw[p_idx] = weighted_aggregate(H_slice, w_raw)
                H_bar_dec[p_idx] = weighted_aggregate(H_slice, w_dec)

                H_ref = H_theoretical.get((int(matched_level), int(matched_j)), float('nan'))
                if H_ref > 0 and torch.isfinite(torch.tensor(H_ref)):
                    H_bar_fire_norm[p_idx] = H_bar_fire[p_idx] / H_ref
                    H_bar_raw_norm[p_idx] = H_bar_raw[p_idx] / H_ref
                    H_bar_dec_norm[p_idx] = H_bar_dec[p_idx] / H_ref

        result['H_bar_fire'] = H_bar_fire
        result['H_bar_raw'] = H_bar_raw
        result['H_bar_dec'] = H_bar_dec
        result['H_bar_fire_norm'] = H_bar_fire_norm
        result['H_bar_raw_norm'] = H_bar_raw_norm
        result['H_bar_dec_norm'] = H_bar_dec_norm

    return result


# ---------------------------------------------------------------------------
# Classification impact (second streaming pass)
# ---------------------------------------------------------------------------

def _make_sae_replacement_hook(sae, mode: str, has_cls: bool, token_idx: int,
                               act_scale: float):
    """Forward-hook that replaces the layer's residuals with the SAE reconstruction
    at the positions the SAE was trained on. Residuals at other positions pass through.

    For 'mean_pooled', this hook is intended to be registered on `model.ln_f`
    (post-ln_f), not on a transformer block. It runs the SAE on the
    mean-pooled output and returns a tensor whose mean over dim=1 equals the
    SAE reconstruction, so the downstream `mean_out = x.mean(dim=1)` in the
    meanclass classifier reads the SAE recon exactly.
    """
    if mode == 'mean_pooled':
        def hook(_m, _i, output):
            pooled = output.mean(dim=1)               # [B, D]
            flat = pooled * act_scale
            recon, _ = sae(flat)                      # [B, D]
            recon = recon / act_scale
            B, T, D = output.size(0), output.size(1), output.size(-1)
            return recon.unsqueeze(1).expand(B, T, D)
        return hook

    def hook(_m, _i, output):
        out = output.clone()
        if mode == 'cls_token':
            sl_shape = output[:, :1, :].shape
            flat = out[:, :1, :].reshape(-1, out.size(-1)) * act_scale
            recon, _ = sae(flat)
            out[:, :1, :] = (recon / act_scale).reshape(sl_shape)
        elif mode == 'one_token':
            offset = 1 if has_cls else 0
            pos = offset + token_idx
            sl_shape = output[:, pos : pos + 1, :].shape
            flat = out[:, pos : pos + 1, :].reshape(-1, out.size(-1)) * act_scale
            recon, _ = sae(flat)
            out[:, pos : pos + 1, :] = (recon / act_scale).reshape(sl_shape)
        else:  # all_tokens
            sl = slice(1, None) if has_cls else slice(0, None)
            sl_shape = output[:, sl, :].shape
            flat = out[:, sl, :].reshape(-1, out.size(-1)) * act_scale
            recon, _ = sae(flat)
            out[:, sl, :] = (recon / act_scale).reshape(sl_shape)
        return out
    return hook


def _classification_pass(model, loader, device, hook_module=None, hook_fn=None):
    """Run through the loader, returning (accuracy, mean_cross_entropy).

    If hook_module and hook_fn are given, the hook is installed for this pass
    only and removed afterward.
    """
    import torch.nn.functional as F_nn
    handle = hook_module.register_forward_hook(hook_fn) if hook_module is not None else None
    correct = total = 0
    total_ce = 0.0
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_ce += F_nn.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
    finally:
        if handle is not None:
            handle.remove()
    return correct / max(total, 1), total_ce / max(total, 1)


@torch.no_grad()
def stream_classification_impact(
    model,
    sae,
    trees: dict,
    *,
    layer_id: int,
    mode: str,
    has_cls: bool,
    token_idx: int,
    act_scale: float,
    batch_size: int,
    device: str,
) -> dict:
    """Measure classification error / cross-entropy with and without the SAE inserted.

    Returns:
        dict with baseline_err, sae_err, norm_err, baseline_ce, sae_ce.
        norm_err = sae_err / random_err, where random_err = 1 - 1/num_classes.
    """
    inputs = trees[max(trees.keys())].long()
    labels = trees[0].long()
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    if mode == 'mean_pooled':
        last_layer = len(model.blocks) - 1
        if layer_id != last_layer:
            raise NotImplementedError(
                f'classification_impact for mean_pooled is only supported on the '
                f'final transformer block (got layer_id={layer_id}, '
                f'num_layers={len(model.blocks)}).'
            )
        if has_cls:
            raise NotImplementedError(
                'mean_pooled is incompatible with has_cls=True.'
            )

    baseline_acc, baseline_ce = _classification_pass(model, loader, device)
    baseline_err = 1.0 - baseline_acc

    hook_fn = _make_sae_replacement_hook(sae, mode, has_cls, token_idx, act_scale)
    sae_acc, sae_ce = _classification_pass(
        model, loader, device,
        hook_module=_hook_module(model, layer_id, mode), hook_fn=hook_fn,
    )
    sae_err = 1.0 - sae_acc
    random_err = 1.0 - 1.0 / float(model.num_classes)
    norm_err = (sae_err / random_err) if random_err > 0 else float('nan')

    return {
        'baseline_err': baseline_err,
        'sae_err': sae_err,
        'norm_err': norm_err,
        'baseline_ce': baseline_ce,
        'sae_ce': sae_ce,
    }
