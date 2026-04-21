"""Direct per-feature analysis of trained SAEs against RHM latents.

For each trained SAE checkpoint in a sweep (or a single --ckpt), this script:

  1. Loads the frozen transformer + SAE that the checkpoint refers to.
  2. Builds a fresh RHM eval split using the SAME rules the transformer and
     the SAE were trained on (verified against sae_dataset_split.rules_source).
  3. Runs one streaming forward pass, capturing the SAE's decoder-weighted
     feature activations at the transformer block the SAE hooks into.
  4. Accumulates the baseline mean / std E[f] and, for every ground truth
     latent triple (level, position, value), the conditional mean
     E[f | trees[level][:, position] == value].
  5. Saves a compact .feature_latent.pt artifact per checkpoint.

"Feature activation" throughout this file means the DECODER-WEIGHTED value
  f_i = z_i * ||W_dec[:, i]||
matching the convention used by eval_sweep.py's activity stats and the old
latent_analysis.py. The artifact's baseline_mean / conditional_mean /
delta_mean / z_score are all in the decoder-weighted scale.

Usage:
    python scripts/sae_direct_analysis/analyze_sae.py \\
        --sweep_dir /path/to/sweep/output/ \\
        --out_dir   /path/to/analysis/out/ \\
        [--ckpt /path/to/one.pt]           \\
        [--eval_size 32768]                \\
        [--eval_seed 99999]                \\
        [--batch_size 512]                 \\
        [--dedupe]                         \\
        [--device cuda]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Add the repo root to sys.path so we can import init, models, datasets,
# and scripts.common.sae_loading
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.random_hierarchy_model import sample_trees
from scripts.common.sae_loading import load_sae, load_transformer


# ---------------------------------------------------------------------------
# Token-position selection (must match eval_sweep.py / train_sae.py exactly)
# ---------------------------------------------------------------------------

def select_activation_tokens(act: torch.Tensor, mode: str, has_cls: bool,
                             token_idx: int):
    """Slice out the (batch, P, emb_dim) view the SAE was trained on.

    Returns (selected, token_positions) where token_positions is a LongTensor
    of 0-based real-token indices, or a LongTensor containing [-1] for
    cls_token mode (since CLS is not a real-token position).
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
        if has_cls:
            selected = act[:, 1:, :]
        else:
            selected = act
        num_real = selected.size(1)
        token_positions = torch.arange(num_real, dtype=torch.long)
    else:
        raise ValueError(f'Unknown sae_activation_source mode: {mode!r}')
    return selected, token_positions


# ---------------------------------------------------------------------------
# Target enumeration
# ---------------------------------------------------------------------------

def _level_values_at_position(level_tensor: torch.Tensor, pos: int) -> torch.Tensor:
    """Return the vector trees[level][:, pos] (or trees[0][:] for level 0)."""
    if level_tensor.ndim == 1:
        return level_tensor
    return level_tensor[:, pos]


def enumerate_targets(trees: dict):
    """Build the list of (level, position, value) targets observed in `trees`.

    Returns:
        targets:      list of dicts with keys 'level', 'position', 'value', 'count'.
        index_layout: list of (level, position, values_tensor, start, end) entries,
                      one per (level, position) group. start:end is the slice into
                      the targets list for that group. values_tensor is on CPU.

    `index_layout` is used to vectorize the conditional-sum accumulation:
    for each (level, position) group, all values at that group are packed into
    a single one-hot matrix and accumulated with a single einsum.
    """
    targets = []
    index_layout = []
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
                'values': unique.clone().long(),  # [num_values_at_this_group]
                'start': int(start),
                'end': int(end),
            })
    return targets, index_layout


# ---------------------------------------------------------------------------
# Core analysis (streaming accumulators)
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_checkpoint(model, sae, trees, layer_id: int, mode: str,
                       has_cls: bool, token_idx: int, act_scale: float,
                       batch_size: int, device: str,
                       compute_cofire: bool = True):
    """Run the streaming analysis for a single (model, sae) pair.

    Returns a dict with all the fields that will be saved to the artifact.

    If compute_cofire is True, also accumulates per-(position, feature)
    firing_rate, firing_count, mean_cofire = E[L0_p | f_i > 0], and the
    per-position L0_mean = E[L0_p].
    """
    # ----- Eval loader: shuffle=False so batch rows line up with `trees` -----
    inputs = trees[max(trees.keys())]  # trees[L], shape [N, s^L]
    labels = trees[0]                  # shape [N]
    dataset = torch.utils.data.TensorDataset(inputs.long(), labels.long())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    num_samples = inputs.size(0)

    # ----- Target enumeration (once, up front) -----
    targets, index_layout = enumerate_targets(trees)
    num_targets = len(targets)

    # Move the per-group value vectors to the device for fast mask construction.
    for group in index_layout:
        group['values_dev'] = group['values'].to(device)

    # ----- Hook the transformer block -----
    buf = []
    hook = model.blocks[layer_id].register_forward_hook(
        lambda _m, _i, o: buf.append(o.detach())
    )

    dec_norms = sae.decoder_feature_norms().to(device)
    latent_dim = int(sae.latent_dim)

    sum_f = None          # [P, F]
    sum_f2 = None         # [P, F]
    count_total = 0
    sum_f_cond = None     # [T, P, F]
    count_cond = None     # [T]  (counts do not depend on the token position)
    num_positions = None
    saved_token_positions = None

    # Co-firing accumulators (only allocated if compute_cofire).
    sum_fire = None       # [P, F]  -- count of times f_i > 0 per (p, f)
    sum_cofire = None     # [P, F]  -- sum over b of 1{f_i>0} * L0_p(b)
    sum_L0 = None         # [P]     -- sum over b of L0_p(b)

    try:
        sample_cursor = 0
        for batch_inputs, _ in loader:
            B = batch_inputs.size(0)
            batch_inputs = batch_inputs.to(device)
            model(batch_inputs)
            if not buf:
                raise RuntimeError(f'No activations captured at layer {layer_id}')
            act = buf.pop(0)                                    # [B, seq_len, emb_dim]
            selected, token_positions = select_activation_tokens(
                act, mode=mode, has_cls=has_cls, token_idx=token_idx
            )                                                   # [B, P, emb_dim]
            P = selected.size(1)
            flat = selected.reshape(B * P, -1) * act_scale
            _, z = sae(flat)                                    # [B*P, F]
            f = (z * dec_norms.unsqueeze(0)).view(B, P, latent_dim).double()
            # ^ accumulate in float64 to avoid precision loss.

            if sum_f is None:
                sum_f = torch.zeros(P, latent_dim, dtype=torch.float64, device=device)
                sum_f2 = torch.zeros_like(sum_f)
                sum_f_cond = torch.zeros(num_targets, P, latent_dim,
                                         dtype=torch.float64, device=device)
                count_cond = torch.zeros(num_targets, dtype=torch.long, device=device)
                num_positions = P
                saved_token_positions = token_positions.clone()
                if compute_cofire:
                    sum_fire = torch.zeros(P, latent_dim, dtype=torch.float64, device=device)
                    sum_cofire = torch.zeros(P, latent_dim, dtype=torch.float64, device=device)
                    sum_L0 = torch.zeros(P, dtype=torch.float64, device=device)
            elif P != num_positions:
                raise RuntimeError(
                    f'Token-count changed between batches: expected {num_positions}, got {P}'
                )

            sum_f += f.sum(dim=0)           # [P, F]
            sum_f2 += (f * f).sum(dim=0)
            count_total += B

            if compute_cofire:
                fire = (f > 0).double()                 # [B, P, F]
                L0_per_pos = fire.sum(dim=2)            # [B, P]
                sum_fire += fire.sum(dim=0)             # [P, F]
                # cofire[p, f] = sum_b fire[b, p, f] * L0_per_pos[b, p]
                sum_cofire += torch.einsum('bpf,bp->pf', fire, L0_per_pos)
                sum_L0 += L0_per_pos.sum(dim=0)         # [P]

            # --- Vectorized conditional accumulation, grouped by (level, pos) ---
            # For each (level, pos) group we materialize an [B, V_g] one-hot
            # matrix over the value axis and do a single einsum into the slice
            # sum_f_cond[start:end, :, :]. Counts are shared across positions.
            batch_end = sample_cursor + B
            for group in index_layout:
                level = group['level']
                pos_g = group['position']
                values_dev = group['values_dev']  # [V_g]
                level_tensor = trees[level]
                col = (level_tensor if level_tensor.ndim == 1
                       else level_tensor[:, pos_g])
                col_batch = col[sample_cursor:batch_end].to(device).long()  # [B]

                # Map each element of col_batch to its index in values_dev.
                # Since values_dev == unique(col), every element of col_batch
                # is guaranteed to appear in values_dev somewhere, but because
                # unique() was computed over the whole tree we also know this.
                # We use == broadcasting: [B, 1] vs [1, V_g] -> [B, V_g] bool.
                onehot = (col_batch.unsqueeze(1) == values_dev.unsqueeze(0)).double()
                # onehot shape [B, V_g]

                # Accumulate conditional sums: for each v in V_g,
                #   sum_f_cond[start+v] += sum over b of onehot[b,v] * f[b]
                # einsum: 'bv,bpf->vpf'
                cond_chunk = torch.einsum('bv,bpf->vpf', onehot, f)
                sum_f_cond[group['start']:group['end']] += cond_chunk
                count_cond[group['start']:group['end']] += onehot.sum(dim=0).long()

            sample_cursor = batch_end
    finally:
        hook.remove()

    if sum_f is None:
        raise RuntimeError('No batches were processed; loader was empty.')

    # ----- Finalize statistics -----
    count_total_t = torch.tensor(float(count_total), device=device, dtype=torch.float64)
    baseline_mean = sum_f / count_total_t                      # [P, F]
    baseline_var = (sum_f2 / count_total_t) - baseline_mean ** 2
    baseline_std = baseline_var.clamp_min(0).sqrt()            # [P, F]

    cc = count_cond.double().clamp_min(1.0)                    # avoid div-by-zero
    conditional_mean = sum_f_cond / cc.view(num_targets, 1, 1) # [T, P, F]

    # Rows where count was truly 0 are set to NaN so plots can mask them.
    zero_mask = (count_cond == 0)
    if zero_mask.any():
        conditional_mean[zero_mask] = float('nan')

    delta_mean = conditional_mean - baseline_mean.unsqueeze(0)
    eps = 1e-8
    z_score = delta_mean / (baseline_std.unsqueeze(0) + eps)

    # Broadcast count_cond to [T, P] so callers can index per-position if the
    # target set ever becomes position-dependent. Right now counts depend
    # only on which tree rows are selected, not on the transformer token
    # position, so we tile along the P dimension.
    count_cond_tp = count_cond.unsqueeze(-1).expand(num_targets, num_positions).contiguous()

    result = {
        'num_samples_used': int(count_total),
        'token_positions': saved_token_positions.cpu(),
        'targets': targets,
        'baseline_mean': baseline_mean.float().cpu(),
        'baseline_std': baseline_std.float().cpu(),
        'conditional_mean': conditional_mean.float().cpu(),
        'conditional_count': count_cond_tp.cpu(),
        'delta_mean': delta_mean.float().cpu(),
        'z_score': z_score.float().cpu(),
        'decoder_norms': dec_norms.float().cpu(),
    }

    if compute_cofire:
        firing_count = sum_fire.long()                         # [P, F]
        firing_rate = sum_fire / count_total_t                 # [P, F]
        mean_cofire = sum_cofire / sum_fire.clamp_min(1.0)     # [P, F]
        mean_cofire[sum_fire == 0] = float('nan')
        L0_mean = sum_L0 / count_total_t                       # [P]
        result['firing_count'] = firing_count.cpu()
        result['firing_rate'] = firing_rate.float().cpu()
        result['mean_cofire'] = mean_cofire.float().cpu()
        result['L0_mean'] = L0_mean.float().cpu()

    return result


# ---------------------------------------------------------------------------
# Dedupe helper
# ---------------------------------------------------------------------------

def dedupe_trees(trees: dict) -> dict:
    """Drop duplicate RHM trees (keyed on trees[L]).

    Returns a new trees dict with the same L+1 keys but fewer rows. The
    returned trees are restricted to the set of unique leaf sequences; the
    corresponding rows of every other level are selected with the
    first-occurrence index of each unique leaf row, ensuring internal
    consistency across levels.
    """
    L = max(trees.keys())
    leaves = trees[L].long()                 # [N, s^L]
    # torch.unique with dim=0 does not return first-occurrence indices, so we
    # recover them manually: sort the inverse mapping and take the first index
    # that maps to each unique row.
    _, inverse = torch.unique(leaves, dim=0, return_inverse=True)
    n_unique = int(inverse.max().item()) + 1
    # first_idx[k] = smallest i such that inverse[i] == k
    first_idx = torch.full((n_unique,), -1, dtype=torch.long)
    order = torch.arange(leaves.size(0), dtype=torch.long)
    # scatter_reduce with amin gives us the first (smallest) row index per unique row.
    first_idx = first_idx.scatter_reduce(
        0, inverse.long(), order, reduce='amin', include_self=False
    )
    keep_t = torch.sort(first_idx).values   # keep trees in original order
    new_trees = {}
    for l in range(L + 1):
        t = trees[l]
        new_trees[l] = t[keep_t]
    return new_trees


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _scan_checkpoints(sweep_dir: Path):
    return sorted(sweep_dir.glob('*.pt'))


def _rhm_params_from_cfg(cfg):
    return {
        'v': int(cfg.num_features),
        'n': int(cfg.num_classes),
        'm': int(cfg.num_synonyms),
        's': int(cfg.tuple_size),
        'L': int(cfg.num_layers),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sweep_dir', type=str, default=None,
                       help='Directory containing SAE .pt checkpoints from the sweep')
    group.add_argument('--ckpt', type=str, default=None,
                       help='Path to a single SAE checkpoint .pt')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to write .feature_latent.pt artifacts into')
    parser.add_argument('--eval_size', type=int, default=32768,
                        help='Number of RHM samples for analysis (default: 32768)')
    parser.add_argument('--eval_seed', type=int, default=99999,
                        help='Random seed for the eval split (default: 99999)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for forward passes (default: 512)')
    parser.add_argument('--dedupe', action='store_true',
                        help='Drop duplicate RHM trees before analysis so '
                             'conditional expectations are exact over unique trees')
    parser.add_argument('--no_cofire', action='store_true',
                        help='Skip per-(position, feature) firing_rate, '
                             'firing_count, mean_cofire, L0_mean accumulation '
                             '(saves a bit of memory / time)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: cuda if available, else cpu)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ckpt:
        ckpt_files = [Path(args.ckpt)]
    else:
        ckpt_files = _scan_checkpoints(Path(args.sweep_dir))
        if not ckpt_files:
            print(f'No .pt files found in {args.sweep_dir}')
            sys.exit(0)

    print(f'Found {len(ckpt_files)} checkpoint file(s).')
    print(f'Eval: size={args.eval_size}, seed={args.eval_seed}, '
          f'batch={args.batch_size}, dedupe={args.dedupe}, device={device}')
    print()

    # Parse metadata without loading weights (to group by transformer source).
    records = []
    for ckpt_file in ckpt_files:
        entry = load_sae(str(ckpt_file), input_dim=None, device='cpu', load_model=False)
        if entry is None:
            print(f'  Skipping (not a valid single-layer SAE checkpoint): {ckpt_file.name}')
            continue
        records.append(entry)

    if not records:
        print('No valid SAE checkpoints found.')
        sys.exit(0)

    # Group by transformer artifact so we load each transformer only once.
    by_transformer = {}
    for r in records:
        by_transformer.setdefault(r['train_output'], []).append(r)

    for train_output_path, group in by_transformer.items():
        print(f'Loading transformer from: {train_output_path}')
        try:
            model, _loader, cfg, rules, rules_source = load_transformer(
                train_output_path, args.eval_size, args.eval_seed,
                args.batch_size, device, shuffle=False,
            )
        except Exception as exc:
            print(f'  ERROR loading transformer: {exc}')
            continue

        print(f'  Rules source: {rules_source}')
        if rules_source == 'seed_rules_resampled':
            print(
                '  WARNING: rules were regenerated from seed_rules because the '
                "transformer checkpoint has no output['rules']. The analysis "
                'is only valid if seed_rules and structural parameters match '
                'training exactly. Re-run training with --save_models to '
                'eliminate this warning.'
            )

        has_cls = hasattr(model, 'cls_token')
        rhm = _rhm_params_from_cfg(cfg)

        # Build the eval trees once; analyze_checkpoint reuses them.
        trees = sample_trees(num_data=args.eval_size, rules=rules,
                             prior=None, probs=None, seed=args.eval_seed)

        if args.dedupe:
            before = trees[rhm['L']].size(0)
            trees = dedupe_trees(trees)
            after = trees[rhm['L']].size(0)
            print(f'  Dedupe: {before} -> {after} unique trees')

        for r in group:
            ckpt_path = r['ckpt_path']
            layer_id = r['layer_id']
            mode = r['setup'].get('sae_activation_source', 'all_tokens')
            token_idx = r.get('sae_token_idx', 0)
            act_scale = r.get('act_scale', 1.0)

            print(f'  Analyzing: {Path(ckpt_path).name}  '
                  f'(layer={layer_id}, mode={mode}'
                  + (f', token={token_idx}' if mode == 'one_token' else '')
                  + f', act_scale={act_scale:.4f})')

            # Rules-source consistency check (same as eval_sweep.py).
            sae_rules_source = r.get('sae_rules_source', None)
            if sae_rules_source is not None and sae_rules_source != rules_source:
                print(
                    f'    ERROR: Rules source mismatch -- SAE trained with '
                    f"'{sae_rules_source}', transformer resolves to "
                    f"'{rules_source}'. Skipping."
                )
                continue

            try:
                entry = load_sae(ckpt_path, input_dim=model.embedding_dim, device=device)
                sae = entry['sae']
            except Exception as exc:
                print(f'    ERROR loading SAE: {exc}')
                continue

            try:
                stats = analyze_checkpoint(
                    model=model, sae=sae, trees=trees, layer_id=layer_id,
                    mode=mode, has_cls=has_cls, token_idx=token_idx,
                    act_scale=act_scale, batch_size=args.batch_size, device=device,
                    compute_cofire=not args.no_cofire,
                )
            except Exception as exc:
                print(f'    ERROR during analysis: {exc}')
                continue

            artifact = {
                'ckpt_path': str(ckpt_path),
                'layer_id': int(layer_id),
                'mode': str(mode),
                'sae_token_idx': int(token_idx),
                'act_scale': float(act_scale),
                'latent_dim': int(entry['latent_dim']),
                'embedding_dim': int(model.embedding_dim),
                'eval_size': int(args.eval_size),
                'eval_seed': int(args.eval_seed),
                'dedupe': bool(args.dedupe),
                'rhm': rhm,
                'rules_source': rules_source,
                'sae_rules_source': sae_rules_source,
                **stats,
            }
            out_path = out_dir / (Path(ckpt_path).stem + '.feature_latent.pt')
            torch.save(artifact, out_path)
            print(f'    saved: {out_path}  '
                  f'(T={len(stats["targets"])}, P={stats["token_positions"].numel()}, '
                  f'F={entry["latent_dim"]}, N={stats["num_samples_used"]})')

    print()
    print('done.')


if __name__ == '__main__':
    main()
