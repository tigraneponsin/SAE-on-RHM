"""Feature-splitting diagnostic for SAE subtree alignment.

Offline post-processing of *.sae_eval.pt artifacts. No SAE re-evaluation.

Builds on min_entropy_diag.py. That script finds, per feature, the unconstrained
whole-tree argmin latent (the "child" (l*, j*)). This script adds the missing
test: does the feature ALSO pin the child's structural PARENT (l*-1, j*//s)
beyond what the RHM tree's leak already forces? If so, the feature is really a
parent feature carved toward one child value (absorption / feature splitting),
and we REASSIGN its label up to the parent while keeping the child as a carve
tag.

Procedure (per SAE position p, per weight scheme, per feature f):
  child  = whole-tree argmin cell (l*, j*).
  parent = (l*-1, j*//s)            (root=level 0, leaf=level L; parent toward
                                     root. l*=0 has no parent -> never reassign.)
  c_i    = the child value the feature specialized to, argmax over the child
           cell's values of P(child = c | f fires) = joint_fire_count /
           firing_count (the value the feature fires most on).
  leak(child, c_i) = H(Z_parent | Z_child = c_i), the VALUE-SPECIFIC structural
                leak from the RHM rules. Same for all lambdas in a sweep ->
                computed ONCE. Indexed by (CHILD cell, value).
  leak_norm[child, c_i] = leak(child, c_i) / H_theoretical[parent].
  ratio_f = (H_per_feature[parent_gidx, p, f] / H_theoretical[parent])
            / leak_norm[child, c_i].
  reassign when ratio_f < alpha (default 0.25): the feature's selectivity on the
  parent is small relative to the value-specific leak, i.e. it pins the parent
  far beyond what the child value alone forces -> it is really a parent feature.

Plots (report-notation aware):
  - entropy vs lambda, parent / level / unconstrained-split-checked (the
    aggregate AFTER reassignment, drawn orange). The raw unconstrained-argmin
    curve is no longer drawn; the split-checked curve replaces it.
  - level-distribution vs lambda, TWO variants: unconstrained (simple argmin)
    and reassigned (after the split check), so the level evolution can be
    compared directly.

Usage:
    python scripts/sae_sweep/feature_splitting_diag.py \\
        --artifacts_dir /path/to/sweep/analysis_files \\
        [--alpha 0.25] [--out_csv ...] [--out_plot_prefix ...] [--xlim 1e-4 1]
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))                       # sibling sweep modules
sys.path.insert(0, str(_here.parent / 'common'))     # sae_loading
sys.path.insert(0, str(_here.parent.parent))         # repo root (datasets)

from notation import report_level, add_report_flag
from plot_entropy_lambda import _find_threshold_lambda
from min_entropy_diag import (weighted_aggregate, weights_for_scheme,
                              torch_nanmin, SCHEMES, sanity_check)
from sae_loading import load_sae, resolve_rules
from datasets.random_hierarchy_model import latent_prior
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_ARTIFACTS_DIR = (
    '/work/pcsl/ponsin/Mean_Transformer/Small_SAE/latent_dim_4*512/'
    'v_16_L_3_m_16_wdecay_0.0001_dropout_0_nores/'
    'sweep_alltokens_layer1_lambda1_zoom/analysis_files'
)

DEFAULT_ALPHA = 0.5
# y-limits for every entropy subplot (small margin so plateaus at 0/1 stay
# visible). Shared by make_plot and the averaged panel.
ENTROPY_YLIM = (-0.02, 1.02)


def _entropy_nats(p):
    """Shannon entropy in nats of a distribution (or batch over last dim)."""
    p64 = p.double()
    return float(-torch.special.xlogy(p64, p64).sum())


def resolve_leak_table(art, artifacts_dir):
    """Value-specific structural leak H(Z_parent | Z_child = c) for every
    (child cell, child value), from the RHM rules. Identical across the whole
    sweep, so resolve once.

    rules are NOT in the *.sae_eval.pt artifact. Chain:
      art['ckpt_path'] name -> <sweep>/sae_checkpoints/<stem>.pt (SAE ckpt)
      load_sae(..., load_model=False)['train_output']  -> transformer ckpt
      resolve_rules(torch.load(transformer), ...)       -> rules (+ source)
    resolve_rules has its own deterministic seed_rules fallback.

    Returns:
      leak_norm : {(l_child, j_child, c_val) -> float}, the value-specific leak
                  H(Z_parent | Z_child = c_val) / H_theoretical[parent]. NaN
                  where the child value is unreachable (P(child=c)=0).
      leak_rows : list of dicts for the leak CSV (raw nats + the normalizer).
      src       : rules source string ('artifact' / 'seed_rules_resampled').
    """
    rhm = art['rhm']
    n, v, s, L = int(rhm['n']), int(rhm['v']), int(rhm['s']), int(rhm['L'])

    # Locate the SAE checkpoint (needed only to recover the transformer path,
    # hence the RHM rules). The eval artifacts may sit one or more levels below
    # the sweep root (e.g. analysis_files/new_analysis/), while sae_checkpoints/
    # lives at the sweep root, so search 'sae_checkpoints/<name>' relative to
    # the artifacts dir and every ancestor, then the literal ckpt_path.
    ckpt_name = Path(art['ckpt_path']).name
    candidates = [d / 'sae_checkpoints' / ckpt_name
                  for d in [artifacts_dir] + list(artifacts_dir.parents)]
    candidates.append(Path(art['ckpt_path']))
    sae_ckpt = next((c for c in candidates if c.exists()), None)
    if sae_ckpt is None:
        raise SystemExit(
            'cannot locate SAE checkpoint to recover RHM rules; tried %s'
            % ', '.join(str(c) for c in candidates[:4] + [candidates[-1]]))
    info = load_sae(str(sae_ckpt), input_dim=None, device='cpu',
                    load_model=False)
    if not info or not info.get('train_output'):
        raise SystemExit('SAE checkpoint %s has no train_output to locate the '
                         'transformer rules' % sae_ckpt)
    train_output = info['train_output']
    if not Path(train_output).exists():
        raise SystemExit('transformer checkpoint %s (from %s) not found'
                         % (train_output, sae_ckpt))
    blob = torch.load(train_output, map_location='cpu', weights_only=False)
    rules, src = resolve_rules(blob, train_output)

    priors = latent_prior(rules, n, v)               # {level -> (s^l, V_l)}
    H_theo = art['H_theoretical']                    # {(l,p) -> float}

    leak_norm = {}
    leak_rows = []
    for l_c in range(1, L + 1):
        l_par = l_c - 1
        m = int(rules[l_par].shape[1])
        for j_c in range(s ** l_c):
            j_par = j_c // s
            c = j_c % s
            P_par = priors[l_par][j_par]              # (V_par,)
            V_next = priors[l_c].shape[1]
            oh = F.one_hot(rules[l_par][:, :, c].long(),
                           num_classes=V_next).double()      # (V_par, m, V_next)
            cond = oh.sum(dim=1) / float(m)           # (V_par, V_next) P(child|par)
            joint = P_par.unsqueeze(1) * cond         # (V_par, V_next) P(par,child)
            P_child = joint.sum(dim=0)                # (V_next,) P(child)
            href_par = H_theo.get((l_par, j_par), float('nan'))
            ok = isinstance(href_par, (int, float)) and href_par > 0 \
                and math.isfinite(href_par)
            # per-value leak H(Z_parent | Z_child = c_val), value-specific.
            for c_val in range(V_next):
                col = joint[:, c_val]                 # (V_par,) P(par, child=c_val)
                pc = float(P_child[c_val])
                if pc > 0:
                    post = col / col.sum()            # P(par | child=c_val)
                    leak = _entropy_nats(post)
                else:
                    leak = float('nan')               # unreachable child value
                ln = (leak / href_par) if (ok and math.isfinite(leak)) \
                    else float('nan')
                leak_norm[(l_c, j_c, c_val)] = ln
                leak_rows.append({
                    'child_level': l_c, 'child_position': j_c,
                    'child_value': c_val, 'parent_level': l_par,
                    'parent_position': j_par, 'P_child': pc,
                    'leak_nats': leak, 'parent_H_theoretical': float(href_par),
                    'leak_norm': ln})
    return leak_norm, leak_rows, src


def process_artifact(art, leak_norm, alpha, collect_ratios=False):
    """Per-position diagnostics for one artifact, with the split check.

    Superset of min_entropy_diag.process_artifact (whole_tree mode). Keeps the
    parent / same-level / whole-tree min curves and the unconstrained level
    distribution, and ADDS the split-checked aggregate, the reassigned level
    distribution, and the reassigned fraction.

    collect_ratios: also stash, per position and scheme, the raw per-feature
        split-check ratio with its weight and reassign-eligibility mask
        (out['ratio_<sc>'][p] = {'ratio','w','eligible'} numpy arrays). Used by
        the alpha-sweep diagnostic to threshold the SAME ratios at many alphas
        without recomputing entropies. Off by default (heavier output).
    """
    rhm = art['rhm']
    s = int(rhm['s'])
    L = int(rhm['L'])
    layer_id = int(art['layer_id'])

    pooled = (art.get('mode') == 'mean_pooled')
    matched_level = 0 if pooled else (L - 1 - layer_id)

    index_layout = art['index_layout']
    H_per_feature = art['H_per_feature']          # [num_groups, P, F]
    firing_rate = art['firing_rate']              # [P, F]
    baseline_mean = art['baseline_mean']          # [P, F]
    decoder_norms = art['decoder_norms']          # [F]
    token_positions = art['token_positions']      # [P]
    H_theoretical = art['H_theoretical']          # {(level,pos)->float}
    firing_count = art['firing_count']            # [P, F]
    joint_fire_count = art['joint_fire_count']    # [num_targets, P, F]
    targets = art['targets']                      # list of {level,position,value,..}

    # row index of value 0 for each (level,pos) target cell; values are
    # contiguous so cell (l,j) value c lives at row start_row[(l,j)] + c.
    start_row = {}
    for t_idx, t in enumerate(targets):
        start_row.setdefault((int(t['level']), int(t['position'])), t_idx)

    P = H_per_feature.shape[1]
    num_groups = H_per_feature.shape[0]

    group_index = {(int(g['level']), int(g['position'])): idx
                   for idx, g in enumerate(index_layout)}

    same_level_groups = [(int(g['level']), int(g['position']))
                         for g in index_layout if int(g['level']) == matched_level]
    if not same_level_groups:
        raise ValueError('no index_layout groups at matched_level=%d' % matched_level)
    whole_tree_groups = [(int(g['level']), int(g['position'])) for g in index_layout]
    cand_sets = {'same_level': same_level_groups, 'whole_tree': whole_tree_groups}
    cand_idxs = {name: [group_index[k] for k in groups]
                 for name, groups in cand_sets.items()}

    def _href_tensor(groups):
        vals = []
        for g in groups:
            h = H_theoretical.get(g, float('nan'))
            vals.append(h if (isinstance(h, (int, float)) and h > 0
                              and math.isfinite(h)) else float('nan'))
        return torch.tensor(vals, dtype=torch.float64)
    cand_href = {name: _href_tensor(groups)
                 for name, groups in cand_sets.items()}

    # whole-tree group cells as level/pos tensors, aligned to that candidate
    # stack, so each feature's argmin row maps straight to its child cell.
    wt_idxs = cand_idxs['whole_tree']
    wt_level = torch.tensor([whole_tree_groups[r][0] for r in range(len(wt_idxs))],
                            dtype=torch.long)
    wt_pos = torch.tensor([whole_tree_groups[r][1] for r in range(len(wt_idxs))],
                          dtype=torch.long)

    # value-specific leak lookup is per (child cell, child value), and the value
    # c_i depends on the feature AND the SAE position (the value it fires most
    # on), so it is resolved inside the position loop, not statically here.
    V_child = {}                                  # (level,pos) -> num values
    for g in index_layout:
        V_child[(int(g['level']), int(g['position']))] = len(g['values'])

    def _value_specific_leakN(child_level_t, child_pos_t, p_idx):
        """[F] normalized leak H(parent|child=c_i)/H_theo[parent], with c_i the
        value each feature fires most on at this position. NaN where the child
        is the root (no parent) or the value/leak is undefined."""
        F_n = child_level_t.shape[0]
        out_ln = torch.full((F_n,), float('nan'), dtype=torch.float64)
        feat_ar = torch.arange(F_n)
        # c_i = argmax over the child cell's value rows of joint_fire_count.
        # Group features by their argmin child cell so each gather is one slice.
        cells = {(int(child_level_t[f]), int(child_pos_t[f])) for f in range(F_n)}
        for (l_c, j_c) in cells:
            if l_c < 1:                           # root child: no parent leak
                continue
            sel = (child_level_t == l_c) & (child_pos_t == j_c)
            feats = feat_ar[sel]
            sr = start_row.get((l_c, j_c))
            nv = V_child.get((l_c, j_c))
            if sr is None or nv is None:
                continue
            jf = joint_fire_count[sr:sr + nv, p_idx, :]      # (nv, F)
            c_i = jf[:, feats].argmax(dim=0)                 # (n_sel,) value
            for k, f in enumerate(feats.tolist()):
                out_ln[f] = leak_norm.get((l_c, j_c, int(c_i[k])), float('nan'))
        return out_ln

    # per-feature gather of H at an arbitrary cell needs a (level,pos)->group row
    # map. A no-parent sentinel row (-1) is masked out before gathering.
    def _cell_to_grow(level_t, pos_t):
        """[F] long: group-index of each (level,pos), or -1 if absent."""
        rows = torch.full_like(level_t, -1)
        for k, gi in group_index.items():
            m = (level_t == k[0]) & (pos_t == k[1])
            rows[m] = gi
        return rows

    out = {
        'lambda_l1': float(art.get('lambda_l1') or 0.0),
        'layer_id': layer_id, 'P': P, 's': s, 'L': L,
        'mode': art.get('mode', ''), 'pooled': pooled,
        'matched_level': matched_level,
        'positions': [], 'parent_group': [],
        'stored_H_bar_fire': [],
    }
    for sc in SCHEMES:
        out['H_bar_%s' % sc] = []
        out['H_bar_%s_norm' % sc] = []
        for cand in cand_sets:
            out['H_bar_min_%s_%s' % (sc, cand)] = []
            out['H_bar_min_%s_%s_norm' % (sc, cand)] = []
        out['H_bar_splitcheck_%s_norm' % sc] = []
        out['reassign_frac_%s' % sc] = []
        out['target_dist_wt_%s' % sc] = []
        out['target_dist_reassigned_%s' % sc] = []
        if collect_ratios:
            out['ratio_%s' % sc] = []

    stored_fire = art.get('H_bar_fire')

    for p_idx in range(P):
        p_real = int(token_positions[p_idx].item())
        if pooled:
            parent_key = (0, 0)
        else:
            matched_j = p_real // (s ** (1 + layer_id))
            parent_key = (matched_level, matched_j)
        parent_gidx = group_index.get(parent_key)
        out['positions'].append(p_real)
        out['parent_group'].append(parent_key)
        out['stored_H_bar_fire'].append(
            float(stored_fire[p_idx]) if stored_fire is not None else float('nan'))

        # per-candidate min and argmin (NaN-aware; dead features stay NaN).
        H_min = {}
        argmins = {}
        for cand, gidxs in cand_idxs.items():
            H_cand = H_per_feature[gidxs, p_idx, :]              # [G, F]
            mn, am = torch_nanmin(H_cand, dim=0)                 # [F], [F]
            H_min[cand] = mn
            argmins[cand] = am
        H_parent = H_per_feature[parent_gidx, p_idx, :]          # [F]

        H_ref = H_theoretical.get(parent_key, float('nan'))
        ref_ok = (H_ref > 0) and math.isfinite(H_ref)

        # --- split check (whole-tree child -> its tree-parent) -------------
        wt_am = argmins['whole_tree']                            # [F] rows
        child_level = wt_level[wt_am]                            # [F]
        child_pos = wt_pos[wt_am]                                # [F]
        # value-specific leak: condition on the value c_i each feature fires
        # most on (argmax joint_fire_count over the child cell's values).
        leakN = _value_specific_leakN(child_level, child_pos, p_idx)  # [F] norm
        can_reassign = child_level >= 1                          # root has no parent
        par_level = child_level - 1
        par_pos = child_pos // s
        par_grow = _cell_to_grow(par_level, par_pos)             # [F] (-1 = none)
        # gather H at each feature's parent cell; masked rows use row 0 then NaN
        safe_grow = par_grow.clamp_min(0)
        H_par_at = H_per_feature[safe_grow, p_idx,
                                 torch.arange(H_per_feature.shape[2])]  # [F]
        H_par_at = torch.where(par_grow >= 0, H_par_at,
                               torch.full_like(H_par_at, float('nan')))
        # H_theoretical at each parent cell (per feature)
        href_par = torch.tensor(
            [H_theoretical.get((int(par_level[f]), int(par_pos[f])), float('nan'))
             for f in range(par_level.shape[0])], dtype=torch.float64)
        href_par = torch.where(href_par > 0, href_par,
                               torch.full_like(href_par, float('nan')))
        ratio = (H_par_at.double() / href_par) / leakN          # [F]
        reassign = can_reassign & torch.isfinite(ratio) & (ratio < alpha)

        # final assigned cell per feature: parent if reassigned else child.
        child_href = cand_href['whole_tree'][wt_am]             # [F]
        final_entropy = torch.where(reassign, H_par_at.double(),
                                    H_min['whole_tree'].double())
        final_href = torch.where(reassign, href_par, child_href)
        final_level = torch.where(reassign, par_level, child_level)

        for sc in SCHEMES:
            w = weights_for_scheme(sc, firing_rate[p_idx], baseline_mean[p_idx],
                                   decoder_norms)
            h_par = weighted_aggregate(H_parent, w)
            out['H_bar_%s' % sc].append(float(h_par))
            out['H_bar_%s_norm' % sc].append(
                float(h_par) / H_ref if ref_ok else float('nan'))

            for cand in cand_sets:
                h_min = weighted_aggregate(H_min[cand], w)
                out['H_bar_min_%s_%s' % (sc, cand)].append(float(h_min))
                href_feat = cand_href[cand][argmins[cand]]
                h_min_norm = weighted_aggregate(
                    H_min[cand] / href_feat.clamp_min(1e-30), w)
                out['H_bar_min_%s_%s_norm' % (sc, cand)].append(float(h_min_norm))

            # split-checked aggregate: normalize EACH feature by its FINAL
            # cell's H_theoretical, then aggregate (mirrors the whole_tree_norm
            # block; a single H_ref would be wrong when levels differ).
            h_split = weighted_aggregate(
                final_entropy / final_href.clamp_min(1e-30), w)
            out['H_bar_splitcheck_%s_norm' % sc].append(float(h_split))

            # weighted fraction of live features reassigned.
            live = torch.isfinite(H_min['whole_tree']) & (w > 0)
            w_live = torch.where(live, w.double(), torch.zeros_like(w.double()))
            tot = w_live.sum()
            re_w = torch.where(live & reassign, w.double(),
                               torch.zeros_like(w.double()))
            out['reassign_frac_%s' % sc].append(
                float(re_w.sum() / tot.clamp_min(1e-30)) if tot > 0 else float('nan'))

            if collect_ratios:
                # raw ratios for the alpha sweep: weighted denominator is all
                # live features (tot), numerator at a given alpha is the live &
                # reassign-eligible features with ratio < alpha. Store ratio, w,
                # and the live/eligible masks so the sweep just thresholds.
                eligible = (live & can_reassign & torch.isfinite(ratio))
                out['ratio_%s' % sc].append({
                    'ratio': ratio.cpu().numpy().copy(),
                    'w': w.double().cpu().numpy().copy(),
                    'live': live.cpu().numpy().copy(),
                    'eligible': eligible.cpu().numpy().copy()})

            # unconstrained child-level distribution (per level).
            out['target_dist_wt_%s' % sc].append(
                _level_votes(child_level, w, live, tot))
            # reassigned distribution: same features, voting at their FINAL
            # level (only difference: reassigned features moved up one level).
            out['target_dist_reassigned_%s' % sc].append(
                _level_votes(final_level, w, live, tot))

    return out


def _level_votes(level_per_feature, w, live, tot):
    """{level -> weighted share} over live features (sums to ~1 when tot>0)."""
    if tot <= 0 or not live.any():
        return {}
    lv = level_per_feature[live].long()
    ww = w.double()[live]
    votes = {}
    for level in torch.unique(lv).tolist():
        votes[int(level)] = float(ww[lv == level].sum() / tot)
    return votes


def write_csv(path, diags):
    cols = ['file', 'lambda_l1', 'layer_id', 'matched_level', 'position',
            'parent_group']
    for sc in SCHEMES:
        cols += ['H_bar_%s' % sc,
                 'H_bar_min_%s_same_level' % sc,
                 'H_bar_min_%s_whole_tree' % sc,
                 'H_bar_%s_norm' % sc,
                 'H_bar_min_%s_same_level_norm' % sc,
                 'H_bar_min_%s_whole_tree_norm' % sc,
                 'H_bar_splitcheck_%s_norm' % sc,
                 'reassign_frac_%s' % sc]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for fname, d in diags:
            for p_idx in range(d['P']):
                row = [fname, d['lambda_l1'], d['layer_id'], d['matched_level'],
                       d['positions'][p_idx], '%d,%d' % d['parent_group'][p_idx]]
                for sc in SCHEMES:
                    row += [d['H_bar_%s' % sc][p_idx],
                            d['H_bar_min_%s_same_level' % sc][p_idx],
                            d['H_bar_min_%s_whole_tree' % sc][p_idx],
                            d['H_bar_%s_norm' % sc][p_idx],
                            d['H_bar_min_%s_same_level_norm' % sc][p_idx],
                            d['H_bar_min_%s_whole_tree_norm' % sc][p_idx],
                            d['H_bar_splitcheck_%s_norm' % sc][p_idx],
                            d['reassign_frac_%s' % sc][p_idx]]
                w.writerow(row)


def write_leak_csv(path, leak_rows):
    cols = ['child_level', 'child_position', 'child_value', 'parent_level',
            'parent_position', 'P_child', 'leak_nats', 'parent_H_theoretical',
            'leak_norm']
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in leak_rows:
            w.writerow(r)


def write_target_dist_csv(path, diags):
    """Long format, one row per (file, position, scheme, variant, target_level).

    variant in {unconstrained, reassigned}. 'share' is the weighted fraction of
    live features whose argmin (unconstrained) or final-assigned (reassigned)
    cell lands on that level.
    """
    cols = ['file', 'lambda_l1', 'layer_id', 'matched_level', 'position',
            'parent_group', 'scheme', 'variant', 'target_level', 'share']
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for fname, d in diags:
            for p_idx in range(d['P']):
                pg = d['parent_group'][p_idx]
                for sc in SCHEMES:
                    for variant, key in (('unconstrained', 'target_dist_wt_%s'),
                                         ('reassigned', 'target_dist_reassigned_%s')):
                        dist = d[key % sc][p_idx]
                        for level, share in sorted(dist.items()):
                            w.writerow([fname, d['lambda_l1'], d['layer_id'],
                                        d['matched_level'], d['positions'][p_idx],
                                        '%d,%d' % pg, sc, variant, level, share])


def _thr_label(tolerance):
    return '%g%% err onset' % (tolerance * 100)


LEVEL_PALETTE = ('#4daf4a', '#984ea3', '#a65628', '#f781bf', '#17becf',
                 '#bcbd22', '#000000')


def make_level_dist_plot(diags, out_prefix, variant, scheme='fire', xlim=None,
                         report=False, threshold_lambda=None, tolerance=0.01):
    """Per-position level-share distribution vs lambda for one variant.

    variant 'unconstrained' uses the simple whole-tree argmin levels;
    'reassigned' uses the split-checked final levels. Curves sum to ~1.
    """
    dkey = {'unconstrained': 'target_dist_wt_%s' % scheme,
            'reassigned': 'target_dist_reassigned_%s' % scheme}[variant]
    diags_sorted = sorted(diags, key=lambda fd: fd[1]['lambda_l1'])
    lambdas = [d['lambda_l1'] for _, d in diags_sorted]
    P = diags_sorted[0][1]['P']
    L = diags_sorted[0][1]['L']

    def _lvl(code_level):
        return report_level(code_level, L) if report else code_level

    levels = sorted({lvl for _, d in diags_sorted for p in range(P)
                     for lvl in d[dkey][p]})
    level_color = {lvl: LEVEL_PALETTE[i % len(LEVEL_PALETTE)]
                   for i, lvl in enumerate(levels)}
    pooled = diags_sorted[0][1].get('pooled', False)

    ncol = min(4, P)
    nrow = math.ceil(P / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    axes = axes.flatten()

    for p_idx in range(P):
        ax = axes[p_idx]
        pg = diags_sorted[0][1]['parent_group'][p_idx]
        for lvl in levels:
            ys = []
            for _, d in diags_sorted:
                dist = d[dkey][p_idx]
                ys.append(dist.get(lvl, float('nan')) if dist else float('nan'))
            ax.plot(lambdas, ys, 'o-', label='level %d' % _lvl(lvl),
                    color=level_color[lvl])
        ax.set_xscale('log')
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_ylim(-0.05, 1.05)
        if threshold_lambda is not None:
            ax.axvline(threshold_lambda, color='black', linestyle='--',
                       linewidth=0.8, alpha=0.7,
                       label=(_thr_label(tolerance) if p_idx == 0 else None))
        ax.set_title('pooled readout (root %d,%d)' % (_lvl(pg[0]), pg[1])
                     if pooled
                     else 'pos %d (parent %d,%d)' % (p_idx, _lvl(pg[0]), pg[1]),
                     fontsize=9)
        ax.set_xlabel(r'$\lambda$', fontsize=12)
        ax.set_ylabel('share', fontsize=12)
        if p_idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    for k in range(P, len(axes)):
        axes[k].axis('off')

    title = ('Unconstrained argmin' if variant == 'unconstrained'
             else 'Split-check reassigned')
    fig.suptitle('%s target level distribution by position' % title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path('%s_leveldist_%s_%s.png' % (out_prefix, variant, scheme))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def make_plot(diags, out_prefix, scheme='fire', xlim=None, report=False,
              plot_avg=False, threshold_lambda=None, tolerance=0.01):
    """Per-position weighted entropy eta-bar vs lambda. Three curves per
    subplot: parent-constrained, level-constrained, and the unconstrained
    whole-tree min BEFORE any reassignment (drawn orange)."""
    diags_sorted = sorted(diags, key=lambda fd: fd[1]['lambda_l1'])
    lambdas = [d['lambda_l1'] for _, d in diags_sorted]
    P = diags_sorted[0][1]['P']
    L = diags_sorted[0][1]['L']

    def _lvl(code_level):
        return report_level(code_level, L) if report else code_level
    total_panels = P + (1 if plot_avg else 0)
    ncol = min(4, total_panels)
    nrow = math.ceil(total_panels / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    axes = axes.flatten()

    pooled = diags_sorted[0][1].get('pooled', False)
    par_key = 'H_bar_%s_norm' % scheme
    same_key = 'H_bar_min_%s_same_level_norm' % scheme
    whole_key = 'H_bar_min_%s_whole_tree_norm' % scheme
    # orange curve: unconstrained whole-tree min, BEFORE reassignment.
    if pooled:
        curves = [(par_key, 'o-', 'C0', 'root constrained'),
                  (whole_key, '^-', 'orange', 'no constraint')]
    else:
        curves = [(par_key, 'o-', 'C0', 'parent constrained'),
                  (same_key, 's-', 'C3', 'level constrained'),
                  (whole_key, '^-', 'orange', 'no constraint')]

    for p_idx in range(P):
        ax = axes[p_idx]
        pg = diags_sorted[0][1]['parent_group'][p_idx]
        for key, style, color, label in curves:
            ys = [d[key][p_idx] for _, d in diags_sorted]
            ax.plot(lambdas, ys, style, color=color, label=label)
        ax.set_xscale('log')
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_ylim(*ENTROPY_YLIM)
        if threshold_lambda is not None:
            ax.axvline(threshold_lambda, color='black', linestyle='--',
                       linewidth=0.8, alpha=0.7,
                       label=(_thr_label(tolerance) if p_idx == 0 else None))
        ax.set_title('pooled readout (root %d,%d)' % (_lvl(pg[0]), pg[1])
                     if pooled
                     else 'pos %d (parent %d,%d)' % (p_idx, _lvl(pg[0]), pg[1]),
                     fontsize=9)
        ax.set_xlabel(r'$\lambda$', fontsize=12)
        ax.set_ylabel(r'$\bar{\eta}$', fontsize=14)
        ax.tick_params(axis='y', labelsize=10)
        if p_idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    if plot_avg:
        ax = axes[P]
        for key, style, color, label in curves:
            avg = [np.nanmean([d[key][p] for p in range(P)])
                   for _, d in diags_sorted]
            ax.plot(lambdas, avg, style, color=color, label=label)
        ax.set_xscale('log')
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_ylim(*ENTROPY_YLIM)
        if threshold_lambda is not None:
            ax.axvline(threshold_lambda, color='black', linestyle='--',
                       linewidth=0.8, alpha=0.7, label=_thr_label(tolerance))
        ax.set_title('avg over positions', fontsize=9)
        ax.set_xlabel(r'$\lambda$', fontsize=12)
        ax.set_ylabel(r'$\bar{\eta}$', fontsize=14)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(fontsize=7, loc='upper left')

    for k in range(total_panels, len(axes)):
        axes[k].axis('off')

    fig.suptitle(r'Parent / level / unconstrained weighted entropy $\bar{\eta}$')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path('%s_%s.png' % (out_prefix, scheme))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def splitcheck_sanity(diag, scheme='fire', tol=1e-6):
    """Assert the split-checked aggregate is at least the unconstrained min, and
    each reassigned/unconstrained level distribution sums to ~1 when live.

    Only whole_tree <= split is a true invariant: reassignment moves a feature
    from its child to that child's tree-parent (entropy can only rise or stay).
    Split is NOT bounded above by the position's NOMINAL parent curve -- a
    reassigned feature lands on its own child's parent cell, which may differ
    from (matched_level, matched_j), so split can exceed the nominal-parent
    aggregate. We therefore do not assert split <= parent.
    """
    for p_idx in range(diag['P']):
        whole = diag['H_bar_min_%s_whole_tree_norm' % scheme][p_idx]
        split = diag['H_bar_splitcheck_%s_norm' % scheme][p_idx]
        if math.isfinite(whole) and math.isfinite(split):
            assert whole <= split + tol, (
                'pos %d: split %.6f < whole_tree %.6f' % (p_idx, split, whole))
        for key in ('target_dist_wt_%s' % scheme,
                    'target_dist_reassigned_%s' % scheme):
            dist = diag[key][p_idx]
            if dist:
                total = sum(dist.values())
                assert abs(total - 1.0) < 1e-4, (
                    'pos %d %s: level shares sum to %.6f' % (p_idx, key, total))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifacts_dir', default=DEFAULT_ARTIFACTS_DIR)
    ap.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                    help='reassign a feature from child to parent when '
                         'ratio = H_norm(parent|f)/leak_norm < alpha '
                         '(default %g)' % DEFAULT_ALPHA)
    ap.add_argument('--out_csv', default=None)
    ap.add_argument('--out_plot_prefix', default=None)
    ap.add_argument('--xlim', nargs=2, type=float, default=None)
    ap.add_argument('--plot_schemes', nargs='+', choices=SCHEMES,
                    default=['raw'],
                    help='subset of weight schemes to plot (default: raw only). '
                         'CSVs always contain all schemes.')
    ap.add_argument('--plot_avg', action='store_true',
                    help='add an extra "avg over positions" panel to the '
                         'entropy plot.')
    ap.add_argument('--csv_path', default=None,
                    help='Path to sweep_metrics.csv for the error-onset '
                         'threshold lambda. Default: <artifacts_dir>/sweep_metrics.csv.')
    ap.add_argument('--err_tolerance', type=float, default=0.01,
                    help='Tolerance on (norm_err - baseline_err) for the '
                         'threshold lambda. Default: 0.01 (1%%).')
    add_report_flag(ap)
    args = ap.parse_args()

    adir = Path(args.artifacts_dir)
    files = sorted(adir.glob('*.sae_eval.pt'))
    if not files:
        raise SystemExit('no *.sae_eval.pt files in %s' % adir)

    out_csv = (Path(args.out_csv) if args.out_csv
               else adir / 'feature_splitting_diag.csv')
    out_prefix = (args.out_plot_prefix
                  if args.out_plot_prefix
                  else str(adir.parent / 'analysis_plots'
                           / 'feature_splitting_diag'))

    needed = ('H_per_feature', 'index_layout', 'firing_rate',
              'baseline_mean', 'decoder_norms', 'token_positions',
              'H_theoretical', 'rhm', 'layer_id',
              'firing_count', 'joint_fire_count', 'targets')

    # resolve the structural leak ONCE from the first usable artifact.
    leak_norm = None
    leak_rows = None
    diags = []
    skipped = []
    for f in files:
        art = torch.load(f, map_location='cpu', weights_only=False)
        if any(k not in art or art[k] is None for k in needed):
            skipped.append(f.name)
            continue
        if leak_norm is None:
            leak_norm, leak_rows, src = resolve_leak_table(art, adir)
            print('resolved RHM rules (source: %s); value-specific leak table '
                  'over %d (child cell, value) entries' % (src, len(leak_norm)))
        d = process_artifact(art, leak_norm, args.alpha)
        sanity_check(art, d)
        splitcheck_sanity(d)
        diags.append((f.name, d))

    if not diags:
        raise SystemExit('no usable artifacts (skipped %d missing keys)'
                         % len(skipped))

    write_csv(out_csv, diags)
    print('wrote %s (%d artifacts x %d positions)'
          % (out_csv, len(diags), diags[0][1]['P']))
    leak_csv = out_csv.with_name(out_csv.stem + '_leak_table.csv')
    write_leak_csv(leak_csv, leak_rows)
    print('wrote %s' % leak_csv)
    dist_csv = out_csv.with_name(out_csv.stem + '_target_dist.csv')
    write_target_dist_csv(dist_csv, diags)
    print('wrote %s' % dist_csv)
    if skipped:
        print('skipped %d artifacts missing required keys' % len(skipped))

    csv_path = args.csv_path or str(adir / 'sweep_metrics.csv')
    layer0 = diags[0][1]['layer_id']
    mode0 = diags[0][1]['mode']
    threshold_lambda = _find_threshold_lambda(
        csv_path, layer0, mode0, args.err_tolerance)

    for sc in args.plot_schemes:
        p = make_plot(diags, out_prefix, scheme=sc, xlim=args.xlim,
                      report=args.report_notation, plot_avg=args.plot_avg,
                      threshold_lambda=threshold_lambda,
                      tolerance=args.err_tolerance)
        print('wrote %s' % p)
        for variant in ('unconstrained', 'reassigned'):
            p = make_level_dist_plot(diags, out_prefix, variant, scheme=sc,
                                     xlim=args.xlim, report=args.report_notation,
                                     threshold_lambda=threshold_lambda,
                                     tolerance=args.err_tolerance)
            print('wrote %s' % p)

    # console summary at the lambda closest to 0.01.
    target = min(diags, key=lambda fd: abs(fd[1]['lambda_l1'] - 0.01))
    fn, d = target
    print('\nsummary at lambda_1=%.4g (%s), alpha=%g:'
          % (d['lambda_l1'], fn, args.alpha))
    print('%4s %10s %9s %9s %9s %9s %7s' %
          ('pos', 'parent', 'H_par', 'H_tree', 'H_split', 'reass', ''))
    for p_idx in range(d['P']):
        pg = d['parent_group'][p_idx]
        print('%4d %10s %9.4f %9.4f %9.4f %9.3f' %
              (d['positions'][p_idx], '%d,%d' % pg,
               d['H_bar_fire_norm'][p_idx],
               d['H_bar_min_fire_whole_tree_norm'][p_idx],
               d['H_bar_splitcheck_fire_norm'][p_idx],
               d['reassign_frac_fire'][p_idx]))


if __name__ == '__main__':
    main()
