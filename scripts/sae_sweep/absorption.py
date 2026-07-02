"""Shared feature-absorption core for SAE feature labeling.

This module holds the per-feature labeling math used by BOTH the sweep
diagnostics (feature_splitting_diag.py, feature_splitting_alpha_sweep.py) and
the circuit-tracing node labeler (circuit_tracing/labels.py). It computes, for
each (SAE position p, feature f) of one *.sae_eval.pt artifact, the latent label
under four schemes:

  parent      : the block-aligned parent latent cell (L-1-k, p//s^(1+k)).
                Value = argmax_v P(value | f fires) at that cell. No argmin.
  level       : argmin entropy over the matched-level cells, then value =
                argmax_v P(value | f fires) at the winning cell.
  whole_tree  : argmin entropy over ALL (level, position) cells (the
                unconstrained "child"), then value at the winning cell.
  reassigned  : whole_tree, but when the absorption ratio < alpha the primary
                label is bumped UP to the child's tree-parent cell
                (l*-1, j*//s) and its value; the child cell/value is kept as a
                precision tag.

The absorption ratio (value-specific leak test) is

    ratio_f = H(Z_parent | f fires) / H(Z_parent | Z_child = c_i)

with c_i the value f fires most on at its child cell. Reassign when
ratio_f < alpha. Both entropies are normalized by H_theoretical[parent] in the
implementation (the normalizer cancels), so the stored ratio is the quotient of
the two raw conditional entropies above.

The core returns PER-FEATURE [F]-shaped tensors and does NOT apply any
weight-scheme aggregation: weighting is an aggregation-only concept (it never
enters the reassign decision), so callers that want a layer summary apply
weights afterward, while circuit tracing reads the per-feature labels directly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure the repo root is importable so `scripts.*` / `datasets.*` resolve when
# this module is imported by a loose script or via `-m`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _entropy_nats(p):
    """Shannon entropy in nats of a distribution (or batch over last dim)."""
    p64 = p.double()
    return float(-torch.special.xlogy(p64, p64).sum())


def _nanmin(x, dim):
    """NaN-aware min + argmin along dim; delegates to entropy_core.torch_nanmin."""
    from scripts.sae_sweep.entropy_core import torch_nanmin
    return torch_nanmin(x, dim=dim)


# ---------------------------------------------------------------------------
# Structural leak table (RHM rules)
# ---------------------------------------------------------------------------

def resolve_leak_table(art, artifacts_dir, *, load_sae=None, resolve_rules=None,
                       latent_prior=None):
    """Value-specific structural leak H(Z_parent | Z_child = c) for every
    (child cell, child value), from the RHM rules. Identical across a whole
    sweep, so resolve once.

    rules are NOT in the *.sae_eval.pt artifact. Chain:
      art['ckpt_path'] name -> <sweep>/sae_checkpoints/<stem>.pt (SAE ckpt)
      load_sae(..., load_model=False)['train_output']  -> transformer ckpt
      resolve_rules(torch.load(transformer), ...)       -> rules (+ source)

    The three helpers (load_sae, resolve_rules, latent_prior) are injected so
    this module does not hard-depend on the sae_loading / datasets import paths;
    callers pass them in. They default to the standard implementations when the
    imports are available.

    Returns:
      leak_norm : {(l_child, j_child, c_val) -> float}, the value-specific leak
                  H(Z_parent | Z_child = c_val) / H_theoretical[parent]. NaN
                  where the child value is unreachable (P(child=c)=0).
      leak_rows : list of dicts for the leak CSV (raw nats + the normalizer).
      src       : rules source string ('artifact' / 'seed_rules_resampled').
    """
    if load_sae is None:
        from scripts.common.sae_loading import load_sae as load_sae
    if resolve_rules is None:
        from scripts.common.sae_loading import resolve_rules as resolve_rules
    if latent_prior is None:
        from datasets.random_hierarchy_model import latent_prior as latent_prior

    artifacts_dir = Path(artifacts_dir)
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

    leak_norm, leak_rows = leak_table_from_rules(
        rules, art['rhm'], art['H_theoretical'], latent_prior=latent_prior)
    return leak_norm, leak_rows, src


def leak_table_from_rules(rules, rhm, H_theoretical, *, latent_prior=None):
    """Value-specific structural leak from RHM rules already in hand.

    Pure-math half of resolve_leak_table: no SAE/transformer checkpoint walking.
    Circuit tracing already loads the rules (and checks rules-consistency), so it
    feeds them here directly instead of re-resolving from the eval artifact's
    ckpt_path. The sweep path keeps using resolve_leak_table (which has no rules
    in hand) and delegates the math here.

    rules        : list of per-level rule tensors (rules[l] shape (V, m, s)).
    rhm          : {'n','v','s','L', ...}.
    H_theoretical: {(level, pos) -> float} from the eval artifact.

    Returns (leak_norm, leak_rows) -- see resolve_leak_table for the schema.
    """
    if latent_prior is None:
        from datasets.random_hierarchy_model import latent_prior as latent_prior

    n, v, s, L = int(rhm['n']), int(rhm['v']), int(rhm['s']), int(rhm['L'])
    priors = latent_prior(rules, n, v)               # {level -> (s^l, V_l)}
    H_theo = H_theoretical                           # {(l,p) -> float}

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
    return leak_norm, leak_rows


# ---------------------------------------------------------------------------
# Per-feature labeling core (no weighting)
# ---------------------------------------------------------------------------

class _ArtifactTables:
    """Pre-resolved per-artifact lookups shared across positions.

    Holds the cell-set / group-index / value-row maps and the small closures
    (_value_specific_leakN, _cell_to_grow, value_at_cell) that the per-position
    decision needs. Built once per artifact in per_feature_labels.
    """

    def __init__(self, art, leak_norm):
        rhm = art['rhm']
        self.s = int(rhm['s'])
        self.L = int(rhm['L'])
        self.layer_id = int(art['layer_id'])
        self.pooled = (art.get('mode') == 'mean_pooled')
        self.matched_level = 0 if self.pooled else (self.L - 1 - self.layer_id)

        self.index_layout = art['index_layout']
        self.H_per_feature = art['H_per_feature']        # [num_groups, P, F]
        self.token_positions = art['token_positions']    # [P]
        self.H_theoretical = art['H_theoretical']        # {(level,pos)->float}
        self.firing_count = art['firing_count']          # [P, F]
        self.joint_fire_count = art['joint_fire_count']  # [num_targets, P, F]
        self.targets = art['targets']
        self.leak_norm = leak_norm

        self.P = self.H_per_feature.shape[1]
        self.F = self.H_per_feature.shape[2]

        # row index of value 0 for each (level,pos) target cell; values are
        # contiguous so cell (l,j) value c lives at row start_row[(l,j)] + c.
        self.start_row = {}
        for t_idx, t in enumerate(self.targets):
            self.start_row.setdefault(
                (int(t['level']), int(t['position'])), t_idx)

        self.group_index = {(int(g['level']), int(g['position'])): idx
                            for idx, g in enumerate(self.index_layout)}

        same_level_groups = [
            (int(g['level']), int(g['position']))
            for g in self.index_layout if int(g['level']) == self.matched_level]
        if not same_level_groups:
            raise ValueError('no index_layout groups at matched_level=%d'
                             % self.matched_level)
        whole_tree_groups = [(int(g['level']), int(g['position']))
                             for g in self.index_layout]
        self.cand_sets = {'same_level': same_level_groups,
                          'whole_tree': whole_tree_groups}
        self.cand_idxs = {name: [self.group_index[k] for k in groups]
                          for name, groups in self.cand_sets.items()}
        self.cand_href = {name: self._href_tensor(groups)
                          for name, groups in self.cand_sets.items()}

        # whole-tree group cells as level/pos tensors, aligned to that candidate
        # stack, so each feature's argmin row maps straight to its child cell.
        wt_idxs = self.cand_idxs['whole_tree']
        self.wt_level = torch.tensor(
            [whole_tree_groups[r][0] for r in range(len(wt_idxs))],
            dtype=torch.long)
        self.wt_pos = torch.tensor(
            [whole_tree_groups[r][1] for r in range(len(wt_idxs))],
            dtype=torch.long)

        self.V_child = {}                             # (level,pos) -> num values
        # index_layout-based value-row map, the sparse-safe source: cell (l,j)
        # occupies rows start:end of joint_fire_count, and values[r] is the
        # actual vocab value at local row r (the observed subset, possibly
        # sparse). dist_at_cell uses this rather than the contiguous start_row
        # assumption so it is correct when a cell's observed values are a strict
        # subset of the vocab.
        self.cell_rows = {}                           # (level,pos) -> (start, end, values LongTensor)
        for g in self.index_layout:
            key = (int(g['level']), int(g['position']))
            self.V_child[key] = len(g['values'])
            self.cell_rows[key] = (int(g['start']), int(g['end']),
                                   g['values'].clone().long())

    def _href_tensor(self, groups):
        vals = []
        for g in groups:
            h = self.H_theoretical.get(g, float('nan'))
            vals.append(h if (isinstance(h, (int, float)) and h > 0
                              and math.isfinite(h)) else float('nan'))
        return torch.tensor(vals, dtype=torch.float64)

    def value_at_cell(self, level_t, pos_t, p_idx):
        """[F] long: argmax_v P(value | f fires) at each feature's (level,pos).

        P(value|fire) = joint_fire_count[value_row, p, f] / firing_count[p, f];
        the firing_count divisor is constant across value rows, so the argmax
        over values reduces to argmax of joint_fire_count over the cell's value
        rows. Returns -1 where the cell is absent or has no value rows.
        """
        F_n = level_t.shape[0]
        out_val = torch.full((F_n,), -1, dtype=torch.long)
        feat_ar = torch.arange(F_n)
        cells = {(int(level_t[f]), int(pos_t[f])) for f in range(F_n)}
        for (l_, j_) in cells:
            sel = (level_t == l_) & (pos_t == j_)
            feats = feat_ar[sel]
            sr = self.start_row.get((l_, j_))
            nv = self.V_child.get((l_, j_))
            if sr is None or nv is None or nv == 0:
                continue
            jf = self.joint_fire_count[sr:sr + nv, p_idx, :]   # (nv, F)
            am = jf[:, feats].argmax(dim=0)                    # (n_sel,) value
            out_val[feats] = am.long()
        return out_val

    def dist_at_cell(self, level_t, pos_t, p_idx):
        """Per-feature conditional P(value | f fires) at each feature's cell.

        Returns a dict feat_idx -> (values LongTensor, prob FloatTensor) for the
        features whose cell exists and that fired (firing_count > 0). Features
        with an absent cell or zero firing are omitted (callers treat them as
        having no distribution). `values` is the cell's observed value subset
        (from index_layout, possibly sparse) and `prob` the matching conditional
        column; the prob vector sums to <= 1 (== 1 when every value of the cell
        was observed for that feature), matching the cond_prob convention.
        """
        F_n = level_t.shape[0]
        out = {}
        feat_ar = torch.arange(F_n)
        cells = {(int(level_t[f]), int(pos_t[f])) for f in range(F_n)}
        for (l_, j_) in cells:
            sel = (level_t == l_) & (pos_t == j_)
            feats = feat_ar[sel]
            rows = self.cell_rows.get((l_, j_))
            if rows is None:
                continue
            st, en, values = rows
            if en <= st:
                continue
            jf = self.joint_fire_count[st:en, p_idx, :].double()  # (V_g, F)
            for f in feats.tolist():
                fc = float(self.firing_count[p_idx, f].item())
                if fc <= 0:
                    continue
                col = jf[:, f] / fc                          # (V_g,) P(value|fire)
                out[int(f)] = (values.clone(), col)          # col stays float64
        return out

    def _value_specific_leakN(self, child_level_t, child_pos_t, p_idx):
        """[F] normalized leak H(parent|child=c_i)/H_theo[parent], with c_i the
        value each feature fires most on at this position. NaN where the child
        is the root (no parent) or the value/leak is undefined."""
        F_n = child_level_t.shape[0]
        out_ln = torch.full((F_n,), float('nan'), dtype=torch.float64)
        feat_ar = torch.arange(F_n)
        cells = {(int(child_level_t[f]), int(child_pos_t[f])) for f in range(F_n)}
        for (l_c, j_c) in cells:
            if l_c < 1:                               # root child: no parent leak
                continue
            sel = (child_level_t == l_c) & (child_pos_t == j_c)
            feats = feat_ar[sel]
            sr = self.start_row.get((l_c, j_c))
            nv = self.V_child.get((l_c, j_c))
            if sr is None or nv is None:
                continue
            jf = self.joint_fire_count[sr:sr + nv, p_idx, :]   # (nv, F)
            c_i = jf[:, feats].argmax(dim=0)                   # (n_sel,) value
            for k, f in enumerate(feats.tolist()):
                out_ln[f] = self.leak_norm.get((l_c, j_c, int(c_i[k])),
                                               float('nan'))
        return out_ln

    def _value_specific_c_i(self, child_level_t, child_pos_t, p_idx):
        """[F] long: the value c_i each feature fires most on at its child cell
        (argmax joint_fire_count over the child cell's values). -1 if undefined
        (root child or absent cell). This is the value the leak conditions on,
        and is kept as the 'precision' tag for reassigned features."""
        return self.value_at_cell(child_level_t, child_pos_t, p_idx)

    def _cell_to_grow(self, level_t, pos_t):
        """[F] long: group-index of each (level,pos), or -1 if absent."""
        rows = torch.full_like(level_t, -1)
        for k, gi in self.group_index.items():
            m = (level_t == k[0]) & (pos_t == k[1])
            rows[m] = gi
        return rows


def per_feature_labels(art, leak_norm, alpha):
    """All four label schemes per (position, feature), BEFORE any weighting.

    Returns a dict:
      {
        'lambda_l1', 'layer_id', 'P', 's', 'L', 'mode', 'pooled',
        'matched_level',
        'positions'    : [P] real token positions,
        'parent_group' : [P] (level, pos) block-aligned parent cell,
        'cand_sets', 'cand_idxs',       # exposed for callers that aggregate
        'per_position' : [P] list of records, each with [F]-shaped tensors:

          # parent-constrained (block-aligned cell)
          'parent_level' (int), 'parent_pos' (int),
          'H_parent' [F], 'parent_href' (float), 'parent_value' [F],

          # level-constrained (argmin over matched-level cells)
          'level_min' [F], 'level_cell_level' [F], 'level_cell_pos' [F],
          'level_href' [F], 'level_value' [F],

          # unconstrained whole-tree (the "child")
          'wt_min' [F], 'child_level' [F], 'child_pos' [F], 'child_href' [F],
          'child_value' [F],

          # absorption check + reassigned label
          'c_i' [F], 'ratio' [F], 'reassign' [F] bool,
          'final_level' [F], 'final_pos' [F], 'final_entropy' [F],
          'final_href' [F], 'final_value' [F],

          # raw min/argmin for both candidate sets (so the sweep aggregator can
          # reproduce its H_bar_min_* curves without recomputing)
          'H_min' {cand -> [F]}, 'argmin' {cand -> [F]},
      }

    No weight-scheme aggregation is performed here.
    """
    T = _ArtifactTables(art, leak_norm)
    s = T.s

    out = {
        'lambda_l1': float(art.get('lambda_l1') or 0.0),
        'layer_id': T.layer_id, 'P': T.P, 's': T.s, 'L': T.L,
        'mode': art.get('mode', ''), 'pooled': T.pooled,
        'matched_level': T.matched_level,
        'cand_sets': T.cand_sets, 'cand_idxs': T.cand_idxs,
        'cand_href': T.cand_href,
        'positions': [], 'parent_group': [],
        'per_position': [],
    }

    Fidx = torch.arange(T.F)

    for p_idx in range(T.P):
        p_real = int(T.token_positions[p_idx].item())
        if T.pooled:
            parent_key = (0, 0)
        else:
            matched_j = p_real // (s ** (1 + T.layer_id))
            parent_key = (T.matched_level, matched_j)
        parent_gidx = T.group_index.get(parent_key)
        out['positions'].append(p_real)
        out['parent_group'].append(parent_key)

        # per-candidate min and argmin (NaN-aware; dead features stay NaN).
        H_min = {}
        argmins = {}
        for cand, gidxs in T.cand_idxs.items():
            H_cand = T.H_per_feature[gidxs, p_idx, :]            # [G, F]
            mn, am = _nanmin(H_cand, dim=0)                      # [F], [F]
            H_min[cand] = mn
            argmins[cand] = am
        H_parent = T.H_per_feature[parent_gidx, p_idx, :]        # [F]
        parent_href = float(T.H_theoretical.get(parent_key, float('nan')))

        # --- level-constrained winning cell (argmin over matched-level) ----
        sl_am = argmins['same_level']                           # [F] rows
        sl_cells = T.cand_sets['same_level']
        sl_level = torch.tensor([sl_cells[r][0] for r in range(len(sl_cells))],
                                dtype=torch.long)
        sl_pos = torch.tensor([sl_cells[r][1] for r in range(len(sl_cells))],
                              dtype=torch.long)
        level_cell_level = sl_level[sl_am]                      # [F]
        level_cell_pos = sl_pos[sl_am]                          # [F]
        level_href = T.cand_href['same_level'][sl_am]           # [F]

        # --- whole-tree (child) -------------------------------------------
        wt_am = argmins['whole_tree']                           # [F] rows
        child_level = T.wt_level[wt_am]                         # [F]
        child_pos = T.wt_pos[wt_am]                             # [F]
        child_href = T.cand_href['whole_tree'][wt_am]           # [F]

        # --- split check (child -> its tree-parent) -----------------------
        leakN = T._value_specific_leakN(child_level, child_pos, p_idx)  # [F]
        c_i = T._value_specific_c_i(child_level, child_pos, p_idx)      # [F]
        can_reassign = child_level >= 1                         # root has no parent
        par_level = child_level - 1
        par_pos = child_pos // s
        par_grow = T._cell_to_grow(par_level, par_pos)          # [F] (-1 none)
        safe_grow = par_grow.clamp_min(0)
        H_par_at = T.H_per_feature[safe_grow, p_idx, Fidx]      # [F]
        H_par_at = torch.where(par_grow >= 0, H_par_at,
                               torch.full_like(H_par_at, float('nan')))
        href_par = torch.tensor(
            [T.H_theoretical.get((int(par_level[f]), int(par_pos[f])),
                                 float('nan'))
             for f in range(par_level.shape[0])], dtype=torch.float64)
        href_par = torch.where(href_par > 0, href_par,
                               torch.full_like(href_par, float('nan')))
        ratio = (H_par_at.double() / href_par) / leakN          # [F]
        reassign = can_reassign & torch.isfinite(ratio) & (ratio < alpha)

        final_entropy = torch.where(reassign, H_par_at.double(),
                                    H_min['whole_tree'].double())
        final_href = torch.where(reassign, href_par, child_href.double())
        final_level = torch.where(reassign, par_level, child_level)
        final_pos = torch.where(reassign, par_pos, child_pos)

        # --- per-scheme values (argmax P(value|fire) at the winning cell) --
        parent_level_t = torch.full((T.F,), int(parent_key[0]), dtype=torch.long)
        parent_pos_t = torch.full((T.F,), int(parent_key[1]), dtype=torch.long)
        parent_value = T.value_at_cell(parent_level_t, parent_pos_t, p_idx)
        level_value = T.value_at_cell(level_cell_level, level_cell_pos, p_idx)
        child_value = T.value_at_cell(child_level, child_pos, p_idx)
        final_value = torch.where(reassign,
                                  T.value_at_cell(final_level, final_pos, p_idx),
                                  child_value)

        out['per_position'].append({
            'parent_level': int(parent_key[0]), 'parent_pos': int(parent_key[1]),
            'H_parent': H_parent, 'parent_href': parent_href,
            'parent_value': parent_value,

            'level_min': H_min['same_level'],
            'level_cell_level': level_cell_level, 'level_cell_pos': level_cell_pos,
            'level_href': level_href, 'level_value': level_value,

            'wt_min': H_min['whole_tree'],
            'child_level': child_level, 'child_pos': child_pos,
            'child_href': child_href, 'child_value': child_value,

            'c_i': c_i, 'ratio': ratio, 'reassign': reassign,
            'final_level': final_level, 'final_pos': final_pos,
            'final_entropy': final_entropy, 'final_href': final_href,
            'final_value': final_value,

            'H_min': H_min, 'argmin': argmins,
        })

    return out
