"""Bottom-up signature-based grouping for the pruned circuit DAG.

After the indirect-influence pruning step, we further simplify the graph
by collapsing feature nodes that share their incoming signature.

Signature definition
--------------------
For a feature node at layer k, the signature is the frozenset
    { (grouped_source_node, sign(weight)) ... }
over the pruned incoming edges from FEATURE (or EMBED for layer 0)
sources. Error sources are excluded. `grouped_source_node` is the
already-collapsed (super-)node at layer k-1 -- this is what makes the
recursion bottom-up.

Two layer-k features merge iff they have the exact same signature.

Empty signatures -- features whose only pruned incoming edges came from
error sources, or that have no incoming pruned edges at all -- do NOT
merge. Each such feature stays as its own singleton group.

Cross-position merges are allowed. Constituents at different positions
project to (level, parent_position) groups that may have different
observed value subsets; we sidestep that by working on the
`value_distribution_full` vector that was already lifted to the full
parent-level vocab (size v at levels 1..L-1, size n at level 0) in
`dag.build_node_table`.

Group attrs
-----------
For a group (super-)node spanning N >= 1 constituent features:
  - kind = 'group'
  - layer = k
  - constituents = [(position, feature_index), ...]  sorted
  - positions = sorted unique positions
  - position_centroid = float(mean(positions))
  - z_sum = sum of constituent z values
  - value_distribution = z-weighted average of `value_distribution_full`,
    renormalized to sum to 1 (if total mass > 0).
  - label_value = int(argmax(value_distribution))  (None if dead)
  - p_value_given_fire = value_distribution[label_value]  (NaN if dead)
  - normalized_entropy = H(value_distribution) / log(V_level), clipped to
    [0, 1] (None if dead). Used for the white -> green color gradient.
  - V_level = vocab size at the parent level (v or n)
  - signature = frozenset((grouped_src, sign), ...) used for hover text

Edges
-----
Original pruned edges are remapped by replacing each endpoint with its
parent group key (embed / err / logit keys map to themselves). Multiple
remapped edges between the same (src, dst) pair are merged by summing
their signed weights.
"""

from __future__ import annotations

import math

import torch

from .dag import feat_key, SCHEME_NAMES


def _sign_of(w: float) -> int:
    if w > 0:
        return 1
    if w < 0:
        return -1
    return 0


def _norm_entropy_of(value_distribution, V_level):
    """Normalized entropy H(dist)/log(V_level) clipped to [0,1]; None if undef."""
    if value_distribution is None or V_level is None or V_level <= 1:
        return None
    p = value_distribution.double()
    H = float(-torch.special.xlogy(p, p).sum().item())
    ref = math.log(V_level)
    if ref <= 0 or not math.isfinite(H):
        return None
    ne = H / ref
    if ne < 0.0:
        ne = 0.0
    elif ne > 1.0:
        ne = 1.0
    return ne


def _group_label_for_scheme(members, nodes, sc):
    """Plurality-cell group label for one scheme.

    Partition the group's members by their scheme-`sc` latent cell
    (level, position); the plurality cell (most members; ties -> larger subset
    z_sum -> lower (level, position)) is the group's label. The value
    distribution is the z-weighted average over the plurality subset ONLY, so
    constituents that point at a different latent do not pollute the label.

    Returns a dict:
      level, position, V_level,
      label_value, p_value_given_fire, normalized_entropy, value_distribution,
      n_plurality, n_labelled,
      disagreement_frac        (1 - n_plurality / n_labelled, by count),
      disagreement_frac_z      (same, z-weighted),
      cell_distribution        {(level,position) -> z-share}  (for hover/diag).
    Fields are None/NaN when no member carries a scheme-`sc` distribution.
    """
    # Collect, per member, its scheme cell + lifted dist + z.
    entries = []  # (level, position, V_level, vd_full, z)
    for m in members:
        sch = nodes[m].get('schemes', {}).get(sc)
        if sch is None:
            continue
        vd = sch.get('value_distribution_full')
        if vd is None:
            continue
        entries.append((sch.get('level'), sch.get('position'),
                        sch.get('V_level'), vd, float(nodes[m]['z'])))
    if not entries:
        return {
            'level': None, 'position': None, 'V_level': None,
            'label_value': None, 'p_value_given_fire': float('nan'),
            'normalized_entropy': None, 'value_distribution': None,
            'n_plurality': 0, 'n_labelled': 0,
            'disagreement_frac': float('nan'),
            'disagreement_frac_z': float('nan'),
            'cell_distribution': {},
        }

    # Bucket members by (level, position).
    cells = {}
    for (lvl, pos, V_level, vd, z) in entries:
        cells.setdefault((lvl, pos), []).append((V_level, vd, z))
    n_labelled = len(entries)
    z_total = sum(z for (_, _, _, _, z) in entries)

    # cell z-share (for diagnostics / hover).
    cell_distribution = {}
    for cell, items in cells.items():
        cell_distribution[cell] = (
            sum(z for (_, _, z) in items) / z_total if z_total > 0 else 0.0)

    # Plurality cell: most members; tie -> larger z_sum -> lower (level, pos).
    def _cell_rank(cell):
        items = cells[cell]
        n = len(items)
        zsum = sum(z for (_, _, z) in items)
        lvl, pos = cell
        # sort key: more members, then more z, then lower level, lower pos.
        return (-n, -zsum, (lvl if lvl is not None else 1 << 30),
                (pos if pos is not None else 1 << 30))
    plurality_cell = min(cells.keys(), key=_cell_rank)
    plur_items = cells[plurality_cell]
    n_plurality = len(plur_items)
    z_plur = sum(z for (_, _, z) in plur_items)

    lvl, pos = plurality_cell
    V_level = plur_items[0][0]

    # z-weighted average of the plurality subset's distributions.
    num = None
    denom = 0.0
    for (V_lv, vd, z) in plur_items:
        if num is None:
            num = z * vd.float().clone()
        else:
            num = num + z * vd.float()
        denom += z
    if num is None or denom <= 0:
        value_distribution = None
        label_value = None
        p_label = float('nan')
        norm_entropy = None
    else:
        value_distribution = num / denom
        total_mass = float(value_distribution.sum().item())
        if total_mass <= 0:
            value_distribution = None
            label_value = None
            p_label = float('nan')
            norm_entropy = None
        else:
            value_distribution = value_distribution / total_mass
            label_value = int(value_distribution.argmax().item())
            p_label = float(value_distribution[label_value].item())
            norm_entropy = _norm_entropy_of(value_distribution, V_level)

    disagree_count = 1.0 - (n_plurality / n_labelled) if n_labelled > 0 else float('nan')
    disagree_z = 1.0 - (z_plur / z_total) if z_total > 0 else float('nan')

    return {
        'level': lvl, 'position': pos, 'V_level': V_level,
        'label_value': label_value, 'p_value_given_fire': p_label,
        'normalized_entropy': norm_entropy,
        'value_distribution': value_distribution,
        'n_plurality': n_plurality, 'n_labelled': n_labelled,
        'disagreement_frac': disagree_count,
        'disagreement_frac_z': disagree_z,
        'cell_distribution': {('%d,%d' % (c[0], c[1]) if c[0] is not None
                               else 'none'): sh
                              for c, sh in cell_distribution.items()},
    }


def group_by_signature(
    nodes: dict,
    pruned_edges: list,
    K: int,
    s: int,
    L: int,
    v: int,
    n: int,
    primary: str = 'reassigned',
) -> tuple[dict, list, dict]:
    """Collapse feature nodes in the pruned graph by incoming signature.

    Grouping (membership/topology) is scheme-INDEPENDENT: features merge by
    incoming signature only. For each of the four label schemes we then compute
    a per-group label by the plurality-cell rule (see _group_label_for_scheme)
    and a disagreement metric, stored under group['schemes'][scheme]. The
    back-compat top-level group label fields mirror `primary`.

    Returns
    -------
    grouped_nodes : dict
        node_key -> attrs. Group keys are tuples
        ('group', layer, group_idx_within_layer). Embed, err, logit keys
        pass through unchanged.
    grouped_edges : list
        list of (src_key, dst_key, signed_weight).
    group_membership : dict
        group_key -> [orig_feat_key, ...] for the constituents.
    """
    # ---- Index incoming edges by destination ----
    # incoming_feat[dst]  = list of (src_feat_or_embed_node, signed_weight)
    # incoming_other[dst] = list of (src_node, signed_weight)  (errors, etc.)
    # `incoming_feat` is what we use to build the signature; we exclude err
    # sources here. Embed sources are kept (they feed only layer 0).
    incoming_feat: dict = {}
    incoming_other: dict = {}
    for (src, dst, w) in pruned_edges:
        if src[0] == 'feat' or src[0] == 'embed':
            incoming_feat.setdefault(dst, []).append((src, float(w)))
        else:
            incoming_other.setdefault(dst, []).append((src, float(w)))

    # ---- parent_of: original-key -> grouped-key ----
    # Pass-through for non-feature kinds.
    parent_of: dict = {}
    for key in nodes:
        if key[0] != 'feat':
            parent_of[key] = key

    grouped_nodes: dict = {k: dict(v) for k, v in nodes.items() if k[0] != 'feat'}
    group_membership: dict = {}

    # ---- Walk layers bottom-up ----
    feats_by_layer: dict[int, list] = {}
    for key in nodes:
        if key[0] == 'feat':
            feats_by_layer.setdefault(int(key[1]), []).append(key)
    for k in sorted(feats_by_layer):
        feats_by_layer[k].sort()

    for k in range(K):
        feats_k = feats_by_layer.get(k, [])
        # Bucket by signature. We allocate a unique bucket key per feature
        # whose signature is empty, so they don't merge.
        sig_to_feats: dict = {}
        # For empty signatures: give each a unique sentinel based on the
        # feature's own key to keep them separate but still flow through
        # the same downstream merging pipeline.
        for fkey in feats_k:
            incoming = incoming_feat.get(fkey, [])
            # Build the signature using the ALREADY-COLLAPSED parents of
            # each non-error source. (parent_of for embed/err keys is the
            # key itself; for feature keys it's set when that layer was
            # processed earlier in this loop, hence bottom-up.)
            sig_items = []
            for (src, w) in incoming:
                grouped_src = parent_of.get(src, src)
                sig_items.append((grouped_src, _sign_of(w)))
            sig = frozenset(sig_items)
            if len(sig) == 0:
                # Singleton bucket; encode feature key in the bucket key
                # so it can never collide with another empty-sig feature.
                bucket = ('__empty__', fkey)
            else:
                bucket = sig
            sig_to_feats.setdefault(bucket, []).append(fkey)

        # ---- Materialize one group per bucket ----
        # Stable bucket order: by (n_constituents desc, first-key tuple).
        bucket_keys_sorted = sorted(
            sig_to_feats.keys(),
            key=lambda b: (
                -len(sig_to_feats[b]),
                tuple(sig_to_feats[b][0]),
            ),
        )
        for g_idx, bucket in enumerate(bucket_keys_sorted):
            members = sorted(sig_to_feats[bucket])
            group_key = ('group', k, g_idx)
            # Aggregate group attrs.
            positions = sorted({int(m[2]) for m in members})
            constituents = sorted([(int(m[2]), int(m[3])) for m in members])
            z_list = [float(nodes[m]['z']) for m in members]
            z_sum = float(sum(z_list))

            # Per-scheme label by the plurality-cell rule (scheme-independent
            # membership, per-scheme label + disagreement overlay).
            scheme_labels = {
                sc: _group_label_for_scheme(members, nodes, sc)
                for sc in SCHEME_NAMES
            }
            prim = scheme_labels[primary]

            # block-aligned ancestor positions, scheme-independent.
            parent_positions = sorted({
                int(nodes[m]['parent_position']) for m in members
            })

            attrs = {
                'kind': 'group',
                'layer': k,
                # top-level label fields mirror the chosen primary scheme.
                'level': prim['level'],
                'V_level': prim['V_level'],
                'positions': positions,
                'position_centroid': float(sum(positions) / len(positions)),
                'parent_positions': parent_positions,
                'constituents': constituents,
                'n_constituents': len(members),
                'z_sum': z_sum,
                'value_distribution': prim['value_distribution'],
                'label_value': prim['label_value'],
                'p_value_given_fire': prim['p_value_given_fire'],
                'normalized_entropy': prim['normalized_entropy'],
                'disagreement_frac': prim['disagreement_frac'],
                'disagreement_frac_z': prim['disagreement_frac_z'],
                'schemes': scheme_labels,
                # Signature as a list of ((kind, ...key tuple...), sign) for
                # serialization friendliness in hover text.
                'signature': sorted(
                    [(tuple(src), int(sgn)) for (src, sgn) in bucket]
                    if not (isinstance(bucket, tuple) and bucket
                            and bucket[0] == '__empty__') else []
                ),
            }
            grouped_nodes[group_key] = attrs
            group_membership[group_key] = list(members)
            for m in members:
                parent_of[m] = group_key

    # ---- Remap edges ----
    edge_accum: dict = {}
    for (src, dst, w) in pruned_edges:
        new_src = parent_of.get(src, src)
        new_dst = parent_of.get(dst, dst)
        if new_src == new_dst:
            # Self-loops shouldn't appear (strict DAG), but drop just in case.
            continue
        key = (new_src, new_dst)
        edge_accum[key] = edge_accum.get(key, 0.0) + float(w)
    grouped_edges = [(s_, d_, w_) for (s_, d_), w_ in edge_accum.items()]

    return grouped_nodes, grouped_edges, group_membership
