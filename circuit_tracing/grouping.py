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

from .dag import feat_key


def _sign_of(w: float) -> int:
    if w > 0:
        return 1
    if w < 0:
        return -1
    return 0


def group_by_signature(
    nodes: dict,
    pruned_edges: list,
    K: int,
    s: int,
    L: int,
    v: int,
    n: int,
) -> tuple[dict, list, dict]:
    """Collapse feature nodes in the pruned graph by incoming signature.

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
            V_level = int(nodes[members[0]]['V_level'])

            # Weighted average of value_distribution_full. Dead constituents
            # (value_distribution_full is None) are skipped from the
            # numerator but still contribute their z to the denominator --
            # actually, better: skip them entirely from BOTH so the
            # average reflects only constituents with eval evidence. If
            # every constituent is dead, the group's distribution is None.
            num = None
            denom = 0.0
            for m, z in zip(members, z_list):
                vd = nodes[m].get('value_distribution_full')
                if vd is None:
                    continue
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
                    # H in nats; reference is log(V_level).
                    p = value_distribution.double()
                    H = float(-torch.special.xlogy(p, p).sum().item())
                    ref = math.log(V_level) if V_level > 1 else 0.0
                    if ref > 0 and math.isfinite(H):
                        norm_entropy = H / ref
                        if norm_entropy < 0.0:
                            norm_entropy = 0.0
                        elif norm_entropy > 1.0:
                            norm_entropy = 1.0
                    else:
                        norm_entropy = None

            # Pull a representative target_level / parent_position. Level
            # is shared across constituents at the same layer (= L - 1 - k).
            target_level = L - 1 - k
            parent_positions = sorted({
                int(nodes[m]['parent_position']) for m in members
            })

            attrs = {
                'kind': 'group',
                'layer': k,
                'level': target_level,
                'V_level': V_level,
                'positions': positions,
                'position_centroid': float(sum(positions) / len(positions)),
                'parent_positions': parent_positions,
                'constituents': constituents,
                'n_constituents': len(members),
                'z_sum': z_sum,
                'value_distribution': value_distribution,
                'label_value': label_value,
                'p_value_given_fire': p_label,
                'normalized_entropy': norm_entropy,
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
