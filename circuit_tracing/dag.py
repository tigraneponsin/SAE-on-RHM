"""DAG construction for circuit tracing.

Node keys:
  ('feat', k, p, i)   feature i at layer k, position p (only if z_k[p, i] > 0)
  ('err',  k, p)      error node at layer k, position p
  ('embed', p)        embedding (tok+pos) input node at position p
  ('logit', c)        logit head node for class c

Edges are tuples of (src_key, dst_key, weight: float).

Node attributes (the 'kind' field is the human-readable spec name; the key
prefix stays compact for backwards compatibility with consumers):
  feat   -> kind='feature'
  err    -> kind='error'
  embed  -> kind='embedding'
  logit  -> kind='logit'
"""

from __future__ import annotations

from typing import Iterable

import torch


def feat_key(k: int, p: int, i: int):
    return ('feat', int(k), int(p), int(i))


def err_key(k: int, p: int):
    return ('err', int(k), int(p))


def embed_key(p: int):
    return ('embed', int(p))


def logit_key(c: int):
    return ('logit', int(c))


def assemble_edges(
    layer_pairs_feat: list[list],   # layer_pairs_feat[k] = list of (i, p, j, q, w) for k -> k+1
    layer_pairs_err: list[list],    # layer_pairs_err[k]  = list of (p, j, q, w) for err k -> feat k+1
    embed_to_l0: list,              # list of (p, j, q, w) for embed -> feat 0
    final_feat_to_logit: list,      # list of (i, p, c, w) at K-1
    final_err_to_logit: list,       # list of (p, c, w) at K-1
    K: int,
) -> list[tuple]:
    """Flatten all per-pair edges into a single (src, dst, weight) list.

    Edge categories produced (in order):
      - feature -> feature  (k -> k+1)
      - error   -> feature  (k -> k+1)
      - embed   -> feature  (-> layer 0)
      - feature -> logit    (K-1 -> per-class logit)
      - error   -> logit    (K-1 -> per-class logit)
    """
    edges = []
    for k, lst in enumerate(layer_pairs_feat):
        for (i, p, j, q, w) in lst:
            edges.append((feat_key(k, p, i), feat_key(k + 1, q, j), w))
    for k, lst in enumerate(layer_pairs_err):
        for (p, j, q, w) in lst:
            edges.append((err_key(k, p), feat_key(k + 1, q, j), w))
    for (p, j, q, w) in embed_to_l0:
        edges.append((embed_key(p), feat_key(0, q, j), w))
    for (i, p, c, w) in final_feat_to_logit:
        edges.append((feat_key(K - 1, p, i), logit_key(c), w))
    for (p, c, w) in final_err_to_logit:
        edges.append((err_key(K - 1, p), logit_key(c), w))
    return edges


def build_node_table(
    z_per_layer: list[torch.Tensor],
    labels_per_layer: list[dict],
    s: int, L: int,
    embed_anchor: torch.Tensor,
    x_input: torch.Tensor,
    logits: torch.Tensor,
    y_true: int,
    v: int,
    n: int,
    pooled_last_layer: bool = False,
) -> dict:
    """Return dict node_key -> attrs for every active feature, every error,
    every embedding (one per position), and every logit class.

    z_per_layer[k]   : [N, F_k] post-ReLU SAE encoder activations. When
                       pooled_last_layer is True, z_per_layer[K-1] is [1, F]
                       (a single pooled node at position 0, mapping to the RHM
                       root, level 0).
    labels_per_layer : list of {(p, f) -> {value, p_value_given_fire, level,
                                            parent_position, firing_count,
                                            value_distribution, values,
                                            normalized_entropy}}.
    embed_anchor     : [N, d] anchors.embed (tok+pos at every leaf position).
    x_input          : [N] long, leaf token ids.
    logits           : [num_classes] anchors.logits.
    y_true           : int.
    v, n             : RHM vocab sizes (per node / root). The feature's
                       value_distribution is lifted to the full parent-
                       level vocab (size v at levels >= 1, size n at level
                       0) by zero-filling unobserved indices.
    """
    K = len(z_per_layer)
    if pooled_last_layer and z_per_layer[K - 1].shape[0] != 1:
        raise ValueError(
            f'pooled_last_layer=True expects z_per_layer[K-1] with a single '
            f'pooled position [1, F], got shape {tuple(z_per_layer[K - 1].shape)}'
        )
    out = {}
    # Feature and error nodes (per layer, per position). The per-position loop
    # is driven by z_k's first dim, so the pooled last layer ([1, F]) yields a
    # single node at position 0 (level 0 = RHM root) automatically.
    for k in range(K):
        z_k = z_per_layer[k]  # [N, F]  ([1, F] for the pooled last layer)
        N, F = z_k.shape
        labels_k = labels_per_layer[k]
        for p in range(N):
            target_level = L - 1 - k
            target_pos = p // (s ** (1 + k))
            V_level = int(n) if target_level == 0 else int(v)
            # Error node always exists.
            out[err_key(k, p)] = {
                'kind': 'error',
                'layer': k,
                'position': p,
                'level': target_level,
                'parent_position': target_pos,
            }
            for i in range(F):
                z = float(z_k[p, i].item())
                if z <= 0:
                    continue
                lab = labels_k.get((p, i), {
                    'value': None, 'p_value_given_fire': float('nan'),
                    'level': target_level, 'parent_position': target_pos,
                    'firing_count': 0,
                    'value_distribution': None, 'values': None,
                    'normalized_entropy': None,
                })
                # Lift the per-feature conditional vector to the full
                # parent-level vocab so cross-position group averaging is
                # easy. Dead features keep value_distribution_full = None.
                vd = lab.get('value_distribution')
                vs = lab.get('values')
                if vd is None or vs is None:
                    vd_full = None
                else:
                    vd_full = torch.zeros(V_level, dtype=torch.float32)
                    vd_full[vs.long()] = vd.float()
                out[feat_key(k, p, i)] = {
                    'kind': 'feature',
                    'layer': k,
                    'position': p,
                    'feature': i,
                    'z': z,
                    'level': lab['level'],
                    'parent_position': lab['parent_position'],
                    'V_level': V_level,
                    'label_value': lab['value'],
                    'p_value_given_fire': lab['p_value_given_fire'],
                    'eval_firing_count': lab['firing_count'],
                    'value_distribution_full': vd_full,
                    'normalized_entropy': lab.get('normalized_entropy'),
                }

    # Embedding nodes: one per leaf position. Group by leaf-group of size s
    # to mirror how layer-k positions group at the target ancestor index.
    N = embed_anchor.size(0)
    for p in range(N):
        out[embed_key(p)] = {
            'kind': 'embedding',
            'position': p,
            'token_id': int(x_input[p].item()),
            'level': 0,
            'parent_position': p // s,
            'e_emb_norm': float(embed_anchor[p].norm().item()),
        }

    # Logit nodes: one per class.
    probs = torch.softmax(logits, dim=0)
    num_classes = int(logits.numel())
    for c in range(num_classes):
        out[logit_key(c)] = {
            'kind': 'logit',
            'class': c,
            'logit': float(logits[c].item()),
            'prob': float(probs[c].item()),
            'is_true_class': bool(c == y_true),
        }
    return out


def to_networkx(nodes: dict, edges: list[tuple]):
    """Build a networkx.DiGraph from nodes + edges.

    All metadata (kind, position, label, etc.) is already on `nodes`.
    Imported lazily so the rest of the pipeline does not require networkx.
    """
    import networkx as nx
    g = nx.DiGraph()
    for key, attrs in nodes.items():
        g.add_node(key, **attrs)
    for (src, dst, w) in edges:
        # If src wasn't added (e.g. an inactive feat that still emitted an
        # edge for some reason), add it on the fly with just `kind`.
        if src not in g:
            g.add_node(src, kind=src[0])
        if dst not in g:
            g.add_node(dst, kind=dst[0])
        g.add_edge(src, dst, weight=float(w))
    return g
