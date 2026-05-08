"""End-to-end smoke test for the full circuit-tracing pipeline.

Builds a tiny RHM, transformer, K SAEs and one .sae_eval.pt-like dict in
memory, then runs the orchestrator's body (sans CLI) on a single input.
Verifies fidelity diagnostics print sensibly and saved files load back.
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import torch

from datasets.random_hierarchy_model import sample_rules, sample_trees
from models.transformer import (
    MeanClassificationTransformer, MeanClassificationTransformerNoResidual,
)
from models.sae import SparseAutoencoder
from circuit_tracing.linearize import (
    capture_anchors, linearized_full_forward,
)
from circuit_tracing.attribution import (
    sae_forward, edges_layer_to_layer,
    edges_embedding_to_layer0,
    edges_final_layer_to_logits_per_class,
)
from circuit_tracing.dag import (
    assemble_edges, build_node_table, to_networkx,
)
from circuit_tracing.prune import prune_indirect_influence


def _build_pseudo_eval_artifact(layer_id, latent_dim, N, s, L, n, v, m,
                                  trees, model, sae, act_scale, device='cpu'):
    """Replicate the bare minimum of stream_sae_eval to produce a dict with
    joint_fire_count, firing_count, index_layout, rhm, latent_dim, layer_id,
    rules_source, sae_rules_source. Used so build_labels_per_layer can run.
    """
    from scripts.sae_eval.streaming import enumerate_targets

    # All token-position activations of x_k.
    x_input = trees[L].long().to(device)  # [num_data, N]
    num_data = x_input.size(0)

    # Run model in batches.
    buf = []
    hook = model.blocks[layer_id].register_forward_hook(
        lambda _m, _i, o: buf.append(o.detach())
    )
    with torch.no_grad():
        model(x_input)
    hook.remove()
    act = buf.pop(0)  # [num_data, N, d]

    z = torch.relu(
        act_scale * act @ sae.encoder.weight.T + sae.encoder.bias
    )  # [num_data, N, F]
    fire = (z > 0).long()  # [num_data, N, F]

    firing_count = fire.sum(dim=0)  # [N, F]

    targets, index_layout = enumerate_targets(trees)
    T = len(targets)
    joint_fire_count = torch.zeros(T, N, latent_dim, dtype=torch.long)
    for group in index_layout:
        level = group['level']
        pos_g = group['position']
        values = group['values'].long()
        level_tensor = trees[level]
        col = level_tensor if level_tensor.ndim == 1 else level_tensor[:, pos_g]
        col = col.long()
        for vi, vval in enumerate(values.tolist()):
            mask = (col == vval).long()  # [num_data]
            # joint[start+vi, p, f] = sum over samples with col == vval of fire[sample, p, f]
            joint_fire_count[group['start'] + vi] = (
                fire * mask.view(-1, 1, 1)
            ).sum(dim=0)

    return {
        'layer_id': layer_id,
        'latent_dim': latent_dim,
        'rhm': {'s': s, 'L': L, 'v': v, 'n': n, 'm': m},
        'firing_count': firing_count,
        'joint_fire_count': joint_fire_count,
        'index_layout': [
            {'level': g['level'], 'position': g['position'],
             'values': g['values'].clone(),
             'start': g['start'], 'end': g['end']}
            for g in index_layout
        ],
        'rules_source': 'artifact',
        'sae_rules_source': 'artifact',
    }


def main():
    torch.manual_seed(0)

    # Tiny RHM
    s, L, v_, n_, m_ = 2, 2, 4, 4, 2
    N = s ** L
    rules = sample_rules(v=v_, n=n_, m=m_, s=s, L=L, seed=0)
    trees = sample_trees(num_data=128, rules=rules, prior=None, probs=None, seed=0)

    # Tiny transformer
    d = 32
    model = MeanClassificationTransformer(
        vocab_size=v_, block_size=N, embedding_dim=d, num_heads=2,
        ffwd_size=2, num_layers=L, num_classes=n_, dropout=0,
    ).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Tiny SAEs (random, just to exercise plumbing)
    K = L
    saes, act_scales = [], []
    for k in range(K):
        sae = SparseAutoencoder(input_dim=d, latent_dim=24).eval()
        for p in sae.parameters():
            p.requires_grad = False
        with torch.no_grad():
            sae.encoder.bias.uniform_(-0.5, 0.8)
        saes.append(sae)
        act_scales.append(1.2 + 0.1 * k)

    # Pseudo-eval artifacts
    eval_artifacts = []
    for k in range(K):
        art = _build_pseudo_eval_artifact(
            layer_id=k, latent_dim=saes[k].latent_dim, N=N, s=s, L=L,
            n=n_, v=v_, m=m_, trees=trees, model=model, sae=saes[k],
            act_scale=act_scales[k],
        )
        eval_artifacts.append(art)

    # Pick an input
    input_idx = 0
    x_input = trees[L][input_idx].long()
    y_true = int(trees[0][input_idx].item())

    # Anchors
    anchors = capture_anchors(model, x_input)

    # SAE forwards
    z_list, x_hat_list, e_list = [], [], []
    for k in range(K):
        x_k = anchors.blocks[k].r_out
        z, x_hat, e = sae_forward(x_k, saes[k], act_scales[k])
        z_list.append(z); x_hat_list.append(x_hat); e_list.append(e)

    # Bit-identity
    spliced = [x_hat_list[k] + e_list[k] for k in range(K)]
    spliced_logits = linearized_full_forward(model, anchors, spliced)
    bit_err = (spliced_logits - anchors.logits).abs().max().item()
    assert bit_err < 1e-4, bit_err
    print(f'bit-identity err = {bit_err:.3e}')

    # Layer-to-layer
    layer_pairs_feat, layer_pairs_err = [], []
    for k in range(K - 1):
        feats, errs = edges_layer_to_layer(
            model, k, anchors,
            z_list[k], e_list[k],
            saes[k].decoder.weight, act_scales[k],
            saes[k + 1].encoder.weight, act_scales[k + 1],
            z_list[k + 1],
            mat_threshold=10000,
        )
        layer_pairs_feat.append(feats)
        layer_pairs_err.append(errs)
        print(f'k={k}: {len(feats)} feat-feat, {len(errs)} err-feat')

    # Embedding -> layer-0 attribution
    embed_to_l0 = edges_embedding_to_layer0(
        model, anchors, anchors.embed,
        z_list[0], saes[0].encoder.weight, act_scales[0],
        mat_threshold=10000,
    )
    print(f'embed -> layer 0: {len(embed_to_l0)} edges')

    # Per-class final-layer attribution
    sorted_idx = torch.argsort(anchors.logits, descending=True)
    y_pred = int(sorted_idx[0].item())
    feat_to_logit, err_to_logit = edges_final_layer_to_logits_per_class(
        model, anchors, z_list[K - 1], e_list[K - 1],
        saes[K - 1].decoder.weight, act_scales[K - 1],
        num_classes=n_,
    )
    print(f'final layer per-class: {len(feat_to_logit)} feat->logit, '
          f'{len(err_to_logit)} err->logit, {n_} classes')

    # Labels
    from circuit_tracing.labels import build_labels_per_layer
    labels_per_layer = build_labels_per_layer(eval_artifacts, s=s, L=L)
    sample_label = next(iter(labels_per_layer[0].values()))
    print(f'sample label at L0: {sample_label}')

    # Build DAG
    nodes = build_node_table(
        z_list, labels_per_layer, s=s, L=L,
        embed_anchor=anchors.embed, x_input=x_input,
        logits=anchors.logits, y_true=y_true,
    )
    # Sanity: there are exactly N embedding nodes and num_classes logit nodes.
    n_embed = sum(1 for k in nodes if k[0] == 'embed')
    n_logit = sum(1 for k in nodes if k[0] == 'logit')
    n_feat = sum(1 for k in nodes if k[0] == 'feat')
    n_err = sum(1 for k in nodes if k[0] == 'err')
    assert n_embed == N, (n_embed, N)
    assert n_logit == n_, (n_logit, n_)
    print(f'nodes by kind: feat={n_feat}, err={n_err}, embed={n_embed}, logit={n_logit}')

    edges = assemble_edges(layer_pairs_feat, layer_pairs_err, embed_to_l0,
                           feat_to_logit, err_to_logit, K=K)
    print(f'total edges: {len(edges)}')

    # Indirect-influence pruning, both sink modes.
    for sink_mode in ('softmax_logits', 'true_class'):
        pruned, kept_keys, diag = prune_indirect_influence(
            nodes, edges, K=K,
            logits=anchors.logits, y_true=y_true,
            sink_mode=sink_mode,
            node_threshold=0.8, edge_threshold=0.98,
        )
        print(f'  sink={sink_mode}: '
              f'edges {diag["n_edges_pre"]} -> {diag["n_edges_post"]}, '
              f'features {diag["n_features_pre"]} -> {diag["n_features_post"]}, '
              f'completeness={diag["completeness_score"]:.3f}, '
              f'replacement={diag["replacement_score"]:.3f}')
        # Embedding/error/logit nodes must never be pruned.
        for k in nodes:
            if k[0] in ('embed', 'err', 'logit'):
                assert k in kept_keys, f'pure-input/output {k!r} was pruned'

    # Round-trip save/load using the last (true_class) result.
    g = to_networkx(nodes, pruned)
    print(f'graph has {g.number_of_nodes()} nodes, {g.number_of_edges()} edges')

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        torch.save(nodes, td / 'nodes.pt')
        torch.save({'all': edges, 'pruned': pruned}, td / 'edges.pt')
        with open(td / 'graph.gpickle', 'wb') as f:
            pickle.dump(g, f)
        n2 = torch.load(td / 'nodes.pt', weights_only=False)
        e2 = torch.load(td / 'edges.pt', weights_only=False)
        with open(td / 'graph.gpickle', 'rb') as f:
            g2 = pickle.load(f)
        assert len(n2) == len(nodes)
        assert len(e2['all']) == len(edges)
        assert g2.number_of_nodes() == g.number_of_nodes()
        print('round-trip OK')

    print('\nE2E smoke test PASSED')


if __name__ == '__main__':
    main()
