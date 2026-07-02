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

from datasets.random_hierarchy_model import (
    sample_rules, sample_trees, latent_prior,
)


def _h_theoretical_from_rules(rules, n, v):
    """{(level, pos) -> H(marginal) in nats} from the RHM rules. Mirrors the
    H_theoretical produced by the real eval so the labeler/leak table run."""
    priors = latent_prior(rules, n, v)  # {level -> (s^level, V_level)}
    out = {}
    for level, marg in priors.items():
        for pos in range(marg.shape[0]):
            p = marg[pos].double()
            out[(int(level), int(pos))] = float(
                -torch.special.xlogy(p, p).sum().item())
    return out


def _h_per_feature_from_joint(joint_fire_count, firing_count, index_layout):
    """[num_groups, P, F] conditional entropy H(latent_cell | f fires) in nats,
    one group per index_layout entry. Mirrors the real eval's H_per_feature so
    per_feature_labels can argmin over cells. Dead (firing==0) -> NaN."""
    G = len(index_layout)
    P, F = firing_count.shape
    H = torch.full((G, P, F), float('nan'), dtype=torch.float64)
    fc = firing_count.double()
    for gi, g in enumerate(index_layout):
        st, en = int(g['start']), int(g['end'])
        jf = joint_fire_count[st:en].double()           # [V_g, P, F]
        cp = jf / fc.clamp_min(1.0).unsqueeze(0)         # [V_g, P, F] P(value|fire)
        ent = -torch.special.xlogy(cp, cp).sum(dim=0)    # [P, F]
        ent = torch.where(fc > 0, ent, torch.full_like(ent, float('nan')))
        H[gi] = ent
    return H
from models.transformer import (
    MeanClassificationTransformer, MeanClassificationTransformerNoResidual,
)
from models.sae import SparseAutoencoder
from scripts.circuit_tracing.linearize import (
    capture_anchors, linearized_full_forward,
)
from scripts.circuit_tracing.attribution import (
    sae_forward, sae_forward_pooled, edges_layer_to_layer,
    edges_layer_to_pooled,
    edges_embedding_to_layer0,
    edges_final_layer_to_logits_per_class,
    edges_pooled_final_to_logits_per_class,
)
from scripts.circuit_tracing.dag import (
    assemble_edges, build_node_table, to_networkx,
)
from scripts.circuit_tracing.prune import prune_indirect_influence


def _build_pseudo_eval_artifact(layer_id, latent_dim, N, s, L, n, v, m,
                                  trees, model, sae, act_scale, rules,
                                  device='cpu'):
    """Replicate the bare minimum of stream_sae_eval to produce a dict with
    joint_fire_count, firing_count, index_layout, targets, H_theoretical, rhm,
    latent_dim, layer_id, rules_source, sae_rules_source. Used so
    build_labels_per_layer (all four schemes + leak table) can run.
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
        'targets': targets,
        'token_positions': torch.arange(N, dtype=torch.long),
        'H_theoretical': _h_theoretical_from_rules(rules, n, v),
        'H_per_feature': _h_per_feature_from_joint(
            joint_fire_count, firing_count, index_layout),
        'rules_source': 'artifact',
        'sae_rules_source': 'artifact',
    }


def _build_pseudo_eval_artifact_pooled(layer_id, latent_dim, N, s, L, n, v, m,
                                        trees, model, sae, act_scale, rules,
                                        device='cpu'):
    """Pooled-last-layer analogue of _build_pseudo_eval_artifact.

    Hooks model.ln_f, mean-pools over the sequence dim, and produces a
    firing_count of shape [1, F] (P=1, single pooled position) with an
    index_layout that still includes the level-0/root group so build_labels
    can map the pooled features to the RHM root.
    """
    from scripts.sae_eval.streaming import enumerate_targets

    x_input = trees[L].long().to(device)  # [num_data, N]

    buf = []
    hook = model.ln_f.register_forward_hook(
        lambda _m, _i, o: buf.append(o.detach())
    )
    with torch.no_grad():
        model(x_input)
    hook.remove()
    act = buf.pop(0)  # [num_data, N, d] post-ln_f
    pooled = act.mean(dim=1, keepdim=True)  # [num_data, 1, d]

    z = torch.relu(
        act_scale * pooled @ sae.encoder.weight.T + sae.encoder.bias
    )  # [num_data, 1, F]
    fire = (z > 0).long()  # [num_data, 1, F]
    firing_count = fire.sum(dim=0)  # [1, F]

    targets, index_layout = enumerate_targets(trees)
    T = len(targets)
    joint_fire_count = torch.zeros(T, 1, latent_dim, dtype=torch.long)
    for group in index_layout:
        level = group['level']
        pos_g = group['position']
        values = group['values'].long()
        level_tensor = trees[level]
        col = level_tensor if level_tensor.ndim == 1 else level_tensor[:, pos_g]
        col = col.long()
        for vi, vval in enumerate(values.tolist()):
            mask = (col == vval).long()  # [num_data]
            joint_fire_count[group['start'] + vi] = (
                fire * mask.view(-1, 1, 1)
            ).sum(dim=0)

    return {
        'layer_id': layer_id,
        'latent_dim': latent_dim,
        'rhm': {'s': s, 'L': L, 'v': v, 'n': n, 'm': m},
        'firing_count': firing_count,        # [1, F]
        'joint_fire_count': joint_fire_count,  # [T, 1, F]
        'index_layout': [
            {'level': g['level'], 'position': g['position'],
             'values': g['values'].clone(),
             'start': g['start'], 'end': g['end']}
            for g in index_layout
        ],
        'targets': targets,
        'token_positions': torch.tensor([-1], dtype=torch.long),
        'H_theoretical': _h_theoretical_from_rules(rules, n, v),
        'H_per_feature': _h_per_feature_from_joint(
            joint_fire_count, firing_count, index_layout),
        'mode': 'mean_pooled',
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
            act_scale=act_scales[k], rules=rules,
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

    # Labels (all four schemes)
    from scripts.circuit_tracing.labels import build_labels_per_layer
    from scripts.sae_sweep.absorption import leak_table_from_rules
    leak_norms = [leak_table_from_rules(rules, art['rhm'],
                                        art['H_theoretical'])[0]
                  for art in eval_artifacts]
    labels_per_layer = build_labels_per_layer(
        eval_artifacts, s=s, L=L, leak_norms=leak_norms)
    sample_label = next(iter(labels_per_layer[0].values()))
    print(f'sample label at L0 (parent scheme): '
          f'{sample_label["schemes"]["parent"]}')

    # Build DAG
    nodes = build_node_table(
        z_list, labels_per_layer, s=s, L=L,
        embed_anchor=anchors.embed, x_input=x_input,
        logits=anchors.logits, y_true=y_true,
        v=v_, n=n_,
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
        # Pure-IO nodes are never removed by the scoring stages, but the
        # reachability trim (intentional) drops embed/err/logit nodes that end
        # up on no surviving input->logit path (e.g. non-true-class logits in
        # true_class mode). Assert at least one logit survives and that every
        # kept pure-IO node is a real node table entry.
        kept_logits = [k for k in kept_keys if k[0] == 'logit']
        assert kept_logits, 'no logit node survived pruning'
        for k in kept_keys:
            if k[0] in ('embed', 'err', 'logit'):
                assert k in nodes, f'kept {k!r} not in node table'

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


def pooled_main():
    """E2E smoke test for the pooled-last-layer mode.

    Layers 0..K-2 are per-position SAEs; layer K-1 is a single mean-pooled
    SAE on anchors.pooled. Asserts pooled bit-identity, completeness of the
    pooled final-layer edges, and that every K-2 -> pooled edge has dst
    position 0.
    """
    print('\n=== Pooled-last-layer E2E ===')
    torch.manual_seed(0)

    s, L, v_, n_, m_ = 2, 2, 4, 4, 2
    N = s ** L
    K = L
    rules = sample_rules(v=v_, n=n_, m=m_, s=s, L=L, seed=0)
    trees = sample_trees(num_data=128, rules=rules, prior=None, probs=None, seed=0)

    d = 32
    model = MeanClassificationTransformer(
        vocab_size=v_, block_size=N, embedding_dim=d, num_heads=2,
        ffwd_size=2, num_layers=L, num_classes=n_, dropout=0,
    ).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Per-position SAEs for 0..K-2, pooled SAE for K-1 (all on [d]-dim input).
    saes, act_scales = [], []
    for k in range(K):
        sae = SparseAutoencoder(input_dim=d, latent_dim=24).eval()
        for p in sae.parameters():
            p.requires_grad = False
        with torch.no_grad():
            sae.encoder.bias.uniform_(-0.5, 0.8)
        saes.append(sae)
        act_scales.append(1.2 + 0.1 * k)

    # Pseudo-eval artifacts: per-position for 0..K-2, pooled for K-1.
    eval_artifacts = []
    for k in range(K - 1):
        eval_artifacts.append(_build_pseudo_eval_artifact(
            layer_id=k, latent_dim=saes[k].latent_dim, N=N, s=s, L=L,
            n=n_, v=v_, m=m_, trees=trees, model=model, sae=saes[k],
            act_scale=act_scales[k], rules=rules,
        ))
    eval_artifacts.append(_build_pseudo_eval_artifact_pooled(
        layer_id=K - 1, latent_dim=saes[K - 1].latent_dim, N=N, s=s, L=L,
        n=n_, v=v_, m=m_, trees=trees, model=model, sae=saes[K - 1],
        act_scale=act_scales[K - 1], rules=rules,
    ))
    assert eval_artifacts[K - 1]['firing_count'].shape[0] == 1, \
        eval_artifacts[K - 1]['firing_count'].shape

    input_idx = 0
    x_input = trees[L][input_idx].long()
    y_true = int(trees[0][input_idx].item())
    anchors = capture_anchors(model, x_input)

    # SAE forwards: per-position 0..K-2, pooled K-1.
    z_list, x_hat_list, e_list = [], [], []
    for k in range(K - 1):
        x_k = anchors.blocks[k].r_out
        z, x_hat, e = sae_forward(x_k, saes[k], act_scales[k])
        z_list.append(z); x_hat_list.append(x_hat); e_list.append(e)
    z_p, x_hat_p, e_p = sae_forward_pooled(anchors.pooled, saes[K - 1], act_scales[K - 1])
    z_list.append(z_p); x_hat_list.append(x_hat_p); e_list.append(e_p)
    assert z_p.shape[0] == 1, z_p.shape
    # Pooled SAE reconstruction must recover anchors.pooled.
    pooled_recon_err = (x_hat_p + e_p - anchors.pooled).abs().max().item()
    assert pooled_recon_err < 1e-5, pooled_recon_err

    # (a) Pooled bit-identity: classifier(x_hat_p + e_p) == logits.
    spliced = [x_hat_list[k] + e_list[k] for k in range(K - 1)]
    pooled_out = x_hat_list[K - 1] + e_list[K - 1]
    spliced_logits = linearized_full_forward(model, anchors, spliced, pooled_out=pooled_out)
    bit_err = (spliced_logits - anchors.logits).abs().max().item()
    assert bit_err < 1e-4, bit_err
    print(f'pooled bit-identity err = {bit_err:.3e}')

    # Layer-to-layer: per-position pairs 0..K-3, then K-2 -> pooled.
    layer_pairs_feat, layer_pairs_err = [], []
    for k in range(K - 2):
        feats, errs = edges_layer_to_layer(
            model, k, anchors,
            z_list[k], e_list[k],
            saes[k].decoder.weight, act_scales[k],
            saes[k + 1].encoder.weight, act_scales[k + 1],
            z_list[k + 1],
            mat_threshold=10000,
        )
        layer_pairs_feat.append(feats); layer_pairs_err.append(errs)
    feats_p, errs_p = edges_layer_to_pooled(
        model, anchors,
        z_list[K - 2], e_list[K - 2],
        saes[K - 2].decoder.weight, act_scales[K - 2],
        saes[K - 1].encoder.weight, act_scales[K - 1],
        z_list[K - 1],
    )
    layer_pairs_feat.append(feats_p); layer_pairs_err.append(errs_p)
    # (c) Every K-2 -> pooled edge has dst position 0.
    assert all(q == 0 for (_, _, _, q, _) in feats_p), 'feat dst pos != 0'
    assert all(q == 0 for (_, _, q, _) in errs_p), 'err dst pos != 0'
    print(f'K-2 -> pooled: {len(feats_p)} feat-feat, {len(errs_p)} err-feat '
          f'(all dst pos 0)')

    embed_to_l0 = edges_embedding_to_layer0(
        model, anchors, anchors.embed,
        z_list[0], saes[0].encoder.weight, act_scales[0],
        mat_threshold=10000,
    )

    # Pooled final-layer attribution + completeness check.
    feat_to_logit, err_to_logit = edges_pooled_final_to_logits_per_class(
        model, z_list[K - 1], e_list[K - 1],
        saes[K - 1].decoder.weight, act_scales[K - 1],
        num_classes=n_,
    )
    # (b) Completeness: sum of all feat + err edges per class equals the
    # logit produced by the classifier on anchors.pooled (== logits, minus the
    # part not mediated... here ALL mediated since x_hat + e == pooled).
    W_cls = model.classifier.weight
    b_cls = model.classifier.bias
    for c in range(n_):
        s_feat = sum(w for (_, _, cc, w) in feat_to_logit if cc == c)
        s_err = sum(w for (_, cc, w) in err_to_logit if cc == c)
        expected = float((W_cls[c] @ anchors.pooled).item())
        got = s_feat + s_err
        assert abs(got - expected) < 1e-3, (c, got, expected)
    # And summed with bias reproduces the true logits.
    for c in range(n_):
        s_feat = sum(w for (_, _, cc, w) in feat_to_logit if cc == c)
        s_err = sum(w for (_, cc, w) in err_to_logit if cc == c)
        recon_logit = s_feat + s_err + float(b_cls[c].item())
        assert abs(recon_logit - float(anchors.logits[c].item())) < 1e-3, \
            (c, recon_logit, float(anchors.logits[c].item()))
    print(f'pooled final-layer completeness OK ({len(feat_to_logit)} feat->logit, '
          f'{len(err_to_logit)} err->logit)')

    from scripts.circuit_tracing.labels import build_labels_per_layer
    from scripts.sae_sweep.absorption import leak_table_from_rules
    leak_norms = [leak_table_from_rules(rules, art['rhm'],
                                        art['H_theoretical'])[0]
                  for art in eval_artifacts]
    labels_per_layer = build_labels_per_layer(
        eval_artifacts, s=s, L=L, leak_norms=leak_norms)
    # Pooled layer labels are keyed (0, f); the parent (block-aligned) scheme
    # maps to the root (level 0). Other schemes may argmin/reassign elsewhere,
    # so check the parent scheme specifically.
    pooled_labels = labels_per_layer[K - 1]
    assert all(p == 0 for (p, _f) in pooled_labels.keys()), 'pooled label pos != 0'
    sample = next((lab for lab in pooled_labels.values()
                   if lab['schemes']['parent']['value'] is not None), None)
    if sample is not None:
        assert sample['schemes']['parent']['level'] == 0, \
            sample['schemes']['parent']['level']
    print(f'pooled labels: {len(pooled_labels)} (all pos 0, parent scheme level 0)')

    nodes = build_node_table(
        z_list, labels_per_layer, s=s, L=L,
        embed_anchor=anchors.embed, x_input=x_input,
        logits=anchors.logits, y_true=y_true,
        v=v_, n=n_, pooled_last_layer=True,
    )
    # Last layer must contribute a single position (p=0) for feat + err.
    feat_km1_pos = {k[2] for k in nodes if k[0] == 'feat' and k[1] == K - 1}
    err_km1_pos = {k[2] for k in nodes if k[0] == 'err' and k[1] == K - 1}
    assert feat_km1_pos.issubset({0}), feat_km1_pos
    assert err_km1_pos == {0}, err_km1_pos
    print(f'pooled K-1 nodes at single position 0 OK')

    edges = assemble_edges(layer_pairs_feat, layer_pairs_err, embed_to_l0,
                           feat_to_logit, err_to_logit, K=K)

    # Pruning still works (DAG ordering holds: K-2 < K-1 < logits at K).
    pruned, kept_keys, diag = prune_indirect_influence(
        nodes, edges, K=K,
        logits=anchors.logits, y_true=y_true,
        sink_mode='softmax_logits',
        node_threshold=0.8, edge_threshold=0.98,
    )
    print(f'pruning OK: edges {diag["n_edges_pre"]} -> {diag["n_edges_post"]}, '
          f'completeness={diag["completeness_score"]:.3f}')

    # Fidelity guards: pooled final feat->feat slot is None.
    from scripts.circuit_tracing.fidelity import (
        compute_pre_prune_fidelity, compute_postprune_alignment,
    )
    pre_fid = compute_pre_prune_fidelity(
        layer_pairs_feat=layer_pairs_feat, layer_pairs_err=layer_pairs_err,
        embed_to_l0=embed_to_l0, feat_to_logit=feat_to_logit,
        err_to_logit=err_to_logit, logits=anchors.logits, y_true=y_true,
        num_classes=n_, sink_mode='softmax_logits', s=s, L=L, N=N,
        pooled_last_layer=True,
    )
    # per_layer_subtree_alignment = [embed, k=0..K-2]; final slot (k=K-2) None.
    assert pre_fid['per_layer_subtree_alignment'][-1] is None, \
        pre_fid['per_layer_subtree_alignment']
    post_fid = compute_postprune_alignment(pruned, s=s, L=L, N=N,
                                           pooled_last_layer=True)
    print(f'fidelity guards OK (final align slot = '
          f'{pre_fid["per_layer_subtree_alignment"][-1]})')

    print('\nPooled-last-layer E2E smoke test PASSED')


if __name__ == '__main__':
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('all', 'perpos'):
        main()
    if mode in ('all', 'pooled'):
        pooled_main()
