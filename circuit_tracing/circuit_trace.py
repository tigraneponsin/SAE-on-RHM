"""Per-input circuit tracing CLI.

Given a transformer checkpoint, K SAE checkpoints (bottom-to-top), and K
matching .sae_eval.pt artifacts (for latent labels), build the linearized
attribution DAG for a single input and save:

  nodes.pt, edges.pt, fidelity.pt, graph.gpickle.

Usage:
  python -m circuit_tracing.circuit_trace \
      --train_output /path/transformer.pt \
      --sae_ckpts L0.pt L1.pt L2.pt \
      --sae_eval_artifacts L0.sae_eval.pt L1.sae_eval.pt L2.sae_eval.pt \
      --input_idx 42 \
      --eval_seed 0 \
      --eval_size 1024 \
      --sink_mode softmax_logits \
      --node_threshold 0.8 --edge_threshold 0.98 \
      --output_dir runs/circuit_trace/example
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch

# Make REPO_ROOT importable for `scripts.*` and `models.*`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import init  # noqa: E402
from datasets.random_hierarchy_model import sample_trees  # noqa: E402
from scripts.common.sae_loading import load_transformer, load_sae  # noqa: E402

from circuit_tracing.linearize import (
    capture_anchors, linearized_full_forward, _unwrap_compiled,
)
from circuit_tracing.attribution import (
    sae_forward, sae_forward_pooled, edges_layer_to_layer,
    edges_layer_to_pooled,
    edges_embedding_to_layer0,
    edges_final_layer_to_logits_per_class,
    edges_pooled_final_to_logits_per_class,
)
from circuit_tracing.labels import build_labels_per_layer
from absorption import leak_table_from_rules
from circuit_tracing.dag import (
    assemble_edges, build_node_table, to_networkx,
)
from circuit_tracing.prune import prune_indirect_influence
from circuit_tracing.fidelity import (
    compute_pre_prune_fidelity, compute_postprune_alignment,
)
from circuit_tracing.grouping import group_by_signature


# ---------------------------------------------------------------------------
# Rules-consistency
# ---------------------------------------------------------------------------

def _check_rules_consistency(transformer_rules_source: str,
                              transformer_seed_rules: int,
                              sae_records: list[dict],
                              eval_artifacts: list[dict]):
    """Hard-fail if any of the loaded artifacts disagree on RHM rules.

    Acceptable cases:
      - All have rules_source == 'artifact': each loaded the saved rules
        object directly. Same rules guaranteed.
      - Some have rules_source == 'seed_rules_resampled' but with the same
        rules_seed as the transformer. Then the regenerated rules are bit-
        identical because sample_rules is deterministic on (v, n, m, s, L,
        seed).

    Anything else is a mismatch.
    """
    msgs = []
    msgs.append(f'  transformer  rules_source = {transformer_rules_source!r}, '
                f'seed_rules = {transformer_seed_rules}')

    def _ok(src: str, seed):
        if src == 'artifact':
            return True
        if src == 'seed_rules_resampled' and int(seed) == int(transformer_seed_rules):
            return True
        return False

    for k, rec in enumerate(sae_records):
        sae_src = rec.get('sae_rules_source')
        sae_seed = None
        ckpt = torch.load(rec['ckpt_path'], map_location='cpu', weights_only=False)
        sae_seed = ckpt.get('sae_dataset_split', {}).get('rules_seed')
        msgs.append(f'  SAE layer {k}: rules_source = {sae_src!r}, rules_seed = {sae_seed}')
        if sae_src is None:
            msgs.append(f'    WARNING: SAE layer {k} predates rules_source field; '
                        f'cannot verify. Proceeding (assume artifact).')
            continue
        if not _ok(sae_src, sae_seed if sae_seed is not None else transformer_seed_rules):
            raise RuntimeError(
                f'Rules-consistency FAILED for SAE layer {k}: '
                f'rules_source={sae_src!r}, seed={sae_seed} vs transformer '
                f'rules_source={transformer_rules_source!r}, seed={transformer_seed_rules}'
            )

    for k, art in enumerate(eval_artifacts):
        art_src = art.get('rules_source')
        sae_art_src = art.get('sae_rules_source')
        msgs.append(f'  artifact L{k}: rules_source = {art_src!r}, '
                    f'sae_rules_source = {sae_art_src!r}')
        if art_src is None:
            msgs.append(f'    WARNING: artifact L{k} has no rules_source field; '
                        f'cannot verify.')
            continue
        if art_src == 'seed_rules_resampled':
            pass
        elif art_src != 'artifact':
            raise RuntimeError(
                f'Rules-consistency FAILED for artifact L{k}: '
                f'rules_source={art_src!r}'
            )

    print('Rules-consistency check:')
    for m in msgs:
        print(m)


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------

class _RemovedFlag(argparse.Action):
    """Argparse action that raises a clear migration error.

    Used for `--prune_fraction`, replaced by `--node_threshold` and
    `--edge_threshold`.
    """
    def __init__(self, option_strings, dest, message: str = '', **kwargs):
        kwargs.setdefault('nargs', '?')
        kwargs.setdefault('default', argparse.SUPPRESS)
        self._migration_message = message
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        raise SystemExit(
            f'\nERROR: {option_string} has been removed.\n  {self._migration_message}'
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """argparse for the single-config CLI. Sweep CLI shares most flags."""
    p = argparse.ArgumentParser()
    p.add_argument('--train_output', required=True,
                   help='Path to transformer checkpoint (with output.model and output.rules).')
    p.add_argument('--sae_ckpts', nargs='+', required=True,
                   help='K SAE checkpoint paths, in bottom-to-top layer order.')
    p.add_argument('--sae_eval_artifacts', nargs='+', required=True,
                   help='K .sae_eval.pt artifact paths matching --sae_ckpts.')
    p.add_argument('--input_idx', type=int, required=True,
                   help='Row index into the freshly sampled eval set.')
    p.add_argument('--eval_seed', type=int, default=0,
                   help='Seed for sampling the eval trees we attribute against.')
    p.add_argument('--eval_size', type=int, default=1024,
                   help='Number of eval samples to draw (so input_idx must be < eval_size).')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--mat_threshold', type=int, default=8192,
                   help='Materialize M as [N*d, N*d] when N*d <= mat_threshold.')
    p.add_argument('--sink_mode', choices=['softmax_logits', 'true_class'],
                   default='softmax_logits',
                   help='How to weight logit nodes for indirect-influence pruning.')
    p.add_argument('--node_threshold', type=float, default=0.8,
                   help='Indirect-influence node-prune threshold.')
    p.add_argument('--edge_threshold', type=float, default=0.98,
                   help='Indirect-influence edge-prune threshold.')
    p.add_argument(
        '--prune_fraction', action=_RemovedFlag,
        message=('Use --node_threshold (default 0.8) and --edge_threshold '
                 '(default 0.98) with --sink_mode {softmax_logits|true_class} '
                 'instead. The old per-node top-fraction-by-|weight| rule '
                 'has been replaced by indirect-influence pruning.'),
    )
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--model_variant', default=None, choices=['best', 'last'],
                   help='Transformer weights to load. If unset, auto-detect from '
                        'SAE checkpoints (all SAEs must agree). If set, must '
                        'match the variant the SAEs were trained on.')
    p.add_argument('--pooled_last_layer', action='store_true', default=False,
                   help='Treat the last-layer SAE as a single mean-pooled SAE '
                        '(post-ln_f pooled [d] space) instead of per-position. '
                        'Auto-detected from the last SAE\'s sae_activation_source '
                        '== mean_pooled; this flag forces it on and is validated '
                        'against the SAEs.')
    p.add_argument('--label_alpha', type=float, default=0.5,
                   help='Absorption reassignment threshold for the '
                        '"reassigned" label scheme (default 0.5). A feature is '
                        'reassigned from its whole-tree child cell up to the '
                        'tree-parent when H(parent|f)/H(parent|child=c_i) < '
                        'label_alpha.')
    p.add_argument('--label_primary', default='reassigned',
                   choices=['parent', 'level', 'whole_tree', 'reassigned'],
                   help='Which label scheme the back-compat top-level node/'
                        'group label fields mirror (default reassigned). All '
                        'four schemes are always stored under node["schemes"].')
    return p


def prepare_pipeline(args) -> dict:
    """Run the threshold-independent stages of the tracer.

    Loads the transformer + SAEs + eval artifacts, captures anchors, runs
    SAE forwards and the bit-identity check, computes all attribution
    edges, builds labels, the node table, the full edge list, and the
    pre-prune fidelity dict. Returns a dict consumed by
    `finalize_one_config`. Safe to call once and re-use across many
    (node_threshold, edge_threshold) cells.

    `args` is the parsed `argparse.Namespace`; the only field consumed
    by per-cell pruning is `args.sink_mode`, which is still captured
    here so it can be threaded through.
    """

    # ---- 0. Resolve model_variant from SAE checkpoints ----
    # The transformer MUST be loaded with the same variant the SAEs were
    # trained on; feeding OOD activations to the SAE inflates active-feature
    # counts and reconstruction error. Peek each SAE checkpoint without
    # instantiating the model (input_dim unknown until the transformer is
    # loaded). All SAEs must agree.
    sae_variants = []
    for path in args.sae_ckpts:
        peek = load_sae(path, input_dim=None, device='cpu', load_model=False)
        if peek is None:
            raise RuntimeError(f'Could not load SAE checkpoint at {path}')
        sae_variants.append(peek.get('model_variant') or 'last')
    unique_variants = sorted(set(sae_variants))
    if len(unique_variants) > 1:
        details = ', '.join(f'{p}: {v!r}' for p, v in zip(args.sae_ckpts, sae_variants))
        raise RuntimeError(
            f'SAE checkpoints disagree on transformer model_variant ({details}). '
            f'All SAEs must be trained on the same transformer variant.'
        )
    sae_variant = unique_variants[0]
    if args.model_variant is None:
        print(f'Auto-detected model_variant={sae_variant!r} from SAE checkpoints.')
    elif args.model_variant != sae_variant:
        raise RuntimeError(
            f'Model-variant mismatch: SAEs were trained on transformer variant '
            f'{sae_variant!r}, but --model_variant is {args.model_variant!r}. '
            f'Rerun with --model_variant {sae_variant}, or omit --model_variant '
            f'to auto-detect.'
        )
    args.model_variant = sae_variant

    # ---- 1. Load transformer (using the variant the SAEs were trained on) ----
    print(f'Loading transformer from {args.train_output} (model_variant={sae_variant})')
    model, _loader, cfg, rules, rules_source = load_transformer(
        train_output_path=args.train_output,
        eval_size=args.eval_size,
        eval_seed=args.eval_seed,
        batch_size=args.eval_size,
        device=args.device,
        shuffle=False,
        model_variant=sae_variant,
    )
    print(f'  cfg.model = {cfg.model}, num_layers = {cfg.num_layers}, '
          f's = {cfg.tuple_size}, L = {cfg.num_layers}, v = {cfg.num_features}, '
          f'n = {cfg.num_classes}, m = {cfg.num_synonyms}')
    K = cfg.num_layers
    s = cfg.tuple_size
    L = cfg.num_layers
    N = s ** L
    num_classes = int(cfg.num_classes)

    if cfg.model not in (
        'transformer_meanclass', 'transformer_meanclass_nores',
        'transformer_freeclass', 'transformer_freeclass_nores',
    ):
        raise RuntimeError(
            f'Unsupported model variant: {cfg.model!r}. circuit_trace currently '
            f'supports only transformer_meanclass, transformer_meanclass_nores, '
            f'transformer_freeclass, and transformer_freeclass_nores.'
        )

    # ---- 2. Load SAEs ----
    if len(args.sae_ckpts) != K:
        raise RuntimeError(
            f'expected {K} SAE checkpoints (one per layer), got {len(args.sae_ckpts)}'
        )
    sae_records = []
    for k, path in enumerate(args.sae_ckpts):
        rec = load_sae(path, input_dim=cfg.embedding_dim, device=args.device)
        if rec is None:
            raise RuntimeError(f'Could not load SAE checkpoint at {path}')
        if rec['layer_id'] != k:
            raise RuntimeError(
                f'SAE checkpoint at position {k} reports layer_id={rec["layer_id"]}; '
                f'expected {k}. Pass --sae_ckpts in bottom-to-top order.'
            )
        # Treat missing model_variant (old checkpoints) as 'last' to match load_sae's default.
        sae_variant = rec.get('model_variant') or 'last'
        if sae_variant != args.model_variant:
            raise RuntimeError(
                f'Model-variant mismatch at SAE layer {k}: SAE was trained on '
                f'transformer variant {sae_variant!r}, but --model_variant is '
                f'{args.model_variant!r}. Feeding OOD activations to the SAE '
                f'inflates active-feature counts and reconstruction error. '
                f'Rerun with --model_variant {sae_variant}.'
            )
        sae_records.append(rec)

    # ---- 2b. Resolve pooled-last-layer mode (flag OR auto-detect) ----
    # A mean_pooled last-layer SAE lives in post-ln_f pooled [d] space. Detect
    # it from the SAE's recorded sae_activation_source and reconcile with the
    # explicit --pooled_last_layer flag.
    sae_sources = [rec.get('sae_activation_source', 'all_tokens') for rec in sae_records]
    detected_pooled = (sae_sources[K - 1] == 'mean_pooled')
    requested_pooled = bool(getattr(args, 'pooled_last_layer', False))
    pooled_last_layer = requested_pooled or detected_pooled
    if pooled_last_layer:
        if sae_sources[K - 1] != 'mean_pooled':
            raise RuntimeError(
                f'--pooled_last_layer was requested but the last SAE (layer '
                f'{K - 1}) has sae_activation_source={sae_sources[K - 1]!r}, not '
                f"'mean_pooled'. Train the last-layer SAE with "
                f'sae_activation_source=mean_pooled.'
            )
        bad = [k for k in range(K - 1) if sae_sources[k] == 'mean_pooled']
        if bad:
            raise RuntimeError(
                f'pooled_last_layer mode requires layers 0..K-2 to be '
                f'per-position SAEs, but layers {bad} have '
                f'sae_activation_source=mean_pooled.'
            )
        if requested_pooled and not detected_pooled:
            # Should be unreachable given the check above, kept for clarity.
            pass
        print(f'Pooled-last-layer mode ON (last SAE source = '
              f'{sae_sources[K - 1]!r}; '
              f'{"flag" if requested_pooled else "auto-detected"}).')
    elif detected_pooled:
        # Unreachable: detected_pooled implies pooled_last_layer True.
        pass
    args.pooled_last_layer = pooled_last_layer

    # ---- 3. Load eval artifacts ----
    if len(args.sae_eval_artifacts) != K:
        raise RuntimeError(
            f'expected {K} eval artifacts, got {len(args.sae_eval_artifacts)}'
        )
    eval_artifacts = []
    for k, path in enumerate(args.sae_eval_artifacts):
        art = torch.load(path, map_location='cpu', weights_only=False)
        if int(art.get('layer_id', -1)) != k:
            raise RuntimeError(
                f'Eval artifact at position {k} has layer_id={art.get("layer_id")}; '
                f'expected {k}.'
            )
        if pooled_last_layer and k == K - 1:
            fc = art.get('firing_count')
            if fc is None or int(fc.shape[0]) != 1:
                raise RuntimeError(
                    f'pooled_last_layer mode expects the last eval artifact to '
                    f'have firing_count [1, F] (single pooled position), got '
                    f'shape {None if fc is None else tuple(fc.shape)}. Re-run the '
                    f'eval on the mean_pooled SAE.'
                )
        rhm_art = art.get('rhm', {})
        if int(rhm_art.get('s', -1)) != s or int(rhm_art.get('L', -1)) != L:
            raise RuntimeError(
                f'Eval artifact at L{k} has rhm={rhm_art}; expected s={s}, L={L}.'
            )
        if int(art.get('latent_dim', -1)) != int(sae_records[k]['latent_dim']):
            raise RuntimeError(
                f'Eval artifact at L{k} latent_dim mismatch: '
                f'{art.get("latent_dim")} vs SAE {sae_records[k]["latent_dim"]}.'
            )
        eval_artifacts.append(art)

    # ---- 4. Rules-consistency ----
    _check_rules_consistency(rules_source, cfg.seed_rules, sae_records, eval_artifacts)

    # ---- 5. Sample trees and pick the single input ----
    trees = sample_trees(num_data=args.eval_size, rules=rules, prior=None,
                         probs=None, seed=args.eval_seed)
    if args.input_idx < 0 or args.input_idx >= args.eval_size:
        raise RuntimeError(
            f'--input_idx {args.input_idx} out of range for eval_size={args.eval_size}'
        )
    x_input = trees[L][args.input_idx].long().to(args.device)  # [N]
    y_true = int(trees[0][args.input_idx].item())
    print(f'Input #{args.input_idx}: leaves={x_input.tolist()}, y_true={y_true}')

    # ---- 6. Anchor capture ----
    model = _unwrap_compiled(model)
    model.eval()
    anchors = capture_anchors(model, x_input)
    print('Anchor capture and sanity check OK.')

    # ---- 7. SAE forwards + error nodes ----
    # Per-position layers run on the post-block residual stream. In pooled
    # mode the last layer (K-1) instead runs on anchors.pooled (post-ln_f
    # mean-pooled [d]) and stores a single pooled node (z [1, F]).
    z_list, x_hat_list, e_list = [], [], []
    n_perpos = K - 1 if pooled_last_layer else K
    for k in range(n_perpos):
        x_k = anchors.blocks[k].r_out
        z, x_hat, e = sae_forward(x_k, sae_records[k]['sae'], sae_records[k]['act_scale'])
        recon_err = (x_hat + e - x_k).abs().max().item()
        if recon_err > 5e-4:
            raise RuntimeError(
                f'SAE bit-identity FAILED at layer {k}: max abs err = {recon_err}'
            )
        z_list.append(z)
        x_hat_list.append(x_hat)
        e_list.append(e)
        print(f'  Layer {k}: z active = {(z > 0).sum().item()} / {z.numel()}, '
              f'||e|| = {e.norm().item():.3f}, ||x|| = {x_k.norm().item():.3f}')

    if pooled_last_layer:
        pooled_vec = anchors.pooled  # [d] post-ln_f mean-pooled
        z_p, x_hat_p, e_p = sae_forward_pooled(
            pooled_vec, sae_records[K - 1]['sae'], sae_records[K - 1]['act_scale'],
        )
        recon_err = (x_hat_p + e_p - pooled_vec).abs().max().item()
        if recon_err > 5e-4:
            raise RuntimeError(
                f'Pooled SAE bit-identity FAILED at layer {K - 1}: '
                f'max abs err = {recon_err}'
            )
        z_list.append(z_p)        # [1, F]
        x_hat_list.append(x_hat_p)  # [d]
        e_list.append(e_p)          # [d]
        print(f'  Layer {K - 1} (pooled): z active = {(z_p > 0).sum().item()} / '
              f'{z_p.numel()}, ||e|| = {e_p.norm().item():.3f}, '
              f'||pooled|| = {pooled_vec.norm().item():.3f}')

    # ---- 8. Whole-pipeline bit-identity ----
    if pooled_last_layer:
        spliced = [x_hat_list[k] + e_list[k] for k in range(K - 1)]
        pooled_out = x_hat_list[K - 1] + e_list[K - 1]  # [d]
        spliced_logits = linearized_full_forward(
            model, anchors, spliced, pooled_out=pooled_out,
        )
    else:
        spliced = [x_hat_list[k] + e_list[k] for k in range(K)]
        spliced_logits = linearized_full_forward(model, anchors, spliced)
    bit_identity_err = (spliced_logits - anchors.logits).abs().max().item()
    print(f'Bit-identity check: max logit err = {bit_identity_err:.3e}')
    if bit_identity_err > 1e-3:
        raise RuntimeError(
            f'Bit-identity check FAILED: max logit err {bit_identity_err}'
        )

    # ---- 9. Layer-to-layer attribution ----
    # Per-position pairs k -> k+1 for k in 0..(last_perpos_pair-1). In pooled
    # mode the final pair (K-2 -> pooled K-1) uses edges_layer_to_pooled.
    layer_pairs_feat = []
    layer_pairs_err = []
    n_perpos_pairs = K - 2 if pooled_last_layer else K - 1
    for k in range(n_perpos_pairs):
        sae_k = sae_records[k]['sae']
        sae_kp1 = sae_records[k + 1]['sae']
        feats, errs = edges_layer_to_layer(
            model, k, anchors,
            z_list[k], e_list[k],
            sae_k.decoder.weight, sae_records[k]['act_scale'],
            sae_kp1.encoder.weight, sae_records[k + 1]['act_scale'],
            z_list[k + 1],
            mat_threshold=args.mat_threshold,
        )
        layer_pairs_feat.append(feats)
        layer_pairs_err.append(errs)
        print(f'  edges k={k} -> k+1: {len(feats)} feat-feat, {len(errs)} err-feat')

    if pooled_last_layer:
        sae_km2 = sae_records[K - 2]['sae']
        sae_pooled = sae_records[K - 1]['sae']
        feats, errs = edges_layer_to_pooled(
            model, anchors,
            z_list[K - 2], e_list[K - 2],
            sae_km2.decoder.weight, sae_records[K - 2]['act_scale'],
            sae_pooled.encoder.weight, sae_records[K - 1]['act_scale'],
            z_list[K - 1],
        )
        layer_pairs_feat.append(feats)
        layer_pairs_err.append(errs)
        print(f'  edges k={K - 2} -> pooled K-1: {len(feats)} feat-feat, '
              f'{len(errs)} err-feat')

    # ---- 10. Embedding -> layer-0 attribution ----
    sae_0 = sae_records[0]['sae']
    embed_to_l0 = edges_embedding_to_layer0(
        model, anchors, anchors.embed,
        z_list[0],
        sae_0.encoder.weight, sae_records[0]['act_scale'],
        mat_threshold=args.mat_threshold,
    )
    print(f'  edges embed -> layer 0: {len(embed_to_l0)}')

    # ---- 11. Per-class final-layer attribution ----
    if pooled_last_layer:
        # Pooled feature/error already live in post-ln_f pooled space: no
        # _lnf_pool_lin, no 1/N, single position 0.
        feat_to_logit, err_to_logit = edges_pooled_final_to_logits_per_class(
            model,
            z_list[K - 1], e_list[K - 1],
            sae_records[K - 1]['sae'].decoder.weight, sae_records[K - 1]['act_scale'],
            num_classes=num_classes,
        )
    else:
        feat_to_logit, err_to_logit = edges_final_layer_to_logits_per_class(
            model, anchors,
            z_list[K - 1], e_list[K - 1],
            sae_records[K - 1]['sae'].decoder.weight, sae_records[K - 1]['act_scale'],
            num_classes=num_classes,
        )
    print(f'  Final-layer per-class edges: {len(feat_to_logit)} feat->logit, '
          f'{len(err_to_logit)} err->logit, {num_classes} classes')

    # ---- 12. Build labels (all four schemes) ----
    # Value-specific structural leak table for the absorption ("reassigned")
    # scheme, built from the rules we already loaded (rules-consistency was
    # checked in step 4), per layer (each uses its artifact's H_theoretical).
    label_alpha = float(getattr(args, 'label_alpha', 0.5))
    label_primary = getattr(args, 'label_primary', 'reassigned')
    leak_norms = []
    for art in eval_artifacts:
        leak_norm, _rows = leak_table_from_rules(
            rules, art['rhm'], art['H_theoretical'])
        leak_norms.append(leak_norm)
    labels_per_layer = build_labels_per_layer(
        eval_artifacts, s=s, L=L, leak_norms=leak_norms,
        alpha=label_alpha, primary=label_primary,
    )

    # ---- 13. Build node table and edges, prune ----
    nodes = build_node_table(
        z_list, labels_per_layer, s=s, L=L,
        embed_anchor=anchors.embed, x_input=x_input,
        logits=anchors.logits, y_true=y_true,
        v=int(cfg.num_features), n=int(cfg.num_classes),
        pooled_last_layer=pooled_last_layer,
        primary=label_primary,
    )
    all_edges = assemble_edges(
        layer_pairs_feat, layer_pairs_err,
        embed_to_l0,
        feat_to_logit, err_to_logit, K=K,
    )

    # Pre-prune fidelity diagnostics (depend on attribution edges only).
    pre_fid = compute_pre_prune_fidelity(
        layer_pairs_feat=layer_pairs_feat,
        layer_pairs_err=layer_pairs_err,
        embed_to_l0=embed_to_l0,
        feat_to_logit=feat_to_logit,
        err_to_logit=err_to_logit,
        logits=anchors.logits,
        y_true=y_true,
        num_classes=num_classes,
        sink_mode=args.sink_mode,
        s=s, L=L, N=N,
        pooled_last_layer=pooled_last_layer,
    )

    # Predicted / runner-up class (threshold-independent).
    sorted_idx = torch.argsort(anchors.logits, descending=True)
    y_pred = int(sorted_idx[0].item())
    y_runner = int(sorted_idx[1].item()) if len(sorted_idx) > 1 else y_pred

    # Per-input RHM tree row (level 0 = root scalar, level L = leaves of
    # length s^L).
    tree_row = {l: trees[l][args.input_idx].cpu() for l in range(L + 1)}

    return {
        'cfg': cfg,
        'K': K, 's': s, 'L': L, 'N': N, 'num_classes': num_classes,
        'nodes': nodes,
        'all_edges': all_edges,
        'logits': anchors.logits,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_runner': y_runner,
        'bit_identity_max_err': bit_identity_err,
        'pre_fid': pre_fid,
        'sink_mode': args.sink_mode,
        'input_idx': args.input_idx,
        'eval_seed': args.eval_seed,
        'tree_row': tree_row,
        'pooled_last_layer': pooled_last_layer,
        'label_alpha': label_alpha,
        'label_primary': label_primary,
    }


def finalize_one_config(prepared: dict,
                        node_threshold: float,
                        edge_threshold: float,
                        out_dir: Path) -> dict:
    """Run the threshold-dependent stages and persist artifacts.

    Pruning + post-prune alignment + signature grouping + on-disk save.
    Returns a small cell-summary dict suitable for inclusion in
    `sweep_summary.json`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = prepared['cfg']
    K = prepared['K']; s = prepared['s']; L = prepared['L']; N = prepared['N']
    num_classes = prepared['num_classes']
    nodes = prepared['nodes']
    all_edges = prepared['all_edges']
    logits = prepared['logits']
    y_true = prepared['y_true']
    y_pred = prepared['y_pred']
    y_runner = prepared['y_runner']
    bit_identity_err = prepared['bit_identity_max_err']
    pre_fid = prepared['pre_fid']
    sink_mode = prepared['sink_mode']
    tree_row = prepared['tree_row']
    pooled_last_layer = prepared.get('pooled_last_layer', False)

    # ---- Prune ----
    pruned_edges, _kept_node_keys, prune_diag = prune_indirect_influence(
        nodes, all_edges, K=K,
        logits=logits, y_true=y_true,
        sink_mode=sink_mode,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        device=logits.device,
    )
    print(f'Pruning ({sink_mode}, node_th={node_threshold}, '
          f'edge_th={edge_threshold}):')
    print(f'  edges: {prune_diag["n_edges_pre"]} pre -> {prune_diag["n_edges_post"]} post')
    print(f'  features: {prune_diag["n_features_pre"]} pre -> '
          f'{prune_diag["n_features_post"]} post')
    print(f'  nodes by kind pre  = {prune_diag["n_nodes_pre_by_kind"]}')
    print(f'  nodes by kind post = {prune_diag["n_nodes_post_by_kind"]}')
    print(f'  completeness_score = {prune_diag["completeness_score"]:.3f}')
    print(f'  replacement_score  = {prune_diag["replacement_score"]:.3f}')

    # ---- Post-prune alignment ----
    post_fid = compute_postprune_alignment(
        pruned_edges, s=s, L=L, N=N, pooled_last_layer=pooled_last_layer,
    )
    subtree_alignment_fraction_postprune = post_fid[
        'subtree_alignment_fraction_postprune'
    ]
    per_layer_subtree_alignment_postprune = post_fid[
        'per_layer_subtree_alignment_postprune'
    ]

    # ---- Bottom-up signature grouping ----
    grouped_nodes, grouped_edges, group_membership = group_by_signature(
        nodes=nodes, pruned_edges=pruned_edges, K=K, s=s, L=L,
        v=int(cfg.num_features), n=int(cfg.num_classes),
        primary=prepared.get('label_primary', 'reassigned'),
    )
    n_groups = sum(1 for k in grouped_nodes if k[0] == 'group')
    n_multi_groups = sum(
        1 for k, v_ in grouped_nodes.items()
        if k[0] == 'group' and int(v_.get('n_constituents', 1)) > 1
    )
    print(f'Grouping (bottom-up by incoming signature):')
    print(f'  {n_groups} group nodes total ({n_multi_groups} with >1 constituent)')
    print(f'  {len(grouped_edges)} grouped edges')

    fidelity = {
        'bit_identity_max_err': bit_identity_err,
        'per_layer_error_fraction': pre_fid['per_layer_error_fraction'],
        'feature_mediated_logit_fraction': pre_fid['feature_mediated_logit_fraction'],
        'subtree_alignment_fraction': pre_fid['subtree_alignment_fraction'],
        'per_layer_subtree_alignment': pre_fid['per_layer_subtree_alignment'],
        'subtree_alignment_fraction_postprune': subtree_alignment_fraction_postprune,
        'per_layer_subtree_alignment_postprune': per_layer_subtree_alignment_postprune,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_runner_up': y_runner,
        'logits': logits.cpu(),
        'rhm': {'s': s, 'L': L, 'v': cfg.num_features,
                'n': cfg.num_classes, 'm': cfg.num_synonyms},
        'input_idx': prepared['input_idx'],
        'eval_seed': prepared['eval_seed'],
        'sink_mode': prune_diag['sink_mode'],
        'node_threshold': prune_diag['node_threshold'],
        'edge_threshold': prune_diag['edge_threshold'],
        'n_nodes_pre_by_kind': prune_diag['n_nodes_pre_by_kind'],
        'n_nodes_post_by_kind': prune_diag['n_nodes_post_by_kind'],
        'n_edges_pre_prune': prune_diag['n_edges_pre'],
        'n_edges_post_prune': prune_diag['n_edges_post'],
        'completeness_score': prune_diag['completeness_score'],
        'replacement_score': prune_diag['replacement_score'],
        'completeness_weight_convention': prune_diag['completeness_weight_convention'],
        'n_grouped_nodes': n_groups,
        'n_grouped_nodes_multi': n_multi_groups,
        'n_grouped_edges': len(grouped_edges),
        'pooled_last_layer': pooled_last_layer,
    }

    print('Fidelity:')
    print(f'  bit_identity_max_err              = {bit_identity_err:.3e}')
    for k, frac in enumerate(pre_fid['per_layer_error_fraction']):
        flag = '  <-- HIGH (>0.2)' if frac > 0.2 else ''
        print(f'  per_layer_error_fraction[k={k}-> k+1] = {frac:.3f}{flag}')
    print(f'  feature_mediated_logit_fraction   = {pre_fid["feature_mediated_logit_fraction"]:.3f}')
    # In pooled mode the final feat->feat slot is degenerate (dst pos == 0)
    # and reported as None; print it as 'n/a (pooled)'.
    def _fmt_frac(frac):
        return 'n/a (pooled)' if frac is None else f'{frac:.3f}'

    align_labels = ['embed -> k=0'] + [f'k={k}-> k+1' for k in range(K - 1)]
    for label, frac in zip(align_labels, pre_fid['per_layer_subtree_alignment']):
        print(f'  subtree_alignment[{label:13s}]      = {_fmt_frac(frac)}  (pre-prune)')
    print(f'  subtree_alignment_fraction        = {pre_fid["subtree_alignment_fraction"]:.3f}  (pre-prune)')
    for label, frac in zip(align_labels, per_layer_subtree_alignment_postprune):
        print(f'  subtree_alignment_pp[{label:13s}]    = {_fmt_frac(frac)}  (post-prune)')
    print(f'  subtree_alignment_fraction_pp     = {subtree_alignment_fraction_postprune:.3f}  (post-prune)')

    # ---- Save artifacts ----
    torch.save(nodes, out_dir / 'nodes.pt')
    torch.save({'all': all_edges, 'pruned': pruned_edges}, out_dir / 'edges.pt')
    torch.save(fidelity, out_dir / 'fidelity.pt')
    torch.save(grouped_nodes, out_dir / 'grouped_nodes.pt')
    torch.save(grouped_edges, out_dir / 'grouped_edges.pt')
    torch.save(group_membership, out_dir / 'group_membership.pt')
    torch.save(tree_row, out_dir / 'tree_for_input.pt')

    g = to_networkx(nodes, pruned_edges)
    with open(out_dir / 'graph.gpickle', 'wb') as f:
        pickle.dump(g, f)

    summary = {
        'input_idx': prepared['input_idx'],
        'y_true': y_true, 'y_pred': y_pred, 'y_runner_up': y_runner,
        'bit_identity_max_err': bit_identity_err,
        'per_layer_error_fraction': pre_fid['per_layer_error_fraction'],
        'feature_mediated_logit_fraction': pre_fid['feature_mediated_logit_fraction'],
        'subtree_alignment_fraction': pre_fid['subtree_alignment_fraction'],
        'per_layer_subtree_alignment': pre_fid['per_layer_subtree_alignment'],
        'subtree_alignment_fraction_postprune': subtree_alignment_fraction_postprune,
        'per_layer_subtree_alignment_postprune': per_layer_subtree_alignment_postprune,
        'sink_mode': sink_mode,
        'node_threshold': float(node_threshold),
        'edge_threshold': float(edge_threshold),
        'n_nodes_pre_by_kind': prune_diag['n_nodes_pre_by_kind'],
        'n_nodes_post_by_kind': prune_diag['n_nodes_post_by_kind'],
        'n_edges_pre_prune': prune_diag['n_edges_pre'],
        'n_edges_post_prune': prune_diag['n_edges_post'],
        'completeness_score': prune_diag['completeness_score'],
        'replacement_score': prune_diag['replacement_score'],
        'n_grouped_nodes': n_groups,
        'n_grouped_nodes_multi': n_multi_groups,
        'n_grouped_edges': len(grouped_edges),
        'pooled_last_layer': pooled_last_layer,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSaved to {out_dir}/  (nodes.pt, edges.pt, fidelity.pt, '
          f'grouped_nodes.pt, grouped_edges.pt, group_membership.pt, '
          f'graph.gpickle, summary.json)')

    return {
        'subdir': str(out_dir),
        'node_threshold': float(node_threshold),
        'edge_threshold': float(edge_threshold),
        'n_features_post': int(prune_diag['n_features_post']),
        'n_edges_post': int(prune_diag['n_edges_post']),
        'n_groups': n_groups,
        'n_groups_multi': n_multi_groups,
        'completeness_score': float(prune_diag['completeness_score']),
        'replacement_score': float(prune_diag['replacement_score']),
        'subtree_alignment_fraction_postprune': float(subtree_alignment_fraction_postprune),
    }


def main():
    args = build_parser().parse_args()
    prepared = prepare_pipeline(args)
    finalize_one_config(
        prepared,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        out_dir=Path(args.output_dir),
    )


if __name__ == '__main__':
    main()
