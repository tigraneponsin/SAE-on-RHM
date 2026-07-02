"""Self-contained smoke test for circuit_tracing.linearize.

Builds a small MeanClassificationTransformer and a small
MeanClassificationTransformerNoResidual from scratch, runs anchor capture on
a random input, and verifies:

  1. Linearized full forward at the anchors reproduces the original logits
     to fp32 tolerance.
  2. M_{k+1}(0) == 0 (the closure cancels the anchor by construction).
  3. M_{k+1} is linear: M(a*v1 + b*v2) == a*M(v1) + b*M(v2) within fp32.
  4. The materialized matrix at small sizes agrees with the closure on
     random vectors.

Run with:  python -m circuit_tracing._smoketest
"""

from __future__ import annotations

import torch

from models.transformer import (
    MeanClassificationTransformer,
    MeanClassificationTransformerNoResidual,
    FreeClassificationTransformer,
    FreeClassificationTransformerNoResidual,
)
from circuit_tracing.linearize import (
    capture_anchors, make_M, materialize_M, linearized_full_forward,
)


def _build_tiny(variant: str, vocab_size=8, block_size=4, embedding_dim=16,
                num_heads=2, ffwd_size=2, num_layers=2, num_classes=4, seed=0):
    torch.manual_seed(seed)
    classes = {
        'meanclass': MeanClassificationTransformer,
        'nores': MeanClassificationTransformerNoResidual,
        'freeclass': FreeClassificationTransformer,
        'freeclass_nores': FreeClassificationTransformerNoResidual,
    }
    if variant not in classes:
        raise ValueError(variant)
    model = classes[variant](
        vocab_size=vocab_size, block_size=block_size,
        embedding_dim=embedding_dim, num_heads=num_heads,
        ffwd_size=ffwd_size, num_layers=num_layers,
        num_classes=num_classes, dropout=0,
    )
    # For freeclass, set NON-uniform pooling logits so the weighted-pool path
    # (w[p] != 1/N) is actually exercised vs the uniform-mean meanclass path.
    if variant.startswith('freeclass'):
        with torch.no_grad():
            model.pool_logits.copy_(torch.randn(block_size))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _check_variant(variant: str):
    print(f'\n=== variant = {variant} ===')
    model = _build_tiny(variant)
    device = 'cpu'
    model = model.to(device)
    N = model.block_size
    x = torch.randint(0, 8, (N,), device=device)

    # Anchor capture (also internally asserts logits match)
    anchors = capture_anchors(model, x)
    print(f'anchor capture OK  (logits = {anchors.logits.tolist()})')

    # 1. Linearized full forward at anchors == original logits.
    layer_outs = [b.r_out for b in anchors.blocks]
    spliced_logits = linearized_full_forward(model, anchors, layer_outs)
    err = (spliced_logits - anchors.logits).abs().max().item()
    print(f'1. linearized full @ anchors max err = {err:.3e}')
    assert err < 1e-4, err

    # 2-4. Per-block tests
    K = len(model.blocks)
    for k in range(K):
        M = make_M(model, k, anchors)
        N_, d = anchors.blocks[k].r_in.shape

        # 2. M(0) == 0
        zero = torch.zeros(N_, d, device=device)
        out0 = M(zero)
        err0 = out0.abs().max().item()
        print(f'  block {k}: M(0) max abs = {err0:.3e}')
        assert err0 < 1e-5, err0

        # 3. linearity
        v1 = torch.randn(N_, d, device=device)
        v2 = torch.randn(N_, d, device=device)
        a, b = 1.7, -2.3
        lhs = M(a * v1 + b * v2)
        rhs = a * M(v1) + b * M(v2)
        err_lin = (lhs - rhs).abs().max().item()
        print(f'  block {k}: linearity max err = {err_lin:.3e}')
        assert err_lin < 1e-3, err_lin

        # 4. matrix vs closure (only for small blocks)
        if N_ * d <= 1024:
            mat = materialize_M(model, k, anchors)
            v = torch.randn(N_, d, device=device)
            out_clo = M(v).reshape(-1)
            out_mat = mat @ v.reshape(-1)
            err_mat = (out_clo - out_mat).abs().max().item()
            print(f'  block {k}: matrix vs closure max err = {err_mat:.3e}')
            assert err_mat < 1e-3, err_mat

    print(f'variant {variant} PASSED')


def main():
    _check_variant('meanclass')
    _check_variant('nores')
    _check_variant('freeclass')
    _check_variant('freeclass_nores')
    print('\nAll smoke tests passed.')


if __name__ == '__main__':
    main()
