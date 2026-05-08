"""Block-level linearization helpers for circuit tracing.

We freeze, on a single forward pass:
- The post-softmax attention pattern at each block.
- The MLP's first-layer ReLU mask at each block.
- Both LayerNorms' (mean, rstd) at each block, plus ln_f.

A LinearizedTransformer applies the same parameter ops as the original model,
but with attention pattern, ReLU mask, and LayerNorm normalization frozen at
their captured anchors. Around the anchor it is exactly the original model.
For an arbitrary perturbation v of the residual stream entering block k+1,
the linearized output is

    M_{k+1}(v) = LinBlock_{k+1}(x_anchor + v) - LinBlock_{k+1}(x_anchor)

which is a true linear function of v (constants from anchors all cancel).

All anchor capture and linearization happens in float32 on the model's device.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import (
    DecoderBlock,
    NoResidualDecoderBlock,
    MeanClassificationTransformer,
    MeanClassificationTransformerNoResidual,
)


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

class BlockAnchors:
    """Captured forward-pass state for a single block, used to linearize it.

    Shapes assume a single-input forward pass (batch is squeezed):
      r_in       : [N, d]    residual stream entering ln1
      ln1_mean   : [N, 1]
      ln1_rstd   : [N, 1]
      attn_pattern : [num_heads, N, N]  post-softmax (also post-dropout, but
                                        dropout is off in eval)
      r_post_attn: [N, d]    residual stream after attention sublayer
                             (= r_in + attn_out if has_residual else attn_out)
      ln2_mean   : [N, 1]
      ln2_rstd   : [N, 1]
      mlp_relu_mask : [N, nn_dim] {0,1} float, the sign of MLP hidden pre-act
      r_out      : [N, d]    block output (post-block residual stream, x_k)
    """

    def __init__(self):
        self.r_in = None
        self.ln1_mean = None
        self.ln1_rstd = None
        self.attn_pattern = None
        self.r_post_attn = None
        self.ln2_mean = None
        self.ln2_rstd = None
        self.mlp_relu_mask = None
        self.r_out = None


class FullAnchors:
    """Captured anchors for the whole forward pass on a single input.

    embed       : [N, d]   token + position embedding (block 0 input)
    blocks      : list of K BlockAnchors
    ln_f_input  : [N, d]   x_{K-1}, equals blocks[K-1].r_out
    ln_f_mean   : [N, 1]
    ln_f_rstd   : [N, 1]
    pooled      : [d]      mean over positions of ln_f(x_{K-1})
    logits      : [num_classes]
    """

    def __init__(self):
        self.embed = None
        self.blocks: list[BlockAnchors] = []
        self.ln_f_input = None
        self.ln_f_mean = None
        self.ln_f_rstd = None
        self.pooled = None
        self.logits = None


# ---------------------------------------------------------------------------
# LayerNorm helpers
# ---------------------------------------------------------------------------

def _layernorm_stats(x: torch.Tensor, eps: float):
    """Compute mean / rstd along the last dim of x. Returns (mean, rstd).

    Matches nn.LayerNorm's internal computation: variance is biased
    (i.e. divided by D, not D-1).
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    return mean, rstd


def _layernorm_apply(x: torch.Tensor, ln: nn.LayerNorm,
                     mean: torch.Tensor, rstd: torch.Tensor) -> torch.Tensor:
    """Apply a LayerNorm with frozen mean and rstd.

    Equivalent to nn.LayerNorm at its anchor only when mean and rstd were
    captured from the same x. Used to linearize: at anchor, exact match;
    for a perturbed x, this stays affine in x.
    """
    normed = (x - mean) * rstd
    if ln.elementwise_affine:
        normed = normed * ln.weight + ln.bias
    return normed


# ---------------------------------------------------------------------------
# Linearized block
# ---------------------------------------------------------------------------

def _attn_linear_apply(block_attn, x_pre_proj: torch.Tensor,
                       attn_pattern: torch.Tensor) -> torch.Tensor:
    """Apply MultiHeadAttention with frozen post-softmax pattern.

    x_pre_proj : [N, d]  input to the attention sublayer (post-ln1).
    attn_pattern : [num_heads, N, N]  frozen post-softmax weights.

    Returns: [N, d]  attention output, same scalings as the original module.
    """
    a = block_attn  # MultiHeadAttention
    N, C = x_pre_proj.shape
    H = a.num_heads
    Dh = a.head_dim

    # Match transformer.py:65 -- v has the C**-0.5 scaling baked in.
    v = F.linear(x_pre_proj, a.value, bias=None) * (C ** -0.5)
    v = v.view(N, H, Dh).transpose(0, 1)  # [H, N, Dh]

    # Frozen pattern is [H, N, N]; we apply it directly.
    out = attn_pattern @ v  # [H, N, Dh]
    out = out.transpose(0, 1).reshape(N, H * Dh)  # [N, d]

    # transformer.py:73 -- projection has its own scaling.
    out = F.linear(out, a.projection, bias=None) * (a.projection.size(-1) ** -0.5)
    return out


def _mlp_linear_apply(block_ffwd, x: torch.Tensor,
                      relu_mask: torch.Tensor) -> torch.Tensor:
    """Apply the MLP with frozen ReLU mask.

    Mirrors models/fcn.py MLP with num_layers=1: a single MyLinear -> ReLU,
    then readout / sqrt(nn_dim). MyLinear divides by sqrt(input_dim).
    """
    mlp = block_ffwd
    # hidden is Sequential([Sequential([MyLinear, ReLU])]) when num_layers=1.
    # Pull out the single MyLinear.
    inner_seq = mlp.hidden[0]  # Sequential([MyLinear, ReLU])
    my_linear = inner_seq[0]  # MyLinear

    # MyLinear.forward: F.linear(x, weight, bias) / sqrt(input_dim).
    pre_act = F.linear(x, my_linear.weight, my_linear.bias) / (x.size(-1) ** 0.5)
    # Frozen ReLU: multiply by mask instead of applying ReLU.
    hidden = pre_act * relu_mask
    # Readout: x @ readout / norm.
    out = hidden @ mlp.readout / mlp.norm
    return out


def _linearized_block_forward(
    block, x: torch.Tensor, anchors: BlockAnchors, has_residual: bool,
):
    """Forward pass through a block with frozen attention pattern, ReLU mask,
    and LayerNorm normalization (mean and rstd) from anchors.

    x : [N, d]  block input.
    Returns block output [N, d].

    At x == anchors.r_in, the result equals anchors.r_out exactly (up to fp).
    For arbitrary x, it is an affine function of x.
    """
    # Attention sublayer
    x_norm = _layernorm_apply(x, block.ln1, anchors.ln1_mean, anchors.ln1_rstd)
    attn_out = _attn_linear_apply(block.attn, x_norm, anchors.attn_pattern)
    if has_residual:
        r1 = x + attn_out
    else:
        r1 = attn_out

    # MLP sublayer
    r1_norm = _layernorm_apply(r1, block.ln2, anchors.ln2_mean, anchors.ln2_rstd)
    mlp_out = _mlp_linear_apply(block.ffwd, r1_norm, anchors.mlp_relu_mask)
    if has_residual:
        r2 = r1 + mlp_out
    else:
        r2 = mlp_out

    return r2


# ---------------------------------------------------------------------------
# Anchor capture
# ---------------------------------------------------------------------------

def _unwrap_compiled(model):
    """Return the underlying module when `model` is a torch.compile wrapper.

    init.init_model wraps the transformer in torch.compile when device='cuda'
    (init.py:280), producing an OptimizedModule whose original module lives at
    `_orig_mod`. We need the unwrapped one for isinstance checks and for
    direct sub-module access in the linearization path.
    """
    return getattr(model, '_orig_mod', model)


@torch.no_grad()
def capture_anchors(model, x: torch.Tensor) -> FullAnchors:
    """Run a single forward pass and capture all anchors needed for
    linearization.

    model : MeanClassificationTransformer or MeanClassificationTransformerNoResidual.
    x : [N] long, single input token sequence.

    Returns FullAnchors. Asserts that re-applying the linearized forward at
    the captured anchors reproduces the original logits to fp tolerance.
    """
    model = _unwrap_compiled(model)
    if not isinstance(
        model,
        (MeanClassificationTransformer, MeanClassificationTransformerNoResidual),
    ):
        raise TypeError(
            f'capture_anchors expected MeanClassification(NoResidual)Transformer, '
            f'got {type(model).__name__}'
        )
    has_residual = not isinstance(model, MeanClassificationTransformerNoResidual)

    device = next(model.parameters()).device
    x = x.to(device)
    if x.dim() != 1:
        raise ValueError(f'x must be 1-D [N], got shape {tuple(x.shape)}')
    N = x.size(0)

    full = FullAnchors()

    # Embedding
    tok = model.token_embedding_table(x)  # [N, d]
    pos = model.position_embedding_table(torch.arange(N, device=device))
    embed = tok + pos
    full.embed = embed.detach().clone()

    r = embed
    for k, block in enumerate(model.blocks):
        anc = BlockAnchors()
        anc.r_in = r.detach().clone()

        # ln1 stats
        m1, s1 = _layernorm_stats(r, block.ln1.eps)
        anc.ln1_mean = m1.detach().clone()
        anc.ln1_rstd = s1.detach().clone()

        # Recompute attention pattern explicitly to capture post-softmax weights
        x_norm = _layernorm_apply(r, block.ln1, m1, s1)
        a = block.attn
        Nq, C = x_norm.shape
        # Match transformer.py:63-69
        k_t = F.linear(x_norm, a.key, bias=None).view(Nq, a.num_heads, a.head_dim).transpose(0, 1) * (C ** -0.5)
        q_t = F.linear(x_norm, a.query, bias=None).view(Nq, a.num_heads, a.head_dim).transpose(0, 1) * (C ** -0.5)
        weight = q_t @ k_t.transpose(-2, -1) * (a.head_dim ** -0.5)
        weight = weight.masked_fill(a.tril[:Nq, :Nq] == 0, float('-inf'))
        attn_pattern = F.softmax(weight, dim=-1)
        anc.attn_pattern = attn_pattern.detach().clone()

        # Apply attention sublayer and (maybe) residual
        attn_out = _attn_linear_apply(block.attn, x_norm, attn_pattern)
        r1 = r + attn_out if has_residual else attn_out
        anc.r_post_attn = r1.detach().clone()

        # ln2 stats
        m2, s2 = _layernorm_stats(r1, block.ln2.eps)
        anc.ln2_mean = m2.detach().clone()
        anc.ln2_rstd = s2.detach().clone()

        # MLP first-layer pre-activation -> ReLU mask
        r1_norm = _layernorm_apply(r1, block.ln2, m2, s2)
        mlp = block.ffwd
        my_linear = mlp.hidden[0][0]
        pre_act = F.linear(r1_norm, my_linear.weight, my_linear.bias) / (r1_norm.size(-1) ** 0.5)
        anc.mlp_relu_mask = (pre_act > 0).to(pre_act.dtype).detach().clone()

        # Apply MLP and (maybe) residual
        mlp_out = _mlp_linear_apply(block.ffwd, r1_norm, anc.mlp_relu_mask)
        r_out = r1 + mlp_out if has_residual else mlp_out
        anc.r_out = r_out.detach().clone()

        full.blocks.append(anc)
        r = r_out

    # ln_f and head
    full.ln_f_input = r.detach().clone()
    mf, sf = _layernorm_stats(r, model.ln_f.eps)
    full.ln_f_mean = mf.detach().clone()
    full.ln_f_rstd = sf.detach().clone()
    r_lnf = _layernorm_apply(r, model.ln_f, mf, sf)
    pooled = r_lnf.mean(dim=0)
    full.pooled = pooled.detach().clone()
    logits = model.classifier(pooled)
    full.logits = logits.detach().clone()

    # Sanity: original model logits (eval mode is required, no dropout).
    if model.training:
        raise RuntimeError(
            'capture_anchors expects model.eval(); dropout would diverge from '
            'the linearization.'
        )
    ref_logits = model(x.unsqueeze(0)).squeeze(0)
    err = (full.logits - ref_logits).abs().max().item()
    if err > 1e-4:
        raise RuntimeError(
            f'capture_anchors: reconstructed logits differ from original by '
            f'{err:.3e} (max abs). Anchor capture is broken.'
        )

    return full


# ---------------------------------------------------------------------------
# Linearized full forward (for bit-identity check and edge computation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def linearized_block_forward(model, k: int, x_in: torch.Tensor,
                              anchors: FullAnchors) -> torch.Tensor:
    """Apply the linearized block k to an arbitrary input x_in [N, d].

    At x_in == anchors.blocks[k].r_in, returns anchors.blocks[k].r_out
    exactly. For other x_in, returns an affine function of x_in (frozen
    attention pattern, frozen ReLU mask, frozen LayerNorm scales).
    """
    model = _unwrap_compiled(model)
    has_residual = not isinstance(model, MeanClassificationTransformerNoResidual)
    block = model.blocks[k]
    anc = anchors.blocks[k]
    return _linearized_block_forward(block, x_in, anc, has_residual)


@torch.no_grad()
def linearized_lnf_pool(model, x_lnf_in: torch.Tensor,
                         anchors: FullAnchors) -> torch.Tensor:
    """Apply ln_f (with frozen mean/rstd) and mean-pool to x_lnf_in [N, d].
    Returns [d]. The classifier itself is just nn.Linear so callers apply it
    directly.
    """
    model = _unwrap_compiled(model)
    r = _layernorm_apply(x_lnf_in, model.ln_f, anchors.ln_f_mean, anchors.ln_f_rstd)
    return r.mean(dim=0)


# ---------------------------------------------------------------------------
# Build M_{k+1} as a closure or as an explicit matrix
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_M(model, k_next: int, anchors: FullAnchors):
    """Return a callable M(v) implementing the linearized block k_next on a
    perturbation v [N, d] of the residual stream entering that block.

    M(v) = LinBlock_{k_next}(anchors.blocks[k_next].r_in + v)
           - LinBlock_{k_next}(anchors.blocks[k_next].r_in)

    This is exactly linear in v: the second term is constant in v and
    cancels constant offsets coming from LayerNorm bias / attention bias.
    """
    model = _unwrap_compiled(model)
    has_residual = not isinstance(model, MeanClassificationTransformerNoResidual)
    block = model.blocks[k_next]
    anc = anchors.blocks[k_next]
    r_in = anc.r_in
    base = _linearized_block_forward(block, r_in, anc, has_residual)

    def M(v: torch.Tensor) -> torch.Tensor:
        return _linearized_block_forward(block, r_in + v, anc, has_residual) - base

    return M


@torch.no_grad()
def materialize_M(model, k_next: int, anchors: FullAnchors,
                  device: str | None = None) -> torch.Tensor:
    """Materialize the linearized block k_next as a [N*d, N*d] matrix.

    Column j of the matrix is M(e_j) flattened, where e_j is the j-th canonical
    basis vector in R^{N*d}. Built one column at a time to avoid holding two
    [N*d, N*d] tensors.

    The caller is responsible for checking N*d is small enough to fit.
    """
    M = make_M(model, k_next, anchors)
    r_in = anchors.blocks[k_next].r_in
    N, d = r_in.shape
    Nd = N * d
    if device is None:
        device = r_in.device
    mat = torch.zeros(Nd, Nd, dtype=r_in.dtype, device=device)
    eye = torch.zeros(N, d, dtype=r_in.dtype, device=device)
    for j in range(Nd):
        p, c = divmod(j, d)
        eye[p, c] = 1.0
        out = M(eye)  # [N, d]
        mat[:, j] = out.reshape(-1)
        eye[p, c] = 0.0
    return mat


@torch.no_grad()
def linearized_full_forward(model, anchors: FullAnchors,
                             x_layer_outputs: list[torch.Tensor]) -> torch.Tensor:
    """Re-run the model in linearized mode by feeding x_layer_outputs[k] as
    the output of block k. Used for the bit-identity check.

    x_layer_outputs : list of length K. x_layer_outputs[k] is fed as the
                      input to block k+1 (and as ln_f input for k = K-1).
                      Pass anchors.blocks[k].r_out at every k to get back
                      the original logits exactly.

    Returns logits [num_classes].
    """
    model = _unwrap_compiled(model)
    has_residual = not isinstance(model, MeanClassificationTransformerNoResidual)
    K = len(model.blocks)
    if len(x_layer_outputs) != K:
        raise ValueError(
            f'expected {K} layer outputs, got {len(x_layer_outputs)}'
        )

    r = anchors.embed
    for k, block in enumerate(model.blocks):
        if k == 0:
            r = anchors.embed
        else:
            r = x_layer_outputs[k - 1]
        # Apply linearized block k to r; we don't actually use the output
        # because we splice in x_layer_outputs[k]. But we need to keep going
        # with the spliced value.
        _ = _linearized_block_forward(block, r, anchors.blocks[k], has_residual)
    # The spliced output of the last block is x_layer_outputs[K-1].
    final = x_layer_outputs[K - 1]
    pooled = linearized_lnf_pool(model, final, anchors)
    return model.classifier(pooled)
