"""Edge-weight computation for circuit tracing.

Conventions (matching .docs/circuit_tracing_handoff.md and our linearize.py):

  - x_k          : [N, d]   post-block residual stream at layer k.
  - z_k          : [N, F_k] raw post-ReLU SAE encoder output (no decoder norm).
  - x_hat_k, e_k : [N, d]   SAE reconstruction (in residual-stream space, after
                            dividing by act_scale_k) and error.
  - W_enc_k      : [F_k, d]   encoder weight (= sae.encoder.weight).
  - b_enc_k      : [F_k]      encoder bias.
  - W_dec_k      : [d, F_k]   decoder weight (= sae.decoder.weight).
  - act_scale_k  : float      scalar multiplied on x_k before the SAE encoder
                              and divided out from the SAE decoder output.

The SAE pre-activation at layer k+1 is computed on the *scaled* input:

    pre_{k+1}[q, j] = W_enc_{k+1}[j, :] @ (act_scale_{k+1} * x_{k+1}[q, :])
                      + b_enc_{k+1}[j].

We decompose x_{k+1} = block_{k+1}(x_k) = block_{k+1}(x_hat_k + e_k) and use
the linearized form M_{k+1} from circuit_tracing.linearize. With M_{k+1}
defined as a centered map (M(0) = 0), only the perturbation parts contribute
to attribution; the constant offset is bundled into the bias.

Source decomposition at layer k:

    x_hat_k = sum_{i, p : z_k[p,i] > 0}  v_{i, p}
        with v_{i, p}[r, :] = delta(r = p) * z_k[p, i] * (W_dec_k[:, i] / act_scale_k).

The error contribution is
    err_p[r, :] = delta(r = p) * e_k[p, :]
applied at position p only.

Per-edge contribution to the SAE pre-activation at (j, q):

    feat -> feat:  c_{j,q <- i,p} = act_scale_{k+1} * W_enc_{k+1}[j] . M_{k+1}(v_{i,p})[q]
    err  -> feat:  c_{j,q <- err,p} = act_scale_{k+1} * W_enc_{k+1}[j] . M_{k+1}(err_p)[q]

Embedding -> layer-0 feature edges use the same pattern with M_0 (= make_M
with k_next=0, anchored at anchors.blocks[0].r_in == anchors.embed) and the
per-position embedding rows as source vectors:

    embed -> feat 0: c_{j,q <- embed,p} = act_scale_0 * W_enc_0[j] . M_0(embed_p)[q]

For the final layer K-1, the "next stage" is ln_f -> mean-pool -> classifier.
We freeze ln_f's mean/rstd (anchors.ln_f_mean, anchors.ln_f_rstd) so it is
also affine. With ln_f_lin(v) the centered linearization of ln_f at its
anchor:

    feat -> logit_c:  W_cls[c] . ln_f_pool_lin(v_{i,p})
                      where ln_f_pool_lin = mean over rows of the centered
                      ln_f linearization. When v is nonzero only at row p,
                      this equals (1/N) * W_cls[c] . ln_f_jvp_at_p(W_dec_{K-1}[:, i] / act_scale_{K-1}) * z_{K-1}[p, i]
                      from the spec (the 1/N comes from mean-pool).
    err  -> logit_c:  W_cls[c] . ln_f_pool_lin(err_p)
                      = (1/N) * W_cls[c] . ln_f_jvp_at_p(e_{K-1}[p]).

One edge per (i, p, c) and (p, c) -- per-class final-layer attribution.
"""

from __future__ import annotations

import torch

from circuit_tracing.linearize import (
    FullAnchors, make_M, materialize_M,
    _layernorm_apply,
)


# ---------------------------------------------------------------------------
# SAE forward + error nodes
# ---------------------------------------------------------------------------

@torch.no_grad()
def sae_forward(x: torch.Tensor, sae, act_scale: float):
    """Run an SAE on residual-stream activations x [N, d] using act_scale.

    Returns (z, x_hat, e):
      z     : [N, F]  raw post-ReLU encoder activations.
      x_hat : [N, d]  reconstruction in residual-stream space (i.e. divided
                       back by act_scale).
      e     : [N, d]  error node (x - x_hat). By construction x_hat + e = x.
    """
    x_scaled = act_scale * x  # [N, d]
    z = torch.relu(x_scaled @ sae.encoder.weight.T + sae.encoder.bias)
    x_hat_scaled = z @ sae.decoder.weight.T + sae.decoder.bias
    x_hat = x_hat_scaled / act_scale
    e = x - x_hat
    return z, x_hat, e


# ---------------------------------------------------------------------------
# Materialized-matrix path (used when N*d is small enough)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _dense_src_edges_matrix(
    M_T: torch.Tensor,          # T = einsum('jc,qcpd->qpjd', W_enc_dst, M4): [N, N, F_dst, d]
    src_vecs: torch.Tensor,     # [N, d] -- source vector at each position p
    act_scale_dst: float,
    active_dst: torch.Tensor,   # [N, F_dst] bool
):
    """Edges (p, j, q, w) where the source vector at position p is given
    directly (no z, no W_dec): one source per position.

    Used for both error -> dst-feature edges (src_vecs = e_k) and
    embedding -> layer-0-feature edges (src_vecs = anchors.embed).

    contribution[j, q <- p] = act_scale_dst * W_enc_dst[j] @ M[q, :, p, :] @ src_vecs[p, :]

    Filters to active dst (active_dst[q, j]).
    """
    N = src_vecs.size(0)
    if not active_dst.any():
        return []
    # E[q, p, j] = T[q, p, j, c] @ src_vecs[p, c]
    E = torch.einsum('qpjd,pd->qpj', M_T, src_vecs) * act_scale_dst
    dst_idx = active_dst.nonzero(as_tuple=False)  # [num_dst, 2] (q, j)
    q_dst = dst_idx[:, 0]
    j_dst = dst_idx[:, 1]
    num_dst = dst_idx.size(0)
    # Gather E at active (q, j) for all p: shape [num_dst, N].
    E_dst = E[q_dst, :, j_dst]  # [num_dst, N]
    p_all = torch.arange(N, device=E.device)
    p_col = p_all.unsqueeze(0).expand(num_dst, N).reshape(-1)
    j_col = j_dst.unsqueeze(1).expand(num_dst, N).reshape(-1)
    q_col = q_dst.unsqueeze(1).expand(num_dst, N).reshape(-1)
    w_col = E_dst.reshape(-1)
    return list(zip(
        p_col.tolist(), j_col.tolist(),
        q_col.tolist(), w_col.tolist(),
    ))


@torch.no_grad()
def _dense_src_edges_closure(
    M,                          # callable v -> M(v), v: [N, d]
    src_vecs: torch.Tensor,     # [N, d]
    W_enc_dst: torch.Tensor,    # [F_dst, d]
    act_scale_dst: float,
    active_dst: torch.Tensor,   # [N, F_dst] bool
):
    """Closure-path version of _dense_src_edges_matrix. One M call per source
    position p (N total). Slower, used when N*d > mat_threshold.
    """
    N, d = src_vecs.shape
    if not active_dst.any():
        return []
    Wenc_scaled = act_scale_dst * W_enc_dst  # [F_dst, d]
    dst_idx = active_dst.nonzero(as_tuple=False)
    edges = []
    v = torch.zeros(N, d, device=src_vecs.device, dtype=src_vecs.dtype)
    for p in range(N):
        v.zero_()
        v[p] = src_vecs[p]
        out = M(v)  # [N, d]
        proj = out @ Wenc_scaled.T  # [N, F_dst]
        for t in range(dst_idx.size(0)):
            q, j = int(dst_idx[t, 0]), int(dst_idx[t, 1])
            edges.append((p, j, q, float(proj[q, j].item())))
    return edges


@torch.no_grad()
def _edges_via_matrix(
    M_mat: torch.Tensor,       # [N*d, N*d]
    z_k: torch.Tensor,          # [N, F_k]
    e_k: torch.Tensor,          # [N, d]
    W_dec_k: torch.Tensor,      # [d, F_k]
    act_scale_k: float,
    W_enc_kp1: torch.Tensor,    # [F_kp1, d]
    act_scale_kp1: float,
    z_kp1: torch.Tensor,        # [N, F_kp1] -- used to filter active dst features
):
    """Compute all attribution edges from layer k to layer k+1 SAE pre-activations
    using the materialized M.

    Returns:
      feat_edges : list of (i, p, j, q, weight)
      err_edges  : list of (p, j, q, weight)
    Only edges to (j, q) with z_kp1[q, j] > 0 are returned.
    """
    N, d = e_k.shape
    F_k = z_k.size(1)
    F_kp1 = W_enc_kp1.size(0)
    Nd = N * d

    # Reshape M as [N, d, N, d] for clarity: M[q, c1, p, c2].
    M4 = M_mat.view(N, d, N, d)

    active_dst = z_kp1 > 0  # [N, F_kp1]
    if not active_dst.any():
        return [], []

    # Shared pre-projection: T[q, p, j, c2] = W_enc_kp1[j, c1] @ M[q, c1, p, c2].
    T = torch.einsum('jc,qcpd->qpjd', W_enc_kp1, M4)

    # ---- feat -> feat ----
    active_src = z_k > 0  # [N, F_k]
    src_idx = active_src.nonzero(as_tuple=False)  # [num_src, 2] (p, i)
    dst_idx = active_dst.nonzero(as_tuple=False)  # [num_dst, 2] (q, j)
    num_src = src_idx.size(0)
    num_dst = dst_idx.size(0)

    if num_src == 0:
        feat_edges = []
    else:
        # B[q, p, j, i] = T[q, p, j, c2] @ W_dec_k[c2, i]
        B = torch.einsum('qpjd,di->qpji', T, W_dec_k)
        scale = act_scale_kp1 / act_scale_k
        contrib = scale * B * z_k.unsqueeze(0).unsqueeze(2)  # [q, p, j, i] * z_k[p, i]

        p_src = src_idx[:, 0]
        i_src = src_idx[:, 1]
        q_dst = dst_idx[:, 0]
        j_dst = dst_idx[:, 1]
        # contrib has shape [N, N, F_kp1, F_k] (axes: q, p, j, i).
        # Step 1: pick along (p, i) axes -> [num_src, N(q), F_kp1]
        sel_src = contrib[:, p_src, :, :].permute(1, 0, 2, 3)  # [num_src, N(q), F_kp1, F_k]
        sel_src = sel_src[torch.arange(num_src, device=contrib.device), :, :, i_src]  # [num_src, N, F_kp1]
        W_edges = sel_src[:, q_dst, j_dst]  # [num_src, num_dst]
        i_col = i_src.unsqueeze(1).expand(num_src, num_dst).reshape(-1)
        p_col = p_src.unsqueeze(1).expand(num_src, num_dst).reshape(-1)
        j_col = j_dst.unsqueeze(0).expand(num_src, num_dst).reshape(-1)
        q_col = q_dst.unsqueeze(0).expand(num_src, num_dst).reshape(-1)
        w_col = W_edges.reshape(-1)
        feat_edges = list(zip(
            i_col.tolist(), p_col.tolist(),
            j_col.tolist(), q_col.tolist(),
            w_col.tolist(),
        ))

    # ---- err -> feat ----  (reuses the dense-source helper)
    err_edges = _dense_src_edges_matrix(T, e_k, act_scale_kp1, active_dst)

    return feat_edges, err_edges


# ---------------------------------------------------------------------------
# Closure path (fallback when N*d is large)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _edges_via_closure(
    M,                          # callable v -> M(v), v: [N, d]
    z_k: torch.Tensor,          # [N, F_k]
    e_k: torch.Tensor,          # [N, d]
    W_dec_k: torch.Tensor,      # [d, F_k]
    act_scale_k: float,
    W_enc_kp1: torch.Tensor,    # [F_kp1, d]
    act_scale_kp1: float,
    z_kp1: torch.Tensor,        # [N, F_kp1]
):
    """Same outputs as _edges_via_matrix, computed by running M on each
    active source vector. One M call per active feature plus one per error
    node per position.
    """
    N, d = e_k.shape
    F_k = z_k.size(1)
    active_src = z_k > 0  # [N, F_k]
    active_dst = z_kp1 > 0  # [N, F_kp1]
    if not active_dst.any():
        return [], []

    Wenc_scaled = act_scale_kp1 * W_enc_kp1  # [F_kp1, d]

    # ---- feat -> feat ----
    feat_edges = []
    src_idx = active_src.nonzero(as_tuple=False)
    dst_idx = active_dst.nonzero(as_tuple=False)
    v = torch.zeros(N, d, device=e_k.device, dtype=e_k.dtype)
    for s in range(src_idx.size(0)):
        p, i = int(src_idx[s, 0]), int(src_idx[s, 1])
        v.zero_()
        v[p] = z_k[p, i] * (W_dec_k[:, i] / act_scale_k)
        out = M(v)  # [N, d]
        proj = out @ Wenc_scaled.T  # [N, F_kp1]
        for t in range(dst_idx.size(0)):
            q, j = int(dst_idx[t, 0]), int(dst_idx[t, 1])
            feat_edges.append((i, p, j, q, float(proj[q, j].item())))

    # ---- err -> feat ----  (reuses the dense-source closure helper)
    err_edges = _dense_src_edges_closure(M, e_k, W_enc_kp1, act_scale_kp1, active_dst)

    return feat_edges, err_edges


@torch.no_grad()
def edges_layer_to_layer(
    model, k: int, anchors: FullAnchors,
    z_k: torch.Tensor, e_k: torch.Tensor,
    W_dec_k: torch.Tensor, act_scale_k: float,
    W_enc_kp1: torch.Tensor, act_scale_kp1: float,
    z_kp1: torch.Tensor,
    mat_threshold: int,
):
    """Dispatch to matrix or closure path based on N*d <= mat_threshold."""
    N, d = e_k.shape
    if N * d <= mat_threshold:
        M_mat = materialize_M(model, k + 1, anchors)
        return _edges_via_matrix(
            M_mat, z_k, e_k, W_dec_k, act_scale_k,
            W_enc_kp1, act_scale_kp1, z_kp1,
        )
    M = make_M(model, k + 1, anchors)
    return _edges_via_closure(
        M, z_k, e_k, W_dec_k, act_scale_k,
        W_enc_kp1, act_scale_kp1, z_kp1,
    )


# ---------------------------------------------------------------------------
# Embedding -> layer-0-feature edges (M_0)
# ---------------------------------------------------------------------------

@torch.no_grad()
def edges_embedding_to_layer0(
    model, anchors: FullAnchors,
    embed_anchor: torch.Tensor,    # [N, d] anchors.embed (= tok+pos rows)
    z_0: torch.Tensor,              # [N, F_0] layer-0 SAE activations (active filter)
    W_enc_0: torch.Tensor,          # [F_0, d] layer-0 SAE encoder
    act_scale_0: float,
    mat_threshold: int,
):
    """Compute embedding -> layer-0-feature attribution edges:

        contribution[j, q <- ('embed', p)]
          = act_scale_0 * W_enc_0[j] @ M_0(delta_p (x) embed_anchor[p])[q]

    where M_0 is the centered linearization of block 0 anchored at
    anchors.blocks[0].r_in == anchors.embed (so injecting the row-p part of
    the embedding gives the per-position embedding contribution to layer-0
    activations).

    Returns: list of (p, j, q, weight). One entry per (p, active dst (j, q)).
    Filtered to z_0[q, j] > 0.
    """
    N, d = embed_anchor.shape
    active_dst = z_0 > 0
    if not active_dst.any():
        return []

    if N * d <= mat_threshold:
        M_mat = materialize_M(model, 0, anchors)
        M4 = M_mat.view(N, d, N, d)
        # T[q, p, j, c2] = W_enc_0[j, c1] @ M0[q, c1, p, c2]
        T = torch.einsum('jc,qcpd->qpjd', W_enc_0, M4)
        return _dense_src_edges_matrix(T, embed_anchor, act_scale_0, active_dst)
    M = make_M(model, 0, anchors)
    return _dense_src_edges_closure(M, embed_anchor, W_enc_0, act_scale_0, active_dst)


# ---------------------------------------------------------------------------
# Final-layer attribution to per-class logits
# ---------------------------------------------------------------------------

@torch.no_grad()
def _lnf_pool_lin(v: torch.Tensor, model, anchors: FullAnchors) -> torch.Tensor:
    """Centered linearization of (ln_f -> mean-pool) at the anchor.

    Given v [N, d], returns a [d] vector equal to
        mean_pool(ln_f_lin(anchor + v)) - mean_pool(ln_f_lin(anchor)).
    Since ln_f_lin is linear-affine and mean-pool is linear, this is a
    linear function of v.

    Note: when v is nonzero only at row p, this equals
        (1/N) * ln_f_jvp_at_p(v[p])
    (the 1/N comes from mean-pool over N rows). So
    `W_cls[c] @ _lnf_pool_lin(v_p_only)` ==
    `(1/N) * W_cls[c] @ ln_f_jvp_at_p(v[p])`, matching the spec.
    """
    base = _layernorm_apply(
        anchors.ln_f_input, model.ln_f, anchors.ln_f_mean, anchors.ln_f_rstd
    ).mean(dim=0)
    pert = _layernorm_apply(
        anchors.ln_f_input + v, model.ln_f, anchors.ln_f_mean, anchors.ln_f_rstd
    ).mean(dim=0)
    return pert - base


@torch.no_grad()
def edges_final_layer_to_logits_per_class(
    model, anchors: FullAnchors,
    z_K: torch.Tensor, e_K: torch.Tensor,
    W_dec_K: torch.Tensor, act_scale_K: float,
    num_classes: int,
):
    """Per-class final-layer attribution.

    For each class c and each active feature (i, p) at layer K-1:
        attr[c <- (K-1, p, i)] = W_cls[c] @ _lnf_pool_lin(v_{i,p})
            (= (1/N) * W_cls[c] @ ln_f_jvp_at_p(W_dec_{K-1}[:,i]/act_scale_{K-1}) * z_{K-1}[p,i])

    For each class c and each error position p at layer K-1:
        attr[c <- (K-1, p, 'error')] = W_cls[c] @ _lnf_pool_lin(err_p)
            (= (1/N) * W_cls[c] @ ln_f_jvp_at_p(e_{K-1}[p]))

    Returns:
      feat_to_logit : list of (i, p, c, weight)
      err_to_logit  : list of (p, c, weight)
    """
    N, d = e_K.shape
    F_K = z_K.size(1)
    W_cls = model.classifier.weight  # [num_classes, d]

    active_src = z_K > 0
    src_idx = active_src.nonzero(as_tuple=False)

    v = torch.zeros(N, d, device=e_K.device, dtype=e_K.dtype)

    feat_to_logit = []
    for s in range(src_idx.size(0)):
        p, i = int(src_idx[s, 0]), int(src_idx[s, 1])
        v.zero_()
        v[p] = z_K[p, i] * (W_dec_K[:, i] / act_scale_K)
        pooled_v = _lnf_pool_lin(v, model, anchors)  # [d]
        # Vectorized over classes: W_cls @ pooled_v gives [num_classes].
        per_class = W_cls @ pooled_v  # [num_classes]
        for c in range(num_classes):
            feat_to_logit.append((i, p, c, float(per_class[c].item())))

    err_to_logit = []
    for p in range(N):
        v.zero_()
        v[p] = e_K[p]
        pooled_v = _lnf_pool_lin(v, model, anchors)
        per_class = W_cls @ pooled_v  # [num_classes]
        for c in range(num_classes):
            err_to_logit.append((p, c, float(per_class[c].item())))

    return feat_to_logit, err_to_logit
