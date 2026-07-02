"""Interactive 3-panel HTML renderer for a saved circuit_trace run.

Produces a single self-contained HTML with three vertically-stacked panels
sharing the leaf-position x-axis:

  Row 1: Pruned UNGROUPED circuit (from nodes.pt + edges.pt['pruned']).
  Row 2: Pruned GROUPED circuit   (from grouped_nodes.pt + grouped_edges.pt).
  Row 3: RHM ground-truth tree    (from tree_for_input.pt).

Features:
  - Zoom / pan / hover via Plotly.
  - Feature / group nodes colored by normalized_entropy (white = uniform,
    green = fully selective; gray fallback when entropy is unavailable).
  - Hover tooltips show layer, position(s), feature index or constituent
    list, z (or z_sum), label_value, P(label_value | fire), normalized
    entropy.

Usage:
  python -m circuit_tracing.visualize_interactive --run_dir <dir>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from circuit_tracing.notation import (
    sae_row_label, report_level, level_ring_color, LEVEL_RING_PALETTE,
)


# ---------------------------------------------------------------------------
# Color helpers (mirror visualize.py)
# ---------------------------------------------------------------------------

_ENTROPY_GREEN = (0x1a, 0x98, 0x50)
_ENTROPY_WHITE = (0xff, 0xff, 0xff)
_ENTROPY_FALLBACK = '#cccccc'


def _entropy_color_hex(norm_entropy) -> str:
    if norm_entropy is None:
        return _ENTROPY_FALLBACK
    t = max(0.0, min(1.0, float(norm_entropy)))
    r = int(round(_ENTROPY_GREEN[0] + (_ENTROPY_WHITE[0] - _ENTROPY_GREEN[0]) * t))
    g = int(round(_ENTROPY_GREEN[1] + (_ENTROPY_WHITE[1] - _ENTROPY_GREEN[1]) * t))
    b = int(round(_ENTROPY_GREEN[2] + (_ENTROPY_WHITE[2] - _ENTROPY_GREEN[2]) * t))
    return f'#{r:02x}{g:02x}{b:02x}'


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _pooled_center_x(N: int) -> float:
    """Horizontal center of the position x-range, used to showcase the single
    pooled last-layer node (matches the single-logit centering convention)."""
    return (N - 1) / 2.0


def _layout_ungrouped(nodes: dict, kept_keys: set, N: int, K: int,
                      num_classes: int,
                      feat_fan_half: float = 0.30,
                      err_x_offset: float = 0.40,
                      pooled_last_layer: bool = False) -> dict:
    """Same layout convention as visualize._compute_positions, kept local
    here to avoid a cross-file import dependency.

    When pooled_last_layer is True, the single layer K-1 node (position 0) is
    centered at x = (N-1)/2 rather than pinned to the left at x=0."""
    pos = {}

    def _base_x(k: int, p: int) -> float:
        if pooled_last_layer and k == K - 1:
            return _pooled_center_x(N)
        return float(p)

    feat_by_cell: dict = {}
    for key in kept_keys:
        if key[0] != 'feat':
            continue
        _, k, p, i = key
        feat_by_cell.setdefault((k, p), []).append(i)
    for cell in feat_by_cell.values():
        cell.sort()

    for (k, p), feat_list in feat_by_cell.items():
        n_feat = len(feat_list)
        if n_feat == 1:
            offsets = [0.0]
        elif n_feat == 2:
            offsets = [-feat_fan_half, +feat_fan_half]
        else:
            offsets = [
                -feat_fan_half + 2 * feat_fan_half * (idx / (n_feat - 1))
                for idx in range(n_feat)
            ]
        base_x = _base_x(k, p)
        for idx, i in enumerate(feat_list):
            pos[('feat', k, p, i)] = (base_x + offsets[idx], float(k))

    for key in kept_keys:
        if key[0] == 'err':
            _, k, p = key
            pos[key] = (_base_x(k, p) + err_x_offset, float(k))
        elif key[0] == 'embed':
            _, p = key
            pos[key] = (float(p), -1.0)
        elif key[0] == 'logit':
            _, c = key
            if num_classes <= 1:
                x = (N - 1) / 2.0
            else:
                x = c * (N - 1) / (num_classes - 1)
            pos[key] = (float(x), float(K))
    return pos


def _layout_grouped(grouped_nodes: dict, kept_keys: set, N: int, K: int,
                    num_classes: int,
                    feat_fan_half: float = 0.30,
                    err_x_offset: float = 0.40,
                    pooled_last_layer: bool = False) -> dict:
    """Same conventions, but for the grouped graph: a 'group' node sits at
    (position_centroid, layer). Multiple groups at the same layer that
    share a centroid get spread horizontally with the same fan rule.

    When pooled_last_layer is True, the layer K-1 group's centroid (0) is
    overridden to the horizontal center x = (N-1)/2."""
    pos = {}

    def _cell_x(layer: int, cx: float) -> float:
        if pooled_last_layer and layer == K - 1:
            return _pooled_center_x(N)
        return cx

    # Bucket group keys by (layer, rounded centroid bucket) to apply fan.
    group_by_cell: dict = {}
    for key in kept_keys:
        if key[0] != 'group':
            continue
        attrs = grouped_nodes[key]
        layer = int(attrs['layer'])
        cx = _cell_x(layer, float(attrs['position_centroid']))
        # Use bucket of integer position so multiple groups at the same
        # integer-ish x don't overlap.
        bucket = (layer, round(cx))
        group_by_cell.setdefault(bucket, []).append((key, cx))
    for (layer, _bucket), members in group_by_cell.items():
        # Sort by centroid for stable left-to-right order.
        members.sort(key=lambda kc: (kc[1], kc[0]))
        n = len(members)
        if n == 1:
            offsets = [0.0]
        elif n == 2:
            offsets = [-feat_fan_half, +feat_fan_half]
        else:
            offsets = [
                -feat_fan_half + 2 * feat_fan_half * (idx / (n - 1))
                for idx in range(n)
            ]
        for idx, (key, cx) in enumerate(members):
            pos[key] = (cx + offsets[idx], float(layer))

    for key in kept_keys:
        if key[0] == 'err':
            _, k, p = key
            pos[key] = (_cell_x(k, float(p)) + err_x_offset, float(k))
        elif key[0] == 'embed':
            _, p = key
            pos[key] = (float(p), -1.0)
        elif key[0] == 'logit':
            _, c = key
            if num_classes <= 1:
                x = (N - 1) / 2.0
            else:
                x = c * (N - 1) / (num_classes - 1)
            pos[key] = (float(x), float(K))
    return pos


# ---------------------------------------------------------------------------
# Edge scaling (mirror visualize._edge_widths_alphas, plotly variant)
# ---------------------------------------------------------------------------

def _edge_widths_alphas(weights, max_width=4.0, min_width=0.2,
                        max_alpha=0.9, min_alpha=0.15):
    if len(weights) == 0:
        return [], []
    abs_w = torch.tensor([abs(w) for w in weights], dtype=torch.float32)
    if abs_w.numel() == 1 or abs_w.max() == 0:
        ref = float(abs_w.max().item()) or 1.0
    else:
        ref = float(torch.quantile(abs_w, 0.95).item())
        if ref <= 0:
            ref = float(abs_w.max().item()) or 1.0
    widths, alphas = [], []
    for w in weights:
        scale = abs(w) / ref if ref > 0 else 0.0
        scale = min(scale, 1.0)
        widths.append(min_width + (max_width - min_width) * scale)
        alphas.append(min_alpha + (max_alpha - min_alpha) * scale)
    return widths, alphas


# ---------------------------------------------------------------------------
# Label-scheme selection
# ---------------------------------------------------------------------------

# A node may carry per-scheme labels under attrs['schemes'][scheme]. When a
# scheme is requested we read the displayed label/entropy/level/position from
# there; otherwise (scheme=None) we use the top-level mirror, which is the
# behavior of all pre-scheme callers.
SCHEME_NAMES = ('parent', 'level', 'whole_tree', 'reassigned')

# Group nodes with disagreement above this fraction get a visible (red) outline
# on the grouped panel, flagging that their signature-merged constituents point
# at different latent cells under the displayed scheme.
DISAGREEMENT_OUTLINE_THRESHOLD = 0.2


def _scheme_view(attrs, scheme):
    """Return the label fields to display for `attrs` under `scheme`.

    Falls back to the node's top-level fields when scheme is None or the node
    has no per-scheme record (errors/embeds/logits, or pre-scheme node tables).
    Returns a dict with at least label_value, p_value_given_fire,
    normalized_entropy, level, position, plus disagreement_frac for groups.
    """
    if scheme is None:
        return {
            'label_value': attrs.get('label_value'),
            'p_value_given_fire': attrs.get('p_value_given_fire', float('nan')),
            'normalized_entropy': attrs.get('normalized_entropy'),
            'level': attrs.get('level'),
            'position': attrs.get('parent_position', attrs.get('position')),
            'disagreement_frac': attrs.get('disagreement_frac'),
            'reassigned': attrs.get('reassigned'),
            'child_level': attrs.get('child_level'),
            'child_position': attrs.get('child_position'),
            'child_value': attrs.get('child_value'),
        }
    sdict = attrs.get('schemes')
    if not sdict or scheme not in sdict:
        return _scheme_view(attrs, None)
    sv = sdict[scheme]
    return {
        'label_value': sv.get('label_value', sv.get('value')),
        'p_value_given_fire': sv.get('p_value_given_fire', float('nan')),
        'normalized_entropy': sv.get('normalized_entropy'),
        'level': sv.get('level'),
        'position': sv.get('position'),
        'disagreement_frac': sv.get('disagreement_frac'),
        'reassigned': sv.get('reassigned'),
        'child_level': sv.get('child_level'),
        'child_position': sv.get('child_position'),
        'child_value': sv.get('child_value'),
    }


# ---------------------------------------------------------------------------
# Hover-text builders
# ---------------------------------------------------------------------------

def _disp_level(code_level, report, L):
    """Level to show in hover: flipped to report convention when report and L
    are available, else the raw code level. None passes through as 'n/a'."""
    if code_level is None:
        return 'n/a'
    if report and L is not None:
        return report_level(code_level, L)
    return int(code_level)


def _hover_feature(key, attrs, report=False, scheme=None, L=None) -> str:
    _, k, p, i = key
    blk = sae_row_label(k, report)
    sv = _scheme_view(attrs, scheme)
    scheme_tag = f'  [{scheme}]' if scheme else ''
    lvl = _disp_level(sv['level'], report, L)
    parts = [
        f'<b>feature · {blk} · pos {p} · f{i}</b>{scheme_tag}',
        f'latent cell  = level {lvl}, pos {sv["position"]}',
        f'label value  = {sv["label_value"]}',
        f'P(label|fire) = {float(sv["p_value_given_fire"]):.3f}',
    ]
    ne = sv['normalized_entropy']
    parts.append(f'entropy      = {ne:.3f}' if ne is not None
                 else 'entropy      = n/a')
    parts.append(f'z            = {float(attrs.get("z", 0)):.3f}')
    if scheme == 'reassigned' and sv.get('reassigned'):
        cl = _disp_level(sv.get('child_level'), report, L)
        parts.append(f'reassigned from child = level {cl}, '
                     f'pos {sv.get("child_position")}, value {sv.get("child_value")}')
    fc = attrs.get('eval_firing_count')
    if fc is not None:
        parts.append(f'firing count = {fc}')
    return '<br>'.join(parts)


def _hover_group(key, attrs, report=False, scheme=None, L=None) -> str:
    _, k, g_idx = key
    blk = sae_row_label(k, report)
    sv = _scheme_view(attrs, scheme)
    scheme_tag = f'  [{scheme}]' if scheme else ''
    lvl = _disp_level(sv['level'], report, L)
    parts = [
        f'<b>group · {blk} · idx {g_idx}</b>{scheme_tag}',
        f'latent cell  = level {lvl}, pos {sv["position"]}',
        f'label value  = {sv["label_value"]}',
        f'P(label|fire) = {float(sv["p_value_given_fire"]):.3f}',
    ]
    ne = sv['normalized_entropy']
    parts.append(f'entropy      = {ne:.3f}' if ne is not None
                 else 'entropy      = n/a')
    parts.append(f'constituents = {attrs["n_constituents"]} '
                 f'@ positions {attrs["positions"]}')
    parts.append(f'z_sum        = {float(attrs["z_sum"]):.3f}')
    constituents = attrs.get('constituents', [])
    if constituents:
        head = constituents[:8]
        more = '' if len(constituents) <= 8 else f' (+{len(constituents) - 8})'
        parts.append('  [' + ', '.join(f'(p{p_},f{f_})' for p_, f_ in head)
                     + more + ']')
    return '<br>'.join(parts)


def _hover_err(key, attrs, report=False) -> str:
    _, k, p = key
    return f'<b>error · {sae_row_label(k, report)} · pos {p}</b>'


def _hover_embed(key, attrs) -> str:
    _, p = key
    tok = attrs.get('token_id')
    return f'<b>embed · pos {p}</b><br>token = {tok}'


def _hover_logit(key, attrs) -> str:
    _, c = key
    prob = float(attrs.get('prob', float('nan')))
    is_true = attrs.get('is_true_class', False)
    return (f'<b>logit · class {c}</b>{"  (TRUE)" if is_true else ""}'
            f'<br>prob = {prob:.3f}')


# ---------------------------------------------------------------------------
# Edge trace builder
# ---------------------------------------------------------------------------

def _edge_traces(edges: list, pos: dict, name_prefix: str):
    """Return two plotly Scatter traces (positive, negative) carrying all
    edges as line segments with NaN separators."""
    import plotly.graph_objects as go

    kept = [(s_, d_, w_) for (s_, d_, w_) in edges
            if s_ in pos and d_ in pos]
    widths, alphas = _edge_widths_alphas([w for (_, _, w) in kept])

    # Plotly Scatter can only have one line width per trace. Sidestep by
    # bucketing widths into a small number of bins so we get visual
    # differentiation without thousands of traces. 5 width bins for each
    # of (positive, negative) -> 10 traces total.
    n_bins = 5
    if not kept:
        return []
    max_w = max(widths) if widths else 1.0
    min_w = min(widths) if widths else 0.2

    def _bin(w):
        if max_w <= min_w:
            return 0
        return int(min(n_bins - 1, round((w - min_w) / (max_w - min_w) * (n_bins - 1))))

    buckets_pos: dict[int, list] = {b: [] for b in range(n_bins)}
    buckets_neg: dict[int, list] = {b: [] for b in range(n_bins)}
    for (s_, d_, w_), lw, alpha in zip(kept, widths, alphas):
        b = _bin(lw)
        seg = (pos[s_], pos[d_], lw, alpha, w_)
        if w_ >= 0:
            buckets_pos[b].append(seg)
        else:
            buckets_neg[b].append(seg)

    traces = []

    def _make_trace(seg_list, color, name):
        if not seg_list:
            return None
        xs, ys = [], []
        ref_lw = float(seg_list[0][2])
        ref_alpha = float(seg_list[0][3])
        for (a, b, lw, al, w_) in seg_list:
            xs.extend([a[0], b[0], None])
            ys.extend([a[1], b[1], None])
        return go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=color, width=ref_lw),
            opacity=ref_alpha,
            hoverinfo='skip',
            showlegend=False,
            name=name,
        )

    for b in range(n_bins):
        t_pos = _make_trace(buckets_pos[b], '#1f77b4',
                            f'{name_prefix} edge+ bin{b}')
        if t_pos is not None:
            traces.append(t_pos)
        t_neg = _make_trace(buckets_neg[b], '#d62728',
                            f'{name_prefix} edge- bin{b}')
        if t_neg is not None:
            traces.append(t_neg)
    return traces


# ---------------------------------------------------------------------------
# Node trace builder
# ---------------------------------------------------------------------------

def _scaled_marker_size(base, value, scale_min=0.6, scale_max=2.4,
                        ref_value=1.0):
    if ref_value <= 0:
        return base
    s = max(scale_min, min(scale_max, value / ref_value))
    return base * s


def _node_traces_ungrouped(nodes: dict, kept: set, pos: dict, K: int,
                           p_ref: float, z_ref: float, report=False,
                           scheme=None):
    import plotly.graph_objects as go
    feat_x, feat_y, feat_c, feat_s, feat_t, feat_l = [], [], [], [], [], []
    feat_ring = []  # per-marker ring color = RHM level
    err_x, err_y, err_t = [], [], []
    emb_x, emb_y, emb_t, emb_l = [], [], [], []
    log_x, log_y, log_t, log_s, log_l = [], [], [], [], []
    for key in kept:
        if key not in pos:
            continue
        x, y = pos[key]
        attrs = nodes[key]
        kind = attrs['kind']
        if kind == 'feature':
            sv = _scheme_view(attrs, scheme)
            feat_x.append(x); feat_y.append(y)
            feat_c.append(_entropy_color_hex(sv['normalized_entropy']))
            feat_ring.append(level_ring_color(sv['level'], K))
            feat_s.append(_scaled_marker_size(18, float(attrs.get('z', 0)),
                                              ref_value=z_ref))
            feat_t.append(_hover_feature(key, attrs, report, scheme=scheme, L=K))
            lv = sv['label_value']
            feat_l.append('' if lv is None else str(int(lv)))
        elif kind == 'error':
            err_x.append(x); err_y.append(y)
            err_t.append(_hover_err(key, attrs, report))
        elif kind == 'embedding':
            emb_x.append(x); emb_y.append(y)
            emb_t.append(_hover_embed(key, attrs))
            tok = attrs.get('token_id')
            emb_l.append('' if tok is None else str(int(tok)))
        elif kind == 'logit':
            log_x.append(x); log_y.append(y)
            log_t.append(_hover_logit(key, attrs))
            log_s.append(_scaled_marker_size(16, float(attrs.get('prob', 0)),
                                             ref_value=p_ref))
            log_l.append(str(int(attrs.get('class', key[1]))))
    traces = []
    if feat_x:
        # Fill marker = entropy (selectivity), with a thin WHITE separator line
        # so the level ring (drawn as an overlay just outside) never blends into
        # a same-hue fill (e.g. green ring over green/selective fill).
        traces.append(go.Scatter(
            x=feat_x, y=feat_y, mode='markers+text',
            marker=dict(symbol='circle', size=feat_s, color=feat_c,
                        line=dict(color='#ffffff', width=1.5)),
            text=feat_l, textposition='middle center',
            textfont=dict(size=8),
            hovertext=feat_t, hoverinfo='text',
            name='feature', showlegend=False,
        ))
        # Level ring overlay: a colored open circle sized just outside the fill,
        # so reading order is fill -> white gap -> level color band. NOTE: for
        # the 'circle-open' symbol the visible ring is marker.color (not
        # marker.line.color); a per-marker line.color would all render the same.
        ring_s = [s + 5.0 for s in feat_s]
        traces.append(go.Scatter(
            x=feat_x, y=feat_y, mode='markers',
            marker=dict(symbol='circle-open', size=ring_s, color=feat_ring,
                        line=dict(width=3.0)),
            hoverinfo='skip', showlegend=False, name='feature level',
        ))
    if err_x:
        traces.append(go.Scatter(
            x=err_x, y=err_y, mode='markers',
            marker=dict(symbol='square', size=8, color='#bdbdbd',
                        line=dict(color='#404040', width=0.7)),
            hovertext=err_t, hoverinfo='text',
            name='error', showlegend=False,
        ))
    if emb_x:
        traces.append(go.Scatter(
            x=emb_x, y=emb_y, mode='markers+text',
            marker=dict(symbol='triangle-up', size=12, color='#ffe5b4',
                        line=dict(color='#7a4f00', width=0.7)),
            text=emb_l, textposition='bottom center',
            textfont=dict(size=8),
            hovertext=emb_t, hoverinfo='text',
            name='embed', showlegend=False,
        ))
    if log_x:
        traces.append(go.Scatter(
            x=log_x, y=log_y, mode='markers+text',
            marker=dict(symbol='diamond', size=log_s, color='#cfe2ff',
                        line=dict(color='#0b3d91', width=0.7)),
            text=log_l, textposition='middle center',
            textfont=dict(size=8),
            hovertext=log_t, hoverinfo='text',
            name='logit', showlegend=False,
        ))
    return traces


def _node_traces_grouped(grouped_nodes: dict, kept: set, pos: dict, K: int,
                         p_ref: float, z_ref: float, report=False,
                         scheme=None):
    import plotly.graph_objects as go
    grp_x, grp_y, grp_c, grp_s, grp_t, grp_l = [], [], [], [], [], []
    grp_line_c, grp_line_w = [], []
    err_x, err_y, err_t = [], [], []
    emb_x, emb_y, emb_t, emb_l = [], [], [], []
    log_x, log_y, log_t, log_s, log_l = [], [], [], [], []
    for key in kept:
        if key not in pos:
            continue
        x, y = pos[key]
        attrs = grouped_nodes[key]
        kind = attrs.get('kind', key[0])
        if kind == 'group':
            sv = _scheme_view(attrs, scheme)
            grp_x.append(x); grp_y.append(y)
            grp_c.append(_entropy_color_hex(sv['normalized_entropy']))
            gsize = _scaled_marker_size(16, float(attrs.get('z_sum', 0)),
                                        ref_value=z_ref)
            grp_s.append(gsize)
            grp_t.append(_hover_group(key, attrs, report, scheme=scheme, L=K))
            lv = sv['label_value']
            grp_l.append('' if lv is None else str(int(lv)))
            # Ring color = RHM level (same palette as the level-dist plots and
            # the RHM tree). Disagreement is shown separately by a red halo.
            grp_line_c.append(level_ring_color(sv['level'], K))
            grp_line_w.append(1.6)
        elif kind == 'error':
            err_x.append(x); err_y.append(y)
            err_t.append(_hover_err(key, attrs, report))
        elif kind == 'embedding':
            emb_x.append(x); emb_y.append(y)
            emb_t.append(_hover_embed(key, attrs))
            tok = attrs.get('token_id')
            emb_l.append('' if tok is None else str(int(tok)))
        elif kind == 'logit':
            log_x.append(x); log_y.append(y)
            log_t.append(_hover_logit(key, attrs))
            log_s.append(_scaled_marker_size(16, float(attrs.get('prob', 0)),
                                             ref_value=p_ref))
            log_l.append(str(int(attrs.get('class', key[1]))))
    traces = []
    if grp_x:
        # Fill = entropy, thin white separator; level ring drawn as an overlay
        # (same fill -> white gap -> level band order as features).
        traces.append(go.Scatter(
            x=grp_x, y=grp_y, mode='markers+text',
            marker=dict(symbol='circle', size=grp_s, color=grp_c,
                        line=dict(color='#ffffff', width=1.5)),
            text=grp_l, textposition='middle center',
            textfont=dict(size=8),
            hovertext=grp_t, hoverinfo='text',
            name='group', showlegend=False,
        ))
        ring_s = [s + 5.0 for s in grp_s]
        traces.append(go.Scatter(
            x=grp_x, y=grp_y, mode='markers',
            marker=dict(symbol='circle-open', size=ring_s, color=grp_line_c,
                        line=dict(width=3.0)),
            hoverinfo='skip', showlegend=False, name='group level',
        ))
    if err_x:
        traces.append(go.Scatter(
            x=err_x, y=err_y, mode='markers',
            marker=dict(symbol='square', size=8, color='#bdbdbd',
                        line=dict(color='#404040', width=0.7)),
            hovertext=err_t, hoverinfo='text',
            name='error', showlegend=False,
        ))
    if emb_x:
        traces.append(go.Scatter(
            x=emb_x, y=emb_y, mode='markers+text',
            marker=dict(symbol='triangle-up', size=12, color='#ffe5b4',
                        line=dict(color='#7a4f00', width=0.7)),
            text=emb_l, textposition='bottom center',
            textfont=dict(size=8),
            hovertext=emb_t, hoverinfo='text',
            name='embed', showlegend=False,
        ))
    if log_x:
        traces.append(go.Scatter(
            x=log_x, y=log_y, mode='markers+text',
            marker=dict(symbol='diamond', size=log_s, color='#cfe2ff',
                        line=dict(color='#0b3d91', width=0.7)),
            text=log_l, textposition='middle center',
            textfont=dict(size=8),
            hovertext=log_t, hoverinfo='text',
            name='logit', showlegend=False,
        ))
    return traces


# ---------------------------------------------------------------------------
# Shared legend builder
# ---------------------------------------------------------------------------

def _add_legend_traces(fig, L: int, report_notation: bool, meta=None):
    """Add dummy (no-data) legend entries to `fig`.

    Legend box 1 (default 'legend'): ring colors + node-type symbols.
    Legend box 2 ('legend2'):        entropy fill + edge directions.

    `meta` is attached to each trace so the sweep JS shim can tag them as
    'shared' (always visible regardless of slider state).
    """
    import plotly.graph_objects as go

    def _tr(trace):
        if meta is not None:
            trace.meta = meta
        return trace

    # -- Ring colors (leaves -> root, ascending report level) ---------------
    for code_l in range(L, -1, -1):
        disp_l = report_level(code_l, L) if report_notation else code_l
        color = level_ring_color(code_l, L)
        is_leaf = (report_notation and disp_l == 0) or (not report_notation and disp_l == L)
        is_root = (report_notation and disp_l == L) or (not report_notation and disp_l == 0)
        lbl = (f'Level {disp_l} (leaves)' if is_leaf
               else f'Level {disp_l} (root)' if is_root
               else f'Level {disp_l}')
        fig.add_trace(_tr(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol='circle-open', size=14, color=color,
                        line=dict(width=3.0)),
            name=lbl, showlegend=True,
        )))
    # -- Node-type symbols --------------------------------------------------
    fig.add_trace(_tr(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol='square', size=10, color='#bdbdbd',
                    line=dict(color='#404040', width=0.7)),
        name='Error node', showlegend=True,
    )))
    fig.add_trace(_tr(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol='triangle-up', size=12, color='#ffe5b4',
                    line=dict(color='#7a4f00', width=0.7)),
        name='Embedding', showlegend=True,
    )))
    fig.add_trace(_tr(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol='diamond', size=12, color='#cfe2ff',
                    line=dict(color='#0b3d91', width=0.7)),
        name='Logit', showlegend=True,
    )))
    # -- Entropy fill (legend2) ---------------------------------------------
    fig.add_trace(_tr(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol='circle', size=14, color='#1a9850',
                    line=dict(color='#ffffff', width=1)),
        name='Selective (low entropy)', showlegend=True, legend='legend2',
    )))
    fig.add_trace(_tr(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol='circle', size=14, color='#ffffff',
                    line=dict(color='#aaaaaa', width=1.5)),
        name='Uniform (high entropy)', showlegend=True, legend='legend2',
    )))
    # -- Edge directions (legend2) ------------------------------------------
    fig.add_trace(_tr(go.Scatter(
        x=[None, None], y=[None, None], mode='lines',
        line=dict(color='#1f77b4', width=2.5),
        name='Positive attribution', showlegend=True, legend='legend2',
    )))
    fig.add_trace(_tr(go.Scatter(
        x=[None, None], y=[None, None], mode='lines',
        line=dict(color='#d62728', width=2.5),
        name='Negative attribution', showlegend=True, legend='legend2',
    )))


# ---------------------------------------------------------------------------
# Tree panel (mirrors visualize._draw_rhm_tree)
# ---------------------------------------------------------------------------

def _tree_traces(tree_row: dict, s: int, L: int, K: int, report=False):
    import plotly.graph_objects as go
    N = s ** L

    def _y_for_level(l: int) -> float:
        return K - (l * (K + 1.0) / L)

    def _x_for_node(l: int, i: int) -> float:
        span = s ** (L - l)
        return (i + 0.5) * span - 0.5

    line_x, line_y = [], []
    for l in range(L):
        for i in range(s ** l):
            xp = _x_for_node(l, i)
            for j in range(s):
                child = i * s + j
                xc = _x_for_node(l + 1, child)
                line_x.extend([xp, xc, None])
                line_y.extend([_y_for_level(l), _y_for_level(l + 1), None])

    node_x, node_y, node_text, hover_text = [], [], [], []
    node_ring = []  # per-node ring color = RHM level (visual key for the graph)
    for l in range(L + 1):
        row = tree_row[l]
        if hasattr(row, 'tolist'):
            vals = row.tolist() if row.ndim > 0 else [int(row.item())]
        else:
            vals = [int(row)] if not isinstance(row, list) else row
        if l == 0 and not isinstance(vals, list):
            vals = [int(vals)]
        y = _y_for_level(l)
        for i, val in enumerate(vals):
            x = _x_for_node(l, i)
            node_x.append(x); node_y.append(y)
            node_text.append(str(int(val)))
            node_ring.append(level_ring_color(l, L))
            disp_l = report_level(l, L) if report else l
            hover_text.append(f'level={disp_l}, idx={i}, value={int(val)}')

    # Match the circuit nodes: fill + thin white separator, level color as an
    # overlay ring, so tree and feature/group rings read identically.
    return [
        go.Scatter(x=line_x, y=line_y, mode='lines',
                   line=dict(color='#888888', width=0.6),
                   opacity=0.7, hoverinfo='skip', showlegend=False,
                   name='tree edges'),
        go.Scatter(x=node_x, y=node_y, mode='markers+text',
                   marker=dict(symbol='circle', size=18,
                               color='#f4f4f4',
                               line=dict(color='#ffffff', width=1.5)),
                   text=node_text, textposition='middle center',
                   textfont=dict(size=12),
                   hovertext=hover_text, hoverinfo='text',
                   showlegend=False, name='tree nodes'),
        go.Scatter(x=node_x, y=node_y, mode='markers',
                   marker=dict(symbol='circle-open', size=23, color=node_ring,
                               line=dict(width=3.0)),
                   hoverinfo='skip', showlegend=False, name='tree level'),
    ]


# ---------------------------------------------------------------------------
# Loading + filtering shared with visualize.py
# ---------------------------------------------------------------------------

def _load_run(run_dir: Path):
    nodes = torch.load(run_dir / 'nodes.pt', weights_only=False)
    edges_blob = torch.load(run_dir / 'edges.pt', weights_only=False)
    fidelity = torch.load(run_dir / 'fidelity.pt', weights_only=False)
    pruned_edges = edges_blob['pruned']
    grouped_nodes_path = run_dir / 'grouped_nodes.pt'
    grouped_edges_path = run_dir / 'grouped_edges.pt'
    has_grouped = grouped_nodes_path.exists() and grouped_edges_path.exists()
    if has_grouped:
        grouped_nodes = torch.load(grouped_nodes_path, weights_only=False)
        grouped_edges = torch.load(grouped_edges_path, weights_only=False)
    else:
        grouped_nodes, grouped_edges = None, None
    tree_path = run_dir / 'tree_for_input.pt'
    tree_row = torch.load(tree_path, weights_only=False) if tree_path.exists() else None
    return nodes, pruned_edges, fidelity, grouped_nodes, grouped_edges, tree_row


def _select_kept_ungrouped(nodes: dict, pruned_edges: list,
                           show_all_logits: bool, show_errors: bool,
                           y_true: int, n_classes: int):
    referenced = set()
    for (s_, d_, _) in pruned_edges:
        referenced.add(s_); referenced.add(d_)
    kept = set(referenced)
    for key, attrs in nodes.items():
        if attrs['kind'] == 'embedding':
            kept.add(key)
    kept.add(('logit', y_true))
    if not show_all_logits:
        kept = {k for k in kept
                if k[0] != 'logit' or (k in referenced) or (k == ('logit', y_true))}
    else:
        for c in range(n_classes):
            kept.add(('logit', c))
    if not show_errors:
        kept = {k for k in kept if k[0] != 'err'}
        pruned_edges = [
            (s_, d_, w_) for (s_, d_, w_) in pruned_edges
            if s_[0] != 'err' and d_[0] != 'err'
        ]
    return kept, pruned_edges


def _select_kept_grouped(grouped_nodes: dict, grouped_edges: list,
                         show_all_logits: bool, show_errors: bool,
                         y_true: int, n_classes: int):
    referenced = set()
    for (s_, d_, _) in grouped_edges:
        referenced.add(s_); referenced.add(d_)
    kept = set(referenced)
    for key, attrs in grouped_nodes.items():
        if attrs.get('kind') == 'embedding':
            kept.add(key)
    kept.add(('logit', y_true))
    if not show_all_logits:
        kept = {k for k in kept
                if k[0] != 'logit' or (k in referenced) or (k == ('logit', y_true))}
    else:
        for c in range(n_classes):
            kept.add(('logit', c))
    if not show_errors:
        kept = {k for k in kept if k[0] != 'err'}
        grouped_edges = [
            (s_, d_, w_) for (s_, d_, w_) in grouped_edges
            if s_[0] != 'err' and d_[0] != 'err'
        ]
    return kept, grouped_edges


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_figure(run_dir: Path,
                 show_all_logits: bool = False,
                 show_errors: bool = True,
                 report_notation: bool = True,
                 scheme: str | None = None):
    """Build and return a Plotly Figure for one circuit-trace run directory.

    `scheme` selects which per-node label scheme to display (None = top-level
    fields, 'parent' / 'level' / 'whole_tree' / 'reassigned' for sweep data).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    (nodes, pruned_edges, fidelity, grouped_nodes, grouped_edges,
     tree_row) = _load_run(run_dir)

    rhm = fidelity['rhm']
    s = int(rhm['s']); L = int(rhm['L']); K = L
    n_classes = int(rhm['n'])
    N = s ** L
    y_true = int(fidelity['y_true'])
    y_pred = int(fidelity['y_pred'])
    pooled_last_layer = bool(fidelity.get('pooled_last_layer', False))

    # ---- Ungrouped panel ----
    kept_u, pruned_edges_u = _select_kept_ungrouped(
        nodes, list(pruned_edges), show_all_logits, show_errors, y_true, n_classes,
    )
    pos_u = _layout_ungrouped(nodes, kept_u, N=N, K=K, num_classes=n_classes,
                              pooled_last_layer=pooled_last_layer)
    z_values = [nodes[k]['z'] for k in kept_u
                if k[0] == 'feat' and 'z' in nodes[k]]
    z_ref = float(torch.tensor(z_values).median().item()) if z_values else 1.0
    if z_ref <= 0:
        z_ref = 1.0
    prob_values = [nodes[k]['prob'] for k in kept_u
                   if k[0] == 'logit' and 'prob' in nodes[k]]
    p_ref = max(prob_values) if prob_values else 1.0

    # ---- Grouped panel ----
    has_grouped = grouped_nodes is not None and grouped_edges is not None
    if has_grouped:
        kept_g, grouped_edges_g = _select_kept_grouped(
            grouped_nodes, list(grouped_edges), show_all_logits, show_errors,
            y_true, n_classes,
        )
        pos_g = _layout_grouped(grouped_nodes, kept_g, N=N, K=K,
                                num_classes=n_classes,
                                pooled_last_layer=pooled_last_layer)
        z_sum_values = [grouped_nodes[k]['z_sum'] for k in kept_g
                        if k[0] == 'group']
        z_ref_g = (float(torch.tensor(z_sum_values).median().item())
                   if z_sum_values else 1.0)
        if z_ref_g <= 0:
            z_ref_g = 1.0
    else:
        kept_g, grouped_edges_g = set(), []
        pos_g = {}
        z_ref_g = 1.0

    # ---- Build figure: 2x2 grid.
    # row 1 col 1 : pruned ungrouped circuit
    # row 2 col 1 : pruned grouped circuit (shares x with row 1 col 1)
    # row 1 col 2 : RHM ground-truth tree (single cell, centered
    #               vertically via yaxis domain override below)
    # row 2 col 2 : empty
    has_tree = tree_row is not None
    if has_tree:
        specs = [[{}, {}],
                 [{}, None]]
        subplot_titles = (
            'Pruned ungrouped circuit',
            'RHM ground-truth tree',
            'Pruned grouped circuit (by incoming signature)',
            None,
        )
    else:
        specs = [[{}, None],
                 [{}, None]]
        subplot_titles = (
            'Pruned ungrouped circuit',
            None,
            'Pruned grouped circuit (by incoming signature)',
            None,
        )
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
        specs=specs,
        subplot_titles=subplot_titles,
    )

    # Row 1, col 1: ungrouped.
    for tr in _edge_traces(pruned_edges_u, pos_u, 'ungrouped'):
        fig.add_trace(tr, row=1, col=1)
    for tr in _node_traces_ungrouped(nodes, kept_u, pos_u, K=K,
                                     p_ref=p_ref, z_ref=z_ref,
                                     report=report_notation, scheme=scheme):
        fig.add_trace(tr, row=1, col=1)

    # Row 2, col 1: grouped (or a placeholder annotation if missing).
    if has_grouped:
        for tr in _edge_traces(grouped_edges_g, pos_g, 'grouped'):
            fig.add_trace(tr, row=2, col=1)
        for tr in _node_traces_grouped(grouped_nodes, kept_g, pos_g, K=K,
                                       p_ref=p_ref, z_ref=z_ref_g,
                                       report=report_notation, scheme=scheme):
            fig.add_trace(tr, row=2, col=1)
    else:
        fig.add_annotation(
            text='grouped_nodes.pt / grouped_edges.pt not found in this '
                 'run_dir; re-run circuit_trace.py to produce them.',
            xref='x3', yref='y3', x=N / 2, y=K / 2,
            showarrow=False, font=dict(size=11, color='#888888'),
        )

    # Row 1, col 2: ground-truth tree.
    if has_tree:
        for tr in _tree_traces(tree_row, s=s, L=L, K=K,
                               report=report_notation):
            fig.add_trace(tr, row=1, col=2)

    # Axes / layout.
    n_nodes_pre = sum(fidelity.get('n_nodes_pre_by_kind', {}).values())
    n_nodes_post = sum(fidelity.get('n_nodes_post_by_kind', {}).values())
    n_edges_pre = fidelity.get('n_edges_pre_prune', '?')
    n_edges_post = fidelity.get('n_edges_post_prune', '?')
    scheme_tag = f'  [{scheme}]' if scheme else ''
    title = (
        f"input {fidelity.get('input_idx', '?')}  "
        f"y_true={y_true}  y_pred={y_pred}  "
        f"completeness={fidelity['completeness_score']:.2f}{scheme_tag}  |  "
        f"nodes {n_nodes_pre}->{n_nodes_post}  "
        f"edges {n_edges_pre}->{n_edges_post}"
    )
    for r in (1, 2):
        fig.update_yaxes(
            row=r, col=1,
            tickmode='array',
            tickvals=[-1] + list(range(K)) + [K],
            ticktext=(['embed']
                      + [sae_row_label(k, report_notation) for k in range(K)]
                      + ['logit']),
            showgrid=True, gridcolor='#eeeeee',
        )
        fig.update_xaxes(
            row=r, col=1,
            tickmode='array',
            tickvals=list(range(N)),
            range=[-0.7, N - 0.3],
            showgrid=True, gridcolor='#eeeeee',
        )
    fig.update_xaxes(row=2, col=1, matches='x')

    if has_tree:
        fig.update_yaxes(
            row=1, col=2,
            tickmode='array',
            tickvals=[K - (l * (K + 1.0) / L) for l in range(L + 1)],
            ticktext=[
                f'level {report_level(l, L) if report_notation else l}'
                for l in range(L + 1)
            ],
            domain=[0.28, 0.97],
        )
        fig.update_xaxes(
            row=1, col=2,
            tickmode='array',
            tickvals=list(range(N)),
            range=[-0.7, N - 0.3],
            showgrid=True, gridcolor='#eeeeee',
        )

    # Legend: ring colors + node types (left box) + entropy + edges (right box).
    _add_legend_traces(fig, L, report_notation)
    _legend_style = dict(
        xanchor='left', yanchor='top',
        bgcolor='rgba(255,255,255,0.97)',
        bordercolor='#cccccc', borderwidth=1,
        font=dict(size=9),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=11)),
        width=1050,
        height=800,
        margin=dict(l=60, r=20, t=80, b=40),
        hovermode='closest',
        plot_bgcolor='#fafafa',
        showlegend=True,
        legend=dict(x=0.63, y=0.25, **_legend_style),
        legend2=dict(x=0.83, y=0.25, **_legend_style),
    )
    return fig


def render_html(run_dir: Path, out_path: Path,
                show_all_logits: bool = False,
                show_errors: bool = True,
                inline_js: bool = False,
                report_notation: bool = True) -> Path:
    fig = build_figure(run_dir, show_all_logits=show_all_logits,
                       show_errors=show_errors, report_notation=report_notation)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out_path),
        include_plotlyjs='inline' if inline_js else 'cdn',
        full_html=True,
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--out', default=None,
                   help='Output HTML path. Defaults to <run_dir>/circuit.html.')
    p.add_argument('--show_all_logits', action='store_true')
    p.add_argument('--hide_errors', action='store_true')
    p.add_argument('--inline_js', action='store_true',
                   help='Embed plotly.js inline (larger file, works offline). '
                        'Default uses the CDN.')
    p.add_argument('--no-report-notation', dest='report_notation',
                   action='store_false',
                   help='Disable report notation (default ON): use Layer k '
                        '(0-based) and top-down RHM levels (root=0, leaf=L) '
                        'instead of SAE k / bottom-up. Display only.')
    p.set_defaults(report_notation=True)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f'--run_dir not a directory: {run_dir}')
    out_path = Path(args.out) if args.out else run_dir / 'circuit.html'
    final = render_html(
        run_dir=run_dir,
        out_path=out_path,
        show_all_logits=args.show_all_logits,
        show_errors=not args.hide_errors,
        inline_js=args.inline_js,
        report_notation=args.report_notation,
    )
    print(f'Wrote {final}')


if __name__ == '__main__':
    main()
