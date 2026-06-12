"""Layered DAG visualization for a saved circuit_trace run.

Loads `nodes.pt`, `edges.pt`, and `fidelity.pt` from --run_dir and renders a
single static figure of the pruned graph.

Conventions
-----------
- Vertical (y) axis = virtual layer:
    embed = -1
    feature/error at layer k = k
    logit = K
  Embed is at the bottom, logits at the top.
- Horizontal (x) axis:
    feature/error/embed: leaf position p in [0, N).
    logit: class index c in [0, num_classes), centered across [0, N).
- Within a (layer, position) cell we stack feature nodes vertically; the error
  node sits in its own slot to the right of that cell. Active features at the
  same (layer, position) are stacked top-to-bottom by feature index for
  reproducibility.
- Node visuals:
    feature -> circle, size scaled by `z`, text label = argmax `label_value`.
    error   -> square, fixed size, no text label.
    embed   -> triangle, fixed size, text label = `token_id`.
    logit   -> diamond, size scaled by `prob`, text label = class index.
- Edge color: blue if signed weight > 0, red if < 0.
- Edge width / alpha: scaled by |w| using the 95th-percentile of |w| so a few
  large edges do not dominate.
- Logit nodes: by default we only draw logits that have any incoming pruned
  edge, plus always the true-class logit. `--show_all_logits` to draw every
  class.

CLI
---
  python -m circuit_tracing.visualize --run_dir runs/circuit_trace/example
  # output: runs/circuit_trace/example/circuit.png

Outputs a single image at --out (PNG by default; format inferred from the
extension).
"""

from __future__ import annotations

import argparse
import pickle

from circuit_tracing.notation import sae_row_label, report_level
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_run(run_dir: Path):
    nodes = torch.load(run_dir / 'nodes.pt', weights_only=False)
    edges_blob = torch.load(run_dir / 'edges.pt', weights_only=False)
    fidelity = torch.load(run_dir / 'fidelity.pt', weights_only=False)
    pruned_edges = edges_blob['pruned']
    return nodes, pruned_edges, fidelity


def _virtual_layer(key, K: int) -> int:
    kind = key[0]
    if kind == 'embed':
        return -1
    if kind == 'logit':
        return K
    return int(key[1])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _compute_positions(nodes: dict, kept_keys: set, N: int, K: int,
                       num_classes: int,
                       feat_fan_half: float = 0.30,
                       err_x_offset: float = 0.40,
                       pooled_last_layer: bool = False):
    """Return dict node_key -> (x, y) for every key in kept_keys.

    Layout:
      y = virtual layer (embed=-1, layers 0..K-1, logit=K).
      Within layer k:
        - Feature nodes at (layer, position) all share y = layer (fan is
          horizontal, not vertical). They are centered on x = position:
            * 1 firing  -> x = position
            * 2 firings -> x = position +/- feat_fan_half
            * n firings -> evenly spaced in [position - feat_fan_half,
                                              position + feat_fan_half]
          Ordered by feature index for reproducibility.
        - Error node at (layer, position) sits at x = position + err_x_offset
          on the layer line (a small offset to the right of the cell).
        - Embed nodes get x = position (no fan), aligned with the position
          cell columns above.
        - Logit nodes evenly spread across the SAME x-range as features
          ([0, N-1]) regardless of num_classes, so the x-axis stays a
          consistent 'position' coordinate.

    When pooled_last_layer is True, the layer K-1 nodes are a single pooled
    SAE node at position 0 (level 0 = root). Instead of pinning it to the
    left at x=0, it is showcased at the horizontal center x = (N-1)/2 (the
    same centering used for a single logit), on the y = K-1 line.
    """
    pos = {}

    def _cell_base_x(k: int, p: int) -> float:
        """Base x for a (layer, position) cell. The pooled last layer's single
        position 0 is centered; everything else sits at its position index."""
        if pooled_last_layer and k == K - 1:
            return (N - 1) / 2.0
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
        base_x = _cell_base_x(k, p)
        for idx, i in enumerate(feat_list):
            pos[('feat', k, p, i)] = (base_x + offsets[idx], float(k))

    for key in kept_keys:
        if key[0] == 'err':
            _, k, p = key
            pos[key] = (_cell_base_x(k, p) + err_x_offset, float(k))
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
# Edge styling
# ---------------------------------------------------------------------------

def _edge_widths_alphas(weights, max_width=4.0, min_width=0.2,
                        max_alpha=0.9, min_alpha=0.15):
    """Scale |w| against its 95th percentile to compute width and alpha.

    Returns two lists aligned with `weights`.
    """
    if len(weights) == 0:
        return [], []
    abs_w = torch.tensor([abs(w) for w in weights], dtype=torch.float32)
    if abs_w.numel() == 1 or abs_w.max() == 0:
        ref = float(abs_w.max().item()) or 1.0
    else:
        ref = float(torch.quantile(abs_w, 0.95).item())
        if ref <= 0:
            ref = float(abs_w.max().item()) or 1.0
    widths = []
    alphas = []
    for w in weights:
        scale = abs(w) / ref if ref > 0 else 0.0
        scale = min(scale, 1.0)
        widths.append(min_width + (max_width - min_width) * scale)
        alphas.append(min_alpha + (max_alpha - min_alpha) * scale)
    return widths, alphas


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _node_visuals():
    """Per-kind defaults: marker, base_size, fill_color, edge_color, label_kw."""
    return {
        'feature':   {'marker': 'o', 'base_size': 80.0,
                      'fill': '#dddddd', 'edge': '#222222'},
        'error':     {'marker': 's', 'base_size': 40.0,
                      'fill': '#bdbdbd', 'edge': '#404040'},
        'embedding': {'marker': '^', 'base_size': 100.0,
                      'fill': '#ffe5b4', 'edge': '#7a4f00'},
        'logit':     {'marker': 'D', 'base_size': 120.0,
                      'fill': '#cfe2ff', 'edge': '#0b3d91'},
    }


# White -> green gradient for normalized_entropy in [0, 1].
# norm_ent == 0  ->  fully selective  -> green
# norm_ent == 1  ->  uniform          -> white
# None           ->  no eval evidence -> light gray fallback
_ENTROPY_GREEN = (0x1a / 255.0, 0x98 / 255.0, 0x50 / 255.0)
_ENTROPY_WHITE = (1.0, 1.0, 1.0)
_ENTROPY_FALLBACK = '#cccccc'


def _entropy_color(norm_entropy):
    """RGB hex for a normalized_entropy in [0, 1] (None -> fallback gray)."""
    if norm_entropy is None:
        return _ENTROPY_FALLBACK
    t = max(0.0, min(1.0, float(norm_entropy)))
    r = _ENTROPY_GREEN[0] + (_ENTROPY_WHITE[0] - _ENTROPY_GREEN[0]) * t
    g = _ENTROPY_GREEN[1] + (_ENTROPY_WHITE[1] - _ENTROPY_GREEN[1]) * t
    b = _ENTROPY_GREEN[2] + (_ENTROPY_WHITE[2] - _ENTROPY_GREEN[2]) * t
    return (r, g, b)


def _draw_rhm_tree(ax, tree_row: dict, s: int, L: int, K: int,
                   report_notation: bool = False):
    """Draw the RHM tree that generated this input next to the circuit.

    tree_row is the per-input slice: tree_row[l] has length s**l, the
    latent values at level l of the hierarchy. Level 0 is the single root,
    level L is the leaf sequence.

    Layout (top-down, leaves at the bottom to mirror the circuit panel):
      root at y = K (same height as logit row).
      level l sits at y = K - l * (K + 1) / L for l = 0..L  -- this puts
      level 0 at y=K and level L at y=-1, so leaves line up with the
      embedding row of the circuit.
      Within level l, nodes are evenly spread across x in [0, N-1] where
      N = s ** L. Node i at level l covers leaf positions
      [i * s ** (L-l), (i+1) * s ** (L-l)), so we center it on the midpoint.
    """
    N = s ** L

    def _y_for_level(l: int) -> float:
        # Map level l in [0, L] -> y in [K, -1] linearly so root aligns with
        # logits and leaves align with embeds.
        return K - (l * (K + 1.0) / L)

    def _x_for_node(l: int, i: int) -> float:
        span = s ** (L - l)
        return (i + 0.5) * span - 0.5

    # Edges first (parent -> children).
    for l in range(L):
        y_parent = _y_for_level(l)
        y_child = _y_for_level(l + 1)
        n_parents = s ** l
        for i in range(n_parents):
            xp = _x_for_node(l, i)
            for j in range(s):
                child_idx = i * s + j
                xc = _x_for_node(l + 1, child_idx)
                ax.plot([xp, xc], [y_parent, y_child],
                        color='#888888', linewidth=0.6, alpha=0.7, zorder=1)

    # Nodes (with text labels carrying the latent value).
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
            ax.scatter([x], [y], marker='o', s=120,
                       facecolor='#f4f4f4', edgecolor='#333333',
                       linewidths=0.7, zorder=3)
            ax.annotate(str(int(val)), (x, y),
                        ha='center', va='center', fontsize=7, zorder=4)

    ax.set_xlim(-0.7, N - 0.3)
    ax.set_ylim(-1.8, K + 0.8)
    yticks = [_y_for_level(l) for l in range(L + 1)]
    yticklabels = [
        f'level {report_level(l, L) if report_notation else l}'
        for l in range(L + 1)
    ]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(list(range(N)))
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title('RHM tree (ground truth)', fontsize=9)


def _scaled_size(base, scale_value, scale_min=0.6, scale_max=2.4,
                 ref_value=1.0):
    """Scale a base size by a value (z, prob) clipped to [scale_min, scale_max]."""
    if ref_value <= 0:
        return base
    s = scale_value / ref_value
    s = max(scale_min, min(scale_max, s))
    return base * s


def render(run_dir: Path, out_path: Path,
           show_all_logits: bool,
           show_errors: bool,
           figsize: tuple,
           dpi: int,
           report_notation: bool = False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    nodes, pruned_edges, fidelity = _load_run(run_dir)

    # Sanity: pull RHM dims from fidelity (saved by circuit_trace.py).
    rhm = fidelity['rhm']
    s = int(rhm['s'])
    L = int(rhm['L'])
    K = L  # transformer has L blocks; matches num_layers in cfg.
    n_classes = int(rhm['n'])
    N = s ** L
    y_true = int(fidelity['y_true'])
    y_pred = int(fidelity['y_pred'])
    pooled_last_layer = bool(fidelity.get('pooled_last_layer', False))

    # Determine which keys are actually kept (only nodes referenced by the
    # pruned edge list, plus always-keep pure-input nodes for context).
    referenced_keys = set()
    for (src, dst, _w) in pruned_edges:
        referenced_keys.add(src)
        referenced_keys.add(dst)

    # Always include pure-input nodes whose presence is meaningful even with
    # zero pruned edges into them: every embedding (so the bottom row reads as
    # the input sequence) and the true-class logit.
    kept = set(referenced_keys)
    for key, attrs in nodes.items():
        if attrs['kind'] == 'embedding':
            kept.add(key)
    kept.add(('logit', y_true))

    # Decide which logits to show.
    if not show_all_logits:
        kept = {
            k for k in kept
            if k[0] != 'logit' or (k in referenced_keys) or (k == ('logit', y_true))
        }
    else:
        for c in range(n_classes):
            kept.add(('logit', c))

    # Drop error nodes if requested.
    if not show_errors:
        kept = {k for k in kept if k[0] != 'err'}
        pruned_edges = [
            (s_, d_, w_) for (s_, d_, w_) in pruned_edges
            if s_[0] != 'err' and d_[0] != 'err'
        ]

    pos = _compute_positions(nodes, kept, N=N, K=K, num_classes=n_classes,
                             pooled_last_layer=pooled_last_layer)
    visuals = _node_visuals()

    # Reference values for size scaling.
    z_values = [
        nodes[k]['z'] for k in kept
        if k[0] == 'feat' and 'z' in nodes[k]
    ]
    z_ref = float(torch.tensor(z_values).median().item()) if z_values else 1.0
    if z_ref <= 0:
        z_ref = 1.0
    prob_values = [
        nodes[k]['prob'] for k in kept
        if k[0] == 'logit' and 'prob' in nodes[k]
    ]
    p_ref = max(prob_values) if prob_values else 1.0

    # Try to load the RHM tree row saved by circuit_trace.py; if missing the
    # tree panel is skipped.
    tree_path = run_dir / 'tree_for_input.pt'
    has_tree = tree_path.exists()
    if has_tree:
        fig, (ax, ax_tree) = plt.subplots(
            1, 2, figsize=figsize, dpi=dpi,
            gridspec_kw={'width_ratios': [1.6, 1.0]},
        )
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax_tree = None

    # --- Edges first (so they sit beneath nodes) ---
    edge_weights = [w for (_, _, w) in pruned_edges]
    widths, alphas = _edge_widths_alphas(edge_weights)
    for (src, dst, w), lw, alpha in zip(pruned_edges, widths, alphas):
        if src not in pos or dst not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        color = '#1f77b4' if w >= 0 else '#d62728'
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle='-|>',
            mutation_scale=6,
            shrinkA=4, shrinkB=4,
            linewidth=lw,
            color=color,
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(arrow)

    # --- Nodes ---
    for key in kept:
        if key not in pos:
            continue
        x, y = pos[key]
        kind = nodes[key]['kind'] if key in nodes else key[0]
        v = visuals[kind]
        if kind == 'logit':
            size = _scaled_size(v['base_size'], nodes[key]['prob'],
                                ref_value=p_ref)
        else:
            size = v['base_size']
        if kind == 'feature':
            face = _entropy_color(nodes[key].get('normalized_entropy'))
        else:
            face = v['fill']
        ax.scatter([x], [y], marker=v['marker'], s=size,
                   facecolor=face, edgecolor=v['edge'],
                   linewidths=0.7, zorder=3)

        # Text label per kind.
        label = None
        if kind == 'feature':
            lv = nodes[key].get('label_value')
            if lv is not None:
                label = str(int(lv))
        elif kind == 'embedding':
            label = str(int(nodes[key]['token_id']))
        elif kind == 'logit':
            label = str(int(nodes[key]['class']))
        if label is not None:
            y_label = y - 0.06 if kind == 'embedding' else y
            ax.annotate(label, (x, y_label),
                        ha='center', va='center',
                        fontsize=6, zorder=4, color='#111111')

    # --- Axes / title ---
    ax.set_xlim(-0.7, N - 0.3)
    ax.set_ylim(-1.8, K + 0.8)
    # Y ticks: -1 = embed, 0..K-1 layers, K = logits.
    yticks = [-1] + list(range(K)) + [K]
    yticklabels = (['embed']
                   + [sae_row_label(k, report_notation) for k in range(K)]
                   + ['logit'])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(list(range(N)))
    ax.set_xlabel('leaf position (logits spread across same x-range)')
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    # Dashed vertical separators between leaf positions.
    for p in range(N + 1):
        ax.axvline(p - 0.5, color='#aaaaaa', linestyle='--',
                   linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Title strip.
    title = (
        f"input_idx={fidelity.get('input_idx', '?')}  "
        f"y_true={y_true}  y_pred={y_pred}  "
        f"prob[y_pred]={float(torch.softmax(fidelity['logits'], dim=0)[y_pred].item()):.3f}\n"
        f"bit_id_err={fidelity['bit_identity_max_err']:.2e}  "
        f"complete={fidelity['completeness_score']:.2f}  "
        f"replace={fidelity['replacement_score']:.2f}  "
        f"subtree_align={fidelity['subtree_alignment_fraction']:.2f}  "
        f"sink={fidelity['sink_mode']}  "
        f"n_th={fidelity['node_threshold']}  e_th={fidelity['edge_threshold']}"
    )
    ax.set_title(title, fontsize=9)

    # Legend (custom patches).
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=visuals['feature']['fill'],
               markeredgecolor=visuals['feature']['edge'], markersize=8, label='feature'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=visuals['error']['fill'],
               markeredgecolor=visuals['error']['edge'], markersize=8, label='error'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=visuals['embedding']['fill'],
               markeredgecolor=visuals['embedding']['edge'], markersize=8, label='embed'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=visuals['logit']['fill'],
               markeredgecolor=visuals['logit']['edge'], markersize=8, label='logit'),
        Line2D([0], [0], color='#1f77b4', lw=2, label='edge w > 0'),
        Line2D([0], [0], color='#d62728', lw=2, label='edge w < 0'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
              frameon=True, framealpha=0.9)

    # --- Entropy colorbar (white = uniform, green = selective) ---
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cmap = LinearSegmentedColormap.from_list(
        'entropy_wg',
        [_ENTROPY_GREEN, _ENTROPY_WHITE],
    )
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01,
                        location='left', shrink=0.5)
    cbar.set_label('feature normalized entropy', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # --- RHM tree panel (right) ---
    if has_tree:
        tree_row = torch.load(tree_path, weights_only=False)
        _draw_rhm_tree(ax_tree, tree_row, s=s, L=L, K=K,
                       report_notation=report_notation)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True,
                   help='Directory containing nodes.pt, edges.pt, fidelity.pt.')
    p.add_argument('--out', default=None,
                   help='Output path prefix. With --format png or html, the '
                        'corresponding extension is appended if missing. With '
                        '--format both, a `.png` and `.html` are written. '
                        'Defaults to <run_dir>/circuit.')
    p.add_argument('--format', dest='fmt',
                   choices=['png', 'html', 'both'], default='both',
                   help='Which renderer(s) to invoke. Default: both.')
    p.add_argument('--show_all_logits', action='store_true',
                   help='Draw every class logit, not just those with incoming '
                        'pruned edges (true-class logit is always drawn).')
    p.add_argument('--hide_errors', action='store_true',
                   help='Drop error nodes and any edges touching them.')
    p.add_argument('--figsize', nargs=2, type=float, default=(14.0, 8.0),
                   metavar=('W', 'H'),
                   help='Matplotlib PNG figure size in inches.')
    p.add_argument('--dpi', type=int, default=150,
                   help='Matplotlib PNG dpi.')
    p.add_argument('--inline_js', action='store_true',
                   help='Embed plotly.js inline in the HTML (larger file, '
                        'works offline). Default uses the CDN.')
    p.add_argument('--report-notation', dest='report_notation',
                   action='store_true',
                   help='Relabel the circuit y-axis as SAE k (1-based) instead '
                        'of layer k, and flip RHM-tree levels to bottom-up '
                        '(leaves=0, root=L). Display only.')
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f'--run_dir not a directory: {run_dir}')

    out_arg = Path(args.out) if args.out else run_dir / 'circuit'
    # Resolve PNG / HTML paths from the prefix / explicit extension.
    suffix = out_arg.suffix.lower()
    if args.fmt == 'png':
        png_path = out_arg if suffix == '.png' else out_arg.with_suffix('.png')
        html_path = None
    elif args.fmt == 'html':
        png_path = None
        html_path = out_arg if suffix == '.html' else out_arg.with_suffix('.html')
    else:  # both
        base = out_arg.with_suffix('') if suffix in ('.png', '.html') else out_arg
        png_path = base.with_suffix('.png')
        html_path = base.with_suffix('.html')

    written = []
    if png_path is not None:
        final = render(
            run_dir=run_dir,
            out_path=png_path,
            show_all_logits=args.show_all_logits,
            show_errors=not args.hide_errors,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            report_notation=args.report_notation,
        )
        written.append(final)
    if html_path is not None:
        from circuit_tracing.visualize_interactive import render_html
        final = render_html(
            run_dir=run_dir,
            out_path=html_path,
            show_all_logits=args.show_all_logits,
            show_errors=not args.hide_errors,
            inline_js=args.inline_js,
            report_notation=args.report_notation,
        )
        written.append(final)
    for f in written:
        print(f'Wrote {f}')


if __name__ == '__main__':
    main()
