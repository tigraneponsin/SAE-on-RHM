"""Interactive 2-slider HTML for a `circuit_trace_sweep` run directory.

Loads every cell under <sweep_dir>/, builds one Plotly figure with three
rows (ungrouped circuit, grouped circuit, RHM ground-truth tree), and
exposes two sliders (`node_threshold`, `edge_threshold`) that act as
independent cursors over the grid.

Two sliders that JOINTLY control trace visibility require a small inline
JS shim (Plotly natively only chains restyles per slider). The shim
listens for slider movement, reads both sliders' current `active`
index from the figure layout, and shows only the trace group whose
`meta.cell == [i, j]` matches.

Usage:
    python -m circuit_tracing.visualize_sweep --sweep_dir <dir>

Outputs:
    <sweep_dir>/sweep_circuit.html  (single self-contained file).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Reuse the single-config helpers verbatim.
from circuit_tracing.visualize_interactive import (
    _select_kept_ungrouped, _select_kept_grouped,
    _layout_ungrouped, _layout_grouped,
    _edge_traces,
    _node_traces_ungrouped, _node_traces_grouped,
    _tree_traces,
)
from circuit_tracing.notation import sae_row_label, report_level

# Label schemes the scheme slider cycles through (order = slider step order).
SCHEMES = ('parent', 'level', 'whole_tree', 'reassigned')


def _subdir_name(node_th: float, edge_th: float) -> str:
    return f'node_{node_th:.4f}__edge_{edge_th:.4f}'


# ---------------------------------------------------------------------------
# The cross-slider JS shim. Reads both sliders' active index from
# figEl.layout.sliders, then sets trace `visible` accordingly. We listen to
# both plotly_sliderchange (fired on every slider movement) and
# plotly_relayout (defensive, in case sliderchange is skipped).
# ---------------------------------------------------------------------------

_JS_SHIM = r"""
(function() {
  var figEl = document.querySelector('.plotly-graph-div');
  if (!figEl) return;

  function _activeIdx(layout, sliderIdx) {
    if (!layout || !layout.sliders || !layout.sliders[sliderIdx]) return 0;
    var a = layout.sliders[sliderIdx].active;
    return (typeof a === 'number') ? a : 0;
  }

  function applyVisibility() {
    var layout = figEl.layout || {};
    var cur_i = _activeIdx(layout, 0);  // node_threshold slider
    var cur_j = _activeIdx(layout, 1);  // edge_threshold slider
    var cur_sc = _activeIdx(layout, 2); // label-scheme slider
    var total = figEl.data ? figEl.data.length : 0;
    var visible = new Array(total);
    var idxs = new Array(total);
    for (var k = 0; k < total; k++) {
      idxs[k] = k;
      var m = (figEl.data[k] && figEl.data[k].meta) || {};
      if (m.cell === 'shared') {
        visible[k] = true;
      } else if (Array.isArray(m.cell)) {
        var cellOk = (m.cell[0] === cur_i && m.cell[1] === cur_j);
        // Edge traces are scheme-independent (scheme === 'any'); node traces
        // carry a numeric scheme index and show only for the active scheme.
        var schemeOk = (m.scheme === 'any') || (m.scheme === cur_sc);
        visible[k] = cellOk && schemeOk;
      } else {
        visible[k] = true;
      }
    }
    Plotly.restyle(figEl, {'visible': visible}, idxs);

    // Update the dynamic status div if we have one.
    var status = document.getElementById('sweep_status');
    if (status && layout.meta && layout.meta.cells_by_idx) {
      var key = cur_i + ',' + cur_j;
      var c = layout.meta.cells_by_idx[key];
      var schemes = (layout.meta && layout.meta.schemes) || [];
      var schemeName = schemes[cur_sc] || ('scheme ' + cur_sc);
      if (c) {
        status.textContent =
          'cell (i=' + cur_i + ', j=' + cur_j + ')  |  ' +
          'label scheme = ' + schemeName + '  |  ' +
          'n_th = ' + Number(c.node_threshold).toFixed(3) + ',  ' +
          'e_th = ' + Number(c.edge_threshold).toFixed(3) + '  |  ' +
          'n_features_post = ' + c.n_features_post + ',  ' +
          'n_edges_post = ' + c.n_edges_post + ',  ' +
          'n_groups = ' + c.n_groups + ' (' + c.n_groups_multi + ' multi),  ' +
          'completeness = ' + Number(c.completeness_score).toFixed(3) + ',  ' +
          'replacement = ' + Number(c.replacement_score).toFixed(3) + ',  ' +
          'align_post = ' + Number(c.subtree_alignment_fraction_postprune).toFixed(3);
      }
    }
  }

  figEl.on('plotly_sliderchange', applyVisibility);
  figEl.on('plotly_relayout', applyVisibility);
  applyVisibility();
})();
"""


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def render_sweep_html(sweep_dir: Path, out_path: Path,
                      show_all_logits: bool = False,
                      show_errors: bool = True,
                      inline_js: bool = False,
                      report_notation: bool = False) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sweep_dir = Path(sweep_dir)
    summary_path = sweep_dir / 'sweep_summary.json'
    if not summary_path.exists():
        raise SystemExit(
            f'No sweep_summary.json in {sweep_dir}. Did you run '
            f'circuit_trace_sweep on this directory?'
        )
    summary = json.load(open(summary_path))

    node_thresholds = list(summary['node_thresholds'])
    edge_thresholds = list(summary['edge_thresholds'])
    K1, K2 = len(node_thresholds), len(edge_thresholds)

    rhm = summary['rhm']
    s = int(rhm['s']); L = int(rhm['L']); K = L
    N = s ** L
    n_classes = int(rhm['n'])
    y_true = int(summary['y_true'])
    y_pred = int(summary['y_pred'])
    pooled_last_layer = bool(summary.get('pooled_last_layer', False))

    # Map [i, j] -> cell metadata so the JS can pull metrics on slider change.
    cells_by_idx: dict[str, dict] = {}
    for c in summary['cells']:
        # Locate (i, j) for this cell.
        i = node_thresholds.index(c['node_threshold'])
        j = edge_thresholds.index(c['edge_threshold'])
        cells_by_idx[f'{i},{j}'] = c

    # ---- Build figure: 2x2 grid. Tree sits in the top-right cell only
    # (no rowspan) so its aspect stays roughly square and the bottom-
    # right region is free for the two sliders.
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
        specs=[[{}, {}],
               [{}, None]],
        subplot_titles=(
            'Pruned ungrouped circuit',
            'RHM ground-truth tree',
            'Pruned grouped circuit (by incoming signature)',
            None,
        ),
    )

    # ---- Tree (shared, top-right, loaded from cell [0, 0]) ----
    first_subdir = sweep_dir / _subdir_name(node_thresholds[0], edge_thresholds[0])
    tree_row = torch.load(first_subdir / 'tree_for_input.pt', weights_only=False)
    for tr in _tree_traces(tree_row, s=s, L=L, K=K, report=report_notation):
        tr.meta = {'cell': 'shared'}
        tr.showlegend = False
        fig.add_trace(tr, row=1, col=2)

    # ---- Rows 1 + 2: per-cell trace groups ----
    for i, n_th in enumerate(node_thresholds):
        for j, e_th in enumerate(edge_thresholds):
            subdir = sweep_dir / _subdir_name(n_th, e_th)
            nodes = torch.load(subdir / 'nodes.pt', weights_only=False)
            edges_blob = torch.load(subdir / 'edges.pt', weights_only=False)
            pruned = edges_blob['pruned']
            g_nodes = torch.load(subdir / 'grouped_nodes.pt', weights_only=False)
            g_edges = torch.load(subdir / 'grouped_edges.pt', weights_only=False)

            # Ungrouped panel.
            kept_u, pruned_edges_u = _select_kept_ungrouped(
                nodes, list(pruned), show_all_logits, show_errors,
                y_true, n_classes,
            )
            pos_u = _layout_ungrouped(nodes, kept_u, N=N, K=K, num_classes=n_classes,
                                      pooled_last_layer=pooled_last_layer)
            z_values = [nodes[k]['z'] for k in kept_u
                        if k[0] == 'feat' and 'z' in nodes[k]]
            z_ref = (float(torch.tensor(z_values).median().item())
                     if z_values else 1.0)
            if z_ref <= 0:
                z_ref = 1.0
            p_values = [nodes[k]['prob'] for k in kept_u
                        if k[0] == 'logit' and 'prob' in nodes[k]]
            p_ref = max(p_values) if p_values else 1.0

            initial_cell = (i == 0 and j == 0)
            # Edge traces are scheme-independent (topology doesn't change with
            # the label scheme); tag them scheme='any' so the shim shows them
            # for the active cell regardless of the scheme slider.
            for tr in _edge_traces(pruned_edges_u, pos_u, f'u_{i}_{j}'):
                tr.meta = {'cell': [i, j], 'scheme': 'any'}
                tr.visible = initial_cell
                tr.showlegend = False
                fig.add_trace(tr, row=1, col=1)
            # Node traces: one set per scheme; only the active scheme is shown.
            for si, scheme in enumerate(SCHEMES):
                vis = initial_cell and (si == 0)
                for tr in _node_traces_ungrouped(
                    nodes, kept_u, pos_u, K=K, p_ref=p_ref, z_ref=z_ref,
                    report=report_notation, scheme=scheme,
                ):
                    tr.meta = {'cell': [i, j], 'scheme': si}
                    tr.visible = vis
                    tr.showlegend = False
                    fig.add_trace(tr, row=1, col=1)

            # Grouped panel.
            kept_g, grouped_edges_g = _select_kept_grouped(
                g_nodes, list(g_edges), show_all_logits, show_errors,
                y_true, n_classes,
            )
            pos_g = _layout_grouped(g_nodes, kept_g, N=N, K=K, num_classes=n_classes,
                                    pooled_last_layer=pooled_last_layer)
            z_sum_values = [g_nodes[k]['z_sum'] for k in kept_g
                            if k[0] == 'group']
            z_ref_g = (float(torch.tensor(z_sum_values).median().item())
                       if z_sum_values else 1.0)
            if z_ref_g <= 0:
                z_ref_g = 1.0

            for tr in _edge_traces(grouped_edges_g, pos_g, f'g_{i}_{j}'):
                tr.meta = {'cell': [i, j], 'scheme': 'any'}
                tr.visible = initial_cell
                tr.showlegend = False
                fig.add_trace(tr, row=2, col=1)
            for si, scheme in enumerate(SCHEMES):
                vis = initial_cell and (si == 0)
                for tr in _node_traces_grouped(
                    g_nodes, kept_g, pos_g, K=K, p_ref=p_ref, z_ref=z_ref_g,
                    report=report_notation, scheme=scheme,
                ):
                    tr.meta = {'cell': [i, j], 'scheme': si}
                    tr.visible = vis
                    tr.showlegend = False
                    fig.add_trace(tr, row=2, col=1)

    # ---- Sliders ----
    # `method='skip'` -> Plotly only updates `active`; the JS shim handles
    # the actual visibility update.
    node_steps = [
        dict(method='skip', label=f'{v:.2f}', args=[]) for v in node_thresholds
    ]
    edge_steps = [
        dict(method='skip', label=f'{v:.2f}', args=[]) for v in edge_thresholds
    ]
    scheme_steps = [
        dict(method='skip', label=sc, args=[]) for sc in SCHEMES
    ]
    # Stack the two sliders vertically beneath the bottom-right empty
    # cell. x, y, len are normalized figure coords. Right column spans
    # roughly x in [0.74, 0.98] given column_widths=[0.7, 0.3] +
    # horizontal_spacing=0.06; y values sit below the tree (which ends
    # roughly at y=0.55 with vertical_spacing=0.10 and height=900).
    sliders = [
        dict(
            active=0, currentvalue=dict(prefix='node_threshold = ',
                                         font=dict(size=11)),
            steps=node_steps,
            name='node_th_slider',
            pad=dict(t=10, b=4),
            x=0.74, y=0.30, len=0.24,
        ),
        dict(
            active=0, currentvalue=dict(prefix='edge_threshold = ',
                                         font=dict(size=11)),
            steps=edge_steps,
            name='edge_th_slider',
            pad=dict(t=10, b=4),
            x=0.74, y=0.10, len=0.24,
        ),
        dict(
            active=0, currentvalue=dict(prefix='label scheme = ',
                                         font=dict(size=11)),
            steps=scheme_steps,
            name='scheme_slider',
            pad=dict(t=10, b=4),
            x=0.40, y=0.10, len=0.28,
        ),
    ]

    # ---- Axes ----
    # Left column (rows 1 + 2, col 1): per-layer y ticks, leaf-position x.
    for r in (1, 2):
        fig.update_yaxes(
            row=r, col=1,
            tickmode='array',
            tickvals=[-1] + list(range(K)) + [K],
            ticktext=(['embed']
                      + [sae_row_label(k_, report_notation) for k_ in range(K)]
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
    # Tie row=2 col=1's x-axis to row=1 col=1's so pan/zoom on the leaf
    # position axis stays in lockstep across the two circuit panels.
    fig.update_xaxes(row=2, col=1, matches='x')

    # Right column (row 1 col 2): RHM tree axis with level labels.
    fig.update_yaxes(
        row=1, col=2,
        tickmode='array',
        tickvals=[K - (l * (K + 1.0) / L) for l in range(L + 1)],
        ticktext=[
            f'level {report_level(l, L) if report_notation else l}'
            for l in range(L + 1)
        ],
    )
    fig.update_xaxes(
        row=1, col=2,
        tickmode='array',
        tickvals=list(range(N)),
        range=[-0.7, N - 0.3],
        showgrid=True, gridcolor='#eeeeee',
    )

    # ---- Layout ----
    title = (
        f'Sweep: node_threshold ({K1}) x edge_threshold ({K2}) x '
        f'label_scheme ({len(SCHEMES)})  |  '
        f"input_idx={summary['input_idx']}  y_true={y_true}  y_pred={y_pred}  |  "
        f"bit_id_err={summary['bit_identity_max_err']:.2e}  "
        f"sink={summary['sink_mode']}"
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=11)),
        sliders=sliders,
        height=900,
        margin=dict(l=60, r=20, t=80, b=80),
        hovermode='closest',
        plot_bgcolor='#fafafa',
        # Bake threshold axes + per-cell metric table into layout.meta so the
        # JS shim can read them without an extra round-trip.
        meta=dict(
            node_thresholds=node_thresholds,
            edge_thresholds=edge_thresholds,
            schemes=list(SCHEMES),
            cells_by_idx=cells_by_idx,
        ),
    )

    # ---- Status div for the title strip + write HTML ----
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out_path),
        include_plotlyjs='inline' if inline_js else 'cdn',
        full_html=True,
        post_script=_JS_SHIM,
    )
    # Splice in a small status div above the plot. fig.write_html doesn't
    # support arbitrary HTML insertion, so we patch the file in place.
    status_html = (
        '<div id="sweep_status" '
        'style="font-family: monospace; font-size: 11px; '
        'padding: 6px 12px; color: #333; background: #f4f4f4; '
        'border-bottom: 1px solid #ddd;">'
        '(loading...)'
        '</div>\n'
    )
    text = out_path.read_text()
    # Insert right after <body>; fall back to before </body>.
    if '<body>' in text:
        text = text.replace('<body>', '<body>\n' + status_html, 1)
    elif '</body>' in text:
        text = text.replace('</body>', status_html + '</body>', 1)
    out_path.write_text(text)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep_dir', required=True,
                   help='Directory created by circuit_trace_sweep.')
    p.add_argument('--out', default=None,
                   help='Output HTML path. Defaults to <sweep_dir>/sweep_circuit.html.')
    p.add_argument('--show_all_logits', action='store_true')
    p.add_argument('--hide_errors', action='store_true')
    p.add_argument('--inline_js', action='store_true',
                   help='Embed plotly.js inline (larger file, works offline).')
    p.add_argument('--report-notation', dest='report_notation',
                   action='store_true',
                   help='Relabel circuit y-axis as SAE k (1-based) and flip '
                        'RHM-tree levels to bottom-up (leaves=0, root=L). '
                        'Display only.')
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        raise SystemExit(f'--sweep_dir not a directory: {sweep_dir}')
    out_path = Path(args.out) if args.out else sweep_dir / 'sweep_circuit.html'

    final = render_sweep_html(
        sweep_dir=sweep_dir,
        out_path=out_path,
        show_all_logits=args.show_all_logits,
        show_errors=not args.hide_errors,
        inline_js=args.inline_js,
        report_notation=args.report_notation,
    )
    print(f'Wrote {final}')


if __name__ == '__main__':
    main()
