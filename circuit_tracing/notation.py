"""Report-notation helpers for circuit-tracing visualizations.

Display only. Mirrors scripts/sae_sweep/notation.py so all figures share one
relabeling convention. Computed indices, graph topology, node values, and saved
artifacts are never changed; only axis ticks, hover text, and titles are remapped
when report_notation is True.

Code conventions (report_notation False):
  - SAE / block index 0-based: circuit y-axis rows read "layer 0 .. layer K-1".
  - RHM-tree panel uses level l in [0..L] top-down (root=0, leaf=L).

Report conventions (report_notation True):
  - SAE / block index 1-based, labeled "SAE k": SAE 1 .. SAE K.
  - RHM levels bottom-up: leaves=0, root=L, via report_level(l, L) = L - l.
"""


def sae_row_label(layer_id, report):
    """Circuit y-axis row label for a 0-based layer index."""
    if report:
        return 'SAE %d' % (int(layer_id) + 1)
    return 'layer %d' % int(layer_id)


def report_level(code_level, L):
    """Flip a tree level l in [0..L] (root=0, leaf=L) to report convention
    (leaf=0, root=L): report = L - l."""
    return int(L) - int(code_level)
