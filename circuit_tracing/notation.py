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


# RHM-level ring palette, shared with the level-distribution plots
# (scripts/sae_sweep/feature_splitting_diag.py LEVEL_PALETTE). Used to color a
# node's marker RING by which RHM level it resolves, so fill = selectivity
# (entropy) and ring = level. Indexed by CODE level (depth from root): root=0 is
# green, deeper levels go purple -> brown -> pink. The leaf level lives at code
# level L: for L=3 that is index 3 (pink); for L=4 it is index 4 (light blue).
# Adding a hierarchy level thus shifts every report level's color up by one
# (green stays at the root) and the new deepest leaf level gets light blue.
LEVEL_RING_PALETTE = ('#4daf4a', '#984ea3', '#8B4513', '#f781bf',
                      '#6baed6', '#bcbd22', '#000000')
LEVEL_RING_FALLBACK = '#999999'


def level_ring_color(code_level, L):
    """Ring color for an RHM code level (root=0, leaf=L).

    Indexed by code level directly so root=green and the leaves are pink (the
    color names the report uses). `L` is accepted for signature symmetry with
    the other notation helpers but is not needed for the index. None or
    out-of-range -> fallback gray (dead / unlabeled nodes)."""
    if code_level is None:
        return LEVEL_RING_FALLBACK
    cl = int(code_level)
    if cl < 0 or cl >= len(LEVEL_RING_PALETTE):
        return LEVEL_RING_FALLBACK
    return LEVEL_RING_PALETTE[cl]
