"""Report-notation helpers, shared by sweep plots and circuit-tracing figures.

Pure display helpers. They never change any computed index or stored data; they
only remap how a layer/level integer is rendered in plot titles, legends, axis
ticks, and hover text when report notation is enabled. This module merges the
former scripts/sae_sweep/notation.py and circuit_tracing/notation.py so every
figure shares one convention.

Code conventions (report off):
  - SAE / transformer block index is 0-based: layer 0 .. layer L-1.
  - "level" integers come from the matched-latent layout level = L-1-layer_id;
    the circuit RHM tree uses level l in [0..L] top-down (root=0, leaf=L).

Report conventions (report on):
  - SAE / block index is 1-based, labeled "SAE k": SAE 1 .. SAE L.
  - RHM levels are bottom-up: leaves = level 0, root = level L. The single flip
    rule is report_level(l, L) = L - l.
"""


def sae_label(layer_id, report, short=False):
    """Label for a 0-based SAE/layer index (sweep-plot style).

    report=False : "Layer {id}"  (or "L{id}" if short)
    report=True  : "SAE {id+1}"  (or "SAE{id+1}" if short)
    """
    if report:
        return ('SAE%d' if short else 'SAE %d') % (int(layer_id) + 1)
    return ('L%d' if short else 'Layer %d') % int(layer_id)


def sae_row_label(layer_id, report):
    """Circuit y-axis row label for a 0-based layer index.

    report=False : "layer {id}"   (lowercase, circuit convention)
    report=True  : "SAE {id+1}"
    """
    if report:
        return 'SAE %d' % (int(layer_id) + 1)
    return 'layer %d' % int(layer_id)


def report_level(code_level, L):
    """Flip a tree level l in [0..L] (root=0, leaf=L) to report convention
    (leaf=0, root=L). Identity-free: report = L - l."""
    return int(L) - int(code_level)


def report_matched_level(layer_id):
    """Report level of the latent resolved by 0-based SAE `layer_id`.

    Code matched level is L-1-layer_id; in report convention (leaf=0, root=L)
    that latent sits at level layer_id+1 (SAE 1 resolves level 1, SAE L the
    root at level L)."""
    return int(layer_id) + 1


def add_report_flag(parser):
    """Add the uniform --report-notation flag to an argparse parser."""
    parser.add_argument(
        '--report-notation', dest='report_notation', action='store_true',
        help='Relabel plots with report notation: SAE k (1-based) instead of '
             'Layer k (0-based), and bottom-up RHM levels (leaves=0, root=L). '
             'Display only; computed values and saved files are unchanged.')
    return parser


# RHM-level ring palette, shared with the level-distribution plots. Used to color
# a node's marker RING by which RHM level it resolves, so fill = selectivity
# (entropy) and ring = level. Indexed by CODE level (depth from root): root=0 is
# green, deeper levels go purple -> brown -> pink. The leaf level lives at code
# level L. Adding a hierarchy level shifts every report level's color up by one
# (green stays at the root) and the new deepest leaf level gets light blue.
LEVEL_RING_PALETTE = ('#4daf4a', '#984ea3', '#8B4513', '#f781bf',
                      '#6baed6', '#bcbd22', '#000000')
LEVEL_RING_FALLBACK = '#999999'


def level_ring_color(code_level, L):
    """Ring color for an RHM code level (root=0, leaf=L).

    Indexed by code level directly so root=green and the leaves are pink. `L` is
    accepted for signature symmetry with the other helpers but is not needed for
    the index. None or out-of-range -> fallback gray (dead / unlabeled nodes)."""
    if code_level is None:
        return LEVEL_RING_FALLBACK
    cl = int(code_level)
    if cl < 0 or cl >= len(LEVEL_RING_PALETTE):
        return LEVEL_RING_FALLBACK
    return LEVEL_RING_PALETTE[cl]
