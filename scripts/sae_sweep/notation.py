"""Report-notation helpers for sweep plots.

Pure display helpers. They never change any computed index or stored data; they
only remap how a layer/level integer is rendered in plot titles, legends, and
axis labels when the optional --report-notation flag is set.

Code conventions (default, flag off):
  - SAE / transformer block index is 0-based: layer 0 .. layer L-1.
  - "level" integers in the sweep diagnostics come from the matched-latent
    layout level = L-1-layer_id, and the circuit RHM tree uses level l in
    [0..L] top-down (root=0, leaf=L).

Report conventions (flag on):
  - SAE / block index is 1-based and labeled "SAE k": SAE 1 .. SAE L.
  - RHM levels are bottom-up: leaves = level 0, root = level L. The single flip
    rule is report_level(l, L) = L - l.
"""


def sae_label(layer_id, report, short=False):
    """Label for a 0-based SAE/layer index.

    report=False : "Layer {id}"  (or "L{id}" if short)
    report=True  : "SAE {id+1}"  (or "SAE{id+1}" if short)
    """
    if report:
        return ('SAE%d' if short else 'SAE %d') % (int(layer_id) + 1)
    return ('L%d' if short else 'Layer %d') % int(layer_id)


def report_level(code_level, L):
    """Flip a tree level (l in [0..L], root=0/leaf=L) to report convention
    (leaf=0/root=L). Identity-free: report = L - l."""
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
