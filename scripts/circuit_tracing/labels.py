"""Per-(layer, position, feature) latent labels from .sae_eval.pt artifacts.

For each (layer k, position p, feature f) we compute the latent label under
FOUR schemes, all from the same per-feature artifact data via the shared core
in scripts/sae_sweep/absorption.py:

  parent      : the block-aligned parent latent cell (level = L-1-k,
                position = p // s^(1+k)). Value = argmax_v P(value | f fires)
                at that cell. No argmin -- the locality assumption, identical to
                the original labeler.
  level       : argmin entropy over the matched-level cells, then value =
                argmax_v P(value | f fires) at the winning cell.
  whole_tree  : argmin entropy over ALL (level, position) cells (the
                unconstrained "child"), then value at the winning cell.
  reassigned  : whole_tree, but when the absorption ratio < alpha the primary
                label is bumped up to the child's tree-parent cell
                (level-1, pos//s) and its value; the child cell/value is kept as
                a precision tag.

Each (p, f) entry carries a `schemes` dict with one record per scheme, plus
top-level back-compat fields that mirror a chosen `primary` scheme so existing
consumers (dag.build_node_table, the visualizers) keep working unchanged.

The relevant block-aligned (level, position) is
    level = L - 1 - k
    parent_position = p // s ** (1 + k)
where (s, L) are the RHM branching factor and depth.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# The shared per-feature labeling core lives under scripts/sae_sweep. Make it
# importable the same way circuit_trace.py makes scripts.* importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT),
           str(_REPO_ROOT / 'scripts' / 'sae_sweep'),
           str(_REPO_ROOT / 'scripts' / 'common')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from absorption import per_feature_labels, _ArtifactTables  # noqa: E402

SCHEME_NAMES = ('parent', 'level', 'whole_tree', 'reassigned')
DEFAULT_PRIMARY = 'reassigned'
DEFAULT_ALPHA = 0.5


def parent_position(p: int, k: int, s: int) -> int:
    """Ancestor position at level L-1-k for leaf position p.

    From CLAUDE.md and the handoff doc: a_{L-1-k}(p) = p // s^(1+k).
    """
    return int(p) // int(s) ** (1 + int(k))


def parent_level(k: int, L: int) -> int:
    return int(L) - 1 - int(k)


def _norm_entropy(h, href):
    """H / H_theoretical clipped to [0, 1]; None when undefined/dead."""
    if h is None or href is None:
        return None
    h = float(h)
    href = float(href)
    if not (math.isfinite(h) and math.isfinite(href) and href > 0):
        return None
    v = h / href
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    return v


def _empty_scheme(level, position):
    """A dead/undefined scheme record (feature didn't fire or no cell)."""
    return {
        'level': level, 'position': position,
        'value': None, 'p_value_given_fire': float('nan'),
        'normalized_entropy': None,
        'value_distribution': None, 'values': None,
    }


def build_labels(eval_artifact: dict, layer_id: int, s: int, L: int,
                 leak_norm: dict, alpha: float = DEFAULT_ALPHA,
                 primary: str = DEFAULT_PRIMARY) -> dict:
    """Per-(p, f) label table for one SAE eval artifact, all four schemes.

    leak_norm is the value-specific structural leak table
    (absorption.leak_table_from_rules / resolve_leak_table); required for the
    'reassigned' scheme. alpha is the absorption reassignment threshold.

    Returns:
        dict[(p, f)] -> {
            'firing_count': int,
            'schemes': {scheme -> {
                'level': int, 'position': int,
                'value': int | None, 'p_value_given_fire': float,
                'normalized_entropy': float | None,
                'value_distribution': FloatTensor[V_g] | None,
                'values': LongTensor[V_g] | None,
                # 'reassigned' scheme only:
                'reassigned': bool, 'ratio': float,
                'child_level': int, 'child_position': int,
                'child_value': int | None,
            }},
            # back-compat top-level fields mirror `primary`:
            'value', 'p_value_given_fire', 'level', 'parent_position',
            'value_distribution', 'values', 'normalized_entropy',
        }

    The 'parent' scheme reproduces the original (locality) labeler exactly.
    """
    if primary not in SCHEME_NAMES:
        raise ValueError(f'unknown primary scheme {primary!r}; '
                         f'choose from {SCHEME_NAMES}')

    core = per_feature_labels(eval_artifact, leak_norm, alpha)
    T = _ArtifactTables(eval_artifact, leak_norm)
    firing = eval_artifact['firing_count'].long()  # [P, F]
    P = core['P']
    Fdim = T.F

    target_level = parent_level(layer_id, L)

    out: dict = {}
    for p_idx in range(P):
        rec = core['per_position'][p_idx]
        p_real = core['positions'][p_idx]
        parent_key = core['parent_group'][p_idx]  # (level, pos) block-aligned

        # Per-scheme winning cell (level, pos) and entropy/href, all [F] except
        # parent which is a single cell shared across features at this position.
        # Distributions are gathered with dist_at_cell at each scheme's cell.
        parent_lvl_t = torch.full((Fdim,), int(parent_key[0]), dtype=torch.long)
        parent_pos_t = torch.full((Fdim,), int(parent_key[1]), dtype=torch.long)

        dist_by_scheme = {
            'parent': T.dist_at_cell(parent_lvl_t, parent_pos_t, p_idx),
            'level': T.dist_at_cell(rec['level_cell_level'],
                                    rec['level_cell_pos'], p_idx),
            'whole_tree': T.dist_at_cell(rec['child_level'],
                                         rec['child_pos'], p_idx),
            'reassigned': T.dist_at_cell(rec['final_level'],
                                         rec['final_pos'], p_idx),
        }

        # Per-scheme entropy + href tensors ([F]) for normalized_entropy.
        H_parent = rec['H_parent']
        href_parent = rec['parent_href']
        ent_by_scheme = {
            'parent': (H_parent, torch.full((Fdim,), float(href_parent),
                                            dtype=torch.float64)),
            'level': (rec['level_min'], rec['level_href'].double()),
            'whole_tree': (rec['wt_min'], rec['child_href'].double()),
            'reassigned': (rec['final_entropy'], rec['final_href'].double()),
        }

        cell_by_scheme = {
            'parent': (parent_lvl_t, parent_pos_t),
            'level': (rec['level_cell_level'], rec['level_cell_pos']),
            'whole_tree': (rec['child_level'], rec['child_pos']),
            'reassigned': (rec['final_level'], rec['final_pos']),
        }

        reassign = rec['reassign']
        ratio = rec['ratio']
        child_level = rec['child_level']
        child_pos = rec['child_pos']
        child_value = rec['child_value']

        for f in range(Fdim):
            fc = int(firing[p_idx, f].item())
            schemes = {}
            for sc in SCHEME_NAMES:
                lvl_t, pos_t = cell_by_scheme[sc]
                lvl = int(lvl_t[f].item())
                pos = int(pos_t[f].item())
                if fc == 0:
                    schemes[sc] = _empty_scheme(lvl, pos)
                else:
                    dist = dist_by_scheme[sc].get(f)
                    H_t, href_t = ent_by_scheme[sc]
                    ne = _norm_entropy(float(H_t[f].item()),
                                       float(href_t[f].item()))
                    if dist is None:
                        schemes[sc] = {
                            'level': lvl, 'position': pos,
                            'value': None, 'p_value_given_fire': float('nan'),
                            'normalized_entropy': ne,
                            'value_distribution': None, 'values': None,
                        }
                    else:
                        # col is float64; pgf is read from it (matching the
                        # original labeler), value_distribution stored float32.
                        values_t, col = dist
                        vi = int(col.argmax().item())
                        schemes[sc] = {
                            'level': lvl, 'position': pos,
                            'value': int(values_t[vi].item()),
                            'p_value_given_fire': float(col[vi].item()),
                            'normalized_entropy': ne,
                            'value_distribution': col.float().clone(),
                            'values': values_t.clone(),
                        }
                # reassigned-only precision tags.
                if sc == 'reassigned':
                    schemes[sc]['reassigned'] = bool(reassign[f].item()) \
                        if fc != 0 else False
                    schemes[sc]['ratio'] = float(ratio[f].item())
                    schemes[sc]['child_level'] = int(child_level[f].item())
                    schemes[sc]['child_position'] = int(child_pos[f].item())
                    cv = int(child_value[f].item())
                    schemes[sc]['child_value'] = cv if (fc != 0 and cv >= 0) else None

            prim = schemes[primary]
            out[(p_idx, f)] = {
                'firing_count': fc,
                'schemes': schemes,
                # back-compat top-level mirror of the primary scheme.
                'value': prim['value'],
                'p_value_given_fire': prim['p_value_given_fire'],
                'level': prim['level'],
                'parent_position': prim['position'],
                'value_distribution': prim['value_distribution'],
                'values': prim['values'],
                'normalized_entropy': prim['normalized_entropy'],
            }
    return out


def build_labels_per_layer(eval_artifacts: list[dict], s: int, L: int,
                           leak_norms: list[dict],
                           alpha: float = DEFAULT_ALPHA,
                           primary: str = DEFAULT_PRIMARY) -> list[dict]:
    """Vectorize build_labels over a list of artifacts (one per layer).

    leak_norms is one leak table per layer (they are identical across a sweep,
    but each artifact's H_theoretical is used for normalization, so we accept a
    per-layer list; callers may pass the same table repeated).
    """
    return [build_labels(art, k, s, L, leak_norms[k], alpha=alpha,
                         primary=primary)
            for k, art in enumerate(eval_artifacts)]
