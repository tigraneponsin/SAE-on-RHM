"""Shared numeric helpers for the SAE entropy diagnostics.

These pieces are used by entropy_diag.py (the unified entropy CLI),
feature_splitting_alpha_sweep.py, and absorption.py (hence, transitively, by
circuit_tracing). They are pure and import-light on purpose: this module must
NOT import absorption (absorption imports torch_nanmin from here), so keep it
free of any sweep/circuit dependency to avoid an import cycle.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

# weight schemes -> (csv suffix, label); 'fire'/'raw'/'dec' mirror
# scripts/sae_eval/streaming.py. 'unweighted' is diag-only (not stored in the
# sae_eval artifact): every feature that ever fires counts equally.
SCHEMES = ('fire', 'raw', 'dec', 'unweighted')


def torch_nanmin(x, dim):
    """NaN-aware min along dim. Returns (values, argmin_index).

    NaNs are treated as +inf for the min; if all entries along dim are NaN the
    value stays NaN and the index is 0 (the aggregate ignores NaN values).
    """
    big = torch.full_like(x, float('inf'))
    filled = torch.where(torch.isnan(x), big, x)
    vals, idx = filled.min(dim=dim)
    all_nan = torch.isnan(x).all(dim=dim)
    vals = torch.where(all_nan, torch.full_like(vals, float('nan')), vals)
    return vals, idx


def weighted_aggregate(H, weights):
    """Weighted mean of H ignoring NaN entries (dead features).

    Copied verbatim semantics from scripts/sae_eval/streaming.py so the
    parent-conditioned recomputation reproduces the stored H_bar_* exactly.
    """
    if H.shape != weights.shape:
        raise ValueError('Shape mismatch: H %s vs weights %s.'
                         % (tuple(H.shape), tuple(weights.shape)))
    H64 = H.double()
    w64 = weights.double()
    mask = torch.isfinite(H64) & (w64 > 0)
    w_masked = torch.where(mask, w64, torch.zeros_like(w64))
    h_masked = torch.where(mask, H64, torch.zeros_like(H64))
    num = (w_masked * h_masked).sum(dim=-1)
    den = w_masked.sum(dim=-1)
    out = torch.where(den > 0, num / den.clamp_min(1e-30),
                      torch.full_like(den, float('nan')))
    return out.float()


def weights_for_scheme(scheme, firing_rate_p, baseline_mean_p, decoder_norms):
    """Per-position [F] weight vector, mirroring streaming.py:748-750."""
    if scheme == 'fire':
        return firing_rate_p
    if scheme == 'raw':
        return baseline_mean_p / decoder_norms.clamp_min(1e-30)
    if scheme == 'dec':
        return baseline_mean_p
    if scheme == 'unweighted':
        return (firing_rate_p > 0).float()
    raise ValueError('unknown scheme %r' % scheme)


def find_threshold_lambda(csv_path, layer, mode, tolerance):
    """Smallest lambda_l1 for which norm_err - baseline_err exceeds tolerance.

    Reads sweep_metrics.csv, filters rows matching (layer, mode), and returns
    the threshold lambda or None if no row crosses it (or the csv is missing /
    malformed).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        print(f'  threshold: csv not found at {csv_path}, skipping line')
        return None
    rows = []
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['layer']) != int(layer):
                    continue
                if mode and row.get('mode', '') and row['mode'] != mode:
                    continue
                try:
                    lam = float(row['lambda_l1'])
                    ne = float(row['norm_err'])
                    be = float(row['baseline_err'])
                except (KeyError, ValueError):
                    continue
                rows.append((lam, ne, be))
    except Exception as ex:
        print(f'  threshold: failed to read {csv_path}: {ex}')
        return None
    if not rows:
        print(f'  threshold: no rows for layer={layer} mode={mode} in {csv_path.name}')
        return None
    rows.sort(key=lambda r: r[0])
    for lam, ne, be in rows:
        if (ne - be) > tolerance:
            print(f'  threshold (layer={layer} mode={mode}): lambda_l1={lam:g} '
                  f'(norm_err={ne:.4g}, baseline_err={be:.4g}, '
                  f'tolerance={tolerance})')
            return lam
    print(f'  threshold: norm_err never exceeds baseline + {tolerance} for '
          f'layer={layer} mode={mode}')
    return None


# Back-compat alias: several callers historically imported this private name.
_find_threshold_lambda = find_threshold_lambda


def thr_label(tolerance):
    """Legend label for the threshold line, e.g. '1% err onset'."""
    return '%g%% err onset' % (tolerance * 100)


def parent_min_sanity(art, diag, tol=1e-4):
    """Assert min<=parent everywhere and stored H_bar_fire matches recompute.

    The parent is one of the same-level candidates, so the same-level min must
    be <= parent, and the recomputed parent H_bar_fire must match the value the
    streaming eval stored.
    """
    import math
    for p_idx in range(diag['P']):
        par = diag['H_bar_fire'][p_idx]
        mn = diag['H_bar_min_fire_same_level'][p_idx]
        if math.isfinite(par) and math.isfinite(mn):
            assert mn <= par + 1e-6, (
                'pos %d: min %.6f > parent %.6f' % (p_idx, mn, par))
        stored = diag['stored_H_bar_fire'][p_idx]
        if math.isfinite(stored) and math.isfinite(par):
            assert abs(stored - par) < tol, (
                'pos %d: recomputed parent H_bar_fire %.6f != stored %.6f'
                % (p_idx, par, stored))
