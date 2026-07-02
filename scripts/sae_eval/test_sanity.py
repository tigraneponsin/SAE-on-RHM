"""Four cheap acceptance checks for the entropy machinery.

Entropy sanity / acceptance checks (see docs/sae_eval_guide.md for the metrics).
All run on synthetic tensors or a small RHM instance; none require a trained
model. Exit code is non-zero if any check fails.

    python scripts/sae_eval/test_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.random_hierarchy_model import (
    latent_entropy,
    latent_prior,
    sample_rules,
    sample_trees,
)
from scripts.sae_eval.streaming import (
    per_feature_entropy,
    shannon_entropy,
    weighted_aggregate,
)


# ---------------------------------------------------------------------------
# Check 1: theoretical vs empirical marginal and entropy
# ---------------------------------------------------------------------------

def check_theoretical_vs_empirical(N: int = 200_000, tol_sigmas: float = 5.0) -> None:
    """P_emp(Z_{l,j}) should agree with P_theo(Z_{l,j}) within ~tol_sigmas / sqrt(N)."""
    n, v, m, s, L = 3, 4, 2, 2, 3
    rules = sample_rules(v, n, m, s, L, seed=42)
    priors = latent_prior(rules, n, v)
    H_theo = latent_entropy(priors)

    trees = sample_trees(N, rules, seed=123)
    tol = tol_sigmas / (N ** 0.5)

    for level, prior_l in priors.items():
        tree_l = trees[level]
        if tree_l.ndim == 1:
            emp = torch.bincount(tree_l, minlength=prior_l.size(1)).double() / N
            diff = (prior_l[0].double() - emp).abs().max().item()
            assert diff < tol, (
                f'level {level}: max|theo - emp| = {diff:.4f} '
                f'exceeds {tol:.4f} (tol_sigmas={tol_sigmas}, N={N})'
            )
        else:
            width = tree_l.size(1)
            for j in range(width):
                emp = torch.bincount(
                    tree_l[:, j], minlength=prior_l.size(1)
                ).double() / N
                diff = (prior_l[j].double() - emp).abs().max().item()
                assert diff < tol, (
                    f'level {level}, j={j}: max|theo - emp| = {diff:.4f} '
                    f'exceeds {tol:.4f}'
                )

        H_emp = shannon_entropy(
            torch.stack([
                (torch.bincount(
                    tree_l if tree_l.ndim == 1 else tree_l[:, j],
                    minlength=prior_l.size(1),
                ).double() / N)
                for j in range(1 if tree_l.ndim == 1 else tree_l.size(1))
            ]),
            dim=-1,
        )
        H_diff = (H_theo[level].double() - H_emp).abs().max().item()
        assert H_diff < 10 * tol, (
            f'level {level}: |H_theo - H_emp| = {H_diff:.4f} '
            f'exceeds {10 * tol:.4f}'
        )


# ---------------------------------------------------------------------------
# Check 2: oracle feature yields H_i = 0
# ---------------------------------------------------------------------------

def check_oracle_feature() -> None:
    """Feature F(x) = 1[Z(x) == z_0] must give H_i = 0 exactly."""
    V, F_dim = 8, 16
    torch.manual_seed(0)
    # Random target counts per value (the marginal distribution of Z given fire).
    firing_count_per_value = torch.randint(10, 1000, (V,)).long()  # [V]
    total_fire = int(firing_count_per_value.sum().item())

    # For each feature, pick a different oracle value z_0. All mass on z_0.
    joint = torch.zeros(V, F_dim, dtype=torch.long)
    for f in range(F_dim):
        z0 = f % V
        joint[z0, f] = total_fire
    marginal = joint.sum(dim=0)  # [F]
    H = per_feature_entropy(joint, marginal)
    assert torch.all(torch.isfinite(H)), f'H has NaN: {H}'
    max_H = H.abs().max().item()
    assert max_H < 1e-12, f'oracle feature: max |H_i| = {max_H} (expected 0)'


# ---------------------------------------------------------------------------
# Check 3: uniform feature yields H_i = H_emp(Z)
# ---------------------------------------------------------------------------

def check_uniform_feature() -> None:
    """Feature that always fires must give H_i == shannon_entropy(empirical P(Z))."""
    V, F_dim = 8, 4
    torch.manual_seed(1)
    counts_per_value = torch.randint(1, 500, (V,)).long()  # empirical count per value
    total = int(counts_per_value.sum().item())
    # Replicate across features: all features fire on every sample, so
    # joint[v, f] = counts_per_value[v] and marginal[f] = total.
    joint = counts_per_value.unsqueeze(-1).expand(V, F_dim).contiguous()
    marginal = torch.full((F_dim,), total, dtype=torch.long)

    H = per_feature_entropy(joint, marginal)
    p_emp = counts_per_value.double() / total
    H_expected = shannon_entropy(p_emp)  # scalar, float64
    diffs = (H.double() - H_expected).abs()
    max_diff = diffs.max().item()
    # per_feature_entropy casts to float32 for storage; float32 precision
    # at an entropy of ~log(V) = ~2 nats is ~1e-7.
    assert max_diff < 1e-6, (
        f'uniform feature: max |H_i - H_emp| = {max_diff} '
        f'(H = {H.tolist()}, H_emp = {H_expected.item():.6f})'
    )


# ---------------------------------------------------------------------------
# Check 4: aggregation degeneracy
# ---------------------------------------------------------------------------

def check_aggregation_degeneracy() -> None:
    """If H_i = h for every alive feature, weighted_aggregate must return h
    for any non-negative weights (with at least one positive)."""
    F_dim = 64
    torch.manual_seed(2)
    h = 1.237  # arbitrary entropy value in nats

    H = torch.full((F_dim,), h)
    # Inject some dead features (NaN H + zero weight).
    dead_idx = torch.tensor([3, 10, 41])
    H[dead_idx] = float('nan')

    for trial in range(5):
        weights = torch.rand(F_dim)
        weights[dead_idx] = 0.0  # dead features have zero weight
        agg = weighted_aggregate(H, weights)
        assert torch.isfinite(agg), f'aggregate is not finite: {agg}'
        assert abs(agg.item() - h) < 1e-6, (
            f'trial {trial}: weighted aggregate = {agg.item()} vs h = {h}'
        )

    # Edge case: all weights zero -> NaN.
    zero_w = torch.zeros(F_dim)
    agg = weighted_aggregate(H, zero_w)
    assert torch.isnan(agg), f'expected NaN for all-zero weights, got {agg}'


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    checks = [
        ('theoretical vs empirical marginal', check_theoretical_vs_empirical),
        ('oracle feature H_i = 0',            check_oracle_feature),
        ('uniform feature H_i = H_emp(Z)',    check_uniform_feature),
        ('aggregation degeneracy',            check_aggregation_degeneracy),
    ]
    failed = 0
    for name, fn in checks:
        try:
            fn()
        except AssertionError as e:
            print(f'FAIL: {name}')
            print(f'      {e}')
            failed += 1
        except Exception as e:
            print(f'ERROR: {name}: {type(e).__name__}: {e}')
            failed += 1
        else:
            print(f'PASS: {name}')
    if failed:
        print(f'\n{failed} check(s) failed.')
    else:
        print('\nAll 4 checks passed.')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
