"""Oracle-conditional unit test for tree reconstruction.

Exercises the cond_prob recovery and per-(l, j) score aggregation in
isolation - no transformer, no SAE.

Build a synthetic feature whose cond_prob[i] is one-hot at value z0 at
the canonical (level, j); feed it into the score formula and assert the
argmax recovers z0.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.sae_tree_reconstruction.run import (
    _build_cond_prob, _full_cond_prob_at, _num_values_for_level,
)


def _make_synthetic_artifact():
    """One layer, P=2 SAE positions, F=4 features, single (level, j) group with V=3.

    Feature 0: fires only when Z=0 (one-hot cond_prob at value 0).
    Feature 1: fires only when Z=1.
    Feature 2: fires only when Z=2.
    Feature 3: dead (firing_count == 0).

    joint_fire_count[v, p, i] = 1 iff i == v else 0  (only counted for live features).
    firing_count[p, i]        = 1 for live, 0 for dead.
    firing_rate[p, i]         = firing_count / 1.
    """
    V = 3
    P = 2
    F = 4
    joint = torch.zeros(V, P, F, dtype=torch.long)
    for v in range(V):
        joint[v, :, v] = 1     # value v <-> feature v
    firing = torch.zeros(P, F, dtype=torch.long)
    firing[:, :3] = 1          # features 0..2 alive at every position
    firing_rate = firing.double()

    targets = [
        {'level': 0, 'position': 0, 'value': 0, 'count': 1},
        {'level': 0, 'position': 0, 'value': 1, 'count': 1},
        {'level': 0, 'position': 0, 'value': 2, 'count': 1},
    ]
    index_layout = [{
        'level': 0, 'position': 0,
        'values': torch.tensor([0, 1, 2], dtype=torch.long),
        'start': 0, 'end': 3,
    }]
    return {
        'joint_fire_count': joint,
        'firing_count': firing,
        'firing_rate': firing_rate,
        'targets': targets,
        'index_layout': index_layout,
    }, V, P, F


def test_cond_prob_recovery():
    art, V, P, F = _make_synthetic_artifact()
    groups = _build_cond_prob(art)
    assert (0, 0) in groups
    cp = groups[(0, 0)]['cond_prob']        # [V, P, F]
    assert cp.shape == (V, P, F)
    # Live features 0..2 have cond_prob[v, *, i] == (1 if v == i else 0).
    for v in range(V):
        for i in range(3):
            expected = 1.0 if v == i else 0.0
            assert torch.allclose(cp[v, :, i].float(),
                                  torch.tensor(expected).expand(P)), (
                f'cond_prob mismatch at v={v}, i={i}'
            )
    # Dead feature 3 -> NaN columns.
    assert torch.isnan(cp[:, :, 3]).all()


def test_full_cond_prob_at_passthrough_when_full():
    art, V, P, F = _make_synthetic_artifact()
    groups = _build_cond_prob(art)
    full = _full_cond_prob_at(groups[(0, 0)], V_full=V)
    assert full.shape == (V, P, F)
    # When V_g == V_full, output equals cp.
    assert torch.equal(
        torch.nan_to_num(full, nan=-1.0),
        torch.nan_to_num(groups[(0, 0)]['cond_prob'], nan=-1.0),
    )


def test_full_cond_prob_at_zero_pads_unobserved():
    """If V_g < V_full, unobserved values get zero (not NaN)."""
    # Same artifact but pretend V_full = 5 (only values 0,1,2 observed).
    art, V, P, F = _make_synthetic_artifact()
    groups = _build_cond_prob(art)
    full = _full_cond_prob_at(groups[(0, 0)], V_full=5)
    assert full.shape == (5, P, F)
    # Live features at unobserved values 3, 4 -> 0.
    for v in (3, 4):
        for i in range(3):
            assert full[v, :, i].abs().sum() == 0.0
    # Dead feature still NaN at every value.
    assert torch.isnan(full[:, :, 3]).all()


def _score_p(cp_p: torch.Tensor, f_act: torch.Tensor, alive: torch.Tensor,
             weighting: str):
    """Reproduce one-position score from run.py for one input vector.

    cp_p:     [V, F]    (NaN at dead features, zero elsewhere if unobserved)
    f_act:    [F]       activations (post decoder weighting)
    alive:    [F]  bool
    """
    fire = (f_act > 0) & alive
    if weighting == 'activation':
        w = torch.where(fire, f_act, torch.zeros_like(f_act))
    else:
        w = fire.double()
    cp_clean = torch.where(torch.isnan(cp_p), torch.zeros_like(cp_p), cp_p)
    num = (w.unsqueeze(0) * cp_clean).sum(dim=-1)   # [V]
    den = w.sum()
    if den.item() == 0:
        return None
    return num / den


def test_oracle_argmax_activation_weighting():
    art, V, P, F = _make_synthetic_artifact()
    groups = _build_cond_prob(art)
    cp = _full_cond_prob_at(groups[(0, 0)], V_full=V)   # [V, P, F]
    alive = (art['firing_rate'] > 0)                    # [P, F]

    # Construct an input where only feature 1 fires at position 0.
    # The oracle conditional says feature 1 -> value 1.
    f_act = torch.zeros(P, F, dtype=torch.float64)
    f_act[0, 1] = 0.7
    sp = _score_p(cp[:, 0, :], f_act[0], alive[0], 'activation')
    assert sp is not None
    assert int(sp.argmax().item()) == 1, f'expected argmax=1, got {sp.argmax()}'

    # Ditto for value 2 via feature 2.
    f_act = torch.zeros(P, F, dtype=torch.float64)
    f_act[0, 2] = 5.0
    sp = _score_p(cp[:, 0, :], f_act[0], alive[0], 'activation')
    assert int(sp.argmax().item()) == 2


def test_oracle_argmax_uniform_weighting():
    art, V, P, F = _make_synthetic_artifact()
    groups = _build_cond_prob(art)
    cp = _full_cond_prob_at(groups[(0, 0)], V_full=V)
    alive = (art['firing_rate'] > 0)

    # Two features fire (1 and 2). Activation weighting would tilt by
    # magnitude; uniform weighting averages cond_prob equally.
    f_act = torch.zeros(P, F, dtype=torch.float64)
    f_act[0, 1] = 0.1
    f_act[0, 2] = 100.0
    sp_act = _score_p(cp[:, 0, :], f_act[0], alive[0], 'activation')
    sp_uni = _score_p(cp[:, 0, :], f_act[0], alive[0], 'uniform')
    # Activation: argmax should be 2 (huge weight on value-2 feature).
    assert int(sp_act.argmax().item()) == 2
    # Uniform: scores at v=1 and v=2 are tied, both 0.5; argmax may break the
    # tie either way, but they should be equal.
    assert torch.isclose(sp_uni[1], sp_uni[2])
    # Value 0 contributed nothing.
    assert sp_uni[0].item() == 0.0


def test_dead_features_excluded():
    """Dead features (alive=False) must not contribute even if 'fired'."""
    art, V, P, F = _make_synthetic_artifact()
    groups = _build_cond_prob(art)
    cp = _full_cond_prob_at(groups[(0, 0)], V_full=V)
    alive = (art['firing_rate'] > 0)

    # Force-fire the dead feature (#3) and nothing else. It must be
    # excluded -> empty A_p -> score returns None.
    f_act = torch.zeros(P, F, dtype=torch.float64)
    f_act[0, 3] = 99.0
    sp = _score_p(cp[:, 0, :], f_act[0], alive[0], 'activation')
    assert sp is None, 'dead feature should not contribute'


def test_position_aggregation_mean():
    """Per-(l, j) score is mean over non-empty positions of normalized
    per-position scores. Empty positions are skipped."""
    art, V, P, F = _make_synthetic_artifact()
    groups = _build_cond_prob(art)
    cp = _full_cond_prob_at(groups[(0, 0)], V_full=V)
    alive = (art['firing_rate'] > 0)

    f_act = torch.zeros(P, F, dtype=torch.float64)
    f_act[0, 0] = 1.0          # position 0 votes for value 0
    f_act[1, 1] = 1.0          # position 1 votes for value 1
    sp0 = _score_p(cp[:, 0, :], f_act[0], alive[0], 'activation')
    sp1 = _score_p(cp[:, 1, :], f_act[1], alive[1], 'activation')
    assert sp0 is not None and sp1 is not None
    avg = (sp0 + sp1) / 2.0
    # values 0 and 1 tied at 0.5 each, value 2 == 0.
    assert torch.isclose(avg[0], avg[1])
    assert avg[2].item() == 0.0


def main():
    tests = [
        test_cond_prob_recovery,
        test_full_cond_prob_at_passthrough_when_full,
        test_full_cond_prob_at_zero_pads_unobserved,
        test_oracle_argmax_activation_weighting,
        test_oracle_argmax_uniform_weighting,
        test_dead_features_excluded,
        test_position_aggregation_mean,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e}')
        except Exception as e:
            failed += 1
            print(f'ERROR {t.__name__}: {type(e).__name__}: {e}')
    if failed:
        print(f'\n{failed} test(s) failed.')
        return 1
    print(f'\nall {len(tests)} tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
