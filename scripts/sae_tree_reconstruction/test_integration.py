"""Integration test: stream_reconstruction end-to-end with mock model + SAEs.

We construct:
  - a tiny mock transformer whose 'blocks[k]' stamps the per-layer one-hot
    encoding of the canonical RHM latent at the matching position into the
    residual stream. Each layer k is responsible for level l = L - 1 - k,
    so position p at layer k 'sees' Z_{l, j} where j = p // s^(1+k).
  - a per-layer 'identity' SAE (encoder = identity, decoder = identity)
    with latent_dim = embedding_dim, so f_act[b, p, i] equals the post-block
    residual at (b, p, i).

The conditional cond_prob[i][z] is built so that feature i corresponds
to value i (one-hot at z=i). Because the residual at position p of layer k
is the one-hot encoding of Z_{l, j}, the reconstructor must recover the
exact latent everywhere.

Asserts:
  - per_level_acc[l] == 1.0 for every level
  - whole_tree_acc == 1.0
  - coverage[l] == 1.0
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from datasets.random_hierarchy_model import sample_rules, sample_trees
from scripts.sae_tree_reconstruction.run import (
    _LayerState, _build_cond_prob, _full_cond_prob_at, _metrics_for_weighting,
    _num_values_for_level, stream_reconstruction,
)


# ---- Mock SAE that satisfies the run.py interface -------------------------

class _IdentitySAE(nn.Module):
    """encoder + decoder are identity. f_act = ReLU(x) elementwise."""
    def __init__(self, dim: int):
        super().__init__()
        self.input_dim = dim
        self.latent_dim = dim
        self.encoder = nn.Linear(dim, dim, bias=True)
        self.decoder = nn.Linear(dim, dim, bias=True)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(dim))
            self.encoder.bias.zero_()
            self.decoder.weight.copy_(torch.eye(dim))
            self.decoder.bias.zero_()

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z

    def decoder_feature_norms(self, eps=1e-12):
        w_dec = self.decoder.weight
        return torch.sqrt((w_dec ** 2).sum(dim=0) + eps)


# ---- Mock transformer block ----------------------------------------------

class _OracleBlock(nn.Module):
    """At layer k for level l = L - 1 - k, position p has j = p // s^(1+k);
    the block writes one-hot(Z_{l,j}) of width V_full into the residual.

    Payload (set on the parent model) holds the full eval-set latents per
    real-token position; the parent passes a per-batch row range via
    `set_batch_range(s_idx, e_idx)`.
    """
    def __init__(self, layer_id: int, L: int, s: int, V_full_for_level: dict,
                 embedding_dim: int):
        super().__init__()
        self.layer_id = layer_id
        self.L = L
        self.s = s
        self.V_full_for_level = V_full_for_level
        self.embedding_dim = embedding_dim
        self.payload = None        # dict {p -> [N_total] long}
        self._range = (0, 0)

    def set_range(self, s_idx: int, e_idx: int):
        self._range = (s_idx, e_idx)

    def forward(self, x):
        B, P, D = x.shape
        out = torch.zeros_like(x)
        if self.payload is None:
            return out
        s_idx, e_idx = self._range
        for p in range(P):
            zs_full = self.payload.get(p)
            if zs_full is None:
                continue
            zs = zs_full[s_idx:e_idx]
            assert zs.size(0) == B
            vv = zs.clamp(min=0, max=self.embedding_dim - 1)
            idx = torch.arange(B, device=x.device)
            out[idx, p, vv] = 1.0
        return out


class _MockTransformer(nn.Module):
    def __init__(self, L: int, s: int, V_full_for_level: dict, embedding_dim: int):
        super().__init__()
        self.L = L
        self.s = s
        self.embedding_dim = embedding_dim
        self.num_layers = L
        self.blocks = nn.ModuleList([
            _OracleBlock(k, L, s, V_full_for_level, embedding_dim)
            for k in range(L)
        ])
        # _trees_payload[k] is set per forward() pass.
        self._batch_inputs_to_trees = None

    def set_payload(self, payloads_per_layer: dict):
        for k, p in payloads_per_layer.items():
            self.blocks[k].payload = p
        self._cursor = 0

    def forward(self, x):
        # x: leaf tokens [B, num_leaves]. Pass through blocks; the blocks
        # use their pre-loaded payload to stamp one-hots.
        B = x.size(0)
        s_idx = self._cursor
        e_idx = s_idx + B
        for blk in self.blocks:
            blk.set_range(s_idx, e_idx)
        self._cursor = e_idx
        h = torch.zeros(B, x.size(1), self.embedding_dim, device=x.device)
        for blk in self.blocks:
            h = blk(h)
        return None      # no classifier head


# ---- Build payloads from trees -------------------------------------------

def _payloads_from_trees(trees: dict, L: int, s: int) -> dict:
    """For each layer k, payload[k][p] = trees[l = L-1-k][:, j=p // s^(1+k)] long."""
    payloads = {}
    num_leaves = s ** L
    for k in range(L):
        l = L - 1 - k
        level_tensor = trees[l]
        per_pos = {}
        for p in range(num_leaves):
            j = p // (s ** (1 + k))
            if level_tensor.ndim == 1:
                per_pos[p] = level_tensor.long()
            else:
                per_pos[p] = level_tensor[:, j].long()
        payloads[k] = per_pos
    return payloads


# ---- Build a fake artifact for one layer ---------------------------------

def _fake_artifact_for_layer(layer_id: int, L: int, s: int, n: int, v: int,
                             trees: dict, embedding_dim: int) -> dict:
    """Construct a .sae_eval.pt-shaped dict for one layer.

    cond_prob[i][z] should be one-hot at z=i, achieved by joint_fire_count
    that's diagonal in (value, feature). All features 0..(V_max-1) are alive.
    """
    level = L - 1 - layer_id
    V_full = n if level == 0 else v
    # SAE position layout = all real tokens.
    num_leaves = s ** L
    P = num_leaves
    F = embedding_dim
    token_positions = torch.arange(P, dtype=torch.long)

    # index_layout: one group per (level, j_in_level).
    j_range = [0] if level == 0 else list(range(s ** level))
    index_layout = []
    targets = []
    start = 0
    for j in j_range:
        # observed values in trees at (level, j)
        if trees[level].ndim == 1:
            col = trees[level].long()
        else:
            col = trees[level][:, j].long()
        unique, counts = torch.unique(col, return_counts=True)
        values = unique.long()
        end = start + values.numel()
        index_layout.append({
            'level': int(level),
            'position': int(j),
            'values': values.clone(),
            'start': start,
            'end': end,
        })
        for vv, cc in zip(values.tolist(), counts.tolist()):
            targets.append({'level': int(level), 'position': int(j),
                            'value': int(vv), 'count': int(cc)})
        start = end

    T_total = len(targets)
    joint_fire_count = torch.zeros(T_total, P, F, dtype=torch.long)
    firing_count = torch.zeros(P, F, dtype=torch.long)
    firing_rate = torch.zeros(P, F, dtype=torch.float32)

    # For each canonical (level, j) group, only positions p with p // s^(1+k) == j
    # actually fire feature `value` when the latent at that group equals `value`.
    for g in index_layout:
        j_in_level = int(g['position'])
        ps = [p for p in range(P) if p // (s ** (1 + layer_id)) == j_in_level]
        for vv_pos, vv in enumerate(g['values'].tolist()):
            t_idx = g['start'] + vv_pos
            for p in ps:
                # feature index = vv (one-hot at value vv)
                joint_fire_count[t_idx, p, vv] = 1   # any positive count
        for p in ps:
            for vv in g['values'].tolist():
                firing_count[p, vv] = 1
                firing_rate[p, vv] = 1.0

    artifact = {
        'ckpt_path': f'/dev/null/layer{layer_id}.pt',
        'layer_id': int(layer_id),
        'mode': 'all_tokens',
        'sae_token_idx': 0,
        'act_scale': 1.0,
        'rhm': {'v': v, 'n': n, 'm': 1, 's': s, 'L': L},
        'eval_size': int(trees[L].size(0)),
        'eval_seed': 0,
        'dedupe': False,
        'joint_fire_count': joint_fire_count,
        'firing_count': firing_count,
        'firing_rate': firing_rate,
        'targets': targets,
        'index_layout': index_layout,
        'token_positions': token_positions,
        'latent_dim': F,
    }
    return artifact


# ---- The integration test -------------------------------------------------

def test_oracle_recovers_full_tree():
    n, v, m, s, L = 4, 4, 1, 2, 3
    embedding_dim = max(n, v)        # one-hot fits
    rules = sample_rules(v=v, n=n, m=m, s=s, L=L, seed=1)
    trees = sample_trees(num_data=64, rules=rules, prior=None, probs=None, seed=2)

    # Build mock transformer + SAEs.
    V_full_for_level = {l: (n if l == 0 else v) for l in range(L)}
    model = _MockTransformer(L=L, s=s, V_full_for_level=V_full_for_level,
                             embedding_dim=embedding_dim)

    # Build per-layer artifacts and _LayerState.
    artifacts = {k: _fake_artifact_for_layer(k, L, s, n, v, trees, embedding_dim)
                 for k in range(L)}
    layer_states = {}
    for k, art in artifacts.items():
        sae = _IdentitySAE(embedding_dim)
        dec_norms = sae.decoder_feature_norms()
        layer_states[k] = _LayerState(art, sae, dec_norms, device='cpu')

    # cond_full: same as in run.main().
    cond_full = {}
    for k, art in artifacts.items():
        cp_groups = _build_cond_prob(art)
        target_level = L - 1 - k
        for (lvl, j), group in cp_groups.items():
            if lvl != target_level:
                continue
            V_full = _num_values_for_level(lvl, {'v': v, 'n': n})
            cond_full[(lvl, j)] = _full_cond_prob_at(group, V_full)

    # Pre-load payloads on the mock blocks.
    payloads = _payloads_from_trees(trees, L, s)
    model.set_payload(payloads)

    rhm = {'v': v, 'n': n, 'm': m, 's': s, 'L': L}
    results = stream_reconstruction(
        model=model, layers=layer_states, cond_full=cond_full,
        trees=trees, rhm=rhm, has_cls=False,
        batch_size=32, device='cpu', classifier_head=False,
    )

    metrics_act = _metrics_for_weighting(
        results['Z_hat_act'], results['Z_true'], results['coverage'], L
    )
    metrics_uni = _metrics_for_weighting(
        results['Z_hat_uni'], results['Z_true'], results['coverage'], L
    )

    for l in range(L):
        assert metrics_act['per_level_acc'][l] == 1.0, (
            f'activation: level {l} acc={metrics_act["per_level_acc"][l]}'
        )
        assert metrics_uni['per_level_acc'][l] == 1.0, (
            f'uniform: level {l} acc={metrics_uni["per_level_acc"][l]}'
        )
        assert metrics_act['coverage'][l] == 1.0, (
            f'coverage at level {l} = {metrics_act["coverage"][l]}'
        )
    assert metrics_act['whole_tree_acc'] == 1.0
    assert metrics_uni['whole_tree_acc'] == 1.0


def main():
    import traceback
    try:
        test_oracle_recovers_full_tree()
        print('PASS  test_oracle_recovers_full_tree')
        return 0
    except AssertionError as e:
        traceback.print_exc()
        print(f'FAIL  test_oracle_recovers_full_tree: {e}')
        return 1
    except Exception as e:
        traceback.print_exc()
        print(f'ERROR test_oracle_recovers_full_tree: {type(e).__name__}: {e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
