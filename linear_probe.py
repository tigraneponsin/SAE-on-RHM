"""Linear probes for intermediate RHM latent recovery.

Train a linear classifier on transformer residual stream activations to predict
the ancestor latent value at a specific hierarchy level.  The default target
level follows the hypothesis that transformer layer k resolves hierarchy level
L-1-k (where L is the number of RHM levels).

Typical usage:

    # 1. Collect activations + ground truth labels
    train_data = collect_probe_data(model, inputs, trees, ...)
    eval_data  = collect_probe_data(model, eval_inputs, eval_trees, ...)

    # 2. Train probe
    probe, result = train_probe(train_data, eval_data=eval_data)

    # 3. Evaluate on SAE-reconstructed activations
    recon_data = collect_probe_data(model, eval_inputs, eval_trees, ..., sae=sae)
    recon_result = eval_probe(probe, recon_data)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProbeData:
    """Pre-collected activations and ground truth labels for linear probing."""
    activations: torch.Tensor   # [N, embedding_dim]
    labels: torch.Tensor        # [N] integer class labels
    num_classes: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ProbeResult:
    """Results from training or evaluating a linear probe."""
    accuracy: float
    per_class_accuracy: torch.Tensor  # [num_classes]
    loss: float
    num_samples: int
    num_classes: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Target level helpers
# ---------------------------------------------------------------------------

def probe_target_level(transformer_layer: int, num_rhm_levels: int) -> int:
    """Return the RHM hierarchy level to probe for a given transformer layer.

    Follows the hypothesis: transformer layer k resolves hierarchy level L-1-k.

    Args:
        transformer_layer: index of the transformer block (0-based).
        num_rhm_levels: L, the number of RHM hierarchy levels.

    Returns:
        Target hierarchy level (0 = class label, L-1 = one above leaves).
    """
    level = num_rhm_levels - 1 - transformer_layer
    if level < 0:
        raise ValueError(
            f"transformer_layer={transformer_layer} exceeds hierarchy depth: "
            f"L-1-k = {num_rhm_levels}-1-{transformer_layer} = {level} < 0"
        )
    return level


def normalized_identification_error(accuracy: float, chance_accuracy: float) -> float:
    """Compute probe identification error normalized by chance level.

    The metric is defined as:

        (1 - accuracy) / (1 - chance_accuracy)

    It has the following anchors:
      - 0.0 at perfect accuracy (accuracy = 1)
      - 1.0 at chance performance (accuracy = chance_accuracy)

    Values greater than 1.0 indicate worse-than-chance performance.

    Args:
        accuracy: Probe accuracy in [0, 1] (or close due to numeric error).
        chance_accuracy: Chance-level accuracy for the task.

    Returns:
        Normalized identification error.
    """
    denom = 1.0 - float(chance_accuracy)
    if denom <= 0.0:
        return 0.0 if float(accuracy) >= 1.0 else float('inf')
    return (1.0 - float(accuracy)) / denom


def ancestor_labels(
    trees: Dict[int, torch.Tensor],
    sample_ids: torch.Tensor,
    leaf_indices: torch.Tensor,
    level: int,
    tuple_size: int,
    num_rhm_levels: int,
) -> torch.Tensor:
    """Compute ancestor latent values for given samples and leaf positions.

    Args:
        trees: RHM tree dict. trees[l] has shape (num_samples, s^l) for l>0,
               and (num_samples,) for l=0.
        sample_ids: [N] indices into trees (which sample each activation comes from).
        leaf_indices: [N] 0-based leaf token indices (position among s^L leaves).
        level: hierarchy level to retrieve (0 = class, L = leaf).
        tuple_size: s parameter of the RHM.
        num_rhm_levels: L parameter of the RHM.

    Returns:
        [N] tensor of integer labels (ancestor values at the requested level).
    """
    sid = sample_ids.long()
    if level == 0:
        return trees[0][sid]
    downscale = tuple_size ** (num_rhm_levels - level)
    ancestor_pos = leaf_indices.long() // int(downscale)
    return trees[level][sid, ancestor_pos]


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_probe_data(
    model: nn.Module,
    inputs: torch.Tensor,
    trees: Dict[int, torch.Tensor],
    layer_id: int,
    token_idx: int,
    model_name: str,
    hierarchy_level: int,
    tuple_size: int,
    num_rhm_levels: int,
    device: torch.device,
    act_scale: float = 1.0,
    sae: Optional[nn.Module] = None,
    batch_size: int = 256,
) -> ProbeData:
    """Collect residual stream activations and matching hierarchy labels.

    Hooks into ``model.blocks[layer_id]`` to capture the post-block residual
    stream at the specified token position, then pairs each activation vector
    with the ground truth ancestor label from the RHM trees.

    Args:
        model: Frozen transformer (already on *device*, in eval mode).
        inputs: [num_samples, seq_len] tensor of token indices (already
                transformed via ``init.transform_inputs``).
        trees: Full RHM tree dict for computing ancestor labels.
        layer_id: Which transformer block to hook (0-based).
        token_idx: 0-based real token index (among the s^L input tokens).
        model_name: ``'transformer_class'``, ``'transformer_meanclass'``,
                or ``'transformer_meanclass_nores'``.
        hierarchy_level: Which RHM level to use as probe target.
        tuple_size: s parameter of the RHM.
        num_rhm_levels: L parameter of the RHM.
        device: Torch device for the forward pass.
        act_scale: Activation scaling factor (from SAE training setup).
        sae: If provided, activations are passed through the SAE and the
             reconstruction is returned instead of raw activations.
        batch_size: Number of samples per forward-pass batch.

    Returns:
        ProbeData with activations and labels aligned row-wise.
    """
    model.eval()
    if sae is not None:
        sae.eval()

    # Determine sequence position of the target real token.
    offset = 1 if model_name == 'transformer_class' else 0
    seq_pos = offset + token_idx

    # The 0-based leaf index is just token_idx (same for all samples in
    # one_token mode).
    leaf_idx_scalar = token_idx

    buf = []
    hook = model.blocks[layer_id].register_forward_hook(
        lambda _m, _i, o: buf.append(o.detach())
    )

    act_chunks = []
    label_chunks = []
    num_samples = inputs.size(0)

    try:
        for b_start in range(0, num_samples, batch_size):
            b_end = min(b_start + batch_size, num_samples)
            batch = inputs[b_start:b_end].to(device)
            _ = model(batch)

            if not buf:
                raise RuntimeError(f"No activations captured for layer {layer_id}")
            act = buf.pop(0)
            # Select the single token position.
            act = act[:, seq_pos, :]  # [B, embedding_dim]

            if sae is not None:
                scaled = act * act_scale
                recon, _z = sae(scaled)
                act = recon / act_scale

            act_chunks.append(act.cpu())

            sample_ids = torch.arange(b_start, b_end, dtype=torch.long)
            leaf_indices = torch.full_like(sample_ids, leaf_idx_scalar)
            labels = ancestor_labels(
                trees, sample_ids, leaf_indices,
                hierarchy_level, tuple_size, num_rhm_levels,
            )
            label_chunks.append(labels)
    finally:
        hook.remove()
        buf.clear()

    all_act = torch.cat(act_chunks, dim=0)
    all_labels = torch.cat(label_chunks, dim=0).long()

    # Determine number of classes at this hierarchy level.
    if hierarchy_level == 0:
        num_classes = int(trees[0].max().item()) + 1
    else:
        num_classes = int(trees[hierarchy_level].max().item()) + 1

    return ProbeData(
        activations=all_act,
        labels=all_labels,
        num_classes=num_classes,
        metadata={
            'layer_id': layer_id,
            'token_idx': token_idx,
            'hierarchy_level': hierarchy_level,
            'num_samples': num_samples,
            'sae_applied': sae is not None,
        },
    )


# ---------------------------------------------------------------------------
# Linear probe model
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """Single linear layer classifier."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_probe(
    probe_data: ProbeData,
    lr: float = 1e-3,
    num_steps: int = 2000,
    batch_size: int = 128,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    eval_data: Optional[ProbeData] = None,
    verbose: bool = True,
) -> Tuple[LinearProbe, ProbeResult]:
    """Train a linear probe on pre-collected activations.

    Args:
        probe_data: Training activations and labels.
        lr: Learning rate for Adam.
        num_steps: Total number of gradient steps.
        batch_size: Minibatch size.
        weight_decay: L2 regularization.
        device: Torch device (defaults to CPU).
        eval_data: If provided, the returned ProbeResult is computed on this
                   split; otherwise on the training data.
        verbose: Print progress every 500 steps.

    Returns:
        (trained_probe, eval_result)
    """
    if device is None:
        device = torch.device('cpu')

    input_dim = probe_data.activations.size(1)
    probe = LinearProbe(input_dim, probe_data.num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(probe_data.activations, probe_data.labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    probe.train()
    step = 0
    while step < num_steps:
        for act_batch, label_batch in loader:
            act_batch = act_batch.to(device)
            label_batch = label_batch.to(device)

            logits = probe(act_batch)
            loss = F.cross_entropy(logits, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if verbose and step % 500 == 0:
                probe.eval()
                train_result = eval_probe(probe, probe_data, device)
                msg = (f"  step {step:>{len(str(num_steps))}}/{num_steps}"
                       f"  train_loss={train_result.loss:.4f}  train_acc={train_result.accuracy:.4f}")
                if eval_data is not None:
                    val_result = eval_probe(probe, eval_data, device)
                    msg += f"  val_loss={val_result.loss:.4f}  val_acc={val_result.accuracy:.4f}"
                print(msg)
                probe.train()
            if step >= num_steps:
                break

    target_data = eval_data if eval_data is not None else probe_data
    result = eval_probe(probe, target_data, device)
    return probe, result


@torch.no_grad()
def eval_probe(
    probe: LinearProbe,
    probe_data: ProbeData,
    device: Optional[torch.device] = None,
    batch_size: int = 1024,
) -> ProbeResult:
    """Evaluate a trained linear probe on collected data.

    Returns:
        ProbeResult with overall accuracy, per-class accuracy, and loss.
    """
    if device is None:
        device = next(probe.parameters()).device

    probe.eval()
    dataset = TensorDataset(probe_data.activations, probe_data.labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_correct = 0
    total_loss = 0.0
    total_samples = 0
    per_class_correct = torch.zeros(probe_data.num_classes, dtype=torch.long)
    per_class_total = torch.zeros(probe_data.num_classes, dtype=torch.long)

    for act_batch, label_batch in loader:
        act_batch = act_batch.to(device)
        label_batch = label_batch.to(device)

        logits = probe(act_batch)
        loss = F.cross_entropy(logits, label_batch, reduction='sum')
        preds = logits.argmax(dim=-1)

        total_loss += float(loss)
        total_correct += int((preds == label_batch).sum())
        total_samples += label_batch.size(0)

        for c in range(probe_data.num_classes):
            mask = label_batch == c
            per_class_total[c] += int(mask.sum())
            per_class_correct[c] += int((preds[mask] == c).sum())

    accuracy = total_correct / max(total_samples, 1)
    avg_loss = total_loss / max(total_samples, 1)
    per_class_acc = per_class_correct.float() / per_class_total.float().clamp(min=1)

    return ProbeResult(
        accuracy=accuracy,
        per_class_accuracy=per_class_acc,
        loss=avg_loss,
        num_samples=total_samples,
        num_classes=probe_data.num_classes,
        metadata=probe_data.metadata,
    )
