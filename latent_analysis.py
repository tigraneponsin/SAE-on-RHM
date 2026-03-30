from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def _validate_layer_ids(layer_ids: Sequence[int]) -> List[int]:
    unique = sorted(set(int(layer_id) for layer_id in layer_ids))
    if not unique:
        raise ValueError("layer_ids must contain at least one transformer layer index")
    return unique


def _token_positions_for_activations(model_name: str, seq_len: int, activation_source: str) -> torch.Tensor:
    if activation_source not in {"all_tokens", "cls_token"}:
        raise ValueError(
            f"activation_source={activation_source} is invalid. Use one of: all_tokens, cls_token"
        )

    if model_name == "transformer_class":
        if activation_source == "cls_token":
            return torch.zeros(1, dtype=torch.long)
        return torch.arange(1, seq_len, dtype=torch.long)

    if model_name == "transformer_meanclass":
        if activation_source == "cls_token":
            raise ValueError("transformer_meanclass has no [CLS] token. Use activation_source=all_tokens.")
        return torch.arange(0, seq_len, dtype=torch.long)

    raise ValueError(
        f"model_name={model_name} is invalid for SAE latent analysis. "
        "Expected transformer_class or transformer_meanclass."
    )


@dataclass
class ActivationIndex:
    sample_id: torch.Tensor
    token_pos: torch.Tensor
    layer_id: torch.Tensor

    def __post_init__(self) -> None:
        n_rows = self.sample_id.numel()
        if self.token_pos.numel() != n_rows or self.layer_id.numel() != n_rows:
            raise ValueError("ActivationIndex tensors must all have the same number of rows")

    def num_rows(self) -> int:
        return int(self.sample_id.numel())


@dataclass
class SAEActivationBatch:
    weighted_activations: torch.Tensor
    raw_activations: torch.Tensor
    index: ActivationIndex

    def __post_init__(self) -> None:
        if self.weighted_activations.shape != self.raw_activations.shape:
            raise ValueError("weighted_activations and raw_activations must have identical shape")
        if self.weighted_activations.ndim != 2:
            raise ValueError("SAE activations must have shape [num_rows, latent_dim]")
        if self.index.num_rows() != self.weighted_activations.size(0):
            raise ValueError("Index row count must match activation row count")


@torch.no_grad()
def collect_weighted_sae_activations(
    model,
    sae_by_layer: Dict[int, torch.nn.Module],
    inputs: torch.Tensor,
    model_name: str,
    activation_source: str = "all_tokens",
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    start_sample_id: int = 0,
) -> SAEActivationBatch:
    """
    Collect SAE feature activations for selected transformer layers with stable row indexing.

    Returns:
        SAEActivationBatch with tensors aligned row-wise:
        row -> (sample_id, token_pos, layer_id, feature_activations)
    """
    layer_ids = _validate_layer_ids(list(sae_by_layer.keys()))
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [num_samples, seq_len] for transformer models")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    for sae in sae_by_layer.values():
        sae.eval()

    captured: Dict[int, List[torch.Tensor]] = {layer_id: [] for layer_id in layer_ids}
    handles = []
    for layer_id in layer_ids:
        handle = model.blocks[layer_id].register_forward_hook(
            lambda _m, _inp, out, layer=layer_id: captured[layer].append(out.detach())
        )
        handles.append(handle)

    weighted_chunks: List[torch.Tensor] = []
    raw_chunks: List[torch.Tensor] = []
    sample_idx_chunks: List[torch.Tensor] = []
    token_idx_chunks: List[torch.Tensor] = []
    layer_idx_chunks: List[torch.Tensor] = []

    try:
        num_samples = inputs.size(0)
        for b_start in range(0, num_samples, batch_size):
            b_end = min(b_start + batch_size, num_samples)
            batch_inputs = inputs[b_start:b_end].to(device)
            _ = model(batch_inputs)

            for layer_id in layer_ids:
                if len(captured[layer_id]) == 0:
                    raise RuntimeError(f"No activations captured for layer {layer_id}")

                act = captured[layer_id].pop(0)
                token_positions = _token_positions_for_activations(model_name, act.size(1), activation_source)
                token_positions = token_positions.to(act.device)
                selected = act[:, token_positions, :].reshape(-1, act.size(-1))

                sae = sae_by_layer[layer_id]
                _, z = sae(selected)
                dec_norms = sae.decoder_feature_norms().to(z.device)
                weighted = z * dec_norms.unsqueeze(0)

                n_local = z.size(0)
                local_batch_size = b_end - b_start
                tokens_per_sample = token_positions.numel()
                if n_local != local_batch_size * tokens_per_sample:
                    raise RuntimeError("Unexpected activation shape after token selection")

                local_sample_ids = (
                    torch.arange(b_start, b_end, device=z.device, dtype=torch.long)
                    .repeat_interleave(tokens_per_sample)
                    + int(start_sample_id)
                )
                local_token_ids = token_positions.repeat(local_batch_size).to(dtype=torch.long)
                local_layer_ids = torch.full((n_local,), int(layer_id), dtype=torch.long, device=z.device)

                weighted_chunks.append(weighted.detach().cpu())
                raw_chunks.append(z.detach().cpu())
                sample_idx_chunks.append(local_sample_ids.detach().cpu())
                token_idx_chunks.append(local_token_ids.detach().cpu())
                layer_idx_chunks.append(local_layer_ids.detach().cpu())

            for layer_id in layer_ids:
                if captured[layer_id]:
                    captured[layer_id].clear()

    finally:
        for handle in handles:
            handle.remove()

    if not weighted_chunks:
        raise RuntimeError("No SAE activations were collected")

    weighted_activations = torch.cat(weighted_chunks, dim=0)
    raw_activations = torch.cat(raw_chunks, dim=0)
    index = ActivationIndex(
        sample_id=torch.cat(sample_idx_chunks, dim=0),
        token_pos=torch.cat(token_idx_chunks, dim=0),
        layer_id=torch.cat(layer_idx_chunks, dim=0),
    )
    return SAEActivationBatch(
        weighted_activations=weighted_activations,
        raw_activations=raw_activations,
        index=index,
    )


def infer_tuple_size_from_trees(trees: Dict[int, torch.Tensor]) -> int:
    max_level = max(int(level) for level in trees.keys())
    if max_level <= 0:
        raise ValueError("trees must include at least two levels to infer tuple_size")

    width_l = trees[max_level].shape[1] if trees[max_level].ndim == 2 else 1
    width_prev = trees[max_level - 1].shape[1] if trees[max_level - 1].ndim == 2 else 1
    if width_prev <= 0 or width_l % width_prev != 0:
        raise ValueError("Could not infer tuple_size from trees")
    return int(width_l // width_prev)


def _level_pos_values(tensor: torch.Tensor, pos: int) -> torch.Tensor:
    if tensor.ndim == 1:
        if pos != 0:
            raise IndexError("level-0 tree has only position 0")
        return tensor
    if pos < 0 or pos >= tensor.size(1):
        raise IndexError(f"position={pos} out of range for level width={tensor.size(1)}")
    return tensor[:, pos]


@dataclass
class LatentRowTable:
    trees: Dict[int, torch.Tensor]
    sample_id: torch.Tensor
    token_pos: torch.Tensor
    tuple_size: int

    def __post_init__(self) -> None:
        if self.sample_id.numel() != self.token_pos.numel():
            raise ValueError("sample_id and token_pos must have identical length")

    def _safe_sample_idx(self) -> torch.Tensor:
        max_sample = self.trees[0].size(0)
        if self.sample_id.min().item() < 0 or self.sample_id.max().item() >= max_sample:
            raise IndexError("sample_id contains out-of-range indices for trees")
        return self.sample_id.long()

    def global_values(self, level: int, position: int) -> torch.Tensor:
        values = _level_pos_values(self.trees[level], int(position))
        return values[self._safe_sample_idx()]

    def global_mask(self, level: int, position: int, value: int) -> torch.Tensor:
        return self.global_values(level, position).eq(int(value))

    def ancestor_position(self, level: int) -> torch.Tensor:
        max_level = max(self.trees.keys())
        if level < 0 or level > max_level:
            raise ValueError(f"level={level} out of bounds [0, {max_level}]")
        if level == 0:
            return torch.zeros_like(self.token_pos, dtype=torch.long)
        downscale = self.tuple_size ** (max_level - level)
        return torch.div(self.token_pos.long(), int(downscale), rounding_mode="floor")

    def ancestor_values(self, level: int) -> torch.Tensor:
        sample_idx = self._safe_sample_idx()
        if level == 0:
            return self.trees[0][sample_idx]
        pos = self.ancestor_position(level)
        return self.trees[level][sample_idx, pos]

    def ancestor_mask(self, level: int, value: int) -> torch.Tensor:
        return self.ancestor_values(level).eq(int(value))


def build_latent_row_table(
    trees: Dict[int, torch.Tensor],
    index: ActivationIndex,
    tuple_size: Optional[int] = None,
) -> LatentRowTable:
    if tuple_size is None:
        tuple_size = infer_tuple_size_from_trees(trees)
    return LatentRowTable(
        trees=trees,
        sample_id=index.sample_id.long(),
        token_pos=index.token_pos.long(),
        tuple_size=int(tuple_size),
    )


@dataclass
class ObservationalMetrics:
    target_names: List[str]
    target_counts: torch.Tensor
    baseline_mean: torch.Tensor
    baseline_presence: torch.Tensor
    conditional_mean: torch.Tensor
    delta_mean: torch.Tensor
    enrichment_ratio: torch.Tensor
    z_score: torch.Tensor
    conditional_presence: torch.Tensor
    delta_presence: torch.Tensor


def compute_observational_metrics(
    activations: torch.Tensor,
    target_masks: Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> ObservationalMetrics:
    """
    Compute observational feature-target statistics for SAE analyses.

    activations: [num_rows, num_features]
    target_masks[name]: [num_rows] boolean
    """
    if activations.ndim != 2:
        raise ValueError("activations must have shape [num_rows, num_features]")
    if not target_masks:
        raise ValueError("target_masks must contain at least one target")

    num_rows, num_features = activations.shape
    x = activations.float()

    baseline_mean = x.mean(dim=0)
    baseline_presence = (x > 0).float().mean(dim=0)
    baseline_std = x.std(dim=0, unbiased=False)

    names: List[str] = []
    counts: List[int] = []
    cond_mean: List[torch.Tensor] = []
    delta_mean: List[torch.Tensor] = []
    ratio: List[torch.Tensor] = []
    z_scores: List[torch.Tensor] = []
    cond_presence: List[torch.Tensor] = []
    delta_presence: List[torch.Tensor] = []

    for name, mask in target_masks.items():
        m = mask.bool()
        if m.numel() != num_rows:
            raise ValueError(f"target mask '{name}' has length {m.numel()}, expected {num_rows}")
        count = int(m.sum().item())
        if count == 0:
            continue

        selected = x[m]
        local_mean = selected.mean(dim=0)
        local_presence = (selected > 0).float().mean(dim=0)
        d = local_mean - baseline_mean
        r = local_mean / (baseline_mean + eps)

        stderr = baseline_std / (count ** 0.5 + eps)
        z = d / (stderr + eps)

        names.append(name)
        counts.append(count)
        cond_mean.append(local_mean)
        delta_mean.append(d)
        ratio.append(r)
        z_scores.append(z)
        cond_presence.append(local_presence)
        delta_presence.append(local_presence - baseline_presence)

    if not names:
        raise ValueError("All target masks were empty")

    return ObservationalMetrics(
        target_names=names,
        target_counts=torch.tensor(counts, dtype=torch.long),
        baseline_mean=baseline_mean,
        baseline_presence=baseline_presence,
        conditional_mean=torch.stack(cond_mean, dim=0),
        delta_mean=torch.stack(delta_mean, dim=0),
        enrichment_ratio=torch.stack(ratio, dim=0),
        z_score=torch.stack(z_scores, dim=0),
        conditional_presence=torch.stack(cond_presence, dim=0),
        delta_presence=torch.stack(delta_presence, dim=0),
    )


def build_full_grid_target_masks(
    latent_table: LatentRowTable,
    value_sets_by_level: Optional[Dict[int, Iterable[int]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build target masks for the full (level, position, value) grid.

    Each target mask is aligned to activation rows.
    """
    masks: Dict[str, torch.Tensor] = {}
    max_level = max(latent_table.trees.keys())

    for level in range(max_level + 1):
        level_tensor = latent_table.trees[level]
        width = 1 if level_tensor.ndim == 1 else level_tensor.size(1)
        for position in range(width):
            values = (
                list(value_sets_by_level[level])
                if value_sets_by_level is not None and level in value_sets_by_level
                else sorted(torch.unique(_level_pos_values(level_tensor, position)).tolist())
            )
            observed = latent_table.global_values(level, position)
            for value in values:
                name = f"global:l{level}:p{position}:v{int(value)}"
                masks[name] = observed.eq(int(value))

    return masks


def build_ancestor_target_masks(
    latent_table: LatentRowTable,
    value_sets_by_level: Optional[Dict[int, Iterable[int]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build row-aligned token-path targets where each level uses the row token's ancestor node.
    """
    masks: Dict[str, torch.Tensor] = {}
    max_level = max(latent_table.trees.keys())

    for level in range(max_level + 1):
        row_values = latent_table.ancestor_values(level)
        values = (
            list(value_sets_by_level[level])
            if value_sets_by_level is not None and level in value_sets_by_level
            else sorted(torch.unique(row_values).tolist())
        )
        for value in values:
            name = f"ancestor:l{level}:v{int(value)}"
            masks[name] = row_values.eq(int(value))

    return masks
