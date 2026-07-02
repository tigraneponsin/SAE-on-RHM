"""Resolve circuit-trace inputs from per-layer SAE sweep folders.

Given a parent directory that contains one sweep subfolder per transformer
layer (e.g. sweep_alltokens_layer0_lambda1_zoom, ..._layer1_..., ...), and one
target lambda1 per layer, this picks the right SAE checkpoint per layer (nearest
lambda1, with a warning if not exact), locates its matching .sae_eval.pt inside
that folder's analysis_files/, and auto-detects the shared transformer
train_output. The ordered checkpoint/eval lists are what circuit_trace_sweep
expects (bottom-to-top, layer k at position k).

Usage:
    python scripts/sae_sweep/resolve_circuit_inputs.py \\
        --parent_dir /work/.../v_16_L_3_m_16_.../ \\
        --lambda1 0:0.01 1:0.02 2:0.005 \\
        --emit text

--emit text  : human-readable summary (default).
--emit shell : KEY=VALUE lines a bash script can `eval` to populate
               TRAIN_OUTPUT, SAE_CKPTS (space-joined), EVAL_ARTS (space-joined).
--emit json  : a JSON object with train_output / sae_ckpts / sae_eval_artifacts.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch


# Relative tolerance for declaring a lambda1 an "exact" match (else we warn).
LAMBDA_REL_TOL = 1e-6


def _err(msg: str) -> None:
    print(f'ERROR: {msg}', file=sys.stderr)
    sys.exit(1)


def _load_ckpt(path: str):
    return torch.load(path, map_location='cpu', weights_only=False)


def _ckpt_layer(blob) -> int | None:
    """Layer this checkpoint trains, from sae_layers (single-element list)."""
    layers = blob.get('sae_layers')
    if isinstance(layers, (list, tuple)) and len(layers) == 1:
        return int(layers[0])
    return None


def _ckpt_lambda(blob) -> float | None:
    cfg = blob.get('config')
    lam = getattr(cfg, 'sae_lambda_l1', None)
    return float(lam) if lam is not None else None


def _ckpt_train_output(blob) -> str | None:
    src = blob.get('source')
    if isinstance(src, dict):
        to = src.get('train_output')
        if to:
            return str(to)
    return None


def _parse_lambda_pairs(pairs: list[str]) -> dict[int, float]:
    """Parse ['0:0.01', '1:0.02'] -> {0: 0.01, 1: 0.02}."""
    out: dict[int, float] = {}
    for p in pairs:
        if ':' not in p:
            _err(f'--lambda1 entries must be LAYER:VALUE, got {p!r}')
        k_str, v_str = p.split(':', 1)
        try:
            k = int(k_str)
            v = float(v_str)
        except ValueError:
            _err(f'--lambda1 entry {p!r} is not LAYER:VALUE with numeric parts')
        if k in out:
            _err(f'--lambda1 specifies layer {k} more than once')
        out[k] = v
    return out


def _ckpt_dir(sweep_folder: Path) -> Path:
    """Where the SAE .pt checkpoints live for a sweep folder.

    New layout puts them in <sweep_folder>/sae_checkpoints/; old (flat) layout
    keeps them directly in <sweep_folder>. Prefer the subfolder when it exists
    and holds at least one .pt.
    """
    sub = sweep_folder / 'sae_checkpoints'
    if sub.is_dir() and any(
        not c.endswith('.sae_eval.pt') for c in glob.glob(str(sub / '*.pt'))
    ):
        return sub
    return sweep_folder


def _list_ckpts(folder: Path) -> list[str]:
    """Raw SAE checkpoints in *folder*'s checkpoint dir (excludes .sae_eval.pt)."""
    ckpts = sorted(glob.glob(str(_ckpt_dir(folder) / '*.pt')))
    return [c for c in ckpts if not c.endswith('.sae_eval.pt')]


def _map_layer_to_folder(parent_dir: Path) -> dict[int, Path]:
    """For each immediate subdir with SAE checkpoints, read one checkpoint to
    learn the layer it covers. Build {layer: folder} (folder = the sweep dir,
    not its sae_checkpoints/ subdir)."""
    layer_to_folder: dict[int, Path] = {}
    for sub in sorted(p for p in parent_dir.iterdir() if p.is_dir()):
        ckpts = _list_ckpts(sub)
        if not ckpts:
            continue
        blob = _load_ckpt(ckpts[0])
        layer = _ckpt_layer(blob)
        if layer is None:
            print(f'  note: {sub.name} has checkpoints but no single-layer '
                  f'sae_layers; skipping', file=sys.stderr)
            continue
        if layer in layer_to_folder:
            _err(f'two folders claim layer {layer}: '
                 f'{layer_to_folder[layer].name} and {sub.name}')
        layer_to_folder[layer] = sub
    return layer_to_folder


def _select_ckpt_for_layer(folder: Path, layer: int,
                           target_lambda: float) -> tuple[str, float]:
    """Pick the checkpoint in *folder* whose stored lambda1 is nearest to
    *target_lambda*. Returns (ckpt_path, actual_lambda)."""
    ckpts = _list_ckpts(folder)
    best = None  # (abs_diff, actual_lambda, path)
    available = []
    for c in ckpts:
        blob = _load_ckpt(c)
        if _ckpt_layer(blob) != layer:
            continue
        lam = _ckpt_lambda(blob)
        if lam is None:
            continue
        available.append(lam)
        diff = abs(lam - target_lambda)
        if best is None or diff < best[0]:
            best = (diff, lam, c)
    if best is None:
        _err(f'no usable checkpoint for layer {layer} in {folder}')
    diff, actual, path = best
    rel = diff / max(abs(target_lambda), 1e-12)
    if rel > LAMBDA_REL_TOL:
        avail_str = ', '.join(f'{x:g}' for x in sorted(available))
        print(f'  WARNING layer {layer}: requested lambda1={target_lambda:g} '
              f'not found; using nearest={actual:g} '
              f'(diff={diff:g}). Available: [{avail_str}]', file=sys.stderr)
    return path, actual


def _eval_artifact_for_ckpt(ckpt_path: str) -> str:
    """The matching .sae_eval.pt lives in <sweep>/analysis_files/<stem>.sae_eval.pt.

    The sweep root is the checkpoint's parent, unless the checkpoint sits in a
    sae_checkpoints/ subfolder, in which case it is one level up.
    """
    ckpt = Path(ckpt_path)
    stem = ckpt.name[:-len('.pt')] if ckpt.name.endswith('.pt') else ckpt.stem
    sweep_root = ckpt.parent.parent if ckpt.parent.name == 'sae_checkpoints' else ckpt.parent
    art = sweep_root / 'analysis_files' / f'{stem}.sae_eval.pt'
    if not art.is_file():
        _err(f'eval artifact not found for {ckpt.name}:\n  expected {art}\n'
             f'  (run the analysis step first, e.g. '
             f'slurm/sae/run_analysis_and_plots.sh on this sweep folder)')
    return str(art)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--parent_dir', required=True,
                   help='Directory containing one per-layer SAE sweep subfolder.')
    p.add_argument('--lambda1', nargs='+', required=True,
                   help='One LAYER:VALUE pair per layer, e.g. 0:0.01 1:0.02 2:0.005')
    p.add_argument('--emit', choices=['text', 'shell', 'json'], default='text',
                   help='Output format (default text).')
    args = p.parse_args()

    parent_dir = Path(args.parent_dir)
    if not parent_dir.is_dir():
        _err(f'--parent_dir is not a directory: {parent_dir}')

    target = _parse_lambda_pairs(args.lambda1)
    requested_layers = sorted(target.keys())

    # Layers must be contiguous from 0 (circuit_trace expects position k == layer k).
    expected = list(range(len(requested_layers)))
    if requested_layers != expected:
        _err(f'--lambda1 layers must be contiguous from 0 (got {requested_layers}, '
             f'expected {expected})')

    print(f'Scanning {parent_dir} for per-layer sweep folders...', file=sys.stderr)
    layer_to_folder = _map_layer_to_folder(parent_dir)
    if not layer_to_folder:
        _err(f'no per-layer SAE sweep folders found under {parent_dir}')

    missing = [k for k in requested_layers if k not in layer_to_folder]
    if missing:
        found = ', '.join(f'{k}->{layer_to_folder[k].name}'
                          for k in sorted(layer_to_folder))
        _err(f'requested layer(s) {missing} have no folder under {parent_dir}. '
             f'Found: {found}')

    sae_ckpts: list[str] = []
    eval_arts: list[str] = []
    train_outputs: list[str] = []
    for layer in requested_layers:
        folder = layer_to_folder[layer]
        ckpt, actual = _select_ckpt_for_layer(folder, layer, target[layer])
        art = _eval_artifact_for_ckpt(ckpt)
        blob = _load_ckpt(ckpt)
        to = _ckpt_train_output(blob)
        if to is None:
            _err(f'checkpoint for layer {layer} has no source.train_output: {ckpt}')
        print(f'  layer {layer}: lambda1={actual:g}  {Path(ckpt).name}',
              file=sys.stderr)
        sae_ckpts.append(ckpt)
        eval_arts.append(art)
        train_outputs.append(to)

    unique_to = sorted(set(train_outputs))
    if len(unique_to) != 1:
        detail = '\n'.join(f'  layer {k}: {t}'
                           for k, t in zip(requested_layers, train_outputs))
        _err(f'selected checkpoints disagree on train_output:\n{detail}')
    train_output = unique_to[0]
    if not Path(train_output).is_file():
        _err(f'detected train_output does not exist: {train_output}')

    if args.emit == 'json':
        print(json.dumps({
            'train_output': train_output,
            'sae_ckpts': sae_ckpts,
            'sae_eval_artifacts': eval_arts,
        }, indent=2))
    elif args.emit == 'shell':
        print(f'TRAIN_OUTPUT={train_output}')
        print(f'SAE_CKPTS={" ".join(sae_ckpts)}')
        print(f'EVAL_ARTS={" ".join(eval_arts)}')
    else:  # text
        print('Resolved circuit-trace inputs:')
        print(f'  train_output: {train_output}')
        print(f'  sae_ckpts ({len(sae_ckpts)}, layer 0 first):')
        for c in sae_ckpts:
            print(f'    {c}')
        print(f'  sae_eval_artifacts ({len(eval_arts)}, layer 0 first):')
        for a in eval_arts:
            print(f'    {a}')


if __name__ == '__main__':
    main()
