"""Quick diagnostic: print the structure of one SAE .pt checkpoint.

Usage:
    python sae_sweep/inspect_ckpt.py /path/to/file.pt
"""
import sys
import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location='cpu')

print("=== Top-level keys ===")
print(list(ckpt.keys()))
print()

layers = ckpt.get('sae_layers', 'MISSING')
print(f"sae_layers: {layers}  (type of each: {[type(x).__name__ for x in layers] if layers != 'MISSING' else '?'})")
print()

for key in ('sae_training_curves', 'sae_eval_curves'):
    d = ckpt.get(key, 'MISSING')
    if d == 'MISSING':
        print(f"{key}: KEY NOT PRESENT IN CHECKPOINT")
    elif not d:
        print(f"{key}: present but EMPTY DICT {{}}")
    else:
        print(f"{key}:")
        for k, v in d.items():
            print(f"  key={k!r}  type(key)={type(k).__name__}", end="")
            if isinstance(v, dict):
                for sk, sv in v.items():
                    length = len(sv) if hasattr(sv, '__len__') else '?'
                    print(f"  |  '{sk}': len={length}", end="")
            print()
    print()
