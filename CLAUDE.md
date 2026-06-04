## Coding Rules
- Read relevant files before editing. Never edit blind.
- Test after writing. Fix errors before moving on.
- Prefer editing over rewriting. Simplest working solution.
- Run code once more before declaring done.

## Output
- No sycophantic openers or closing fluff.
- No em dashes, smart quotes, or Unicode. ASCII only.
- Be concise. If unsure, say so. Never guess.

## Override Rule
User instructions always override this file.

## Project: SAE on RHM

Study how deep transformers learn hierarchically compositional data via SAEs.

### RHM Parameters
- `n`: number of classes (root vocabulary)
- `v`: vocabulary size per node
- `m`: synonymic rules per node
- `s`: branching factor (tuple size)
- `L`: number of hierarchy levels
- Input: `s^L` leaf tokens; `trees[l]` has shape `(N, s^l)` for l=1..L, `(N,)` for l=0.

### Transformer-to-RHM Level Mapping
Layer `k` is expected to resolve RHM level `L-1-k` (bottom-up composition).

### Token Position Offset (non-obvious)
- `transformer_class`: CLS at position 0; real token `i` is at sequence position `i+1`.
- `transformer_meanclass`: real token `i` is at sequence position `i`.
- `transformer_meanclass_nores`: real token `i` is at sequence position `i`.
- `transformer_freeclass`: real token `i` is at sequence position `i` (no CLS).
- `transformer_freeclass_nores`: real token `i` is at sequence position `i` (no CLS).

### Pooling Head (meanclass vs freeclass)
- `transformer_meanclass[_nores]`: readout is a uniform mean over sequence positions after `ln_f`.
- `transformer_freeclass[_nores]`: readout is a learned softmax-weighted sum over positions, `pooled = sum_p softmax(pool_logits)[p] * ln_f(x)[p]`. The `pool_logits` parameter (length `s^L`) is zero-initialized (uniform at init = meanclass) and learned during training. Models expose `pool_weights()`; meanclass models do not. SAE/circuit-tracing code detects freeclass via `getattr(model, 'pool_weights', None)` and falls back to uniform mean otherwise. For `sae_activation_source=mean_pooled`, the pooling uses these learned weights (vs uniform mean for meanclass).

### SAE Setup
- Hooks into `model.blocks[layer_id]` post-block residual stream.
- Token modes: `one_token`, `all_tokens`, `cls_token`.
- `act_scale = sqrt(d_model) / E[||x||]`, stored in checkpoint at `sae_training_setup.act_scale`.

### Checkpoint Keys
- Transformer: `config`, `output.model`, `output.rules`, `output.best`, `output.dynamics`
- SAE: `config`, `sae_layers`, `sae_state[layer_id]`, `sae_metrics[layer_id]`, `sae_training_setup.{sae_activation_source, sae_token_idx, act_scale}`

### Linear Probe Target Mapping
For probe at layer `k`, token `p` (0-based real index):
- Target level: `L-1-k`
- Ancestor index: `p // s^(1+k)`
- Label vocab: `v` (levels 1..L-1) or `n` (level 0)

### Entry Points
- `main.py`: train transformer
- `train_sae.py`: train SAE on frozen transformer
- `sae_sweep/generate_sweep.py` / `eval_sweep.py`: sweep generation and evaluation
