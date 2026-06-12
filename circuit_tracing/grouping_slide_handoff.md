# Node Grouping Procedure (handoff for slide creation)

This document explains the node-grouping step in the circuit-tracing pipeline.
The goal: produce one slide that explains *what* grouping does and *how*, plus a
small worked schema/example. Everything below is ASCII-only.

---

## 1. Where grouping sits in the pipeline

The circuit tracer builds a layered DAG of SAE feature nodes:

  embed -> feat(layer 0) -> feat(layer 1) -> ... -> feat(layer K-1) -> logits

Node kinds:
  - embed : input embedding node, one per token position p
  - feat  : an SAE feature that fired, keyed ('feat', layer k, position p, feature index i)
  - err   : reconstruction-error node (the part of the residual the SAE missed)
  - logit : output class node

Pipeline order:
  1. Build full edge set (every feature-to-feature weight).
  2. Indirect-influence PRUNING: keep only edges that matter for the prediction.
  3. GROUPING (this step): collapse feature nodes that play the same role.
  4. Visualize.

Grouping runs AFTER pruning and operates on the pruned edge set only.

Why group at all: after pruning there are still many feature nodes that are
functionally redundant. The RHM data is built from synonym rules, so the
transformer often learns several SAE features that detect "the same thing"
(same parents, same role) at different positions or as duplicate detectors.
Grouping merges those into a single super-node so the final graph is readable.

---

## 2. The core idea: group by INCOMING SIGNATURE

Two feature nodes are merged if and only if they have the *exact same signature*.

Signature of a layer-k feature node =

    frozenset{ (grouped_source_node, sign_of_weight) for each pruned incoming
               edge whose source is a FEATURE node (or EMBED at layer 0) }

Key points:
  - It is a SET, so edge order does not matter.
  - Only the SIGN of each weight is kept (+1 or -1), not the magnitude.
  - ERROR sources are EXCLUDED from the signature.
  - The source used is the ALREADY-GROUPED parent of the source node, not the
    raw node. This is what makes the procedure recursive / bottom-up.

Plain-English version: "Two features belong together if they listen to the same
upstream super-nodes, with the same excitatory/inhibitory sign on each."

---

## 3. Why bottom-up matters

Layers are processed in order k = 0, 1, ..., K-1.

When we compute the signature for a layer-k feature, its incoming edges come
from layer k-1. By the time we reach layer k, layer k-1 has ALREADY been
grouped, so each incoming source is replaced by its super-node.

Consequence: a merge low in the graph propagates upward. If two layer-0
features merge into group G, then any layer-1 feature that pointed to either of
them now points to the SAME source G, which makes those layer-1 features more
likely to share a signature and merge too. The simplification cascades up.

(Processing top-down would not work: you cannot reference a parent group that
has not been formed yet.)

---

## 4. The empty-signature rule (important edge case)

A feature has an EMPTY signature if its only surviving pruned incoming edges
came from error sources, or it has no incoming pruned edges at all.

Rule: empty-signature features do NOT merge. Each one stays as its own
singleton group.

Rationale: an empty signature carries no information about "what this feature
listens to," so merging two such features would be unjustified. In the code
this is done by giving each empty-sig feature a unique bucket key derived from
its own node key, so it can never collide with another.

---

## 5. Cross-position merges

Features at DIFFERENT token positions can merge. RHM structure means a feature
detecting "synonym group X" at position 2 and the same detector at position 5
are genuinely the same role. To make their value distributions comparable, the
node table stores `value_distribution_full`, already lifted to the full
parent-level vocabulary (size v at levels 1..L-1, size n at level 0), so
position-specific value subsets are not a problem.

---

## 6. What a group super-node carries

Each merged group spanning N >= 1 constituent features stores:

  - kind = 'group', layer = k, level = L-1-k
  - constituents = list of (position, feature_index)
  - positions, position_centroid
  - z_sum = sum of constituent activation (z) values
  - value_distribution = z-WEIGHTED average of the constituents'
    value_distribution_full, renormalized to sum to 1
  - label_value = argmax of that distribution (the value this group encodes)
  - p_value_given_fire = probability mass on label_value
  - normalized_entropy = H(distribution) / log(V_level), clipped to [0,1]
    (drives the white->green color gradient in the viz; low entropy = green =
    a confident, single-value detector)
  - signature (kept for hover text)

Dead constituents (no eval evidence, value_distribution_full = None) are skipped
from the weighted average. If every constituent is dead, the group's
distribution is None.

---

## 7. Edge remapping after grouping

Once every node has a parent group, edges are rebuilt:
  - Replace each edge endpoint (src, dst) with its parent group key.
    (embed / err / logit keys map to themselves.)
  - Drop any edge that becomes a self-loop.
  - Merge duplicate (src, dst) edges by SUMMING their signed weights.

Result: a smaller DAG of group super-nodes with summed signed edges.

---

## 8. Algorithm summary (for a "how it works" box on the slide)

  for k in 0 .. K-1:                       # bottom-up
      for each feature f at layer k:
          sig = { (parent_of(src), sign(w))
                  for (src, w) in pruned incoming feature/embed edges of f }
          if sig is empty: put f in its own singleton bucket
          else:            put f in the bucket keyed by sig
      for each bucket -> create one group super-node
          aggregate z, value_distribution, entropy, ...
          record parent_of(member) = this group   # used by next layer up
  remap all pruned edges through parent_of, sum duplicates

---

## 9. Suggested worked example for the schema/diagram

Claude should draw a tiny before/after DAG. Concrete numbers to use:

Setup: 3 layers shown, weights labeled with sign only.

BEFORE (pruned, ungrouped) -- layer 0 has 4 features, layer 1 has 2:

  embed_A ---+--(+)--> f0_p2_i7  ---(+)--> f1_q0_i3
             |
  embed_A ---+--(+)--> f0_p5_i7  ---(+)--> f1_q0_i9
                                              ^
  embed_B ------(-)--> f0_p2_i4  ---(+)-------+   (also feeds f1_q0_i3)
                       (err) ----> f0_p9_i1       <- empty signature

Walk:
  - Layer 0:
      f0_p2_i7 signature = {(embed_A, +)}
      f0_p5_i7 signature = {(embed_A, +)}   -> SAME -> merge into G0
      f0_p2_i4 signature = {(embed_B, -)}   -> different -> own group G1
      f0_p9_i1 signature = {} (only error in) -> singleton G2 (no merge)
  - Layer 1 (sources now refer to G0, G1, ...):
      f1_q0_i3 signature = {(G0, +), (G1, +)}
      f1_q0_i9 signature = {(G0, +)}        -> different from i3 -> stays separate
        (note: both i3 and i9 originally pointed at distinct layer-0 features
         that collapsed into G0, illustrating the bottom-up cascade)

AFTER (grouped):

  embed_A --(+)--> [G0: f0 x2 detectors] --(+)--> f1_q0_i3
  embed_B --(-)--> [G1] -----------------(+)----^
                                          (+)--> f1_q0_i9
  [G2: empty-sig singleton]   (isolated)

Talking point for the slide: two layer-0 detectors that both read embed_A with a
positive sign are the same role, so they collapse to G0; the error-only feature
G2 refuses to merge; and because layer 0 collapsed first, layer 1 signatures are
computed against the new super-nodes.

---

## 10. One-line takeaway (slide title candidate)

"Grouping collapses feature nodes that share the same signed set of upstream
super-nodes, bottom-up, so functionally identical detectors become one node."
