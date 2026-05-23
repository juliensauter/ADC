"""Boundary IoU @ 5px (Tier 2, dormant until segmenter available).

Edge-based IoU: `edge_mask = mask XOR binary_erosion(mask, iters=5)`, then
standard IoU on the resulting edge masks. Used only when a held-out segmenter
exists to produce predicted masks from generated images (Phase C).
Implementation lands in commit A.5.
"""

# TODO(A.5): implement Boundary IoU @ 5px helper.
