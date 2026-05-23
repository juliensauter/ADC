"""KID with DINOv2 ViT-L/14 features.

Implements an `nn.Module` wrapping `facebook/dinov2-large` that exposes
CLS-token features (1024-d) for use with
`torchmetrics.image.kid.KernelInceptionDistance(feature=<wrapper>)`.

Implementation lands in commit A.2 (Phase A.2 of metrics_implementation_plan.md).
"""

# TODO(A.2): implement DINOv2 feature wrapper + KID setup helper.
