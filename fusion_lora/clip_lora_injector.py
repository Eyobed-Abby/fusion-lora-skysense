# fusion_lora/clip_lora_injector.py

import math
from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    True LoRA wrapper for an existing nn.Linear layer.

    - Clones pretrained weight & bias from `base_linear` and freezes them.
    - Adds low-rank A, B matrices as a trainable update.
    - Output:  base(x) + (alpha/r) * (x @ A^T @ B^T)
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Frozen pretrained weight & bias
        self.weight = nn.Parameter(
            base_linear.weight.data.clone(), requires_grad=False
        )
        if base_linear.bias is not None:
            self.bias = nn.Parameter(
                base_linear.bias.data.clone(), requires_grad=False
            )
        else:
            self.bias = None

        # LoRA factors (trainable)
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))

        # Optional input dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Init LoRA weights (small)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base linear: [*, in] -> [*, out]
        base_out = F.linear(x, self.weight, self.bias)

        # LoRA update: [*, in] @ A^T @ B^T
        lora_out = self.dropout(x) @ self.A.T
        lora_out = lora_out @ self.B.T

        return base_out + self.scaling * lora_out


def _apply_lora_to_linears(module: nn.Module, r: int, alpha: float) -> None:
    """
    Recursively walk a module and replace all nn.Linear with LoRALinear.
    Used on selected Swin stages so we don't have to know mmcv.FFN internals.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            _apply_lora_to_linears(child, r=r, alpha=alpha)


def add_lora_to_skysense_visual(
    clip_model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_stages: Optional[Iterable[int]] = None,
) -> nn.Module:
    """
    Inject LoRA into the *visual* Swin backbone inside SkySenseCLIP.

    Assumes:
      - `clip_model` is an instance of SkySenseCLIP
      - `clip_model.visual` is SwinTransformerV2 with:
            visual.stages: ModuleList of SwinBlockV2Sequence
            each stage.blocks: ModuleList of SwinBlockV2
            each SwinBlockV2.ffn: mmcv.FFN composed of nn.Linear layers

    Strategy:
      - Pick some Swin stages (by index) and replace *all* nn.Linear inside
        those stages with LoRALinear.

    Args:
        clip_model: SkySenseCLIP instance.
        r: LoRA rank.
        alpha: LoRA scaling factor.
        target_stages: indices of Swin stages to adapt (e.g., (2, 3)).
                       If None, use only the last stage (e.g., (3,)).

    Returns:
        The same clip_model with modified visual encoder (in-place).
    """
    if not hasattr(clip_model, "visual"):
        raise ValueError("clip_model has no attribute `visual` (expected SkySenseCLIP).")

    visual = clip_model.visual  # SwinTransformerV2

    # Sanity check for type by class name to avoid import cycles
    if visual.__class__.__name__ != "SwinTransformerV2":
        print(
            f"[clip_lora_injector] clip_model.visual is not SwinTransformerV2 "
            f"({visual.__class__.__name__}); skipping LoRA injection."
        )
        return clip_model

    num_swin_stages = len(visual.stages)

    if target_stages is None:
        # Default: only last Swin stage (highest-level features)
        target_stages = (num_swin_stages - 1,)

    # Normalize to list
    target_stages = list(target_stages)

    print(
        f"[clip_lora_injector] Injecting LoRA (r={r}, alpha={alpha}) into "
        f"Swin stages {target_stages} (0-based) out of {num_swin_stages} total."
    )

    for stage_idx, stage in enumerate(visual.stages):
        if stage_idx in target_stages:
            # Recursively replace all nn.Linear in this stage (blocks + FFN + norms if any)
            _apply_lora_to_linears(stage, r=r, alpha=alpha)

    return clip_model


def freeze_clip_except_lora(clip_model: nn.Module) -> None:
    """
    Freeze ALL parameters in SkySenseCLIP, then unfreeze only LoRALinear A/B.
    Call this AFTER `add_lora_to_skysense_visual`.
    """
    # Freeze everything first
    for p in clip_model.parameters():
        p.requires_grad = False

    # Unfreeze LoRA parameters
    for module in clip_model.modules():
        if isinstance(module, LoRALinear):
            # A and B are the trainable LoRA factors
            module.A.requires_grad = True
            module.B.requires_grad = True
            # The frozen base weights remain requires_grad=False
