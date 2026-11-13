# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT
#
# DEFM_MEAN, DEFM_STD, make_norm_transform, get_target_size, preprocess_depth_batch and replace_bn_with_gn are copied
# unchanged from DeFM (https://github.com/leggedrobotics/defm, defm/utils/utils.py and defm/models/regnet_bifpn.py):
# Copyright (c) 2026, ETH Zurich, Manthan Patel, licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0).

"""Frozen depth encoder of the parkour policy: DeFM RegNetY-400MF.

The front camera renders one small depth image per env. ``DepthEncoder`` turns it into a FEATURE_DIM vector that
the high-level actor reads next to its other observations; the critic does not get it. The encoder is never trained
and is not part of the policy networks, so checkpoints do not carry its weights.

The preprocessing and the model are DeFM's own code, copied below: ``preprocess_depth_batch`` builds DeFM's
three-channel metric input, and ``DepthEncoder`` computes the global feature of DeFM's RegNetBiFPN,
``forward_no_bifpn(x)["global_backbone"]``.
"""

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.v2 as tt
from torch import nn
from torchvision.models import regnet_y_400mf
from torchvision.models.feature_extraction import create_feature_extractor

DEPTH_SENSOR = "depth_front"  # Name of the CameraSensorCfg in the scene.
DEPTH_CAMERA = "robot/depth_front"  # MJCF camera it wraps.
DEPTH_SIZE = (60, 80)  # Rendered image: rows, columns.

UPSAMPLE = 3
FEATURE_DIM = 440  # Channels of the last backbone stage.
CHUNK = 256  # Images per forward pass; an image takes about 6 MB of activations at 180 x 240.
DEFM_WEIGHTS = Path(__file__).parent / "depth_model" / "defm_regnet_y_400mf.pth"


# Default MEAN and STD
DEFM_MEAN = [0.248880, 0.495620, 0.492858]
DEFM_STD = [0.139357, 0.271314, 0.297177]


def make_norm_transform(height: int, width: int, to_tensor: bool = True) -> tt.Compose:
    """
    Create a DeFM normalization transform for a given (H, W).

    Parameters:
      - height, width: target size for Resize
      - to_tensor: if True include tt.ToTensor() as the first step.
                   If False assume input is already a Tensor and only Resize+Normalize are applied.

    Usage:
        norm_transform = make_norm_transform(final_h, final_w)
        norm_transform_no_to_tensor = make_norm_transform(final_h, final_w, to_tensor=False)
    """
    transforms = []
    if to_tensor:
        transforms.append(tt.ToTensor())
    transforms.extend(
        [
            tt.Resize((height, width)),
            tt.Normalize(mean=DEFM_MEAN, std=DEFM_STD),
        ]
    )
    return tt.Compose(transforms)


def get_target_size(
    current_size: Tuple[int, int],
    target_size: Optional[Union[int, Tuple[int, int]]] = None,
    patch_size: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Determines the final (H, W) based on target_size and patch_size constraints.
    """
    curr_h, curr_w = current_size

    # 1. Handle target_size
    if target_size is None:
        new_h, new_w = curr_h, curr_w
    elif isinstance(target_size, int):
        new_h, new_w = target_size, target_size
    else:
        new_h, new_w = target_size

    # 2. Adjust for patch_size divisibility
    if patch_size is not None:
        new_h = (new_h // patch_size) * patch_size
        new_w = (new_w // patch_size) * patch_size

    return new_h, new_w


def preprocess_depth_batch(
    input_batch: Union[np.ndarray, torch.Tensor],
    target_size: Optional[Union[int, Tuple[int, int]]] = None,
    cnn_padding: Optional[bool] = False,
    patch_size: Optional[int] = None,
    max_depth_c1: float = 100.0,
    max_depth_c2: float = 9.0,
    device: Optional[Union[str, torch.device]] = "cpu",
) -> torch.Tensor:
    """
    Vectorized batch preprocessing for DeFM depth images (no Python loop).
    Accepts input shapes:
      - (N, H, W)
      - (N, 1, H, W)
    Returns:
      - torch.Tensor of shape (N, 3, H_out, W_out), dtype=torch.float32 on `device`.
    Behavior:
      - Builds the 3-channel metric representation (C1, C2, C3) in a vectorized way.
      - Resizes using torch.nn.functional.interpolate (fast, batched).
      - Applies DeFM per-channel mean/std normalization.
    """

    # Convert numpy -> torch
    if isinstance(input_batch, np.ndarray):
        t = torch.from_numpy(input_batch)
    elif isinstance(input_batch, torch.Tensor):
        t = input_batch
    else:
        raise TypeError("input_batch must be a numpy array or torch.Tensor")

    # Ensure float32
    t = t.to(dtype=torch.float32)

    # Normalize shapes to (N,1,H,W)
    if t.ndim == 3:
        # (N,H,W) -> (N,1,H,W)
        t = t.unsqueeze(1)
    elif t.ndim == 4:
        if t.shape[1] != 1:
            # take first channel if multi-channel provided
            t = t[:, :1, :, :]
    else:
        raise ValueError("input_batch must have shape (N,H,W) or (N,1,H,W)")

    # Move to device
    device = torch.device(device)
    t = t.to(device)

    N, _, H, W = t.shape

    # Determine final target size
    final_h, final_w = get_target_size((H, W), target_size, patch_size)

    # Replace NaN/inf with 0
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip to max depth for C1 computation
    t_clamped = torch.clamp(t, min=0.0, max=max_depth_c1)

    # log1p
    log_depth = torch.log1p(t_clamped)

    # C1: normalized by log1p(max_depth_c1)
    denom_c1 = math.log1p(max_depth_c1)
    c1 = log_depth / denom_c1

    # C2: clipped ratio using max_depth_c2
    denom_c2 = math.log1p(max_depth_c2)
    c2 = torch.clamp(log_depth / denom_c2, min=0.0, max=1.0)

    # C3: per-image min-max normalization over spatial dims
    log_flat = log_depth.view(N, -1)
    min_log = log_flat.min(dim=1)[0].view(N, 1, 1, 1)
    max_log = log_flat.max(dim=1)[0].view(N, 1, 1, 1)
    denom = max_log - min_log
    # avoid divide-by-zero: where denom <= 0 set denom=1 and result will be zero after subtract
    denom_safe = torch.where(denom > 0.0, denom, torch.ones_like(denom))
    c3 = (log_depth - min_log) / denom_safe
    # zero out images where denom was <= 0
    c3 = torch.where(denom > 0.0, c3, torch.zeros_like(c3))

    # Stack channels -> (N,3,H,W)
    x = torch.cat([c1, c2, c3], dim=1)

    norm_transform = make_norm_transform(final_h, final_w, to_tensor=False)
    x = norm_transform(x)

    if cnn_padding:
        # Pad to next multiple of 32 (Since this is needed for the BiFPNs)
        pad_h = (32 - (final_h % 32)) % 32
        pad_w = (32 - (final_w % 32)) % 32
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    return x


# Copied unchanged from DeFM, defm/models/regnet_bifpn.py.

def replace_bn_with_gn(model, default_num_groups=32):
    """
    Recursively replaces all BatchNorm layers in a model with GroupNorm.
    Automatically adjusts num_groups so that num_channels % num_groups == 0.
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            num_channels = module.num_features

            # Choose a valid num_groups dynamically
            num_groups = default_num_groups
            while num_groups > 1 and num_channels % num_groups != 0:
                num_groups //= 2
            if num_channels % num_groups != 0:
                num_groups = 1  # fallback to LayerNorm-like behavior

            gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
            setattr(model, name, gn)

        else:
            replace_bn_with_gn(module, default_num_groups=default_num_groups)

    return model


class DepthEncoder(nn.Module):
    """Depth images (N, H, W) in metres, 0 = no return -> features (N, FEATURE_DIM).

    DeFM's RegNetBiFPN without its BiFPN neck: the neck does not feed the global feature used here, and it needs the
    image padded to a multiple of 32. All weights are frozen.
    """

    def __init__(self) -> None:
        super().__init__()
        # As in RegNetBiFPN.__init__, with only the backbone stage that the global feature reads.
        backbone = replace_bn_with_gn(regnet_y_400mf(weights=None), default_num_groups=32)
        self.backbone = create_feature_extractor(backbone, return_nodes={"trunk_output.block4": "c5"})
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.norm_cls = nn.LayerNorm(FEATURE_DIM, eps=1e-6)

        checkpoint = torch.load(DEFM_WEIGHTS, map_location="cpu", weights_only=True)["model"]
        # The remaining keys (fpn.*, norm_patch.*) belong to the BiFPN neck.
        state = {key: value for key, value in checkpoint.items() if key.startswith(("backbone.", "norm_cls."))}
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False).eval()

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Encode depth images (N, H, W) in metres, 0 = no return, into features (N, FEATURE_DIM)."""
        return torch.cat([self._features(chunk) for chunk in depth.split(CHUNK)])

    def _features(self, depth: torch.Tensor) -> torch.Tensor:
        """RegNetBiFPN.forward_no_bifpn(preprocess_depth_batch(depth))["global_backbone"] for depth (n, H, W)."""
        height, width = depth.shape[-2:]
        # DeFM's default clipping (100 m for C1, 9 m for C2) is kept: clipping at 4 m instead detected obstacles
        # beyond 2.5 m less often (0.71 against 0.78) at the same near-range scores.
        x = preprocess_depth_batch(depth, target_size=(UPSAMPLE * height, UPSAMPLE * width), device=depth.device)
        feats = self.backbone(x)
        return self.norm_cls(self.global_pool(feats["c5"]).flatten(1))
