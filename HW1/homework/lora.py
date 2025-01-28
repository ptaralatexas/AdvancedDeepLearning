from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear,FullPrecisionLayerNorm


import torch
import torch.nn as nn

class LoRALinear(HalfLinear):
    """
    Implements LoRA on top of a frozen half-precision Linear.
    The base weight (and optional bias) are half precision and frozen (no grad).
    The LoRA layers are float32, trainable, and low-rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        Args:
            in_features:  Dimension of input.
            out_features: Dimension of output.
            lora_dim:     Rank of the LoRA decomposition.
            bias:         Whether to have a bias in the base linear layer.
        """
        # Initialize the parent half-precision linear
        super().__init__(in_features, out_features, bias=bias)

        # 1) Freeze the parent's main weight/bias (which are in float16)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # 2) Define LoRA layers in float32
        #    Typically, these do NOT include a bias (bias=False).
        self.lora_a = nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_b = nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        # 3) Initialize LoRA layers so that their contribution is near zero initially
        #    Common practice is a small std on lora_a, zero on lora_b.
        nn.init.normal_(self.lora_a.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1) Base half-precision path from HalfLinear
          2) LoRA path in float32
          3) Sum them
          4) Final output is float32
        """
        # (a) Get the base output from the frozen half-precision linear
        out_base = super().forward(x)  # In float16 internally, returns float32

        # (b) Compute LoRA path in float32
        out_lora = self.lora_b(self.lora_a(x))  # Both lora layers are float32

        # (c) Combine the two paths (float32 + float32) -> float32
        return out_base + out_lora



class LoraBigNet(torch.nn.Module):
    """
    A BigNet variant that applies LoRA to each linear sub-layer.
    """

    class Block(torch.nn.Module):
        """
        Each block is:
            LoRALinear -> ReLU
            LoRALinear -> ReLU
            LoRALinear
        with a residual connection (x + block(x)).
        """
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()
            self.model = nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
                nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        """
        Args:
            lora_dim: Rank of LoRA adapters. 
                      Higher = more trainable parameters, lower = smaller memory overhead.
        """
        super().__init__()
        
        # Stack multiple (Block -> LayerNorm) pairs, similar to the original BigNet.
        self.model = nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),
            # Use your full-precision normalization layer:
            FullPrecisionLayerNorm(BIGNET_DIM),
            
            self.Block(BIGNET_DIM, lora_dim),
            FullPrecisionLayerNorm(BIGNET_DIM),
            
            self.Block(BIGNET_DIM, lora_dim),
            FullPrecisionLayerNorm(BIGNET_DIM),
            
            self.Block(BIGNET_DIM, lora_dim),
            FullPrecisionLayerNorm(BIGNET_DIM),
            
            self.Block(BIGNET_DIM, lora_dim),
            FullPrecisionLayerNorm(BIGNET_DIM),
            
            # Final residual block (no norm after).
            self.Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
