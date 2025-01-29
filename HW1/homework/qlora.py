from pathlib import Path

import torch
import torch.nn as nn

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)

        self.lora_a = nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_b = nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        # 4) Initialize them so their initial contribution is near zero
        nn.init.normal_(self.lora_a.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1) Convert input x to float32 for the quantized linear base path
           (since Linear4Bit typically dequantizes to float32).
        2) Compute the base path using super().forward(...)  (4-bit quant).
        3) Compute LoRA path in float32.
        4) Return the sum, cast back to the original input dtype.
        """
        orig_dtype = x.dtype

        # a) Cast to float32 for the base linear's internal matmul
        x_32 = x.to(torch.float32)

        # b) Base path: uses the 4-bit weight (dequantized to float32)
        out_base = super().forward(x_32)  # float32 result

        # c) LoRA path: also in float32
        out_lora = self.lora_b(self.lora_a(x_32))  # float32

        # d) Sum the two outputs in float32, cast to original dtype
        out = out_base + out_lora
        return out.to(orig_dtype)


class QLoRABigNet(nn.Module):
    """
    A BigNet variant that uses 4-bit quantization + LoRA
    for each linear sub-layer.
    """

    class Block(nn.Module):
        def __init__(self, channels: int, lora_dim: int, group_size: int):
            super().__init__()
            # MLP with 3 quantized + LoRA linear layers,
            # each ReLU in between the first two layers.
            self.model = nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size),
                nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
                nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Residual connection: x + block(x)
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        # We replicate the same "Block -> LayerNorm" stacking
        # as in the original BigNet.
        self.model = nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
