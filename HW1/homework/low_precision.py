from pathlib import Path

import torch

import torch.nn as nn

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def block_quantize_4bit(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize the input tensor to 4-bit precision along the last dimension.
    Always quantize group_size value together and store their absolute value first.
    To keep things simple, we require x to be a 1D tensor, and the size divisible by group_size.
    Return the quantized tensor and scaling factor.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + normalization) / (2 * normalization)
    x_quant_8 = (x_norm * 15).round().to(torch.int8)
    x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
    return x_quant_4, normalization.to(torch.float16)


def block_dequantize_4bit(x_quant_4: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    """
    The reverse operation of block_quantize_4bit.
    """
    assert x_quant_4.dim() == 2

    normalization = normalization.to(torch.float32)
    x_quant_8 = x_quant_4.new_empty(x_quant_4.size(0), x_quant_4.shape[1] * 2)
    x_quant_8[:, ::2] = x_quant_4 & 0xF
    x_quant_8[:, 1::2] = (x_quant_4 >> 4) & 0xF
    x_norm = x_quant_8.to(torch.float32) / 15
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear4Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16) -> None:
        super().__init__()
        # Let's store all the required information to load the weights from a checkpoint
        self._shape = (out_features, in_features)
        self._group_size = group_size

        # self.register_buffer is used to store the weights in the model, but not as parameters
        # This makes sure weights are put on the correct device when calling `model.to(device)`.
        # persistent=False makes sure the buffer is not saved or loaded. The bignet has a parameters
        # called "weight" that we need to quantize when the model is loaded.
        self.register_buffer(
            "weight_q4",
            torch.zeros(out_features * in_features // group_size, group_size // 2, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        # Register a hook to load the weights from a checkpoint. This function reaches deep into
        # PyTorch internals. It makes sure that Linear4Bit._load_state_dict_pre_hook is called
        # every time the model is loaded from a checkpoint. We will quantize the weights in that function.
        self._register_load_state_dict_pre_hook(Linear4Bit._load_state_dict_pre_hook, with_module=True)
        # Add in an optional bias
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """
        This hook is called before the state_dict is loaded. Here, we read the original 'weight'
        from the state_dict, quantize it to 4-bit chunks, and store in 'weight_q4' + 'weight_norm'.
        """
        weight_key = f"{prefix}weight"
        if weight_key in state_dict:
            # 1) Extract the full-precision weight from the state_dict
            weight = state_dict[weight_key]  # shape [out_features, in_features]
            del state_dict[weight_key]       # remove it so it won't be loaded by PyTorch

            # 2) Flatten the weight to 1D
            weight_1d = weight.view(-1)

            # 3) Quantize in 4-bit chunks using your block_quantize_4bit
            x_quant_4, normalization = block_quantize_4bit(weight_1d, group_size=self._group_size)
            # x_quant_4 has shape [num_chunks, group_size//2]
            # normalization has shape [num_chunks, 1]

            # 4) Copy data into buffers
            self.weight_q4.copy_(x_quant_4)
            self.weight_norm.copy_(normalization.half())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1) Dequantize weights from 4-bit to float32
        2) Perform matmul with the input x
        3) Add bias if present
        4) Return result in float32
        """
        # 1) Dequantize
        #    block_dequantize_4bit expects shape [num_chunks, group_size//2]
        #    => produces a 1D or 2D output which we then reshape to [out_features, in_features].
        weight_dequant = block_dequantize_4bit(self.weight_q4, self.weight_norm)
        weight_dequant = weight_dequant.view(self._shape)  # [out_features, in_features]

        # 2) Linear op (x is [batch_size, in_features], or some shape with last dim = in_features)
        out = torch.nn.functional.linear(x, weight_dequant, self.bias)

        return out


class BigNet4Bit(nn.Module):
    """
    A BigNet where all weights are in 4bit precision. Use the Linear4Bit module for this.
    It is fine to keep all computation in float32.
    """

    class Block(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            # A 3-layer MLP with ReLU activations between them,
            # then add a residual connection.
            self.model = nn.Sequential(
                Linear4Bit(channels, channels),
                nn.ReLU(),
                Linear4Bit(channels, channels),
                nn.ReLU(),
                Linear4Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Residual connection
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # Stack multiple (Block -> LayerNorm) pairs, replicating the original BigNet design.
        self.model = nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNet4Bit:
    net = BigNet4Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
