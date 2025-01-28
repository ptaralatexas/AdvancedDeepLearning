from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


class HalfLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """
        A Linear layer whose parameters and internal computation are done in float16,
        while input and output remain float32.
        """
        # Use nn.Linear constructor for convenient weight/bias creation
        super().__init__(in_features, out_features, bias=bias)

        # Cast parameters to float16
        self.weight = torch.nn.Parameter(self.weight.half())
        if bias:
            self.bias = torch.nn.Parameter(self.bias.half())

        # (Optional) If you truly do not need gradient updates, disable gradient:
        self.weight.requires_grad = False
        if bias:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x is float32. Internally, we cast x to float16 and
        perform the linear operation in float16, then cast back to float32.
        """
        # Cast input to float16
        x_half = x.half()

        # Perform linear in half precision
        out_half = torch.nn.functional.linear(
            x_half, 
            self.weight, 
            self.bias
        )

        # Cast the output back to float32
        return out_half.float()


class FullPrecisionLayerNorm(torch.nn.Module):
    """
    A LayerNorm that runs in float32 even if the input is float16. 
    """
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            # Keep parameters in float32
            self.weight = torch.nn.Parameter(torch.ones(num_channels, dtype=torch.float32))
            self.bias   = torch.nn.Parameter(torch.zeros(num_channels, dtype=torch.float32))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Cast input to float32
        x_32 = x.float()

        # 2) Optionally cast weight/bias to float32 (they already are)
        w_32 = self.weight if self.weight is not None else None
        b_32 = self.bias   if self.bias  is not None else None

        # 3) Apply group_norm in float32
        out_32 = torch.nn.functional.group_norm(
            x_32, 
            num_groups=1, 
            weight=w_32, 
            bias=b_32, 
            eps=self.eps
        )

        # 4) Cast back to x’s original dtype (float16 or float32) if needed
        return out_32.to(x.dtype)





class HalfBigNet(torch.nn.Module):
    """
    A BigNet where all linear weights are in half precision, but
    the layer normalization runs in float32.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            # Each block is a 3-layer MLP with ReLU, all in half precision
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Residual connection: y = x + MLP(x)
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        
        # For convenience, define the dimension in a global or pass it in:
        BIGNET_DIM = 1024
        
        # Stack multiple (Block + LayerNorm) layers, 
        # similar to the original BigNet design.
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            FullPrecisionLayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            FullPrecisionLayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            FullPrecisionLayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            FullPrecisionLayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            FullPrecisionLayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



def load(path: Path | None) -> HalfBigNet:
    # You should not need to change anything here
    # PyTorch can load float32 states into float16 models
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
