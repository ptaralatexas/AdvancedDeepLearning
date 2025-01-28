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

class HalfBigNet(torch.nn.Module):
    """
    A BigNet where all weights are in half precision. Make sure that the normalization uses full
    precision though to avoid numerical instability.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            raise NotImplementedError()

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> HalfBigNet:
    # You should not need to change anything here
    # PyTorch can load float32 states into float16 models
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
