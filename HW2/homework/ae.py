import abc
import torch
from pathlib import Path
import torch.nn.functional as F

def load() -> torch.nn.Module:
    """
    Load the saved PatchAutoEncoder model.
    """
    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    
    model = PatchAutoEncoder()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an arbitrary tensor from (H, W, C) to (C, H, W) format.
    Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    """
    The opposite of hwc_to_chw. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


class PatchifyLinear(torch.nn.Module):
    """
    Patchifies an image tensor from (B, H, W, 3) to (B, H//patch_size, W//patch_size, latent_dim).
    Uses a linear transformation (a 2D convolution) to embed each patch.
    """

    def __init__(self, patch_size: int = 16, latent_dim: int = 256):
        super().__init__()
        self.patch_conv = torch.nn.Conv2d(3, latent_dim, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return chw_to_hwc(self.patch_conv(hwc_to_chw(x)))


class UnpatchifyLinear(torch.nn.Module):
    """
    Converts a patchified embedding tensor (B, H//patch_size, W//patch_size, latent_dim)
    back into an image of shape (B, H, W, 3).
    """

    def __init__(self, patch_size: int = 16, latent_dim: int = 256):
        super().__init__()
        self.unpatch_conv = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return chw_to_hwc(self.unpatch_conv(hwc_to_chw(x)))


class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes input image x (B, H, W, 3) into a tensor (B, h, w, bottleneck),
        where h = H // patch_size, w = W // patch_size.
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes a tensor (B, h, w, bottleneck) back into an image (B, H, W, 3).
        """


class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """
    Implement a PatchLevel AutoEncoder

    Hint: Convolutions work well enough, no need to use a transformer unless you really want.
    Hint: See PatchifyLinear and UnpatchifyLinear for how to use convolutions with the input and
          output dimensions given.
    Hint: You can get away with 3 layers or less.
    Hint: Many architectures work here (even a just PatchifyLinear / UnpatchifyLinear).
          However, later parts of the assignment require both non-linearities (i.e. GeLU) and
          interactions (i.e. convolutions) between patches.
    """

    class PatchEncoder(torch.nn.Module):
        """
        (Optionally) Use this class to implement an encoder.
                     It can make later parts of the homework easier (reusable components).
        """

        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.patchify = PatchifyLinear(patch_size, latent_dim)
            self.conv1 = torch.nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(latent_dim)
            self.conv2 = torch.nn.Conv2d(latent_dim, bottleneck, kernel_size=3, stride=1, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(bottleneck)
            self.activation = torch.nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.patchify(x)
            x = hwc_to_chw(x)
            x = self.activation(self.bn1(self.conv1(x)))
            x = self.activation(self.bn2(self.conv2(x)))
            return chw_to_hwc(x)

    class PatchDecoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
           super().__init__()
           self.conv1 = torch.nn.Conv2d(bottleneck, latent_dim, kernel_size=3, stride=1, padding=1)
           self.bn1 = torch.nn.BatchNorm2d(latent_dim)
           self.conv2 = torch.nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)
           self.bn2 = torch.nn.BatchNorm2d(latent_dim)
           self.unpatchify = UnpatchifyLinear(patch_size, latent_dim)
           self.activation = torch.nn.GELU()

        def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:  # ✅ Fixed
           x = hwc_to_chw(x)
           x = self.activation(self.bn1(self.conv1(x)))
           x = self.activation(self.bn2(self.conv2(x)))
           x = chw_to_hwc(x)
           x = self.unpatchify(x)

        # ✅ Dynamically resize to match the original input size
           x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
           return x


    def __init__(self, patch_size: int = 25, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        self.encoder = self.PatchEncoder(patch_size, latent_dim, bottleneck)
        self.decoder = self.PatchDecoder(patch_size, latent_dim, bottleneck)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        You can return an empty dictionary if you don't have any additional terms.
        """
        encoded = self.encode(x)
        target_size = x.shape[-3:-1]  # ✅ Dynamically extract (H, W)
        reconstructed = self.decode(encoded, target_size=target_size)  # ✅ No hardcoded size
        return reconstructed, {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        return self.decoder(x, target_size)