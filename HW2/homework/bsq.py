import abc
import torch
import torch.nn.functional as F
import math

from .ae import PatchAutoEncoder


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self.codebook_bits = codebook_bits
        self.embedding_dim = embedding_dim

        # ✅ Project from embedding_dim (128) to codebook_bits (10)
        self.linear_proj = torch.nn.Linear(self.embedding_dim, self.codebook_bits)

        # ✅ Project back from codebook_bits (10) to embedding_dim (128)
        self.linear_recon = torch.nn.Linear(self.codebook_bits, self.embedding_dim)



    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder with additional debugging prints.
        """
        B, H, W, C = x.shape
        print(f"Before reshaping: {x.shape}")  # Debugging
        x = x.view(B, H * W, C)  # Flatten
        print(f"After reshaping: {x.shape}")  # Debugging
        x = self.linear_proj(x)  # Linear projection
        print(f"After projection: {x.shape}")  # Debugging
        x = F.normalize(x, p=2, dim=-1)  # L2 Normalization
        x = diff_sign(x)  # Differentiable sign function
        print(f"Final shape before returning: {x.shape}")  # Debugging
        return x.view(B, H, W, self.codebook_bits)  # Reshape correctly



    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a BSQ-encoded representation back into an image.
        """
        B, HW, C = x.shape  # 🚀 Get current shape (e.g., 16, 600, 10)
        H, W = int(math.sqrt(HW)), HW // int(math.sqrt(HW))  # Dynamically determine height & width

        x = x.view(B, H, W, C)  # ✅ Reshape to (B, 20, 30, 10) before decoding
        x = self.linear_recon(x)  # ✅ Project back to 128-dim embeddings
        return x  # Pass to autoencoder for final reconstruction


    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert BSQ-encoded representation into discrete integer tokens.
        """
        x = self.encode(x)  # Get BSQ representation
        return self._code_to_index(x)  # Convert to indices

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete integer tokens back into an image representation.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * (1 << torch.arange(x.size(-1)).to(x.device))).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (1 << torch.arange(self.codebook_bits).to(x.device))) > 0).float() - 1




class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim, bottleneck=latent_dim)
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)
        self.codebook_bits = codebook_bits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image x using the autoencoder and then apply BSQ quantization.
        """
        encoded = super().encode(x)  # Get encoded representation
        return self.bsq.encode(encoded)  # Apply BSQ quantization

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a BSQ-encoded representation back into an image.
        """
        B, HW, C = x.shape  # Get shape
        H, W = int(math.sqrt(HW)), HW // int(math.sqrt(HW))  # Dynamically infer H, W

        x = x.view(B, H, W, C)  # ✅ Reshape before feeding to autoencoder
        return super().decode(x)  # Call PatchAutoEncoder's decode()


    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input image into a set of discrete integer tokens.
        """
        return self.bsq.encode_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of discrete integer tokens into an image.
        """
        return self.decode(self.bsq.decode_index(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        """
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)

        # Compute codebook usage statistics
        token_counts = torch.bincount(self.encode_index(x).flatten(), minlength=2 ** self.codebook_bits)
        loss_terms = {
            "cb0": (token_counts == 0).float().mean().detach(),
            "cb2": (token_counts <= 2).float().mean().detach(),
        }

        return reconstructed, loss_terms
