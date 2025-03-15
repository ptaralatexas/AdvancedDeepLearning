import abc
import torch
import torch.nn.functional as F
from pathlib import Path
import math

from .ae import PatchAutoEncoder



def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    loaded_obj = torch.load(model_path, map_location=torch.device("cpu"), weights_only= False)

    # Decide how to handle the loaded object
    if isinstance(loaded_obj, torch.nn.Module):
        # If it’s already a full model, extract the state dict
        print("Loaded a full BSQPatchAutoEncoder model, extracting state_dict...")
        state_dict = loaded_obj.state_dict()
    elif isinstance(loaded_obj, dict):
        # If it’s just a dict, assume it’s already a state_dict
        print("Loaded a state_dict...")
        state_dict = loaded_obj
    else:
        raise TypeError(
            f"The file {model_path} must contain either a nn.Module or a "
            f"state_dict, but got type {type(loaded_obj)}."
        )

    # Now initialize a fresh model and load the state_dict
    model = BSQPatchAutoEncoder()
    model.load_state_dict(state_dict)
    return model


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
        x = x.view(B, H * W, C)  # ✅ Ensure (B, HW, 128) before projection
        print(f"After reshaping: {x.shape}")  # Debugging

        if x.shape[-1] != self.embedding_dim:  # 🚀 Check before projection
            raise ValueError(f"Expected last dimension {self.embedding_dim}, but got {x.shape[-1]}")

        x = self.linear_proj(x)  # ✅ Linear projection (B, HW, 10)
        print(f"After projection: {x.shape}")  # Debugging

        x = F.normalize(x, p=2, dim=-1)  # L2 Normalization
        x = diff_sign(x)  # Differentiable sign function
        print(f"Final shape before returning: {x.shape}")  # Debugging

        return x.view(B, H, W, self.codebook_bits)  # ✅ Reshape correctly




    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a BSQ-encoded representation back into an image.
        """
        print(f"Decoding - Input shape: {x.shape}")  # Debugging print

        if x.dim() == 4:  # ✅ If already (B, H, W, C), no need to reshape
            B, H, W, C = x.shape  
        elif x.dim() == 3:  # ✅ If flattened (B, HW, C), infer H & W
            B, HW, C = x.shape
            H, W = int(math.sqrt(HW)), HW // int(math.sqrt(HW))  # Dynamically determine H & W
            x = x.view(B, H, W, C)  # ✅ Reshape from (B, HW, C) → (B, H, W, C)

        print(f"Decoding - After reshape: {x.shape}")  # Debugging print

        x = self.linear_recon(x)  # ✅ Map from 10 back to 128 channels
        print(f"Decoding - After projection: {x.shape}")  # Debugging print

        return x  # Correctly return reshaped tensor




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
        Encode an input image `x` using the autoencoder and then apply BSQ quantization.
        """
        if x.shape[-1] == 3:  # Raw image, needs full encoding
            print(f"BSQPatchAutoEncoder.encode() - Running PatchAutoEncoder on raw image {x.shape}")
            
            # Add batch dimension if necessary
            if x.dim() == 3:  # [H, W, C]
                x = x.unsqueeze(0)  # [1, H, W, C]
                
            print(f"After adding batch dim: {x.shape}")
            latent = super().encode(x)  # Get latent representation, expected shape: [B, H, W, 128]
            
            # Apply BSQ quantization to convert embeddings (128) to quantized codes (10)
            quantized = self.bsq.encode(latent)  # Now shape: [B, H, W, 10]
            return quantized
        
        # Optionally, handle other cases if needed...

        




    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a BSQ-encoded representation back into an image.
        """
        x = self.bsq.decode(x)  # ✅ Ensure BSQ decoding is handled correctly
        print(f"BSQPatchAutoEncoder.decode() - After BSQ decode: {x.shape}")  # Debugging

        return super().decode(x)  # ✅ Call PatchAutoEncoder's decode()






    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert an image into discrete integer tokens.
        """
        if x.shape[-1] == 3:  # Raw image
            # Process through full encoder to get quantized codes
            x = self.encode(x)  # (B, H, W, 10)
            return self.bsq._code_to_index(x)
        
        if x.shape[-1] == 128:  # Embedding
            # Quantize embeddings then convert to indices
            x = self.bsq.encode(x)
            return self.bsq._code_to_index(x)
        
        if x.shape[-1] == 10:  # Pre-quantized
            # Directly convert to indices
            return self.bsq._code_to_index(x)
        
        raise ValueError(f"Unexpected input channels: {x.shape[-1]}")







    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        # Convert indices → BSQ codes without decoding
        codes = self.bsq._index_to_code(x)  # ✅ Direct code conversion
        return self.decode(codes)  # ✅ Proper single decoding pass


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