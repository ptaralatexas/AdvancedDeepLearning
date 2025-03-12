import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, H, W) of integers as input.
        Produce a probability over the next token as an output (B, H, W, n_tokens).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(nn.Module, Autoregressive):
    """
    An autoregressive model that predicts token logits.
    
    The model expects an input image tensor of shape (B, H, W, 3) if you have an RGB image
    and want to convert it to tokens by averaging the channels. If your input is already
    tokenized (B, H, W) of integers, it will use those directly.
    
    The output is token logits of shape (B, H, W, n_tokens), which you can train with
    next-token cross-entropy.
    """
    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent

        # Embed discrete tokens into a continuous latent space
        self.embedding = nn.Embedding(n_tokens, d_latent)

        # A Transformer-Encoder used as a decoder with a causal mask
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=6
        )

        # Final projection from latent to token logits
        self.fc_out = nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: (B, H, W, 3) if float, or (B, H, W) if integer tokens.
        Returns:
          token_logits: (B, H, W, n_tokens)
          dict with:
            "target_tokens": the ground-truth tokens for cross-entropy,
            "target_tokens_mean": a scalar (mean of target tokens) for logging,
            "original_input_mean": a scalar for logging/debug.
        """
        print(f"DEBUG: Input shape to forward(): {x.shape}")

        # Store original input for logging
        original_input = x

        # If x is a float image with 4D shape, convert to tokens by averaging channels
        if x.dim() == 4:
            if x.is_floating_point():
                # Convert to tokens by averaging the last dimension and casting to long
                x = x.mean(dim=-1).long()  # shape: (B, H, W)
                print(f"DEBUG: Tokenized x shape (from float): {x.shape}")
            else:
                # If it's already integer but shape is (B, H, W, C),
                # assume the first channel has tokens
                x = x[..., 0]
                print(f"DEBUG: Input x is integer; using first channel: {x.shape}")

        if x.dim() != 3:
            raise ValueError(f"Unexpected input shape {x.shape}, expected (B, H, W)")

        B, H, W = x.shape

        # Ground truth tokens for cross-entropy
        target_tokens = x.clone()

        # Flatten the (H, W) dims into a single sequence
        x = x.view(B, -1)        # shape: (B, H*W)
        x = self.embedding(x)    # shape: (B, H*W, d_latent)

        # Shift the sequence by prepending a zero embedding and dropping the last element
        x = torch.cat([
            torch.zeros(B, 1, self.d_latent, device=x.device),
            x[:, :-1, :]
        ], dim=1)

        # Create a causal mask so each position only sees itself and previous tokens
        B, L, _ = x.shape
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=x.device),
            diagonal=1
        )

        # Pass through the Transformer
        x = self.transformer_encoder(x, mask=causal_mask)  # (B, H*W, d_latent)

        # Project to token logits
        x = self.fc_out(x)       # (B, H*W, n_tokens)
        token_logits = x.view(B, H, W, self.n_tokens)

        # Prepare the additional info dictionary.
        additional_info = {
            "target_tokens": target_tokens,  # For loss computation (do not log this directly)
            "target_tokens_mean": target_tokens.float().mean(),  # Scalar for logging
            "original_input_mean": original_input.float().mean()
        }

        return token_logits, additional_info

    def generate(self, B: int = 1, H: int = 30, W: int = 20, device=None) -> torch.Tensor:
        """
        Generate a grid of tokens (B, H, W) in an autoregressive manner,
        sampling each position from the model's predicted distribution.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()

        # Initialize an empty token grid
        output = torch.zeros((B, H, W), dtype=torch.long, device=device)

        with torch.no_grad():
            for i in range(H):
                for j in range(W):
                    # Get token logits for the entire grid so far
                    token_logits, _ = self.forward(output)
                    # Focus on the distribution for position (i, j)
                    probs = F.softmax(token_logits[:, i, j, :], dim=-1)
                    # Sample one token from this distribution
                    output[:, i, j] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return output
