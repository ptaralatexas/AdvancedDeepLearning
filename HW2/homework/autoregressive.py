import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


def load() -> torch.nn.Module:
    """
    Load the saved AutoRegressiveModel model.
    """
    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")

    # Initialize model with default hyperparameters
    model = AutoregressiveModel()

    # Load state_dict safely
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    if not isinstance(state_dict, dict):  # Ensure state_dict is valid
        raise TypeError(
            f"Expected state_dict to be dict-like, but got {type(state_dict)}. "
            f"Try re-saving the model using `torch.save(model.state_dict(), 'BSQPatchAutoEncoder.pth')`."
        )

    model.load_state_dict(state_dict)  # Load weights
    return model

class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, H, W) or (B, C, H, W) of integers as input.
        Produce a probability over the next token as an output (B, H, W, n_tokens).
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        """
        Use your generative model to produce B new token images of size (B, h, w).
        """


class AutoregressiveModel(nn.Module, Autoregressive):
    """
    An autoregressive model that predicts token logits of shape (B, H, W, n_tokens).
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent

        # Embed discrete tokens into a continuous latent space
        self.embedding = nn.Embedding(n_tokens, d_latent)

        # A Transformer-Encoder used as a "decoder" with a causal mask
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
        x: (B, H, W) or (B, C, H, W).
        Returns:
          token_logits: (B, H, W, n_tokens)
          info_dict: dict with:
            "target_tokens": the ground-truth tokens,
            "target_tokens_mean": a scalar for logging,
            "original_input_mean": a scalar for logging/debug.
        """
        print(f"DEBUG: Input shape to forward(): {x.shape}")

        # For logging/debug
        original_input = x

        # If input has 4 dimensions, handle accordingly
        if x.dim() == 4:  # (B, C, H, W)
            if x.is_floating_point():
                # Convert floats to discrete tokens by averaging channels (if you truly want that)
                # Then cast to long
                x = x.mean(dim=1).long()  # (B, H, W)
                print(f"DEBUG: Tokenized x shape (from float): {x.shape}")
            else:
                # If it's already integer tokens but shape is (B, C, H, W),
                # remove the channel dimension (assuming C=1 or that the first channel is the tokens)
                x = x[:, 0, :, :]  # (B, H, W)
                print(f"DEBUG: Reshaped integer input to: {x.shape}")

        # Now x should be (B, H, W)
        if x.dim() != 3:
            raise ValueError(f"Unexpected input shape {x.shape}, expected (B, H, W)")

        B, H, W = x.shape

        # Ground truth tokens for cross-entropy
        target_tokens = x.clone()

        # Flatten the 2D tokens into a single sequence
        x = x.view(B, -1)         # (B, H*W)
        x = self.embedding(x)     # (B, H*W, d_latent)

        # Shift the sequence right for next-token prediction
        x = torch.cat([
            torch.zeros(B, 1, self.d_latent, device=x.device, dtype=x.dtype),
            x[:, :-1, :]
        ], dim=1)                 # still (B, H*W, d_latent)

        # Create a causal mask so each position sees itself and previous tokens only
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1
        )

        # Pass through the Transformer
        x = self.transformer_encoder(x, mask=causal_mask)  # (B, H*W, d_latent)

        # Project to token logits
        x = self.fc_out(x)  # (B, H*W, n_tokens)
        token_logits = x.view(B, H, W, self.n_tokens)

        # Prepare info dictionary
        info_dict = {
            "target_tokens": target_tokens,   # shape (B, H, W)
            "target_tokens_mean": target_tokens.float().mean(),
            "original_input_mean": original_input.float().mean()
        }

        return token_logits, info_dict

    def generate(self, B: int = 1, H: int = 30, W: int = 20, device=None) -> torch.Tensor:
        """
        Generate a grid of tokens (B, H, W) in an autoregressive manner.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()

        # Initialize output tokens to 0
        output = torch.zeros((B, H, W), dtype=torch.long, device=device)

        with torch.no_grad():
            for i in range(H):
                for j in range(W):
                    # Get token logits for the grid so far
                    token_logits, _ = self.forward(output)
                    # Distribution for position (i, j)
                    probs = F.softmax(token_logits[:, i, j, :], dim=-1)
                    # Sample from this distribution
                    output[:, i, j] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return output
