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
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(nn.Module):
    """
    An autoregressive model that predicts token logits.
    
    The model expects an input image tensor of shape (B, H, W, 3). It tokenizes
    the image by averaging channels and casting to long, then treats the result as token indices.
    The output is transformed back to shape (B, H, W, 3) to match the input for MSE loss.
    """
    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent

        self.embedding = nn.Embedding(n_tokens, d_latent)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_latent, nhead=8, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=6
        )
        self.fc_out = nn.Linear(d_latent, n_tokens)
        
        # Add a final projection layer to convert from n_tokens to 3 channels
        self.to_rgb = nn.Linear(n_tokens, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        print(f"DEBUG: Input shape to forward(): {x.shape}")
        
        # Store original input for debugging (we reduce it to a scalar for logging)
        original_input = x
        
        # If x is an RGB image, tokenize it by averaging channels and converting to long
        if x.dim() == 4:
            x = x.mean(dim=-1).long()  # now shape becomes (B, H, W)
            print(f"DEBUG: Tokenized x shape: {x.shape}")

        if x.dim() != 3:
            raise ValueError(f"Unexpected input shape {x.shape}, expected (B, H, W)")

        B, H, W = x.shape

        # Flatten spatial dimensions into a sequence.
        x = x.view(B, -1)  # shape: (B, H*W)
        x = self.embedding(x)  # shape: (B, H*W, d_latent)

        # Shift the sequence by prepending a zero embedding and dropping the last element.
        x = torch.cat([torch.zeros(B, 1, self.d_latent, device=x.device), x[:, :-1, :]], dim=1)
        x = self.transformer_encoder(x)  # shape: (B, H*W, d_latent)
        x = self.fc_out(x)  # shape: (B, H*W, n_tokens)
        
        # Store token logits for debugging
        token_logits = x.view(B, H, W, self.n_tokens)
        
        # Convert to RGB format (B, H, W, 3) for compatibility with MSE loss
        x = self.to_rgb(x)  # shape: (B, H*W, 3)
        x = x.view(B, H, W, 3)  # reshape back to (B, H, W, 3)
        
        # Instead of returning the full original_input tensor, return its mean for logging.
        return x, {"token_logits": token_logits, "original_input_mean": original_input.mean()}

    def generate(self, B: int = 1, H: int = 30, W: int = 20, device=None) -> torch.Tensor:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        output = torch.zeros((B, H, W), dtype=torch.long, device=device)
        
        with torch.no_grad():
            for i in range(H):
                for j in range(W):
                    # Forward pass to get token logits
                    rgb_output, info = self.forward(output)
                    token_logits = info["token_logits"]
                    
                    # Get probabilities for the current position and sample a token.
                    probs = F.softmax(token_logits[:, i, j, :], dim=-1)
                    output[:, i, j] = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Convert final tokens to RGB
        final_rgb, _ = self.forward(output)
        return final_rgb
