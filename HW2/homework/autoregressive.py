import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoregressiveModel(nn.Module):
    """
    An autoregressive model for sequence-based image generation.
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, n_layers: int = 6, n_heads: int = 8):
        """
        Args:
            d_latent (int): Embedding dimension for token representation.
            n_tokens (int): Number of unique tokens.
            n_layers (int): Number of Transformer encoder layers.
            n_heads (int): Number of attention heads.
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent

        # Token embedding layer
        self.embedding = nn.Embedding(n_tokens, d_latent)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_latent, nhead=n_heads, dim_feedforward=4 * d_latent)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: maps latent representation to token probabilities
        self.output_head = nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, h, w).

        Returns:
            tuple: (logits of shape (B, h, w, n_tokens), auxiliary dictionary)
        """
        B, h, w = x.shape
        x = x.view(B, -1)  # Flatten to (B, h*w)

        # Embedding
        x = self.embedding(x)  # Shape: (B, h*w, d_latent)

        # Shift sequence by one (prepend a zero token)
        x_shifted = torch.cat([torch.zeros((B, 1, self.d_latent), device=x.device), x[:, :-1, :]], dim=1)

        # Pass through Transformer
        x_encoded = self.transformer(x_shifted)  # Shape: (B, h*w, d_latent)

        # Decode output
        logits = self.output_head(x_encoded)  # Shape: (B, h*w, n_tokens)

        # Reshape to (B, h, w, n_tokens)
        logits = logits.view(B, h, w, self.n_tokens)

        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        """
        Generates an image token-by-token using autoregressive inference.

        Args:
            B (int): Batch size.
            h (int): Height of the image.
            w (int): Width of the image.
            device (torch.device): Device for computation.

        Returns:
            torch.Tensor: Generated tensor of shape (B, h, w).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval()
        with torch.no_grad():
            generated = torch.zeros((B, h, w), dtype=torch.long, device=device)

            for i in range(h):
                for j in range(w):
                    logits, _ = self.forward(generated)
                    probs = F.softmax(logits[:, i, j, :], dim=-1)  # Get probability distribution
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # Sample from distribution
                    generated[:, i, j] = next_token  # Assign sampled token

        return generated
