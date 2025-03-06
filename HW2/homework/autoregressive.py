import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pass

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        pass


class AutoregressiveModel(nn.Module):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent
        
        self.embedding = nn.Embedding(n_tokens, d_latent)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_latent, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=6)
        self.fc_out = nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        print(f"DEBUG: Input shape to forward(): {x.shape}")  # 🔍 Debugging

        if x.dim() == 4:  # If x has 4 dimensions (B, H, W, C), remove the channel dimension
            x = x.mean(dim=-1)  # Convert (B, H, W, C) → (B, H, W) by averaging channels
            print(f"DEBUG: Reshaped x to {x.shape}")  # 🔍 Debugging

        if x.dim() != 3:  # Ensure we now have (B, H, W)
            raise ValueError(f"Unexpected input shape {x.shape}, expected (B, H, W)")

        B, h, w = x.shape  # ✅ Now safe to unpack

        x = x.view(B, -1)  # Flatten into sequence
        x = self.embedding(x)  # Convert tokens to embeddings
        x = torch.cat([torch.zeros(B, 1, self.d_latent, device=x.device), x[:, :-1, :]], dim=1)  # Shift by 1
        x = self.transformer_encoder(x)  # Process through transformer
        x = self.fc_out(x)  # Compute output logits

        x = x.view(B, h, w, self.n_tokens)  # Reshape back to spatial dimensions
        return x, {}




    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        output = torch.zeros((B, h, w), dtype=torch.long, device=device)
        with torch.no_grad():
            for i in range(h):
                for j in range(w):
                    logits, _ = self.forward(output)
                    probs = F.softmax(logits[:, i, j, :], dim=-1)
                    output[:, i, j] = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return output