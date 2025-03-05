from pathlib import Path
from typing import cast
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .bsq import Tokenizer, BSQPatchAutoEncoder

def tokenize(tokenizer: Path, output: Path, *images: Path):
    """
    Tokenize images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    output: Path to save the tokenize image tensor.
    images: Path to the image / images to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the state_dict safely
    state_dict = torch.load(tokenizer, map_location=device, weights_only=True)

    # Instantiate the model
    tk_model = BSQPatchAutoEncoder()

    # Load the model weights
    tk_model.load_state_dict(state_dict)

    # Move model to device
    tk_model.to(device)
    tk_model.eval()  # Set model to evaluation mode

    # Load and compress all images
    # Load and compress all images
    compressed_tensors = []
    for image_path in tqdm(images):
        image = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)
        
        # Convert image to numpy array (H, W, C)
        x = np.array(image, dtype=np.uint8)

        # Convert to torch tensor and move to device
        x = torch.tensor(x, dtype=torch.float32, device=device)

        # Permute from (H, W, C) → (C, H, W) and add batch dimension (1, C, H, W)
        x = x.permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize to [0,1]
        x = x - 0.5  # Center at 0

        print(f"Processed image shape before encoding: {x.shape}")  # Debugging

        with torch.inference_mode():
            cmp_image = tk_model.encode_index(x)  # Encode image
            compressed_tensors.append(cmp_image.cpu())

    # Stack tensors
    compressed_tensor = torch.stack(compressed_tensors)


    # Convert to smallest possible integer type
    np_compressed_tensor = compressed_tensor.numpy()
    if np_compressed_tensor.max() < 2**8:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint8)
    elif np_compressed_tensor.max() < 2**16:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint16)
    else:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint32)

    # Save tokenized images
    torch.save(np_compressed_tensor, output)

if __name__ == "__main__":
    from fire import Fire
    Fire(tokenize)
