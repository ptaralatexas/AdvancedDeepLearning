# file: train.py

import inspect
import math
from datetime import datetime
from pathlib import Path
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your model modules
from . import ae, autoregressive, bsq

# Identify patch-based models
patch_models = {
    n: m
    for M in [ae, bsq]
    for n, m in inspect.getmembers(M)
    if inspect.isclass(m) and issubclass(m, torch.nn.Module)
}

# Identify autoregressive models
ar_models = {
    n: m
    for M in [autoregressive]
    for n, m in inspect.getmembers(M)
    if inspect.isclass(m) and issubclass(m, torch.nn.Module)
}


def train(model_name_or_path: str, epochs: int = 5, batch_size: int = 4):
    """
    Train either a patch-based autoencoder (from ae.py or bsq.py)
    or an autoregressive model (from autoregressive.py).
    """

    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger

    # Import your dataset code
    from .data import ImageDataset, TokenDataset

    # -------------------------------------------------------------------------
    # A helper function to safely log only scalar values with self.log(...)
    # -------------------------------------------------------------------------
    def log_all_scalars(prefix: str, logger_fn, metrics_dict: dict, prog_bar: bool = False):
        """
        For each item in metrics_dict:
          - If it's a multi-dimensional tensor, log the mean
          - If it's already scalar (int, float, 0-dim tensor), log directly
        """
        for k, v in metrics_dict.items():
            # If it's a tensor
            if isinstance(v, torch.Tensor):
                if v.dim() > 0:
                    # log mean if multi-dimensional
                    logger_fn(f"{prefix}/{k}", v.float().mean(), prog_bar=prog_bar)
                else:
                    # 0-dim (scalar) => log directly
                    logger_fn(f"{prefix}/{k}", v, prog_bar=prog_bar)
            # If it's a number
            elif isinstance(v, (int, float)):
                logger_fn(f"{prefix}/{k}", v, prog_bar=prog_bar)
            # otherwise skip or handle as you wish

    # -------------------------------------------------------------------------
    # Trainer for patch-based models like autoencoders
    # -------------------------------------------------------------------------
    class PatchTrainer(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def training_step(self, x, batch_idx):
            # Scale/shift input into [-0.5, +0.5]
            x = x.float() / 255.0 - 0.5

            # Forward pass
            x_hat, additional_losses = self.model(x)

            # Main reconstruction loss
            loss = F.mse_loss(x_hat, x)  # shape () => scalar
            # If additional_losses has scalar items, you can sum them here:
            total_loss = loss + sum(additional_losses.values())

            # Log the main MSE
            self.log("train/loss", loss, prog_bar=True)
            # Log all other scalar items in additional_losses
            log_all_scalars("train", self.log, additional_losses)

            return total_loss

        def validation_step(self, x, batch_idx):
            x = x.float() / 255.0 - 0.5

            with torch.no_grad():
                x_hat, additional_losses = self.model(x)
                loss = F.mse_loss(x_hat, x)

            self.log("validation/loss", loss, prog_bar=True)
            log_all_scalars("validation", self.log, additional_losses)

            # Optionally log images (only once per epoch at batch_idx=0)
            if batch_idx == 0:
                # show original
                self.logger.experiment.add_images(
                    "input", 
                    (x[:64] + 0.5).clamp(min=0, max=1).permute(0, 3, 1, 2), 
                    self.global_step
                )
                # show prediction
                self.logger.experiment.add_images(
                    "prediction", 
                    (x_hat[:64] + 0.5).clamp(min=0, max=1).permute(0, 3, 1, 2), 
                    self.global_step
                )
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

        def train_dataloader(self):
            dataset = ImageDataset("train")
            return torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=2, shuffle=True)

        def val_dataloader(self):
            dataset = ImageDataset("valid")
            return torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=2, shuffle=True)

    # -------------------------------------------------------------------------
    # Trainer for autoregressive models (e.g. from autoregressive.py)
    # -------------------------------------------------------------------------
    class AutoregressiveTrainer(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def training_step(self, x, batch_idx):
            # Forward pass => x_hat: (B, H, W, n_tokens)
            # info: dictionary with possibly multi-dim items
            x_hat, info = self.model(x)

            # Cross-entropy => scalar
            ce_loss = (
                F.cross_entropy(
                    x_hat.view(-1, x_hat.shape[-1]), 
                    x.view(-1),
                    reduction="mean"
                )
                / math.log(2)  
                / x.shape[0]
            )

            # If you had extra scalar losses in info, sum them here if needed
            # e.g. total_loss = ce_loss + sum_of_scalar_regularizers
            total_loss = ce_loss

            # Log the main CE loss
            self.log("train/loss", ce_loss, prog_bar=True)
            # Log the rest (turn multi-dim into .mean())
            log_all_scalars("train", self.log, info)

            return total_loss

        def validation_step(self, x, batch_idx):
            with torch.no_grad():
                x_hat, info = self.model(x)
                ce_loss = (
                    F.cross_entropy(
                        x_hat.view(-1, x_hat.shape[-1]), 
                        x.view(-1),
                        reduction="sum"
                    )
                    / math.log(2) 
                    / x.shape[0]
                )

            self.log("validation/loss", ce_loss, prog_bar=True)
            log_all_scalars("validation", self.log, info)
            return ce_loss

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

        def train_dataloader(self):
            dataset = TokenDataset("train")
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

        def val_dataloader(self):
            dataset = TokenDataset("valid")
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # -------------------------------------------------------------------------
    # A simple checkpoint callback
    # -------------------------------------------------------------------------
    class CheckPointer(L.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fn = Path(f"checkpoints/{timestamp_str}_{model_name}.pth")
            fn.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model, fn)

    # -------------------------------------------------------------------------
    # Load or create the model
    # -------------------------------------------------------------------------
    # If there's a file with the given name, load it.
    if Path(model_name_or_path).exists():
        model = torch.load(model_name_or_path, weights_only=False)
        model_name = model.__class__.__name__
    else:
        # Otherwise interpret model_name_or_path as a class name
        model_name = model_name_or_path
        if model_name in patch_models:
            model = patch_models[model_name]()
        elif model_name in ar_models:
            model = ar_models[model_name]()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # Pick which LightningModule to use based on model type
    if isinstance(model, (autoregressive.Autoregressive)):
        l_model = AutoregressiveTrainer(model)
    else:
        l_model = PatchTrainer(model)

    # -------------------------------------------------------------------------
    # Create a logger and trainer
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger("logs", name=f"{timestamp}_{model_name}")

    trainer = L.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[CheckPointer()],
        # You can add accelerator="gpu" if you want:
        # accelerator="gpu",
    )

    # -------------------------------------------------------------------------
    # Fit the model
    # -------------------------------------------------------------------------
    trainer.fit(l_model)


if __name__ == "__main__":
    from fire import Fire
    Fire(train)
