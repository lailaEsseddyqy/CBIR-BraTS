# src/models/autoencoder_supervised.py
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.training.supcon_loss import SupConLoss

class BraTSAutoencoderSupervised(pl.LightningModule):
    def __init__(self, latent_dim=256, lr=1e-3,
                 alpha_recon=0.0, beta_supcon=1.0, temperature=0.07):
        super().__init__()
        self.save_hyperparameters()

        # ENCODEUR PLUS PROFOND
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.embedding_layer = nn.Linear(256 * 4 * 4, latent_dim)

        self.unflatten = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # DÉCODEUR PLUS PROFOND
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

        self.recon_criterion = nn.MSELoss()
        self.ssim_metric     = StructuralSimilarityIndexMeasure(data_range=1.0)
        # CORRECTION : Utilisation directe de la variable locale
        self.supcon_criterion = SupConLoss(temperature=temperature)

    def forward(self, x):
        enc    = self.encoder(x)
        flat   = self.flatten(enc)
        emb    = self.embedding_layer(flat)
        emb_n  = nn.functional.normalize(emb, p=2, dim=1)
        
        unflat = self.unflatten(emb).view(-1, 256, 4, 4)
        recon  = self.decoder(unflat)
        return recon, emb_n

    def _shared_step(self, batch, stage: str):
        imgs, labels = batch
        recon, emb_n = self(imgs)

        loss_recon = self.recon_criterion(recon, imgs)
        ssim_val   = self.ssim_metric(recon, imgs)
        loss_supcon = self.supcon_criterion(emb_n, labels)

        # CORRECTION : Utilisation de la syntaxe dictionnaire ["..."]
        loss = (self.hparams["alpha_recon"] * loss_recon
                + self.hparams["beta_supcon"] * loss_supcon)

        self.log(f"{stage}_loss",        loss,        prog_bar=True)
        self.log(f"{stage}_loss_recon",  loss_recon)
        self.log(f"{stage}_loss_supcon", loss_supcon)
        self.log(f"{stage}_ssim",        ssim_val)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        # CORRECTION : Utilisation de la syntaxe dictionnaire ["..."]
        optimizer = optim.Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }