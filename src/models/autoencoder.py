# src/models/autoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure


class BraTSAutoencoderLightning(pl.LightningModule):
    def __init__(self, latent_dim=256, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.embedding_layer = nn.Linear(64 * 16 * 16, latent_dim)

        self.unflatten = nn.Linear(latent_dim, 64 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16,  1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        self.criterion = nn.MSELoss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x):
        enc    = self.encoder(x)
        flat   = self.flatten(enc)
        emb    = self.embedding_layer(flat)
        unflat = self.unflatten(emb).view(-1, 64, 16, 16)
        recon  = self.decoder(unflat)
        return recon, emb

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }