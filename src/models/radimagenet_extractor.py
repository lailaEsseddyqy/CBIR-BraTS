# src/models/radimagenet_extractor.py
"""
Extracteur RadImageNet — VERSION 2048D BRUTS (sans projection apprise)

Changement vs version 256D :
  - Suppression de la couche de projection (2048→256)
  - Sortie directe : vecteur 2048D normalisé L2, features brutes du backbone
  - Avantage : pas de collapse possible, espace latent riche et discriminant
  - Collection Qdrant : 'radimagenet_embeddings' (2048D, distance cosinus)

Note : unified_engine.py et multimodal_search.py sont mis à jour
       pour passer LATENT_DIM=2048 lors des requêtes Qdrant.
"""

import os, sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from radimagenet_models.models.resnet import radimagenet_resnet50


class RadImageNetExtractor(nn.Module):
    """
    Extracteur de features RadImageNet 2D — sortie 2048D bruts.

    Pipeline :
      [B, 1, 128, 128]
          → repeat → [B, 3, 128, 128]
          → ResNet-50 backbone RadImageNet (gelé)
          → [B, 2048, H', W']
          → AdaptiveAvgPool → [B, 2048]
          → normalisation L2 → [B, 2048]
    """

    EMBED_DIM = 2048

    def __init__(
        self,
        weights_dir : str  = None,
        freeze      : bool = True,
        # projection_path gardé pour compatibilité ascendante mais ignoré
        projection_path: str = None,
    ):
        super().__init__()

        self.backbone    = radimagenet_resnet50(model_dir=weights_dir)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if projection_path:
            print("[RadImageNet] INFO : projection_path ignoré — mode 2048D bruts.")
        print(f"[RadImageNet] Backbone 2D chargé — sortie {self.EMBED_DIM}D bruts.")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, 128, 128]
        Retourne : [B, 2048] normalisé L2
        """
        x_rgb    = x.repeat(1, 3, 1, 1)            # [B, 3, 128, 128]
        feat     = self.backbone(x_rgb)             # [B, 2048, H', W']
        feat     = self.global_pool(feat)           # [B, 2048, 1, 1]
        feat     = torch.flatten(feat, 1)           # [B, 2048]
        # Normalisation L2
        norms    = feat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return feat / norms                         # [B, 2048]

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> list:
        """
        Encode une image [1, 128, 128] → vecteur 2048D normalisé L2.
        Interface identique aux autres extracteurs.
        """
        self.eval()
        img = image_tensor.unsqueeze(0)             # [1, 1, 128, 128]
        emb = self.forward(img).cpu().numpy().astype("float32")
        return emb[0].tolist()


if __name__ == "__main__":
    print("Test RadImageNetExtractor 2048D...")
    model = RadImageNetExtractor()
    model.eval()

    dummy = torch.rand(1, 128, 128)
    vec   = model.extract(dummy)

    print(f"  Input  : [1, 128, 128]")
    print(f"  Output : {len(vec)}D")
    print(f"  Norme  : {np.linalg.norm(vec):.6f} (doit être ≈ 1.0)")
    print(f"  Extrait: [{', '.join([f'{v:.4f}' for v in vec[:5]])}  ...]")
    print("RadImageNetExtractor 2048D opérationnel !")