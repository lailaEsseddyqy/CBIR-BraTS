# src/models/medicalnet_extractor.py
"""
Extracteur de features basé sur MedicalNet (ResNet-50 pré-entraîné
sur 8 datasets médicaux dont BraTS 2021).

Adaptation 3D → 2D :
  MedicalNet attend des volumes 3D [B, 1, D, H, W]
  Nos coupes BraTS sont en 2D [B, 1, H, W]
  Solution : répliquer la coupe 2D sur une profondeur D=16
             pour simuler un mini-volume 3D

Sortie : vecteur 256D normalisé L2
         (même espace que l'auto-encodeur baseline)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


# ─────────────────────────────────────────────────────────
# Bloc résiduel ResNet 3D (identique à MedicalNet officiel)
# ─────────────────────────────────────────────────────────

class ResidualBlock3D(nn.Module):
    """Bloc résiduel avec connexion skip — cœur de ResNet."""

    expansion = 4  # facteur d'expansion des canaux (ResNet-50 bottleneck)

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                                kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels,
                                kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_channels)

        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion,
                                kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm3d(out_channels * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity          # connexion résiduelle
        return self.relu(out)


# ─────────────────────────────────────────────────────────
# Backbone ResNet-50 3D (architecture MedicalNet)
# ─────────────────────────────────────────────────────────

class ResNet3D(nn.Module):
    """
    ResNet-50 3D — architecture identique à MedicalNet.
    Utilisé uniquement comme backbone (sans couche de classification).
    """

    def __init__(self):
        super().__init__()

        self.in_channels = 64

        # Couche d'entrée
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7,
                                stride=(2, 2, 2), padding=3, bias=False)
        self.bn1   = nn.BatchNorm3d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 4 étages résiduels (ResNet-50 : 3-4-6-3 blocs)
        self.layer1 = self._make_layer(64,  blocks=3)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

        # Pooling adaptatif → vecteur 2048D
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        expanded   = out_channels * ResidualBlock3D.expansion

        if stride != 1 or self.in_channels != expanded:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, expanded,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(expanded),
            )

        layers = [ResidualBlock3D(self.in_channels, out_channels,
                                   stride, downsample)]
        self.in_channels = expanded

        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # [B, 2048]


# ─────────────────────────────────────────────────────────
# Extracteur MedicalNet complet
# ─────────────────────────────────────────────────────────

class MedicalNetExtractor(nn.Module):
    """
    Extracteur de features MedicalNet pour coupes IRM 2D.

    Pipeline :
      Coupe 2D [B, 1, 128, 128]
          → répliquer sur D=16 → [B, 1, 16, 128, 128]
          → ResNet-50 3D MedicalNet (backbone gelé)
          → [B, 2048]
          → couche de projection linéaire
          → [B, 256]  normalisé L2

    Args:
        weights_path : chemin vers resnet_50_23dataset.pth
        output_dim   : dimension de sortie (256 par défaut)
        depth        : nombre de répétitions de la coupe (16 par défaut)
        freeze       : geler le backbone (True par défaut)
    """

    def __init__(
        self,
        weights_path   : str,
        output_dim     : int  = 256,
        depth          : int  = 16,
        freeze         : bool = True,
        projection_path: str  = None,
    ):
        super().__init__()
        self.depth = depth

        # ── Backbone ResNet-50 3D ─────────────────────────
        self.backbone = ResNet3D()
        self._load_medicalnet_weights(weights_path)

        # ── Couche de projection 2048D → 256D ─────────────
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
        )

        if projection_path and os.path.exists(projection_path):
            self.projection.load_state_dict(
                torch.load(projection_path, map_location="cpu", weights_only=True)
            )
            print(f"[MedicalNet] Poids de la projection chargés : {projection_path}")
        else:
            print("[MedicalNet] Projection non entraînée (poids aléatoires).")

        # Geler le backbone si demandé
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[MedicalNet] Backbone gelé — seule la projection est entraînable.")

    def _load_medicalnet_weights(self, weights_path: str) -> None:
        """
        Charge les poids MedicalNet depuis le fichier .pth.
        Gère les différences de noms de clés entre MedicalNet
        et notre implémentation ResNet3D.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Poids MedicalNet introuvables : {weights_path}\n"
                f"Lance d'abord : python download_medicalnet.py"
            )

        checkpoint = torch.load(weights_path,
                                 map_location="cpu",
                                 weights_only=False)

        # MedicalNet sauvegarde sous la clé 'state_dict'
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Nettoyer le préfixe 'module.' ajouté par DataParallel
        clean_state = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "")
            # Supprimer la couche de classification (fc)
            if not key.startswith("fc"):
                clean_state[key] = v

        missing, unexpected = self.backbone.load_state_dict(
            clean_state, strict=False
        )

        print(f"[MedicalNet] Poids chargés depuis : {weights_path}")
        if missing:
            print(f"  Clés manquantes  : {len(missing)}")
        if unexpected:
            print(f"  Clés inattendues : {len(unexpected)}")
        print("[MedicalNet] Backbone initialisé avec succès.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, 128, 128] — coupe IRM 2D
        Retourne : [B, 256] — embedding normalisé L2
        """
        # Simuler un volume 3D en répétant la coupe D fois
        # [B, 1, 128, 128] → [B, 1, D, 128, 128]
        x3d = x.unsqueeze(2).repeat(1, 1, self.depth, 1, 1)
        with torch.enable_grad():
            features = self.backbone(x3d).detach()  # [B, 2048], coupe le graphe du backbone gelé
            features.requires_grad_(False)
            emb = self.projection(features)         # [B, 256] -- grad actif ici
        return emb

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> list:
        """
        Encode une image [1, 128, 128] → vecteur 256D normalisé L2.
        Interface identique à l'auto-encodeur baseline.
        """
        self.eval()
        img = image_tensor.unsqueeze(0)     # [1, 1, 128, 128]
        emb = self.forward(img).cpu().numpy().astype("float32")

        # Normalisation L2
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb[0].tolist()


# ─────────────────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    WEIGHTS = "./pretrain/resnet_50_23dataset.pth"

    print("Initialisation MedicalNetExtractor...")
    model = MedicalNetExtractor(weights_path=WEIGHTS)
    model.eval()

    # Test avec une coupe aléatoire
    dummy = torch.rand(1, 128, 128)
    vec   = model.extract(dummy)

    print(f"\nTest extraction :")
    print(f"  Input  : [1, 128, 128]")
    print(f"  Output : {len(vec)}D")
    print(f"  Norme  : {np.linalg.norm(vec):.4f} (doit être ≈ 1.0)")
    print(f"  Extrait: [{', '.join([f'{v:.4f}' for v in vec[:5]])}...]")
    print("\nMedicalNetExtractor opérationnel !")