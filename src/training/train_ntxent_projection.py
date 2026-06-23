# src/training/train_ntxent_projection.py
"""
Entraînement de la projection RadImageNet avec NT-Xent Loss (SimCLR).

Pourquoi NT-Xent plutôt que Triplet Loss ?
  - Triplet Loss : collapse fréquent (loss = margin en permanence)
  - NT-Xent : contrastif sur tout le batch → moins sujet au collapse
  - Chaque image est augmentée deux fois → la paire (aug1, aug2) est positive
  - Toutes les autres paires du batch sont négatives
  - La température τ contrôle la "dureté" des négatifs

Architecture : backbone RadImageNet (2048D) → MLP projection (2048 → 256D)
               → normalisation L2 → NT-Xent
Le backbone reste gelé. Seul le MLP de projection est entraîné.
Après entraînement, la projection est sauvegardée séparément.
"""

import os, sys, json, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.radimagenet_extractor import RadImageNetExtractor

# ── Hyperparamètres ───────────────────────────────────────
WEIGHTS_DIR     = os.getenv("RADIMAGENET_WEIGHTS_DIR", "./pretrain")
DATA_DIR        = os.getenv("DATA_DIR", "./Data/brats_subset_5k")
CACHE_JSON      = os.getenv("CACHE_JSON", "./Data/brats_subset_5k_cache.json")
OUTPUT_PATH     = os.getenv("RADIMAGENET_PROJECTION", "./pretrain/radimagenet_projection_ntxent.pth")

BATCH_SIZE      = 32      # NT-Xent bénéficie de grands batches (au moins 32)
EPOCHS          = 15
LR              = 3e-4
TEMPERATURE     = 0.07    # τ — standard SimCLR, descendre à 0.05 si collapse
OUTPUT_DIM      = 256     # dimension de sortie de la projection


# ── Augmentations ─────────────────────────────────────────
def augment(tensor: torch.Tensor) -> torch.Tensor:
    """
    Augmentations légères adaptées aux IRM (pas de color jitter fort).
    tensor : [1, 128, 128]
    """
    x = tensor.clone()

    # 1. Flip horizontal aléatoire
    if random.random() > 0.5:
        x = torch.flip(x, dims=[-1])

    # 2. Flip vertical aléatoire
    if random.random() > 0.5:
        x = torch.flip(x, dims=[-2])

    # 3. Bruit gaussien léger (σ ≤ 0.02 — imperceptible sur IRM)
    if random.random() > 0.5:
        x = x + torch.randn_like(x) * 0.015

    # 4. Variation de contraste légère [0.85, 1.15]
    if random.random() > 0.5:
        factor = 0.85 + random.random() * 0.30
        x = (x * factor).clamp(0, 1)

    return x


# ── Dataset ───────────────────────────────────────────────
class AugPairDataset(Dataset):
    """
    Retourne deux vues augmentées de la même coupe (paire positive).
    Pas besoin de labels : les paires sont construites par augmentation.
    """
    def __init__(self, data_dir: str, cache_json: str):
        self.data_dir = data_dir
        with open(cache_json) as f:
            self.cache = json.load(f)
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])
        assert len(self.files) == len(self.cache)

        # Filtrer les coupes invalides avec is_valid_slice
        from src.evaluation.medical_metrics import is_valid_slice
        valid = []
        for i, fname in enumerate(self.files):
            t = torch.load(os.path.join(data_dir, fname), weights_only=True)
            if is_valid_slice(t.squeeze().numpy()):
                valid.append(i)
        self.valid_indices = valid
        print(f"[NT-Xent Dataset] {len(valid)}/{len(self.files)} coupes valides")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        file_idx = self.valid_indices[idx]
        path     = os.path.join(self.data_dir, self.files[file_idx])
        t        = torch.load(path, weights_only=True)          # [1, 128, 128]
        return augment(t), augment(t)                           # deux vues


# ── NT-Xent Loss ──────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) — SimCLR.
    Pour un batch de N paires, on a 2N images.
    Chaque image a 1 positif (son augmentation sœur) et 2N-2 négatifs.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2 : [N, D] — embeddings normalisés L2 des deux vues
        """
        N   = z1.size(0)
        z   = torch.cat([z1, z2], dim=0)                # [2N, D]
        z   = F.normalize(z, p=2, dim=1)

        # Matrice de similarité cosinus [2N, 2N]
        sim = torch.mm(z, z.T) / self.tau

        # Masque diagonal : exclure sim(i,i) (même image avec elle-même)
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim  = sim.masked_fill(mask, float("-inf"))

        # Labels positifs : z1[i] est positif avec z2[i] → index N+i et i
        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N,     device=z.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss


# ── MLP de projection ─────────────────────────────────────
def build_projection_head(input_dim: int = 2048,
                           hidden_dim: int = 512,
                           output_dim: int = 256) -> nn.Module:
    """
    MLP 2 couches : 2048 → 512 → 256
    BatchNorm sur la couche cachée (stabilise NT-Xent).
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


# ── Entraînement ──────────────────────────────────────────
def train_ntxent(
    epochs    : int   = EPOCHS,
    batch_size: int   = BATCH_SIZE,
    lr        : float = LR,
    temperature: float = TEMPERATURE,
    output_dim: int   = OUTPUT_DIM,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device} | τ={temperature} | batch={batch_size}")

    # Backbone RadImageNet gelé
    backbone = RadImageNetExtractor(weights_dir=WEIGHTS_DIR, freeze=True)
    backbone.eval().to(device)

    # MLP de projection entraînable
    projection = build_projection_head(
        input_dim=2048, hidden_dim=512, output_dim=output_dim
    ).to(device)

    dataset = AugPairDataset(DATA_DIR, CACHE_JSON)
    loader  = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, drop_last=True
    )

    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.Adam(projection.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    for epoch in range(1, epochs + 1):
        projection.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for aug1, aug2 in pbar:
            aug1, aug2 = aug1.to(device), aug2.to(device)

            # Features 2048D brutes (backbone gelé)
            with torch.no_grad():
                f1 = backbone.forward(aug1)   # [B, 2048] normalisé L2
                f2 = backbone.forward(aug2)

            # Projection 2048 → 256
            z1 = projection(f1)
            z2 = projection(f2)

            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                              lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        avg = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{epochs} — loss : {avg:.4f} "
              f"| lr : {scheduler.get_last_lr()[0]:.2e}")

    # Sauvegarde
    os.makedirs(os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else ".", exist_ok=True)
    torch.save(projection.state_dict(), OUTPUT_PATH)
    print(f"\nProjection NT-Xent sauvegardée : {OUTPUT_PATH}")
    print("→ Relancer index_radimagenet.py pour ré-indexer avec ce modèle.")


if __name__ == "__main__":
    train_ntxent()