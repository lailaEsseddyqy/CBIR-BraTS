# src/training/train_radimagenet_projection.py
"""
Entraînement de la couche de projection (2048D -> 256D) de RadImageNetExtractor.

Stratégie : Triplet Loss
  - Ancre / Positif  : deux coupes proches en Z (même patient+modalité)
  - Négatif          : une coupe d'un autre patient/modalité

Le backbone reste gelé. Seuls les poids de `model.projection` sont mis à jour.
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

# ── Constantes ────────────────────────────────────────────
WEIGHTS_DIR    = os.getenv("RADIMAGENET_WEIGHTS_DIR", "./pretrain")
DATA_DIR       = os.getenv("DATA_DIR", "./Data/brats_subset_5k")
CACHE_JSON     = os.getenv("CACHE_JSON", "./Data/brats_subset_5k_cache.json")
OUTPUT_PATH    = os.getenv("RADIMAGENET_PROJECTION", "./pretrain/radimagenet_projection.pth")

BATCH_SIZE     = 16
EPOCHS         = 10
LR             = 1e-4
MARGIN         = 0.3
Z_NEIGHBOR_MAX = 2


# ── Dataset triplet (identique à la version MedicalNet) ────
class TripletSliceDataset(Dataset):
    def __init__(self, data_dir, cache_json):
        self.data_dir = data_dir

        with open(cache_json) as f:
            self.cache = json.load(f)

        self.groups = {}
        for idx, (file_path, slice_z) in enumerate(self.cache):
            parts      = Path(file_path).stem.replace(".nii", "").split("_")
            patient_id = "_".join(parts[:2]) if len(parts) >= 2 else "unknown"
            modalite   = parts[-1] if len(parts) >= 3 else "unknown"
            key = (patient_id, modalite)
            self.groups.setdefault(key, []).append((idx, int(slice_z)))

        self.groups = {k: v for k, v in self.groups.items() if len(v) >= 2}
        self.group_keys = list(self.groups.keys())

        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])

        assert len(self.files) == len(self.cache), (
            f"Incohérence cache/fichiers : {len(self.files)} fichiers vs "
            f"{len(self.cache)} entrées de cache"
        )

    def __len__(self):
        return sum(len(v) for v in self.groups.values())

    def _load(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        return torch.load(path, weights_only=True)

    def __getitem__(self, _):
        key = random.choice(self.group_keys)
        items = self.groups[key]

        anchor_idx, anchor_z = random.choice(items)

        candidates = [
            (i, z) for (i, z) in items
            if i != anchor_idx and abs(z - anchor_z) <= Z_NEIGHBOR_MAX
        ]
        if not candidates:
            candidates = [(i, z) for (i, z) in items if i != anchor_idx]
        pos_idx, _ = random.choice(candidates)

        neg_key = random.choice(self.group_keys)
        tries = 0
        while neg_key == key and tries < 10:
            neg_key = random.choice(self.group_keys)
            tries += 1
        neg_idx, _ = random.choice(self.groups[neg_key])

        anchor = self._load(anchor_idx)
        pos    = self._load(pos_idx)
        neg    = self._load(neg_idx)

        return anchor, pos, neg


# ── Entraînement ────────────────────────────────────────────
def train_projection(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Chargement RadImageNetExtractor (backbone gelé)...")
    model = RadImageNetExtractor(weights_dir=WEIGHTS_DIR, freeze=True)
    model.to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.projection.parameters():
        p.requires_grad = True

    dataset = TripletSliceDataset(DATA_DIR, CACHE_JSON)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    triplet_loss = nn.TripletMarginLoss(margin=MARGIN, p=2)
    optimizer    = torch.optim.Adam(model.projection.parameters(), lr=lr)

    model.backbone.eval()

    for epoch in range(1, epochs + 1):
        model.projection.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for anchor, pos, neg in pbar:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            optimizer.zero_grad()

            emb_a = model(anchor)
            emb_p = model(pos)
            emb_n = model(neg)

            emb_a = F.normalize(emb_a, p=2, dim=1)
            emb_p = F.normalize(emb_p, p=2, dim=1)
            emb_n = F.normalize(emb_n, p=2, dim=1)

            loss = triplet_loss(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{epochs} — loss moyenne : {avg_loss:.4f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(model.projection.state_dict(), OUTPUT_PATH)
    print(f"\nPoids de la projection sauvegardés : {OUTPUT_PATH}")


if __name__ == "__main__":
    train_projection()