# src/indexation/index_supcon.py
"""
Indexation avec le modèle SupCon (apprentissage métrique supervisé)
— collection Qdrant séparée : 'brats_supcon_embeddings'

Contrairement au Baseline classique, ce modèle a été entraîné
avec une supervision sur le grade tumoral (SupConLoss), en plus
de la reconstruction. L'espace latent résultant est organisé
par pertinence clinique plutôt que par seule apparence visuelle.
"""

import os, sys, json, uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from qdrant_client.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.autoencoder_supervised import BraTSAutoencoderSupervised
from src.db.connections import get_qdrant_client, get_slices_collection

# ── Constantes ────────────────────────────────────────────
SUPCON_COLLECTION = "brats_supcon_embeddings"
CKPT_PATH   = os.getenv("CHECKPOINT_PATH_SUPCON", "./saved_models/brats_supcon_best.ckpt")
DATA_DIR    = os.getenv("DATA_DIR",   "./Data/brats_subset_5k")
CACHE_JSON  = os.getenv("CACHE_JSON", "./Data/brats_subset_5k_cache.json")
LATENT_DIM  = int(os.getenv("LATENT_DIM", 256))
FLUSH_EVERY = 50


# ── Dataset ───────────────────────────────────────────────
# ── Dataset ───────────────────────────────────────────────
class BraTSDatasetFast(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files    = sorted([
            f for f in os.listdir(data_dir) if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        # 1. On charge et on passe en float
        img = torch.load(path, weights_only=True).float()
        
        # 2. LA NORMALISATION OBLIGATOIRE (comme à l'entraînement)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min: 
            img = (img - img_min) / (img_max - img_min)
            
        return img, idx


# ── Création collection Qdrant ────────────────────────────
def ensure_supcon_collection(client) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if SUPCON_COLLECTION not in existing:
        client.create_collection(
            collection_name=SUPCON_COLLECTION,
            vectors_config=VectorParams(size=LATENT_DIM, distance=Distance.COSINE),
        )
        print(f"[Qdrant] Collection '{SUPCON_COLLECTION}' créée ({LATENT_DIM}D).")
    else:
        print(f"[Qdrant] Collection '{SUPCON_COLLECTION}' déjà existante.")


# ── Flush ─────────────────────────────────────────────────
def _flush(client, buf: list) -> None:
    if buf:
        client.upsert(collection_name=SUPCON_COLLECTION, points=buf)


# ── Métadonnées depuis le cache ───────────────────────────
def parse_metadata(local_idx: int, cache: list) -> dict:
    file_path, slice_z = cache[local_idx]
    parts      = Path(file_path).stem.replace(".nii", "").split("_")
    patient_id = "_".join(parts[:2]) if len(parts) >= 2 else "unknown"
    modalite   = parts[-1] if len(parts) >= 3 else "unknown"
    return {
        "slice_id"  : f"{patient_id}_{modalite}_z{int(slice_z):03d}",
        "patient_id": patient_id,
        "modalite"  : modalite,
        "slice_z"   : int(slice_z),
    }


# ── Indexation principale ─────────────────────────────────
def run_indexation_supcon(batch_size: int = 64):

    # 1. Cache JSON
    with open(CACHE_JSON) as f:
        cache = json.load(f)
    print(f"Cache : {len(cache)} coupes")

    # 2. Modèle SupCon
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BraTSAutoencoderSupervised.load_from_checkpoint(CKPT_PATH)
    model.eval().to(device)
    print(f"[SupCon] Modèle chargé — device: {device}")

    # 3. Bases de données
    qdrant = get_qdrant_client()
    ensure_supcon_collection(qdrant)
    mongo  = get_slices_collection()

    # 4. DataLoader (num_workers=0 — obligatoire sur Windows)
    dataset = BraTSDatasetFast(DATA_DIR)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0)

    # 5. Extraction + insertion par lots
    qdrant_buf = []

    with torch.no_grad():
        for imgs, local_idxs in tqdm(loader, desc="Indexation SupCon"):
            imgs = imgs.to(device)

            # forward() retourne (recon, emb_n) — emb_n déjà normalisé L2
            _, embs = model(imgs)
            embs = embs.cpu().numpy().astype("float32")

            for i in range(len(imgs)):
                idx  = local_idxs[i].item()
                meta = parse_metadata(idx, cache)

                qdrant_buf.append(PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"supcon_{meta['slice_id']}")),
                    vector=embs[i].tolist(),
                    payload={
                        "slice_id"  : meta["slice_id"],
                        "patient_id": meta["patient_id"],
                        "modalite"  : meta["modalite"],
                        "slice_z"   : meta["slice_z"],
                        "model"     : "baseline_supcon",
                    },
                ))

            if len(qdrant_buf) >= FLUSH_EVERY:
                _flush(qdrant, qdrant_buf)
                qdrant_buf.clear()

    _flush(qdrant, qdrant_buf)

    count = qdrant.count(SUPCON_COLLECTION).count
    print(f"\nIndexation SupCon terminée !")
    print(f"  Qdrant '{SUPCON_COLLECTION}' : {count} vecteurs")


if __name__ == "__main__":
    run_indexation_supcon()