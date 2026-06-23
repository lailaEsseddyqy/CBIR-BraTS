# src/indexation/index_embeddings.py
import os
import sys
import json
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from qdrant_client.models import PointStruct
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.autoencoder import BraTSAutoencoderLightning
from src.db.connections import (
    get_qdrant_client,
    ensure_qdrant_collection,
    get_slices_collection,
    QDRANT_COLLECTION,
)

# ── Chemins depuis .env ───────────────────────────────────
CKPT_PATH  = os.getenv("CHECKPOINT_PATH", "./saved_models/my_brats_model.ckpt")
DATA_DIR   = os.getenv("DATA_DIR",  "./Data/brats_subset_5k")
CACHE_JSON = os.getenv("CACHE_JSON", "./Data/brats_subset_5k_cache.json")
LATENT_DIM = int(os.getenv("LATENT_DIM", 256))

# BUG FIX 1 : 512 causait des timeouts Qdrant → réduit à 50
FLUSH_EVERY = 50


# ── Dataset ───────────────────────────────────────────────
class BraTSDatasetFast(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files    = sorted([
            f for f in os.listdir(data_dir) if f.endswith('.pt')
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        # BUG FIX 2 : on retourne aussi le nom du fichier .pt
        # pour construire le chemin local correct dans MongoDB
        return torch.load(path, weights_only=True), idx, self.files[idx]


# ── Métadonnées depuis le cache ───────────────────────────
def parse_metadata(local_idx: int, cache: list) -> dict:
    """
    local_idx = position dans le sous-cache (0 à N-1).
    Retourne les métadonnées extraites du chemin NIfTI original.
    """
    file_path, slice_z = cache[local_idx]
    parts      = Path(file_path).stem.replace(".nii", "").split("_")
    patient_id = "_".join(parts[:2]) if len(parts) >= 2 else "unknown"
    modalite   = parts[-1]           if len(parts) >= 3 else "unknown"
    return {
        "slice_id"  : f"{patient_id}_{modalite}_z{int(slice_z):03d}",
        "patient_id": patient_id,
        "modalite"  : modalite,
        "slice_z"   : int(slice_z),
        "global_idx": local_idx,
    }


# ── Boucle principale ─────────────────────────────────────
def run_indexation(batch_size: int = 64):
    """
    BUG FIX 3 : num_workers=0 obligatoire sur Windows
    (num_workers > 0 cause des erreurs de multiprocessing)
    """

    # 1. Cache JSON
    with open(CACHE_JSON) as f:
        cache = json.load(f)
    print(f"Cache : {len(cache)} coupes")

    # 2. Modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BraTSAutoencoderLightning.load_from_checkpoint(CKPT_PATH)
    model.eval().to(device)
    print(f"Modele charge — device: {device}")

    # 3. Bases de données
    qdrant = get_qdrant_client()
    ensure_qdrant_collection(qdrant)
    mongo  = get_slices_collection()

    # 4. DataLoader — num_workers=0 pour Windows
    dataset = BraTSDatasetFast(DATA_DIR)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,       # BUG FIX 3 : 0 sur Windows
    )

    # 5. Extraction + insertion
    qdrant_buf, mongo_buf = [], []

    with torch.no_grad():
        for imgs, local_idxs, filenames in tqdm(loader, desc="Indexation"):
            imgs    = imgs.to(device)
            enc     = model.encoder(imgs)
            flat    = model.flatten(enc)
            embs    = model.embedding_layer(flat).cpu().numpy().astype("float32")
            imgs_np = imgs.cpu().numpy()

            for i in range(len(imgs)):
                idx  = local_idxs[i].item()
                meta = parse_metadata(idx, cache)

                # BUG FIX 2 : chemin .pt LOCAL correct
                # (avant : stockait le chemin NIfTI de Lightning Studios)
                local_pt_path = os.path.abspath(
                    os.path.join(DATA_DIR, filenames[i])
                )

                # Normalisation L2
                vec  = embs[i]
                norm = np.linalg.norm(vec)
                vec  = (vec / norm if norm > 0 else vec).tolist()

                # Point Qdrant
                qdrant_buf.append(PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, meta["slice_id"])),
                    vector=vec,
                    payload={
                        "slice_id"  : meta["slice_id"],
                        "patient_id": meta["patient_id"],
                        "modalite"  : meta["modalite"],
                        "slice_z"   : meta["slice_z"],
                    },
                ))

                # Document MongoDB avec chemin .pt LOCAL correct
                img_flat = imgs_np[i, 0]
                mongo_buf.append({
                    "slice_id"  : meta["slice_id"],
                    "patient_id": meta["patient_id"],
                    "modalite"  : meta["modalite"],
                    "slice_z"   : meta["slice_z"],
                    "file_path" : local_pt_path,   # ← chemin local correct
                    "global_idx": meta["global_idx"],
                    "stats": {
                        "mean"       : float(img_flat.mean()),
                        "std"        : float(img_flat.std()),
                        "nonzero_pct": float(
                            np.count_nonzero(img_flat) / img_flat.size
                        ),
                    },
                })

            if len(qdrant_buf) >= FLUSH_EVERY:
                _flush(qdrant, mongo, qdrant_buf, mongo_buf)
                qdrant_buf.clear()
                mongo_buf.clear()

    # Dernier lot
    if qdrant_buf:
        _flush(qdrant, mongo, qdrant_buf, mongo_buf)

    # Rapport final
    qdrant_count = qdrant.count(QDRANT_COLLECTION).count
    mongo_count  = mongo.count_documents({})
    print(f"\nIndexation terminee !")
    print(f"  Qdrant  : {qdrant_count} vecteurs")
    print(f"  MongoDB : {mongo_count} documents")

    # Vérification cohérence
    if qdrant_count != mongo_count:
        print(f"  ATTENTION : Qdrant ({qdrant_count}) != MongoDB ({mongo_count})")
    else:
        print(f"  OK : Qdrant et MongoDB sont synchronises")

    # Vérifier un chemin au hasard
    sample = mongo.find_one({}, {"file_path": 1})
    if sample:
        exists = os.path.exists(sample["file_path"])
        print(f"  Chemin exemple : {sample['file_path']}")
        print(f"  Fichier existe : {'OUI' if exists else 'NON - verifier DATA_DIR'}")


def _flush(qdrant, mongo, qbuf: list, mbuf: list) -> None:
    """Insere un lot dans Qdrant + MongoDB (idempotent)."""
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=qbuf)
    try:
        mongo.insert_many(mbuf, ordered=False)
    except Exception:
        pass  # BulkWriteError attendu pour les doublons (index unique slice_id)


if __name__ == "__main__":
    run_indexation()