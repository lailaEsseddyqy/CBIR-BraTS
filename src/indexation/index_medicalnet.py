# src/indexation/index_medicalnet.py
"""
Indexation avec MedicalNet — collection Qdrant séparée : 'medicalnet_embeddings'
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

from src.models.medicalnet_extractor import MedicalNetExtractor
from src.db.connections import get_qdrant_client, get_slices_collection

# ── Constantes ────────────────────────────────────────────
MEDICALNET_COLLECTION = "medicalnet_embeddings"
WEIGHTS_PATH    = os.getenv("MEDICALNET_WEIGHTS", "./pretrain/resnet_50_23dataset.pth")
PROJECTION_PATH = os.getenv("PROJECTION_OUT", "./pretrain/medicalnet_projection.pth")
DATA_DIR     = os.getenv("DATA_DIR",   "./Data/brats_subset_5k")
CACHE_JSON   = os.getenv("CACHE_JSON", "./Data/brats_subset_5k_cache.json")
LATENT_DIM   = 256
FLUSH_EVERY  = 50


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
        return torch.load(path, weights_only=True), idx


# ── Création collection Qdrant ────────────────────────────
def ensure_medicalnet_collection(client) -> None:
    existing = {c.name for c in client.get_collections().collections}
    
    # Supprimer l'ancienne collection 256D si elle existe
    if MEDICALNET_COLLECTION in existing:
        print(f"[Qdrant] Suppression de l'ancienne collection {MEDICALNET_COLLECTION}...")
        client.delete_collection(collection_name=MEDICALNET_COLLECTION)

    client.create_collection(
        collection_name=MEDICALNET_COLLECTION,
        vectors_config=VectorParams(
            size=LATENT_DIM,
            distance=Distance.COSINE
        ),
    )
    print(f"[Qdrant] Collection '{MEDICALNET_COLLECTION}' créée en {LATENT_DIM}D.")


# ── Flush ─────────────────────────────────────────────────
def _flush(client, buf: list) -> None:
    if buf:
        client.upsert(
            collection_name=MEDICALNET_COLLECTION,
            points=buf
        )


# ── Indexation principale ─────────────────────────────────
def run_indexation_medicalnet(batch_size: int = 8) -> None:
    """
    batch_size=8 car MedicalNet (3D) est plus lourd en mémoire
    que l'auto-encodeur 2D — adapter selon ta RAM disponible.
    """

    # 1. Cache JSON
    with open(CACHE_JSON) as f:
        cache = json.load(f)
    print(f"Cache : {len(cache)} coupes")

    # 2. Modèle MedicalNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MedicalNetExtractor(
        weights_path=WEIGHTS_PATH,
        depth=16,
        freeze=True,
        projection_path=PROJECTION_PATH,
    )
    model.eval().to(device)
    print(f"[MedicalNet] Chargé — device: {device}")

    # 3. Bases de données
    qdrant    = get_qdrant_client()
    ensure_medicalnet_collection(qdrant)
    mongo_col = get_slices_collection()

    # 4. DataLoader
    dataset = BraTSDatasetFast(DATA_DIR)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 5. Extraction + insertion par lots
    qdrant_buf = []

    with torch.no_grad():
        for imgs, local_idxs in tqdm(loader, desc="Indexation MedicalNet"):
            imgs = imgs.to(device)

            # Extraction embeddings MedicalNet
            embs = model(imgs).cpu().numpy().astype("float32")  # [B, 256]

            for i in range(len(imgs)):
                idx       = local_idxs[i].item()
                file_path, slice_z = cache[idx]

                # Métadonnées depuis le cache
                parts      = Path(file_path).stem.replace(".nii", "").split("_")
                patient_id = "_".join(parts[:2]) if len(parts) >= 2 else "unknown"
                modalite   = parts[-1] if len(parts) >= 3 else "unknown"
                slice_id   = f"{patient_id}_{modalite}_z{int(slice_z):03d}"

                # Normalisation L2
                vec  = embs[i]
                norm = np.linalg.norm(vec)
                vec  = (vec / norm if norm > 0 else vec).tolist()

                qdrant_buf.append(PointStruct(
                    id=str(uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"medicalnet_{slice_id}"
                    )),
                    vector=vec,
                    payload={
                        "slice_id"  : slice_id,
                        "patient_id": patient_id,
                        "modalite"  : modalite,
                        "slice_z"   : int(slice_z),
                        "model"     : "medicalnet_resnet50",
                    },
                ))

            if len(qdrant_buf) >= FLUSH_EVERY:
                _flush(qdrant, qdrant_buf)
                qdrant_buf.clear()

    # Dernier lot
    _flush(qdrant, qdrant_buf)

    count = qdrant.count(MEDICALNET_COLLECTION).count
    print(f"\nIndexation MedicalNet terminée !")
    print(f"  Qdrant '{MEDICALNET_COLLECTION}' : {count} vecteurs")


if __name__ == "__main__":
    run_indexation_medicalnet()