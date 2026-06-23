# src/recherche/search_engine.py
import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from dataclasses import dataclass, field
from typing import Optional
from src.evaluation.medical_metrics import compute_all, interpret
from src.db.connections import (
    get_qdrant_client,
    get_slices_collection,
    QDRANT_COLLECTION,
)
from src.models.autoencoder import BraTSAutoencoderLightning
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from dotenv import load_dotenv

load_dotenv()

CKPT_PATH = os.getenv("CHECKPOINT_PATH", "./saved_models/my_brats_model.ckpt")


# ─────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    rank           : int
    score          : float
    slice_id       : str
    patient_id     : str
    modalite       : str
    slice_z        : int
    file_path      : str
    stats          : dict = field(default_factory=dict)
    metrics        : dict = field(default_factory=dict)
    interpretation : str  = ""


@dataclass
class SearchQuery:
    image_tensor       : torch.Tensor   # [1, 128, 128]
    k                  : int            = 10
    modalite           : Optional[str]  = None
    patient_id         : Optional[str]  = None
    slice_z_min        : Optional[int]  = None
    slice_z_max        : Optional[int]  = None
    score_threshold    : float          = 0.0
    exclude_patient_id : Optional[str]  = None   # ← exclure un patient


# ─────────────────────────────────────────────────────────
# Moteur de recherche
# ─────────────────────────────────────────────────────────

class CBMIRSearchEngine:

    def __init__(self, checkpoint_path: str = CKPT_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = BraTSAutoencoderLightning.load_from_checkpoint(checkpoint_path)
        self.model.eval().to(self.device)
        print(f"[SearchEngine] Modele charge — device: {self.device}")

        self.qdrant = get_qdrant_client()
        self.mongo  = get_slices_collection()

    # ── Encodage ──────────────────────────────────────────
    @torch.no_grad()
    def encode(self, image_tensor: torch.Tensor) -> list:
        """Encode une image [1, 128, 128] en vecteur 256D normalise L2."""
        img  = image_tensor.unsqueeze(0).to(self.device)  # [1,1,128,128]
        enc  = self.model.encoder(img)
        flat = self.model.flatten(enc)
        emb  = self.model.embedding_layer(flat).cpu().numpy().astype("float32")
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb[0].tolist()

    # ── Construction du filtre Qdrant ─────────────────────
    @staticmethod
    def _build_filter(query: "SearchQuery") -> Optional[Filter]:
        """
        Construit le filtre Qdrant selon les criteres de la requete.
        Supporte : inclusion (modalite, patient_id, slice_z)
                   exclusion (exclude_patient_id)
        """
        must     = []
        must_not = []

        if query.modalite:
            must.append(
                FieldCondition(key="modalite",
                               match=MatchValue(value=query.modalite))
            )

        if query.patient_id:
            must.append(
                FieldCondition(key="patient_id",
                               match=MatchValue(value=query.patient_id))
            )

        if query.slice_z_min is not None or query.slice_z_max is not None:
            must.append(
                FieldCondition(
                    key="slice_z",
                    range=Range(gte=query.slice_z_min, lte=query.slice_z_max)
                )
            )

        if query.exclude_patient_id:
            must_not.append(
                FieldCondition(key="patient_id",
                               match=MatchValue(value=query.exclude_patient_id))
            )

        if must or must_not:
            return Filter(
                must=must if must else None,
                must_not=must_not if must_not else None,
            )
        return None

    # ── Recherche classique ───────────────────────────────
    def search(self, query: "SearchQuery") -> list:
        """
        Recherche Top-K dans Qdrant + enrichissement MongoDB.
        Retourne des coupes (peut inclure plusieurs coupes
        du meme patient).
        """
        vector = self.encode(query.image_tensor)

        hits = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=query.k,
            query_filter=self._build_filter(query),
            score_threshold=query.score_threshold,
            with_payload=True,
        ).points

        if not hits:
            return []

        slice_ids  = [hit.payload["slice_id"] for hit in hits]
        mongo_docs = {
            doc["slice_id"]: doc
            for doc in self.mongo.find({"slice_id": {"$in": slice_ids}})
        }

        results = []
        for rank, hit in enumerate(hits):
            sid  = hit.payload["slice_id"]
            mdoc = mongo_docs.get(sid, {})
            results.append(SearchResult(
                rank       = rank + 1,
                score      = round(float(hit.score), 4),
                slice_id   = sid,
                patient_id = hit.payload.get("patient_id", ""),
                modalite   = hit.payload.get("modalite", ""),
                slice_z    = hit.payload.get("slice_z", -1),
                file_path  = mdoc.get("file_path", ""),
                stats      = mdoc.get("stats", {}),
            ))
        return results

    # ── Recherche par patients distincts ──────────────────
    def search_similar_patients(self, query: "SearchQuery",
                                 exclude_patient_id: str = None) -> list:
        """
        Recherche CBMIR correcte medicalement :
          1. Recupere Top-50 pour avoir une large marge
          2. Exclut toutes les coupes du meme patient
          3. Garde la meilleure coupe par patient (group by)
          4. Retourne Top-K patients distincts

        C'est la methode principale a utiliser dans l'interface.
        """
        vector = self.encode(query.image_tensor)

        # Etape 1 : Recuperer beaucoup plus que K
        fetch_limit = max(query.k * 10, 50)

        # Construire le filtre avec exclusion du patient
        query_with_exclusion = SearchQuery(
            image_tensor       = query.image_tensor,
            k                  = fetch_limit,
            modalite           = query.modalite,
            score_threshold    = query.score_threshold,
            exclude_patient_id = exclude_patient_id,
        )

        hits = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=fetch_limit,
            query_filter=self._build_filter(query_with_exclusion),
            score_threshold=query.score_threshold,
            with_payload=True,
        ).points

        if not hits:
            return []

        # Etape 2 : Securite supplementaire — exclure le patient
        if exclude_patient_id:
            hits = [
                h for h in hits
                if h.payload.get("patient_id") != exclude_patient_id
            ]

        # Etape 3 : Garder 1 seule meilleure coupe par patient
        best_per_patient = {}
        for hit in hits:
            pid = hit.payload.get("patient_id", "")
            if pid not in best_per_patient:
                best_per_patient[pid] = hit
            # hits deja tries par score decroissant → premier = meilleur

        # Etape 4 : Trier par score et prendre Top-K
        top_hits = sorted(
            best_per_patient.values(),
            key=lambda h: h.score,
            reverse=True
        )[:query.k]

        # Etape 5 : Enrichissement MongoDB
        slice_ids  = [h.payload["slice_id"] for h in top_hits]
        mongo_docs = {
            doc["slice_id"]: doc
            for doc in self.mongo.find({"slice_id": {"$in": slice_ids}})
        }

        # Etape 6 : Assembler les resultats
        results = []
        for rank, hit in enumerate(top_hits):
            sid  = hit.payload["slice_id"]
            mdoc = mongo_docs.get(sid, {})
            results.append(SearchResult(
                rank       = rank + 1,
                score      = round(float(hit.score), 4),
                slice_id   = sid,
                patient_id = hit.payload.get("patient_id", ""),
                modalite   = hit.payload.get("modalite", ""),
                slice_z    = hit.payload.get("slice_z", -1),
                file_path  = mdoc.get("file_path", ""),
                stats      = mdoc.get("stats", {}),
            ))
        return results

    # ── Statistiques ──────────────────────────────────────
    def stats(self) -> dict:
        """Statistiques globales de l'index."""
        return {
            "qdrant_vectors" : self.qdrant.count(QDRANT_COLLECTION).count,
            "mongo_documents": self.mongo.count_documents({}),
            "modalites"      : self.mongo.distinct("modalite"),
            "n_patients"     : len(self.mongo.distinct("patient_id")),
        }


# ─────────────────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = CBMIRSearchEngine()

    print("\n── Statistiques ──")
    for k, v in engine.stats().items():
        print(f"  {k}: {v}")

    print("\n── Test recherche classique (image aleatoire) ──")
    dummy   = torch.rand(1, 128, 128)
    query   = SearchQuery(image_tensor=dummy, k=5)
    results = engine.search(query)
    for r in results:
        print(f"  Rang {r.rank} | score={r.score} | {r.slice_id}")

    print("\n── Test patients distincts ──")
    results2 = engine.search_similar_patients(
        query=SearchQuery(image_tensor=dummy, k=5),
        exclude_patient_id=results[0].patient_id if results else None,
    )
    patients = [r.patient_id for r in results2]
    print(f"  Patients uniques : {len(set(patients))}/{len(results2)}")
    for r in results2:
        print(f"  Rang {r.rank} | {r.patient_id} | score={r.score}")