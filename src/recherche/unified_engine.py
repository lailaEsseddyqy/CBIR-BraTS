# src/recherche/unified_engine.py
"""
Moteur unifié CBMIR — 3 configurations :
  - "baseline" : Auto-encodeur BraTS 2021 (Non supervisé)
  - "supcon"   : Auto-encodeur entraîné avec Supervised Contrastive Loss
  - "guided"   : Classification-Guided Retrieval (SupCon + filtre grade prédit)
"""

import os, sys, uuid, torch, numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.autoencoder import BraTSAutoencoderLightning
from src.models.autoencoder_supervised import BraTSAutoencoderSupervised
from src.models.brats_classifier import BraTSClassifierGuided
from src.training.grade_constants import IDX_TO_GRADE
from src.db.connections import (
    get_qdrant_client,
    get_slices_collection,
    QDRANT_COLLECTION,
)
from qdrant_client.models import Filter, FieldCondition, MatchValue, HasIdCondition
from dotenv import load_dotenv

load_dotenv()

CKPT_PATH = os.getenv("CHECKPOINT_PATH")
CKPT_PATH_SUPCON = os.getenv("CHECKPOINT_PATH_SUPCON", "./saved_models/brats_supcon_best.ckpt")
CLASSIFIER_CKPT  = os.getenv("CLASSIFIER_CKPT", "./saved_models/brats_guided_classifier-v1.ckpt")
SUPCON_COLL      = "brats_supcon_embeddings"


@dataclass
class UnifiedResult:
    rank           : int
    score          : float
    score_baseline : float = 0.0
    score_supcon   : float = 0.0
    score_guided   : float = 0.0
    slice_id       : str   = ""
    patient_id     : str   = ""
    modalite       : str   = ""
    slice_z        : int   = -1
    file_path      : str   = ""
    stats          : dict  = field(default_factory=dict)
    metrics        : dict  = field(default_factory=dict)
    interpretation : str   = ""
    model_used     : str   = ""
    grade          : str   = ""
    predicted_grade: str   = ""


class UnifiedSearchEngine:
    def __init__(self):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qdrant  = get_qdrant_client()
        self.mongo   = get_slices_collection()

        self.baseline = BraTSAutoencoderLightning.load_from_checkpoint(CKPT_PATH)
        self.baseline.eval().to(self.device)
        print(f"[Baseline] Chargé — device: {self.device}")

        self._supcon = None
        self._classifier = None
        self._last_guided_info: dict = {}

    @property
    def supcon(self) -> BraTSAutoencoderSupervised:
        if self._supcon is None:
            self._supcon = BraTSAutoencoderSupervised.load_from_checkpoint(CKPT_PATH_SUPCON)
            self._supcon.eval().to(self.device)
            print(f"[SupCon] Chargé — device: {self.device}")
        return self._supcon

    @property
    def classifier(self) -> BraTSClassifierGuided:
        if self._classifier is None:
            if not os.path.exists(CLASSIFIER_CKPT):
                raise FileNotFoundError(
                    f"Checkpoint classifieur introuvable : {CLASSIFIER_CKPT}\n"
                    "Définissez CLASSIFIER_CKPT dans .env"
                )
            self._classifier = BraTSClassifierGuided.load_from_checkpoint(CLASSIFIER_CKPT)
            self._classifier.eval().to(self.device)
            print(f"[CGR] Classifieur chargé — device: {self.device}")
        return self._classifier

    @property
    def last_guided_info(self) -> dict:
        return self._last_guided_info

    @torch.no_grad()
    def encode_baseline(self, tensor: torch.Tensor) -> list:
        img  = tensor.unsqueeze(0).to(self.device)
        enc  = self.baseline.encoder(img)
        flat = self.baseline.flatten(enc)
        emb  = self.baseline.embedding_layer(flat).cpu().numpy().astype("float32")
        norm = np.linalg.norm(emb)
        return (emb / norm if norm > 0 else emb)[0].tolist()

    @torch.no_grad()
    def encode_supcon(self, tensor: torch.Tensor) -> list:
        img = tensor.unsqueeze(0).to(self.device)
        _, emb_n = self.supcon(img)
        emb_n = emb_n.cpu().numpy().astype("float32")
        return emb_n[0].tolist()

    @torch.no_grad()
    def predict_grade(self, tensor: torch.Tensor) -> dict:
        img = tensor.unsqueeze(0).to(self.device)
        embedding, pred_idx, probs = self.classifier.predict_grade(img)
        idx = int(pred_idx.item())
        return {
            "predicted_grade": IDX_TO_GRADE.get(idx, "Inconnu"),
            "confidence"     : round(float(probs[0, idx].item()), 4),
            "vector"         : embedding.cpu().numpy().astype("float32")[0].tolist(),
        }

    def _slice_ids_for_grade(self, grade: str, modalite: Optional[str] = None) -> list:
        query = {"patient.grade": grade}
        if modalite:
            query["modalite"] = modalite
        return [doc["slice_id"] for doc in self.mongo.find(query, {"slice_id": 1})]

    def _search_collection_with_ids(
        self, collection: str, vector: list, eligible_ids: list,
        k: int, exclude_patient_id: Optional[str] = None,
        score_threshold: float = 0.0, namespace: str = "supcon_",
    ) -> dict:
        fetch_limit = max(k * 10, 50)
        if not eligible_ids:
            return {}

        max_ids = 1000
        if len(eligible_ids) > max_ids:
            eligible_ids = eligible_ids[:max_ids]

        qdrant_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}{sid}"))
            for sid in eligible_ids
        ]
        qdrant_filter = Filter(must=[HasIdCondition(has_id=qdrant_ids)])

        hits = self.qdrant.query_points(
            collection_name=collection, query=vector, limit=fetch_limit,
            query_filter=qdrant_filter, score_threshold=score_threshold,
            with_payload=True,
        ).points

        if exclude_patient_id:
            hits = [h for h in hits if h.payload.get("patient_id") != exclude_patient_id]

        best = {}
        for h in hits:
            pid = h.payload.get("patient_id", "")
            if pid not in best:
                best[pid] = h
        return best

    @staticmethod
    def _build_filter(modalite: Optional[str] = None, exclude_patient_id: Optional[str] = None) -> Optional[Filter]:
        must, must_not = [], []
        if modalite:
            must.append(FieldCondition(key="modalite", match=MatchValue(value=modalite)))
        if exclude_patient_id:
            must_not.append(FieldCondition(key="patient_id", match=MatchValue(value=exclude_patient_id)))
        if must or must_not:
            return Filter(must=must if must else None, must_not=must_not if must_not else None)
        return None

    def _search_collection(self, collection: str, vector: list, k: int, modalite: Optional[str] = None, exclude_patient_id: Optional[str] = None, score_threshold: float = 0.0) -> dict:
        fetch_limit = max(k * 10, 50)
        hits = self.qdrant.query_points(
            collection_name=collection, query=vector, limit=fetch_limit,
            query_filter=self._build_filter(modalite, exclude_patient_id),
            score_threshold=score_threshold, with_payload=True,
        ).points
        best = {}
        for h in hits:
            pid = h.payload.get("patient_id", "")
            if pid not in best:
                best[pid] = h
        return best

    def search(self, image_tensor: torch.Tensor, model: str = "baseline", k: int = 5, modalite: Optional[str] = None, exclude_patient_id: Optional[str] = None, score_threshold: float = 0.0) -> list:
        if model == "baseline":
            self._last_guided_info = {}
            return self._search_baseline(image_tensor, k, modalite, exclude_patient_id, score_threshold)
        elif model == "supcon":
            self._last_guided_info = {}
            return self._search_supcon(image_tensor, k, modalite, exclude_patient_id, score_threshold)
        elif model == "guided":
            return self._search_guided(image_tensor, k, modalite, exclude_patient_id, score_threshold)
        else:
            raise ValueError(f"Modèle inconnu : {model} — valeurs acceptées : baseline, supcon, guided")

    def _search_baseline(self, tensor, k, modalite, exclude_patient_id, score_threshold) -> list:
        vec  = self.encode_baseline(tensor)
        best = self._search_collection(QDRANT_COLLECTION, vec, k, modalite, exclude_patient_id, score_threshold)
        top  = sorted(best.values(), key=lambda h: h.score, reverse=True)[:k]
        return self._build_results(top, "baseline")

    def _search_supcon(self, tensor, k, modalite, exclude_patient_id, score_threshold) -> list:
        vec  = self.encode_supcon(tensor)
        best = self._search_collection(SUPCON_COLL, vec, k, modalite, exclude_patient_id, score_threshold)
        top  = sorted(best.values(), key=lambda h: h.score, reverse=True)[:k]
        return self._build_results(top, "supcon")

    def _search_guided(self, tensor, k, modalite, exclude_patient_id, score_threshold) -> list:
        grade_info = self.predict_grade(tensor)
        self._last_guided_info = {
            "predicted_grade": grade_info["predicted_grade"],
            "confidence"     : grade_info["confidence"],
        }
        eligible = self._slice_ids_for_grade(grade_info["predicted_grade"], modalite)
        best = self._search_collection_with_ids(
            SUPCON_COLL, grade_info["vector"], eligible, k,
            exclude_patient_id, score_threshold,
        )
        top = sorted(best.values(), key=lambda h: h.score, reverse=True)[:k]
        return self._build_results(
            top, "guided", predicted_grade=grade_info["predicted_grade"],
        )

    def _build_results(self, top_hits: list, model: str, predicted_grade: str = "") -> list:
        slice_ids  = [h.payload["slice_id"] for h in top_hits]
        mongo_docs = {doc["slice_id"]: doc for doc in self.mongo.find({"slice_id": {"$in": slice_ids}})}

        results = []
        for rank, hit in enumerate(top_hits):
            sid  = hit.payload["slice_id"]
            mdoc = mongo_docs.get(sid, {})
            score = round(float(hit.score), 4)
            results.append(UnifiedResult(
                rank           = rank + 1,
                score          = score,
                score_baseline = score if model == "baseline" else 0.0,
                score_supcon   = score if model == "supcon" else 0.0,
                score_guided   = score if model == "guided" else 0.0,
                slice_id       = sid,
                patient_id     = hit.payload.get("patient_id", ""),
                modalite       = hit.payload.get("modalite", ""),
                slice_z        = hit.payload.get("slice_z", -1),
                file_path      = mdoc.get("file_path", ""),
                stats          = mdoc.get("stats", {}),
                model_used     = model,
                grade          = mdoc.get("patient", {}).get("grade", ""),
                predicted_grade= predicted_grade,
            ))
        return results

    def stats(self) -> dict:
        counts = {}
        for name in [QDRANT_COLLECTION, SUPCON_COLL]:
            try:
                counts[name] = self.qdrant.count(name).count
            except Exception:
                counts[name] = 0
        return {
            "baseline_vectors": counts.get(QDRANT_COLLECTION, 0),
            "supcon_vectors"  : counts.get(SUPCON_COLL, 0),
            "mongo_documents" : self.mongo.count_documents({}),
            "modalites"       : self.mongo.distinct("modalite"),
            "n_patients"      : len(self.mongo.distinct("patient_id")),
        }
