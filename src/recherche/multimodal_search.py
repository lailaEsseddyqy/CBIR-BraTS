# src/recherche/multimodal_search.py
"""
Recherche multimodale CBMIR — 3 modes :
  - "baseline" : Auto-encodeur BraTS 2021 (collection brats_embeddings)
  - "supcon"   : Auto-encodeur Supervised Contrastive (brats_supcon_embeddings)
  - "guided"   : Classification-Guided Retrieval (SupCon + filtre grade prédit)

Combine la similarité visuelle (Qdrant) avec les filtres cliniques (MongoDB).
"""

import os, sys, uuid
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.db.connections import (
    get_qdrant_client,
    get_slices_collection,
    QDRANT_COLLECTION,
)
from src.models.autoencoder import BraTSAutoencoderLightning
from src.models.autoencoder_supervised import BraTSAutoencoderSupervised
from src.models.brats_classifier import BraTSClassifierGuided
from src.training.grade_constants import IDX_TO_GRADE
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    Range, HasIdCondition
)
from dotenv import load_dotenv

load_dotenv()

CKPT_PATH        = os.getenv("CHECKPOINT_PATH", "./saved_models/my_brats_model.ckpt")
CKPT_PATH_SUPCON = os.getenv("CHECKPOINT_PATH_SUPCON", "./saved_models/brats_supcon_best.ckpt")
CLASSIFIER_CKPT  = os.getenv("CLASSIFIER_CKPT", "./saved_models/brats_guided_classifier-v1.ckpt")
SUPCON_COLL      = "brats_supcon_embeddings"


@dataclass
class PatientFilter:
    sexe       : Optional[str] = None
    age_min    : Optional[int] = None
    age_max    : Optional[int] = None
    modalite   : Optional[str] = None
    hopital    : Optional[str] = None
    diagnostic : Optional[str] = None
    grade      : Optional[str] = None
    annee_min  : Optional[int] = None
    annee_max  : Optional[int] = None


@dataclass
class MultimodalQuery:
    image_tensor       : torch.Tensor
    k                  : int           = 10
    model              : str           = "baseline"   # "baseline" | "supcon" | "guided"
    patient_filter     : PatientFilter = field(default_factory=PatientFilter)
    score_threshold    : float         = 0.0
    exclude_patient_id : Optional[str] = None


@dataclass
class MultimodalResult:
    rank         : int
    score        : float
    score_baseline: float = 0.0
    model_used   : str   = ""
    slice_id     : str   = ""
    patient_id   : str   = ""
    modalite     : str   = ""
    slice_z      : int   = -1
    file_path    : str   = ""
    sexe         : str   = ""
    age          : int   = 0
    hopital      : str   = ""
    diagnostic   : str   = ""
    machine_irm  : str   = ""
    annee_exam   : int   = 0
    antecedents  : str   = ""
    grade        : str   = ""


class MultimodalSearchEngine:

    def __init__(self, checkpoint_path: str = CKPT_PATH):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = BraTSAutoencoderLightning.load_from_checkpoint(checkpoint_path)
        self.model.eval().to(self.device)

        self.qdrant = get_qdrant_client()
        self.mongo  = get_slices_collection()

        self._supcon = None
        self._classifier = None
        self._last_guided_info: dict = {}

        print(f"[Multimodal] Moteur chargé — device: {self.device}")

    @property
    def supcon(self) -> BraTSAutoencoderSupervised:
        if self._supcon is None:
            self._supcon = BraTSAutoencoderSupervised.load_from_checkpoint(CKPT_PATH_SUPCON)
            self._supcon.eval().to(self.device)
            print(f"[Multimodal] SupCon chargé — device: {self.device}")
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
            print(f"[Multimodal] Classifieur CGR chargé — device: {self.device}")
        return self._classifier

    @property
    def last_guided_info(self) -> dict:
        return self._last_guided_info

    @torch.no_grad()
    def encode_baseline(self, image_tensor: torch.Tensor) -> list:
        img  = image_tensor.unsqueeze(0).to(self.device)
        enc  = self.model.encoder(img)
        flat = self.model.flatten(enc)
        emb  = self.model.embedding_layer(flat).cpu().numpy().astype("float32")
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb[0].tolist()

    @torch.no_grad()
    def encode_supcon(self, image_tensor: torch.Tensor) -> list:
        img = image_tensor.unsqueeze(0).to(self.device)
        _, emb_n = self.supcon(img)
        return emb_n.cpu().numpy().astype("float32")[0].tolist()

    @torch.no_grad()
    def predict_grade(self, image_tensor: torch.Tensor) -> dict:
        img = image_tensor.unsqueeze(0).to(self.device)
        embedding, pred_idx, probs = self.classifier.predict_grade(img)
        idx = int(pred_idx.item())
        return {
            "predicted_grade": IDX_TO_GRADE.get(idx, "Inconnu"),
            "confidence"     : round(float(probs[0, idx].item()), 4),
            "probs"          : {
                IDX_TO_GRADE[i]: round(float(probs[0, i].item()), 4)
                for i in range(probs.shape[1])
            },
            "vector"         : embedding.cpu().numpy().astype("float32")[0].tolist(),
        }

    def _filter_by_patient(self, pf: PatientFilter) -> list:
        mongo_query = {}

        if pf.sexe:
            mongo_query["patient.sexe"] = pf.sexe

        if pf.age_min is not None or pf.age_max is not None:
            age_q = {}
            if pf.age_min is not None: age_q["$gte"] = pf.age_min
            if pf.age_max is not None: age_q["$lte"] = pf.age_max
            mongo_query["patient.age"] = age_q

        if pf.modalite:
            mongo_query["modalite"] = pf.modalite

        if pf.hopital:
            mongo_query["patient.hopital"] = pf.hopital

        if pf.diagnostic:
            mongo_query["patient.diagnostic"] = {
                "$regex": pf.diagnostic, "$options": "i"
            }

        if pf.grade:
            mongo_query["patient.grade"] = pf.grade

        if pf.annee_min is not None or pf.annee_max is not None:
            annee_q = {}
            if pf.annee_min is not None: annee_q["$gte"] = pf.annee_min
            if pf.annee_max is not None: annee_q["$lte"] = pf.annee_max
            mongo_query["patient.annee_exam"] = annee_q

        docs = self.mongo.find(mongo_query, {"slice_id": 1})
        return [doc["slice_id"] for doc in docs]

    def _search_qdrant_with_ids(
        self,
        collection      : str,
        vector          : list,
        eligible_ids    : list,
        fetch_limit     : int,
        score_threshold : float,
        namespace       : str = "",
    ) -> list:
        MAX_IDS = 1000
        if len(eligible_ids) > MAX_IDS:
            eligible_ids = eligible_ids[:MAX_IDS]

        qdrant_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}{sid}"))
            for sid in eligible_ids
        ]

        qdrant_filter = Filter(
            must=[HasIdCondition(has_id=qdrant_ids)]
        )

        return self.qdrant.query_points(
            collection_name=collection,
            query=vector,
            limit=fetch_limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

    def _search_qdrant_direct(
        self,
        collection      : str,
        vector          : list,
        fetch_limit     : int,
        score_threshold : float,
        modalite_filter : Optional[str] = None,
    ) -> list:
        qdrant_filter = None
        if modalite_filter:
            qdrant_filter = Filter(
                must=[FieldCondition(
                    key="modalite",
                    match=MatchValue(value=modalite_filter)
                )]
            )

        return self.qdrant.query_points(
            collection_name=collection,
            query=vector,
            limit=fetch_limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

    def _search_one_model(
        self,
        collection      : str,
        vector          : list,
        fetch_limit     : int,
        score_threshold : float,
        pf              : PatientFilter,
        exclude_patient_id : Optional[str],
        namespace       : str = "",
    ) -> dict:
        has_clinical_filters = any([
            pf.sexe, pf.age_min, pf.age_max,
            pf.hopital, pf.diagnostic, pf.grade,
            pf.annee_min, pf.annee_max,
        ])

        if has_clinical_filters:
            eligible_ids = self._filter_by_patient(pf)
            if not eligible_ids:
                return {}
            hits = self._search_qdrant_with_ids(
                collection, vector, eligible_ids,
                fetch_limit, score_threshold, namespace=namespace,
            )
        else:
            hits = self._search_qdrant_direct(
                collection, vector, fetch_limit, score_threshold,
                modalite_filter=pf.modalite,
            )

        if exclude_patient_id:
            hits = [
                h for h in hits
                if h.payload.get("patient_id") != exclude_patient_id
            ]

        best_per_patient = {}
        for hit in hits:
            pid = hit.payload.get("patient_id", "")
            if pid not in best_per_patient:
                best_per_patient[pid] = hit
        return best_per_patient

    def _build_results(self, entries: list) -> list:
        slice_ids  = [e["hit"].payload["slice_id"] for e in entries]
        mongo_docs = {
            doc["slice_id"]: doc
            for doc in self.mongo.find({"slice_id": {"$in": slice_ids}})
        }

        results = []
        for rank, entry in enumerate(entries):
            hit  = entry["hit"]
            sid  = hit.payload["slice_id"]
            mdoc = mongo_docs.get(sid, {})
            pat  = mdoc.get("patient", {})

            results.append(MultimodalResult(
                rank          = rank + 1,
                score         = round(float(entry["score"]), 4),
                score_baseline= round(float(entry.get("score_baseline", 0.0)), 4),
                model_used    = entry.get("model_used", ""),
                slice_id      = sid,
                patient_id    = hit.payload.get("patient_id", ""),
                modalite      = hit.payload.get("modalite", ""),
                slice_z       = hit.payload.get("slice_z", -1),
                file_path     = mdoc.get("file_path", ""),
                sexe          = pat.get("sexe", ""),
                age           = pat.get("age", 0),
                hopital       = pat.get("hopital", ""),
                diagnostic    = pat.get("diagnostic", ""),
                machine_irm   = pat.get("machine_irm", ""),
                annee_exam    = pat.get("annee_exam", 0),
                antecedents   = pat.get("antecedents", ""),
                grade         = pat.get("grade", ""),
            ))
        return results

    def _search_guided(self, query: MultimodalQuery, fetch_limit: int) -> list:
        grade_info = self.predict_grade(query.image_tensor)
        self._last_guided_info = {
            "predicted_grade": grade_info["predicted_grade"],
            "confidence"     : grade_info["confidence"],
            "probs"          : grade_info["probs"],
        }

        pf = PatientFilter(
            sexe       = query.patient_filter.sexe,
            age_min    = query.patient_filter.age_min,
            age_max    = query.patient_filter.age_max,
            modalite   = query.patient_filter.modalite,
            hopital    = query.patient_filter.hopital,
            diagnostic = query.patient_filter.diagnostic,
            grade      = grade_info["predicted_grade"],
            annee_min  = query.patient_filter.annee_min,
            annee_max  = query.patient_filter.annee_max,
        )

        best = self._search_one_model(
            SUPCON_COLL, grade_info["vector"], fetch_limit, query.score_threshold,
            pf, query.exclude_patient_id, namespace="supcon_",
        )
        top = sorted(best.values(), key=lambda h: h.score, reverse=True)[:query.k]
        entries = [
            {"hit": h, "score": h.score, "score_baseline": 0.0, "model_used": "guided"}
            for h in top
        ]
        return self._build_results(entries)

    def search(self, query: MultimodalQuery) -> list:
        pf = query.patient_filter
        fetch_limit = max(query.k * 10, 50)

        if query.model == "baseline":
            self._last_guided_info = {}
            vec  = self.encode_baseline(query.image_tensor)
            best = self._search_one_model(
                QDRANT_COLLECTION, vec, fetch_limit, query.score_threshold,
                pf, query.exclude_patient_id, namespace="",
            )
            top = sorted(best.values(), key=lambda h: h.score, reverse=True)[:query.k]
            entries = [
                {"hit": h, "score": h.score, "score_baseline": h.score, "model_used": "baseline"}
                for h in top
            ]
            return self._build_results(entries)

        elif query.model == "supcon":
            self._last_guided_info = {}
            vec  = self.encode_supcon(query.image_tensor)
            best = self._search_one_model(
                SUPCON_COLL, vec, fetch_limit, query.score_threshold,
                pf, query.exclude_patient_id, namespace="supcon_",
            )
            top = sorted(best.values(), key=lambda h: h.score, reverse=True)[:query.k]
            entries = [
                {"hit": h, "score": h.score, "score_baseline": 0.0, "model_used": "supcon"}
                for h in top
            ]
            return self._build_results(entries)

        elif query.model == "guided":
            return self._search_guided(query, fetch_limit)

        else:
            raise ValueError(f"Modèle inconnu : {query.model} — valeurs acceptées : baseline, supcon, guided")

    def stats_by_filter(self, pf: PatientFilter) -> dict:
        eligible = self._filter_by_patient(pf)
        total    = self.mongo.count_documents({})
        return {
            "total_eligible"  : len(eligible),
            "total_collection": total,
            "pourcentage"     : round(len(eligible) / max(total, 1) * 100, 1),
        }
