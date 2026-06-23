# src/recherche/multimodal_search.py
"""
Recherche multimodale CBMIR — VERSION MULTI-MODELES

Supporte 3 modes (comme UnifiedSearchEngine) :
  - "baseline"    : Auto-encodeur BraTS 2021 (collection brats_embeddings)
  - "radimagenet" : ResNet-50 RadImageNet     (collection radimagenet_embeddings)
  - "combined"    : Moyenne des scores des deux modèles

Combine la similarité visuelle (Qdrant) avec les filtres cliniques (MongoDB Atlas).

Bugs corriges (version precedente) :
  1. Sans filtres : requete directe Qdrant (pas de HasIdCondition sur 5000 IDs)
  2. Logique patients distincts ajoutee
  3. fetch_limit = k * 10 pour avoir assez apres deduplication
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
from src.models.radimagenet_extractor import RadImageNetExtractor
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    Range, HasIdCondition
)
from dotenv import load_dotenv

load_dotenv()

CKPT_PATH        = os.getenv("CHECKPOINT_PATH", "./saved_models/my_brats_model.ckpt")
RADIMAGENET_WEIGHTS_DIR     = os.getenv("RADIMAGENET_WEIGHTS_DIR", "./pretrain")
RADIMAGENET_PROJECTION_PATH = os.getenv("RADIMAGENET_PROJECTION", "./pretrain/radimagenet_projection.pth")
RADIMAGENET_COLL = "radimagenet_embeddings"


# ─────────────────────────────────────────────────────────
# Structures
# ─────────────────────────────────────────────────────────

@dataclass
class PatientFilter:
    sexe       : Optional[str] = None
    age_min    : Optional[int] = None
    age_max    : Optional[int] = None
    modalite   : Optional[str] = None
    hopital    : Optional[str] = None
    diagnostic : Optional[str] = None
    annee_min  : Optional[int] = None
    annee_max  : Optional[int] = None


@dataclass
class MultimodalQuery:
    image_tensor       : torch.Tensor
    k                  : int           = 10
    model              : str           = "baseline"   # "baseline" | "radimagenet" | "combined"
    patient_filter     : PatientFilter = field(default_factory=PatientFilter)
    score_threshold    : float         = 0.0
    exclude_patient_id : Optional[str] = None   # patient de la requete a exclure


@dataclass
class MultimodalResult:
    rank            : int
    score           : float
    score_baseline  : float = 0.0
    score_radimagenet: float = 0.0
    model_used      : str   = ""
    slice_id        : str   = ""
    patient_id      : str   = ""
    modalite        : str   = ""
    slice_z         : int   = -1
    file_path       : str   = ""
    sexe            : str   = ""
    age             : int   = 0
    hopital         : str   = ""
    diagnostic      : str   = ""
    machine_irm     : str   = ""
    annee_exam      : int   = 0
    antecedents     : str   = ""


# ─────────────────────────────────────────────────────────
# Moteur multimodal
# ─────────────────────────────────────────────────────────

class MultimodalSearchEngine:

    def __init__(self, checkpoint_path: str = CKPT_PATH):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Baseline — toujours chargé
        self.model = BraTSAutoencoderLightning.load_from_checkpoint(checkpoint_path)
        self.model.eval().to(self.device)

        self.qdrant = get_qdrant_client()
        self.mongo  = get_slices_collection()

        # RadImageNet — chargement lazy (lourd en RAM)
        self._radimagenet = None

        print(f"[Multimodal] Moteur charge — device: {self.device}")

    @property
    def radimagenet(self) -> RadImageNetExtractor:
        """Charge RadImageNet à la première utilisation."""
        if self._radimagenet is None:
            self._radimagenet = RadImageNetExtractor(
                weights_dir=RADIMAGENET_WEIGHTS_DIR,
                freeze=True,
                projection_path=RADIMAGENET_PROJECTION_PATH,
            )
            self._radimagenet.eval().to(self.device)
            print(f"[Multimodal] RadImageNet charge — device: {self.device}")
        return self._radimagenet

    # ── Encodage ──────────────────────────────────────────
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

    def encode_radimagenet(self, image_tensor: torch.Tensor) -> list:
        tensor_device = image_tensor.to(self.device)
        return self.radimagenet.extract(tensor_device)

    # ── Filtrage MongoDB ───────────────────────────────────
    def _filter_by_patient(self, pf: PatientFilter) -> list:
        """Retourne les slice_id eligibles selon les filtres cliniques."""
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

        if pf.annee_min is not None or pf.annee_max is not None:
            annee_q = {}
            if pf.annee_min is not None: annee_q["$gte"] = pf.annee_min
            if pf.annee_max is not None: annee_q["$lte"] = pf.annee_max
            mongo_query["patient.annee_exam"] = annee_q

        docs = self.mongo.find(mongo_query, {"slice_id": 1})
        return [doc["slice_id"] for doc in docs]

    # ── Recherche Qdrant avec filtre HasId ─────────────────
    def _search_qdrant_with_ids(
        self,
        collection      : str,
        vector          : list,
        eligible_ids    : list,
        fetch_limit     : int,
        score_threshold : float,
        namespace       : str = "",
    ) -> list:
        """
        Recherche vectorielle parmi un sous-ensemble de IDs.
        namespace : préfixe utilisé lors de la génération des UUID
                    (vide pour baseline, "radimagenet_" pour RadImageNet)
        """
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

    # ── Recherche Qdrant directe (sans filtre ID) ──────────
    def _search_qdrant_direct(
        self,
        collection      : str,
        vector          : list,
        fetch_limit     : int,
        score_threshold : float,
        modalite_filter : Optional[str] = None,
    ) -> list:
        """
        Recherche directe dans Qdrant (pas de HasIdCondition sur tous les IDs).
        """
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

    # ── Recherche brute pour une collection donnée ─────────
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
        """
        Retourne un dict {patient_id: hit} avec la meilleure coupe
        par patient pour la collection donnée.
        """
        has_clinical_filters = any([
            pf.sexe, pf.age_min, pf.age_max,
            pf.hopital, pf.diagnostic,
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

    # ── Enrichissement MongoDB + assemblage résultats ──────
    def _build_results(self, entries: list) -> list:
        """
        entries : liste de dicts avec au minimum
                  {"hit", "score", "score_baseline", "score_radimagenet", "model_used"}
        """
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
                rank             = rank + 1,
                score            = round(float(entry["score"]), 4),
                score_baseline   = round(float(entry.get("score_baseline", 0.0)), 4),
                score_radimagenet= round(float(entry.get("score_radimagenet", 0.0)), 4),
                model_used       = entry.get("model_used", ""),
                slice_id         = sid,
                patient_id       = hit.payload.get("patient_id", ""),
                modalite         = hit.payload.get("modalite", ""),
                slice_z          = hit.payload.get("slice_z", -1),
                file_path        = mdoc.get("file_path", ""),
                sexe             = pat.get("sexe", ""),
                age              = pat.get("age", 0),
                hopital          = pat.get("hopital", ""),
                diagnostic       = pat.get("diagnostic", ""),
                machine_irm      = pat.get("machine_irm", ""),
                annee_exam       = pat.get("annee_exam", 0),
                antecedents      = pat.get("antecedents", ""),
            ))
        return results

    # ── Recherche principale ───────────────────────────────
    def search(self, query: MultimodalQuery) -> list:
        """
        Recherche multimodale avec patients distincts.
        model : "baseline" | "radimagenet" | "combined"
        """
        pf = query.patient_filter
        fetch_limit = max(query.k * 10, 50)

        if query.model == "baseline":
            vec  = self.encode_baseline(query.image_tensor)
            best = self._search_one_model(
                QDRANT_COLLECTION, vec, fetch_limit, query.score_threshold,
                pf, query.exclude_patient_id, namespace="",
            )
            top = sorted(best.values(), key=lambda h: h.score, reverse=True)[:query.k]
            entries = [
                {"hit": h, "score": h.score, "score_baseline": h.score,
                 "score_radimagenet": 0.0, "model_used": "baseline"}
                for h in top
            ]
            return self._build_results(entries)

        elif query.model == "radimagenet":
            vec  = self.encode_radimagenet(query.image_tensor)
            best = self._search_one_model(
                RADIMAGENET_COLL, vec, fetch_limit, query.score_threshold,
                pf, query.exclude_patient_id, namespace="radimagenet_",
            )
            top = sorted(best.values(), key=lambda h: h.score, reverse=True)[:query.k]
            entries = [
                {"hit": h, "score": h.score, "score_baseline": 0.0,
                 "score_radimagenet": h.score, "model_used": "radimagenet"}
                for h in top
            ]
            return self._build_results(entries)

        elif query.model == "combined":
            vec_b = self.encode_baseline(query.image_tensor)
            vec_r = self.encode_radimagenet(query.image_tensor)

            best_b = self._search_one_model(
                QDRANT_COLLECTION, vec_b, query.k * 2, query.score_threshold,
                pf, query.exclude_patient_id, namespace="",
            )
            best_r = self._search_one_model(
                RADIMAGENET_COLL, vec_r, query.k * 2, query.score_threshold,
                pf, query.exclude_patient_id, namespace="radimagenet_",
            )

            all_pids = set(best_b.keys()) | set(best_r.keys())
            combined = {}
            for pid in all_pids:
                hit_b = best_b.get(pid)
                hit_r = best_r.get(pid)

                if hit_b and hit_r:
                    avg_score = (hit_b.score + hit_r.score) / 2.0
                    combined[pid] = {
                        "hit": hit_b, "score": avg_score,
                        "score_baseline": hit_b.score,
                        "score_radimagenet": hit_r.score,
                        "model_used": "combined",
                    }
                elif hit_b:
                    combined[pid] = {
                        "hit": hit_b, "score": hit_b.score * 0.85,
                        "score_baseline": hit_b.score,
                        "score_radimagenet": 0.0,
                        "model_used": "combined",
                    }
                else:
                    combined[pid] = {
                        "hit": hit_r, "score": hit_r.score * 0.85,
                        "score_baseline": 0.0,
                        "score_radimagenet": hit_r.score,
                        "model_used": "combined",
                    }

            top = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:query.k]
            return self._build_results(top)

        else:
            raise ValueError(f"Modèle inconnu : {query.model}")

    # ── Statistiques ──────────────────────────────────────
    def stats_by_filter(self, pf: PatientFilter) -> dict:
        eligible = self._filter_by_patient(pf)
        total    = self.mongo.count_documents({})
        return {
            "total_eligible"  : len(eligible),
            "total_collection": total,
            "pourcentage"     : round(len(eligible) / max(total, 1) * 100, 1),
        }


# ─────────────────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = MultimodalSearchEngine()
    dummy  = torch.rand(1, 128, 128)

    print("\n── Test : Hommes + 50 ans + FLAIR (baseline) ──")
    query = MultimodalQuery(
        image_tensor   = dummy,
        k              = 5,
        model          = "baseline",
        patient_filter = PatientFilter(sexe="Homme", age_min=50, modalite="flair"),
    )
    results = engine.search(query)
    patients = [r.patient_id for r in results]
    print(f"Patients distincts : {len(set(patients))}/{len(results)}")
    for r in results:
        print(f"  Rang {r.rank} | {r.score} | {r.patient_id} | {r.sexe} {r.age}ans")

    print("\n── Test : Sans filtres, combined ──")
    query2   = MultimodalQuery(image_tensor=dummy, k=5, model="combined")
    results2 = engine.search(query2)
    patients2 = [r.patient_id for r in results2]
    print(f"Patients distincts : {len(set(patients2))}/{len(results2)}")
    for r in results2:
        print(f"  Rang {r.rank} | score={r.score} (b={r.score_baseline}, r={r.score_radimagenet}) | {r.patient_id}")