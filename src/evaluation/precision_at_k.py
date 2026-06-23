# src/evaluation/precision_at_k.py
"""
Évaluation Precision@K par grade tumoral — VERSION STRATIFIÉE

Problème du dataset BraTS 2021 : 94% Grade IV → un système aléatoire
atteint P@K ≈ 0.94, ce qui masque la vraie capacité de discrimination.

Solution : deux métriques complémentaires présentées au jury :
  1. Precision@K STRATIFIÉE  : N requêtes par grade équitablement réparti
     (ex: 10 requêtes × 3 grades = 30 requêtes, indépendamment de la fréquence)
     → mesure la capacité à retrouver chaque grade même minoritaire
  2. Balanced Precision@K    : moyenne des P@K par grade (macro-average)
     → non biaisée par le déséquilibre, comparable entre modèles
  3. Precision@K GLOBALE     : P@K classique sur toutes les requêtes
     → fournie pour comparaison mais annotée "biaisée dataset"
"""

import os, sys, json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.recherche.unified_engine import UnifiedSearchEngine
from src.db.connections import get_slices_collection

# ── Constantes ────────────────────────────────────────────
DATA_DIR        = os.getenv("DATA_DIR", "./Data/brats_subset_5k")
CACHE_JSON      = os.getenv("CACHE_JSON", "./Data/brats_subset_5k_cache.json")
OUTPUT_DIR      = "./evaluation_results"
N_PER_GRADE     = 15     # requêtes PAR grade (stratification forcée)
K_VALUES        = [1, 3, 5, 10]
MODELS          = ["baseline", "radimagenet", "supcon"]
GRADE_FIELD     = "patient.grade"
GRADES_ORDER    = ["Grade IV", "Grade III", "Grade II"]

GRADE_COLORS = {
    "Grade IV"  : "#ef4444",
    "Grade III" : "#f59e0b",
    "Grade II"  : "#3b82f6",
}
MODEL_COLORS = {
    "baseline"    : "#60a5fa",
    "radimagenet" : "#fb923c",
    "supcon"      : "#a855f7",  
}


def load_cache():
    with open(CACHE_JSON) as f:
        return json.load(f)


import time
import random as _random

def preprocess_slice(path: str) -> torch.Tensor:
    return torch.load(path, weights_only=True)


def _search_with_retry(engine, tensor, model, k, exclude_pid, max_retries=3):
    """Wrapper avec retry + backoff exponentiel pour les erreurs Qdrant."""
    for attempt in range(max_retries):
        try:
            return engine.search(
                image_tensor       = tensor,
                model              = model,
                k                  = k,
                exclude_patient_id = exclude_pid,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + _random.uniform(0.5, 1.5)
                print(f"\n  ⚠️  Qdrant error (tentative {attempt+1}/{max_retries}): "
                      f"{type(e).__name__} — retry dans {wait:.1f}s")
                time.sleep(wait)
            else:
                print(f"\n  ❌ Abandon après {max_retries} tentatives : {e}")
                return []
    return []


# ─────────────────────────────────────────────────────────
# Sélection STRATIFIÉE des requêtes
# ─────────────────────────────────────────────────────────

def pick_query_slices_stratified(mongo_col, cache,
                                  n_per_grade: int = N_PER_GRADE) -> list:
    """
    Sélectionne exactement N_PER_GRADE requêtes par grade.
    Garantit un ensemble de requêtes équilibré indépendamment
    de la distribution du dataset (94% Grade IV ne biaise plus).

    Retourne une liste de dicts :
      {file_path, patient_id, grade, modalite, slice_z}
    """
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pt")])
    assert len(files) == len(cache), \
        f"Cache ({len(cache)}) ≠ fichiers ({len(files)})"

    # Construire patient_id → {grade, indices[]}
    patient_info = defaultdict(lambda: {"grade": None, "indices": []})
    for idx, (file_path, slice_z) in enumerate(cache):
        parts      = Path(file_path).stem.replace(".nii", "").split("_")
        patient_id = "_".join(parts[:2]) if len(parts) >= 2 else "unknown"
        patient_info[patient_id]["indices"].append((idx, int(slice_z)))

    # Récupérer les grades depuis MongoDB en une seule requête
    all_pids = list(patient_info.keys())
    for doc in mongo_col.find(
        {"patient_id": {"$in": all_pids}, GRADE_FIELD: {"$exists": True}},
        {"patient_id": 1, "patient.grade": 1, "modalite": 1}
    ):
        pid   = doc["patient_id"]
        grade = doc.get("patient", {}).get("grade", "")
        if grade and grade != "Inconnu":
            patient_info[pid]["grade"] = grade

    # Grouper par grade
    by_grade = defaultdict(list)
    for pid, info in patient_info.items():
        if info["grade"]:
            by_grade[info["grade"]].append(pid)

    print(f"\n[Precision@K] Patients disponibles par grade :")
    for g in GRADES_ORDER:
        print(f"  {g:12s} : {len(by_grade.get(g, []))} patients "
              f"→ {min(n_per_grade, len(by_grade.get(g, [])))} requêtes sélectionnées")

    # Sélection stratifiée : exactement n_per_grade par grade
    selected = []
    for grade in GRADES_ORDER:
        pids   = by_grade.get(grade, [])
        chosen = pids[:n_per_grade]   # prendre les N premiers (ordre stable)
        for pid in chosen:
            info  = patient_info[pid]
            # Coupe médiane en Z (plus informative que la première)
            slices_sorted = sorted(info["indices"], key=lambda x: x[1])
            mid_idx, mid_z = slices_sorted[len(slices_sorted) // 2]
            fp, _ = cache[mid_idx]
            parts = Path(fp).stem.replace(".nii", "").split("_")
            modalite = parts[-1] if len(parts) >= 3 else "unknown"
            selected.append({
                "file_path"  : os.path.join(DATA_DIR, files[mid_idx]),
                "patient_id" : pid,
                "grade"      : grade,
                "modalite"   : modalite,
                "slice_z"    : mid_z,
            })

    print(f"\n  Total : {len(selected)} requêtes stratifiées "
          f"({n_per_grade} × {len([g for g in GRADES_ORDER if by_grade.get(g)])} grades)")
    return selected


# ─────────────────────────────────────────────────────────
# Calcul Precision@K
# ─────────────────────────────────────────────────────────

def compute_precision_at_k(engine, queries, mongo_col,
                            k: int, model: str) -> dict:
    """
    Calcule Precision@K + Balanced Precision@K.

    Balanced P@K = macro-average des P@K par grade
                 = (P@K_GradeIV + P@K_GradeIII + P@K_GradeII) / 3
    Non biaisé par la distribution du dataset.
    """
    grade_hits = defaultdict(list)
    total_hits = []

    for q in tqdm(queries, desc=f"  P@{k:<3d} [{model}]", leave=False):
        tensor = preprocess_slice(q["file_path"])

        results = _search_with_retry(engine, tensor, model, k, q["patient_id"])

        # Pause légère entre requêtes pour éviter le rate-limiting Qdrant Cloud
        time.sleep(0.15)

        if not results:
            grade_hits[q["grade"]].append(0.0)
            total_hits.append(0.0)
            continue

        # Grades des résultats depuis MongoDB (batch)
        pids = [r.patient_id for r in results]
        grade_map = {
            doc["patient_id"]: doc.get("patient", {}).get("grade", "")
            for doc in mongo_col.find(
                {"patient_id": {"$in": pids}},
                {"patient_id": 1, "patient.grade": 1}
            )
        }

        # Debug premier appel : vérifier que les grades sont bien récupérés
        if not hasattr(compute_precision_at_k, "_debug_done"):
            compute_precision_at_k._debug_done = True
            missing = [p for p in pids if not grade_map.get(p)]
            if missing:
                print(f"\n  ⚠️  DEBUG: {len(missing)}/{len(pids)} résultats sans grade dans MongoDB")
                print(f"       Exemples patient_id résultats : {pids[:3]}")
                print(f"       grade_map : {dict(list(grade_map.items())[:3])}")

        n_correct = sum(
            1 for r in results
            if grade_map.get(r.patient_id, "") == q["grade"]
        )
        precision = n_correct / len(results)
        grade_hits[q["grade"]].append(precision)
        total_hits.append(precision)

    # Precision par grade
    by_grade = {
        g: round(float(np.mean(v)), 4)
        for g, v in grade_hits.items() if v
    }

    # Balanced Precision@K = macro-average
    balanced = round(float(np.mean(list(by_grade.values()))), 4) \
               if by_grade else 0.0

    # Global (biaisé si dataset déséquilibré)
    global_p = round(float(np.mean(total_hits)), 4) if total_hits else 0.0

    return {
        "model"     : model,
        "k"         : k,
        "global"    : global_p,      # biaisé — affiché avec avertissement
        "balanced"  : balanced,      # métrique principale
        "by_grade"  : by_grade,
        "n_queries" : len(queries),
        "n_per_grade": {g: len(v) for g, v in grade_hits.items()},
    }


# ─────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────

def build_report_figure(all_results: list) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#f4f7fb")
    fig.suptitle("Évaluation Precision@K — BraTS 2021 (stratifiée par grade)",
                  fontsize=13, fontweight="bold", y=1.01)

    # ── Graphe 1 : Balanced Precision@K par modèle ────────
    ax1 = axes[0]
    ax1.set_facecolor("#ffffff")
    ax1.set_title("Balanced Precision@K\n(macro-average, non biaisé)", fontsize=10, pad=10)
    for model in MODELS:
        ks   = [r["k"] for r in all_results if r["model"] == model]
        prec = [r["balanced"] for r in all_results if r["model"] == model]
        if ks:
            ax1.plot(ks, prec, marker="o", linewidth=2.5, markersize=7,
                      label=model.upper(), color=MODEL_COLORS[model])
    ax1.axhline(y=1/3, color="#94a3b8", linestyle="--", linewidth=1,
                alpha=0.7, label=f"Hasard (1/3 = {1/3:.2f})")
    ax1.set_xlabel("K"); ax1.set_ylabel("Balanced Precision@K")
    ax1.set_xticks(K_VALUES); ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # ── Graphe 2 : Precision@5 par grade ──────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#ffffff")
    ax2.set_title("Precision@5 par grade tumoral\n(stratifié)", fontsize=10, pad=10)
    p5  = [r for r in all_results if r["k"] == 5]
    x   = np.arange(len(GRADES_ORDER))
    w   = 0.25
    for i, model in enumerate(MODELS):
        r = next((r for r in p5 if r["model"] == model), None)
        if not r:
            continue
        vals = [r["by_grade"].get(g, 0.0) for g in GRADES_ORDER]
        bars = ax2.bar(x + i * w, vals, w, label=model.upper(),
                        color=MODEL_COLORS[model], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.01,
                          f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x + w); ax2.set_xticklabels(GRADES_ORDER, fontsize=9)
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("Precision@5")
    ax2.axhline(y=1/3, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.6)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")

    # ── Graphe 3 : Global vs Balanced comparison ──────────
    ax3 = axes[2]
    ax3.set_facecolor("#ffffff")
    ax3.set_title("Global vs Balanced Precision@5\n(impact du déséquilibre dataset)",
                   fontsize=10, pad=10)
    p5_models = [r for r in p5]
    models_lb  = [r["model"].upper() for r in p5_models if r["k"] == 5]
    g_vals     = [r["global"]   for r in p5_models if r["k"] == 5]
    b_vals     = [r["balanced"] for r in p5_models if r["k"] == 5]
    xm = np.arange(len(models_lb))
    ax3.bar(xm - 0.2, g_vals, 0.35, label="Global (biaisé)",
             color="#94a3b8", alpha=0.7)
    ax3.bar(xm + 0.2, b_vals, 0.35, label="Balanced (non biaisé)",
             color="#6366f1", alpha=0.85)
    for i, (gv, bv) in enumerate(zip(g_vals, b_vals)):
        ax3.text(i - 0.2, gv + 0.01, f"{gv:.2f}", ha="center", fontsize=8)
        ax3.text(i + 0.2, bv + 0.01, f"{bv:.2f}", ha="center", fontsize=8)
    ax3.set_xticks(xm); ax3.set_xticklabels(models_lb, fontsize=9)
    ax3.set_ylim(0, 1.1); ax3.set_ylabel("Precision@5")
    ax3.axhline(y=1/3, color="#ef4444", linestyle="--", linewidth=1,
                alpha=0.6, label=f"Hasard équilibré (0.33)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(pad=1.5)
    return fig


# ─────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────

def run_evaluation(
    n_per_grade : int   = N_PER_GRADE,
    k_values    : list  = K_VALUES,
    models      : list  = MODELS,
    save_report : bool  = True,
) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    engine    = UnifiedSearchEngine()
    mongo_col = get_slices_collection()
    cache     = load_cache()

    queries = pick_query_slices_stratified(mongo_col, cache, n_per_grade)
    if not queries:
        print("[Precision@K] Aucune requête valide — "
              "vérifiez que generate_fake_metadata.py a été exécuté.")
        return {}

    all_results = []
    print()
    for model in models:
        print(f"[{model.upper()}]")
        for k in k_values:
            result = compute_precision_at_k(engine, queries, mongo_col, k, model)
            all_results.append(result)
            print(f"  P@{k:<3d} global={result['global']:.4f}  "
                  f"balanced={result['balanced']:.4f}  "
                  f"| {result['by_grade']}")

    report = {
        "n_per_grade" : n_per_grade,
        "k_values"    : k_values,
        "models"      : models,
        "results"     : all_results,
    }

    if save_report:
        path_json = os.path.join(OUTPUT_DIR, "precision_at_k.json")
        with open(path_json, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        fig = build_report_figure(all_results)
        path_fig = os.path.join(OUTPUT_DIR, "precision_at_k.png")
        fig.savefig(path_fig, dpi=150, bbox_inches="tight", facecolor="#f4f7fb")
        plt.close(fig)
        print(f"\nRapport JSON : {path_json}")
        print(f"Figure       : {path_fig}")

    # ── Synthèse console ──────────────────────────────────
    print("\n" + "═" * 68)
    print("SYNTHÈSE PRECISION@K — STRATIFIÉE (Balanced = métrique principale)")
    print("═" * 68)
    print(f"{'Modèle':14s}" +
          "".join(f"  Bal@{k:<3d}" for k in k_values) +
          "".join(f"  Glb@{k:<3d}" for k in k_values))
    print("─" * 68)
    for model in models:
        row = f"{model.upper():14s}"
        for k in k_values:
            r = next((r for r in all_results
                      if r["model"] == model and r["k"] == k), None)
            row += f"  {r['balanced']:.4f} " if r else "  —      "
        for k in k_values:
            r = next((r for r in all_results
                      if r["model"] == model and r["k"] == k), None)
            row += f"  {r['global']:.4f} " if r else "  —      "
        print(row)
    print("═" * 68)
    print(f"Hasard (1/3 classes équilibrées) = {1/3:.4f}")
    print(f"Requêtes : {n_per_grade} par grade × "
          f"{len(set(q['grade'] for q in queries))} grades "
          f"= {len(queries)} total")

    return report


if __name__ == "__main__":
    run_evaluation()