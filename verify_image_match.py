# verify_image_match.py
import sys, torch, os, cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")

from src.recherche.search_engine import CBMIRSearchEngine, SearchQuery
from src.evaluation.medical_metrics import compute_all
from src.db.connections import get_slices_collection

engine    = CBMIRSearchEngine()
mongo_col = get_slices_collection()
os.makedirs("./results", exist_ok=True)


def safe_load(file_path: str):
    """Charge un .pt en toute sécurité — retourne None si problème."""
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        return torch.load(file_path, weights_only=True).squeeze().numpy()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# TEST 1 — Image identique comme requête
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 1 : Image identique comme requête")
print("Rang 1 attendu : score ≈ 1.0 et SSIM ≈ 1.0")
print("="*60)

doc    = mongo_col.find_one({"modalite": "flair", "file_path": {"$ne": ""}})
tensor = torch.load(doc["file_path"], weights_only=True)
print(f"\nRequête : {doc['slice_id']}")
print(f"Chemin  : {doc['file_path']}")

query   = SearchQuery(image_tensor=tensor, k=5)
results = engine.search(query)
query_np = tensor.squeeze().numpy()

print(f"\n{'Rang':<5} {'Score':<8} {'SSIM':<8} {'PSNR':<8} "
      f"{'MSE':<10} {'Slice ID':<35} {'Image OK?'}")
print("-" * 85)

for r in results:
    result_img = safe_load(r.file_path)

    if result_img is None:
        print(f"{r.rank:<5} {r.score:<8} {'N/A':<8} {'N/A':<8} "
              f"{'N/A':<10} {r.slice_id:<35} ⚠️ fichier manquant")
        continue

    q_res   = cv2.resize(query_np, (result_img.shape[1], result_img.shape[0]))
    metrics = compute_all(q_res, result_img)

    is_same = r.slice_id == doc["slice_id"]
    flag    = "✅ IDENTIQUE" if is_same else "🔍 similaire"

    print(f"{r.rank:<5} {r.score:<8} "
          f"{metrics['ssim']:<8} "
          f"{metrics['psnr']:<8} "
          f"{metrics['mse']:<10} "
          f"{r.slice_id:<35} {flag}")


# ─────────────────────────────────────────────────────────
# TEST 2 — Vérification visuelle (3 requêtes × 5 résultats)
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2 : Vérification visuelle — figure sauvegardée")
print("="*60)

# Prendre 3 coupes avec fichiers valides
docs = list(mongo_col.find(
    {
        "modalite" : {"$in": ["flair", "t1", "t2"]},
        "file_path": {"$ne": ""}
    }
).limit(3))

fig, axes = plt.subplots(3, 6, figsize=(20, 10))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle(
    "Vérification : Requête vs Résultats Top-5",
    color="white", fontsize=14, fontweight="bold"
)

for row, doc in enumerate(docs):
    tensor   = torch.load(doc["file_path"], weights_only=True)
    query_np = tensor.squeeze().numpy()
    results  = engine.search(SearchQuery(image_tensor=tensor, k=5))

    # Colonne 0 : image requête
    axes[row, 0].imshow(query_np, cmap="gray")
    axes[row, 0].set_title(
        f"REQUÊTE\n{doc['modalite']} z={doc['slice_z']}",
        color="#ff6b6b", fontsize=8, fontweight="bold"
    )
    axes[row, 0].axis("off")

    # Colonnes 1-5 : résultats
    for col_idx, r in enumerate(results):
        result_img = safe_load(r.file_path)
        ax         = axes[row, col_idx + 1]

        if result_img is None:
            ax.text(0.5, 0.5, "Manquant",
                    ha="center", va="center",
                    color="gray", transform=ax.transAxes)
            ax.set_title(f"#{r.rank} ⚠️", color="gray", fontsize=8)
            ax.axis("off")
            continue

        q_res   = cv2.resize(query_np, (result_img.shape[1], result_img.shape[0]))
        metrics = compute_all(q_res, result_img)
        ssim    = metrics["ssim"]

        ax.imshow(result_img, cmap="gray")

        color = "#4ecdc4" if ssim > 0.7 else "#ffe66d" if ssim > 0.5 else "#ff6b6b"
        ax.set_title(
            f"#{r.rank} cos={r.score}\n"
            f"SSIM={ssim} | {r.modalite}",
            color=color, fontsize=7, fontweight="bold"
        )
        ax.axis("off")

plt.tight_layout()
plt.savefig("./results/verify_image_match.png", dpi=150, bbox_inches="tight")
print("Figure sauvegardée : ./results/verify_image_match.png")


# ─────────────────────────────────────────────────────────
# TEST 3 — Cohérence globale sur 20 requêtes
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 3 : Cohérence globale sur 20 requêtes")
print("="*60)

# Seulement les docs avec fichiers valides
valid_docs = list(mongo_col.find(
    {"file_path": {"$ne": ""}}
).limit(20))

ssim_scores, cos_scores = [], []

for doc in valid_docs:
    result_img_query = safe_load(doc["file_path"])
    if result_img_query is None:
        continue

    tensor  = torch.load(doc["file_path"], weights_only=True)
    results = engine.search(SearchQuery(image_tensor=tensor, k=1))

    if not results:
        continue

    r          = results[0]
    result_img = safe_load(r.file_path)
    if result_img is None:
        continue

    q_res   = cv2.resize(
        tensor.squeeze().numpy(),
        (result_img.shape[1], result_img.shape[0])
    )
    metrics = compute_all(q_res, result_img)
    ssim_scores.append(metrics["ssim"])
    cos_scores.append(r.score)

if cos_scores:
    print(f"\n  Cosinus moyen : {np.mean(cos_scores):.4f}")
    print(f"  SSIM moyen    : {np.mean(ssim_scores):.4f}")
    print(f"  Cosinus min   : {np.min(cos_scores):.4f}")
    print(f"  SSIM min      : {np.min(ssim_scores):.4f}")

    cos_moy  = np.mean(cos_scores)
    ssim_moy = np.mean(ssim_scores)

    print("\n── Interprétation ──")
    if cos_moy > 0.5 and ssim_moy > 0.5:
        print("  ✅ Système retourne des images visuellement cohérentes")
    elif cos_moy > 0.3:
        print("  🟡 Résultats acceptables — plus de données améliorerait")
    else:
        print("  ❌ Résultats faibles — vérifier l'indexation")
else:
    print("  ⚠️ Aucune requête valide trouvée")


# ─────────────────────────────────────────────────────────
# DIAGNOSTIC — Compter les chemins vides
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC : État des chemins MongoDB")
print("="*60)
total  = mongo_col.count_documents({})
empty  = mongo_col.count_documents({"file_path": ""})
valids = total - empty
print(f"  Total documents : {total}")
print(f"  Chemins valides : {valids}")
print(f"  Chemins vides   : {empty}")
if empty > 0:
    print(f"\n  ⚠️ {empty} documents sans fichier local")
    print(f"  → Lance : python fix_empty_paths.py")
else:
    print(f"\n  ✅ Tous les chemins sont valides")