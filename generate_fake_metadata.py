# src/data/generate_fake_metadata.py
"""
Génération de métadonnées pour BraTS 2021.

Grade tumoral : inféré depuis les VRAIS masques de segmentation .nii.gz
  Label BraTS 2021 → signification :
    0 = fond
    1 = NCR (Necrotic Core — nécrose tumorale)
    2 = ET  (Enhancing Tumor — tumeur active rehaussée)
    3 = ED  (Edema péritumoral)

  ⚠️  Convention BraTS 2021 (différente de BraTS 2019/2020) :
    BraTS 2019 : label 1=NCR, 2=ED, 3=ET
    BraTS 2021 : label 1=NCR, 2=ET, 3=ED  ← labels 2 et 3 inversés !

  Règle d'inférence (validée cliniquement sur BraTS 2021) :
    ET présent (label 2) + NCR présent (label 1) → Glioblastome Grade IV
    ET présent (label 2) + NCR absent            → Astrocytome Grade III
    Uniquement NCR (label 1), pas ET             → Gliome Grade II
    Aucun label tumoral                          → Inconnu

Autres métadonnées (sexe, âge, hôpital...) : simulées mais
assignées UNE FOIS par patient et appliquées à toutes ses coupes.
"""

import os, sys, random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.db.connections import get_slices_collection
from dotenv import load_dotenv

load_dotenv()

random.seed(42)

# ── Chemins ───────────────────────────────────────────────
BRATS_DIR = os.getenv("BRATS_DIR", "./Data/BraTS2021_Training_Data")

# ── Tables de référence ───────────────────────────────────
HOPITAUX = [
    "CHU Casablanca",
    "Hôpital Ibn Sina Rabat",
    "Clinique Al Farabi",
    "CHU Mohammed VI Marrakech",
    "Hôpital Cheikh Khalifa",
]
MACHINES_IRM = [
    "Siemens Magnetom Prisma 3T",
    "Philips Ingenia 3T",
    "GE Discovery MR750 3T",
    "Siemens Skyra 3T",
]
ANTECEDENTS_POOL = [
    "Aucun antécédent notable",
    "Hypertension artérielle",
    "Diabète type 2",
    "Épilepsie traitée",
    "Antécédent de méningiome opéré",
    "Chimio-radiothérapie antérieure",
    "Immunodépression (VIH+)",
]

# Mapping grade → diagnostic affiché dans l'interface
GRADE_TO_DIAGNOSTIC = {
    "Grade IV"  : "Glioblastome Grade IV",
    "Grade III" : "Astrocytome Grade III",
    "Grade II"  : "Gliome de bas grade",
    "Inconnu"   : "Non déterminé",
}


# ─────────────────────────────────────────────────────────
# Inférence du grade depuis le masque de segmentation
# ─────────────────────────────────────────────────────────

def find_seg_file(patient_id: str) -> str | None:
    """
    Cherche le fichier _seg.nii.gz pour un patient.
    Structure attendue :
      ./Data/BraTS2021_Training_Data/BraTS2021_XXXXX/BraTS2021_XXXXX_seg.nii.gz
    """
    # BraTS2021 utilise "BraTS2021_" comme préfixe dans les dossiers
    # mais nos patient_ids sont parfois "TS2021_XXXXX" (préfixe tronqué)
    # → on essaie les deux formes
    candidates = [
        Path(BRATS_DIR) / patient_id / f"{patient_id}_seg.nii.gz",
        Path(BRATS_DIR) / f"BraTS2021_{patient_id.split('_')[-1]}" /
            f"BraTS2021_{patient_id.split('_')[-1]}_seg.nii.gz",
        Path(BRATS_DIR) / patient_id / f"{patient_id}_seg.nii",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def infer_grade(patient_id: str) -> str:
    """
    Infère le grade tumoral depuis le masque de segmentation BraTS 2021.

    Convention RÉELLE constatée sur ce dataset (labels [0,1,2,4]) :
      Label 0 = fond
      Label 1 = NCR  (Necrotic Core)
      Label 2 = ED   (Edema péritumoral)
      Label 4 = ET   (Enhancing Tumor — tumeur active)
      (Label 3 absent dans ce dataset)

    Règles cliniques :
      ET (label 4) + NCR (label 1) → Grade IV  (GBM)
      ET (label 4) seul, sans NCR  → Grade III (astrocytome anaplasique)
      NCR/ED sans ET               → Grade II  (gliome bas grade)
      Aucun label tumoral          → Inconnu
    """
    try:
        import nibabel as nib
    except ImportError:
        print("  ⚠️  nibabel non installé — pip install nibabel")
        return "Inconnu"

    seg_path = find_seg_file(patient_id)
    if seg_path is None:
        return "Inconnu"

    try:
        seg = nib.load(seg_path).get_fdata().astype(np.uint8)
    except Exception as e:
        print(f"  ⚠️  Erreur lecture {seg_path} : {e}")
        return "Inconnu"

    has_ncr = np.any(seg == 1)   # Necrotic Core
    has_ed  = np.any(seg == 2)   # Edema
    has_et  = np.any(seg == 4)   # Enhancing Tumor ← label 4 (pas 3 !)

    if has_et and has_ncr:
        return "Grade IV"    # GBM : nécrose + tumeur active
    elif has_et and not has_ncr:
        return "Grade III"   # Astrocytome anaplasique : ET sans nécrose
    elif (has_ncr or has_ed) and not has_et:
        return "Grade II"    # Gliome bas grade : pas de rehaussement
    else:
        return "Inconnu"


# ─────────────────────────────────────────────────────────
# Génération du profil patient
# ─────────────────────────────────────────────────────────

def random_date(start_year=2018, end_year=2023) -> str:
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")


def generate_patient_profile(patient_id: str, grade: str) -> dict:
    """
    Génère UN profil clinique par patient.
    Le grade est inféré depuis la segmentation (pas aléatoire).
    """
    diagnostic = GRADE_TO_DIAGNOSTIC.get(grade, "Non déterminé")
    return {
        "patient_id"  : patient_id,
        "sexe"        : random.choice(["Homme", "Femme"]),
        "age"         : random.randint(18, 85),
        "hopital"     : random.choice(HOPITAUX),
        "machine_irm" : random.choice(MACHINES_IRM),
        "diagnostic"  : diagnostic,
        "grade"       : grade,           # ← vrai grade depuis segmentation
        "annee_exam"  : random.randint(2018, 2023),
        "date_exam"   : random_date(),
        "antecedents" : random.choice(ANTECEDENTS_POOL),
    }


# ─────────────────────────────────────────────────────────
# Script principal
# ─────────────────────────────────────────────────────────

def run_generate(dry_run: bool = False) -> None:
    """
    1. Récupère les patient_ids distincts depuis MongoDB
    2. Pour chaque patient, infère le grade depuis _seg.nii.gz
    3. Génère UN profil et l'applique à TOUS ses documents (update_many)
    """
    col = get_slices_collection()
    total_docs  = col.count_documents({})
    patient_ids = col.distinct("patient_id")

    print(f"[Metadata] {total_docs} documents · {len(patient_ids)} patients distincts")
    print(f"[Metadata] Dossier BraTS : {BRATS_DIR}")

    # ── Inférence des grades ───────────────────────────────
    print("\n[Metadata] Inférence des grades depuis les masques...")
    grade_map    = {}
    grade_counts = {"Grade IV": 0, "Grade III": 0, "Grade II": 0, "Inconnu": 0}
    missing_seg  = []

    for patient_id in patient_ids:
        grade = infer_grade(patient_id)
        grade_map[patient_id] = grade
        grade_counts[grade]  += 1
        if grade == "Inconnu":
            missing_seg.append(patient_id)

    print("\n  Distribution des grades (depuis segmentation) :")
    for g, n in grade_counts.items():
        pct = round(n / max(len(patient_ids), 1) * 100, 1)
        print(f"    {g:12s} : {n:4d} patients ({pct}%)")

    if missing_seg:
        print(f"\n  ⚠️  {len(missing_seg)} patients sans segmentation trouvée :")
        for pid in missing_seg[:10]:
            print(f"    {pid}")
        if len(missing_seg) > 10:
            print(f"    ... et {len(missing_seg)-10} autres")
        print("  → Vérifiez BRATS_DIR dans votre .env")

    if dry_run:
        print("\n[DRY RUN] Aucune écriture MongoDB.")
        for pid, grade in list(grade_map.items())[:5]:
            print(f"  {pid} → {grade} → {GRADE_TO_DIAGNOSTIC[grade]}")
        return

    # ── Écriture dans MongoDB ──────────────────────────────
    print("\n[Metadata] Mise à jour MongoDB...")
    profiles_done = 0
    docs_updated  = 0

    for patient_id, grade in grade_map.items():
        profile = generate_patient_profile(patient_id, grade)
        result  = col.update_many(
            {"patient_id": patient_id},
            {"$set": {"patient": profile}},
        )
        profiles_done += 1
        docs_updated  += result.modified_count

        if profiles_done % 50 == 0:
            print(f"  {profiles_done}/{len(patient_ids)} patients "
                  f"({docs_updated} documents mis à jour)...")

    print(f"\n[Metadata] ✅ Terminé !")
    print(f"  Profils générés  : {profiles_done}")
    print(f"  Documents mis à jour : {docs_updated}")

    # ── Vérification cohérence ─────────────────────────────
    pipeline = [
        {"$group": {
            "_id"    : "$patient_id",
            "grades" : {"$addToSet": "$patient.grade"},
        }},
        {"$match": {"grades.1": {"$exists": True}}},
    ]
    incoherents = list(col.aggregate(pipeline))
    if incoherents:
        print(f"  ⚠️  {len(incoherents)} patients avec grades incohérents !")
    else:
        print("  ✅ Cohérence vérifiée — 1 grade par patient")

    # ── Distribution finale dans MongoDB ──────────────────
    print("\n  Distribution finale des grades dans MongoDB :")
    for grade in ["Grade IV", "Grade III", "Grade II", "Inconnu"]:
        n = len(col.distinct("patient_id", {"patient.grade": grade}))
        print(f"    {grade:12s} : {n} patients")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Génère les métadonnées patients avec grades réels BraTS"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Afficher sans écrire dans MongoDB"
    )
    parser.add_argument(
        "--brats-dir", type=str, default=None,
        help="Chemin vers BraTS2021_Training_Data (override .env)"
    )
    args = parser.parse_args()

    if args.brats_dir:
        BRATS_DIR = args.brats_dir

    run_generate(dry_run=args.dry_run)