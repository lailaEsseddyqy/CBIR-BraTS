# src/training/grade_labels.py
"""
Construction / export des labels de grade tumoral (1 label par coupe .pt).

Sources (par priorité) :
  1. MongoDB  — champ patient.grade
  2. Segmentation BraTS — infer_grade() depuis generate_fake_metadata.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.grade_constants import GRADE_TO_IDX


def _lookup_patient_grade(mongo, patient_id: str) -> str:
    """Cherche patient.grade dans MongoDB (variantes d'ID BraTS / TS)."""
    candidates = [
        patient_id,
        patient_id.replace("BraTS2021_", "TS2021_"),
        f"TS2021_{patient_id.split('_')[-1]}",
    ]
    for pid in dict.fromkeys(candidates):
        doc = mongo.find_one({"patient_id": pid}, {"patient.grade": 1})
        if doc and doc.get("patient", {}).get("grade"):
            return doc["patient"]["grade"]
    return "Inconnu"


def patient_id_from_nifti_path(file_path: str) -> str:
    parts = Path(file_path).stem.replace(".nii", "").split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else "unknown"


def export_grade_labels(
    data_dir: str,
    cache_json: str,
    output_path: str,
    use_mongo: bool = True,
) -> list[int]:
    """
    Génère un JSON list[int] aligné sur les fichiers .pt triés du data_dir.
    Retourne la liste des labels (-1 = Inconnu, exclu à l'entraînement).
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pt"))
    with open(cache_json, encoding="utf-8") as f:
        cache = json.load(f)

    if len(files) != len(cache):
        raise ValueError(
            f"Mismatch : {len(files)} fichiers .pt vs {len(cache)} entrées cache"
        )

    mongo = None
    if use_mongo:
        try:
            from src.db.connections import get_slices_collection
            mongo = get_slices_collection()
        except Exception as exc:
            print(f"[grade_labels] MongoDB indisponible ({exc}) — fallback segmentation.")

    infer_grade = None
    if mongo is None:
        from src.training.grade_inference import infer_grade_from_segmentation as infer_grade

    patient_grade_idx: dict[str, int] = {}
    labels: list[int] = []

    for idx, (nifti_path, _slice_z) in enumerate(cache):
        pid = patient_id_from_nifti_path(nifti_path)

        if pid not in patient_grade_idx:
            grade_str = "Inconnu"

            if mongo is not None:
                grade_str = _lookup_patient_grade(mongo, pid)

            if grade_str == "Inconnu" and infer_grade is not None:
                grade_str = infer_grade(pid)

            patient_grade_idx[pid] = GRADE_TO_IDX.get(grade_str, -1)

        labels.append(patient_grade_idx[pid])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f)

    valid = sum(1 for l in labels if l != -1)
    print(f"[grade_labels] {valid}/{len(labels)} coupes labellisees -> {output_path}")
    for c in sorted(set(labels)):
        print(f"  Classe {c} : {labels.count(c)} coupes")

    if valid == 0:
        raise RuntimeError(
            "Aucun label de grade valide. Verifiez :\n"
            "  1. MongoDB peuple (python generate_fake_metadata.py)\n"
            "  2. nibabel installe + masques BraTS dans BRATS_DIR\n"
            "  3. GRADES_LABELS_JSON pointe vers un fichier existant"
        )

    return labels


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    data_dir   = os.getenv("DATA_DIR", "./Data/brats_subset_5k")
    cache_json = os.getenv("CACHE_JSON", "./Data/brats_subset_5k_cache.json")
    out_path   = os.getenv("GRADES_LABELS_JSON", "./Data/brats_grade_labels.json")

    export_grade_labels(data_dir, cache_json, out_path)
