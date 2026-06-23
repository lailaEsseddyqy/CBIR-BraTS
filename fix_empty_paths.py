# fix_empty_paths.py
import sys, os
sys.path.append(".")
from src.db.connections import get_slices_collection

col       = get_slices_collection()
LOCAL_DIR = r"C:\Users\HP\Desktop\PROJET_PFE\CBIR-SYS\Data\brats_subset_5k"

# Compter les chemins vides
empty = col.count_documents({"file_path": ""})
print(f"Documents avec file_path vide : {empty}")

# Lister les .pt disponibles
pt_files = sorted([f for f in os.listdir(LOCAL_DIR) if f.endswith(".pt")])
print(f"Fichiers .pt disponibles      : {len(pt_files)}")

# Récupérer TOUS les docs triés par global_idx
docs = list(col.find({}, {"_id": 1, "global_idx": 1, "file_path": 1})
              .sort("global_idx", 1))
print(f"Total documents               : {len(docs)}")

# Réassigner tous les chemins proprement
updated = 0
for doc, fname in zip(docs, pt_files):
    new_path = os.path.join(LOCAL_DIR, fname)
    col.update_one(
        {"_id": doc["_id"]},
        {"$set": {"file_path": new_path}}
    )
    updated += 1

print(f"✅ {updated} chemins mis à jour")

# Vérification
empty_after = col.count_documents({"file_path": ""})
print(f"Chemins vides après fix       : {empty_after}")

# Vérifier que les fichiers existent
sample = list(col.find({}, {"file_path": 1}).limit(3))
print(f"\nVérification existence fichiers :")
for s in sample:
    exists = os.path.exists(s["file_path"])
    print(f"  {'✅' if exists else '❌'} {s['file_path']}")