# fix_all_paths.py
import sys, os
sys.path.append(".")
from src.db.connections import get_slices_collection
from pathlib import Path

col       = get_slices_collection()
LOCAL_DIR = r"C:\Users\HP\Desktop\PROJET_PFE\CBIR-SYS\Data\brats_subset_5k"

# Lister tous les .pt disponibles localement
pt_files = sorted([f for f in os.listdir(LOCAL_DIR) if f.endswith(".pt")])
print(f"Fichiers .pt locaux : {len(pt_files)}")

# Vérifier combien ont encore de mauvais chemins
bad = col.count_documents({
    "file_path": {"$not": {"$regex": "brats_subset_5k"}}
})
print(f"Documents avec mauvais chemin : {bad}")

# Corriger TOUS les documents par ordre global_idx
docs = list(col.find({}, {"_id": 1, "global_idx": 1}).sort("global_idx", 1))
print(f"Total documents : {len(docs)}")

updated = 0
for doc, fname in zip(docs, pt_files):
    new_path = os.path.join(LOCAL_DIR, fname)
    col.update_one(
        {"_id": doc["_id"]},
        {"$set": {"file_path": new_path}}
    )
    updated += 1

print(f"✅ {updated} chemins corrigés")

# Vérification finale
sample = col.find_one({}, {"slice_id": 1, "file_path": 1})
print(f"\nExemple : {sample['slice_id']}")
print(f"Chemin  : {sample['file_path']}")

# Vérifier que le fichier existe bien
exists = os.path.exists(sample["file_path"])
print(f"Fichier existe : {'✅ OUI' if exists else '❌ NON'}")