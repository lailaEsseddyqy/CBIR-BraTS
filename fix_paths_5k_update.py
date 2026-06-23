# fix_paths_5k_update.py
import sys, os
sys.path.append(".")
from src.db.connections import get_slices_collection
from pathlib import Path

col       = get_slices_collection()
LOCAL_DIR = r"C:\Users\HP\Desktop\PROJET_PFE\CBIR-SYS\data\brats_subset_5k"

pt_files = sorted([f for f in os.listdir(LOCAL_DIR) if f.endswith(".pt")])
docs     = list(col.find({}, {"_id": 1, "global_idx": 1}).sort("global_idx", 1))

print(f"Documents MongoDB : {len(docs)}")
print(f"Fichiers .pt      : {len(pt_files)}")

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
sample = col.find_one({}, {"slice_id": 1, "file_path": 1})
print(f"Exemple : {sample['slice_id']} → {sample['file_path']}")