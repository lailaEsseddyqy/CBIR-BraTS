# fix_paths_5k.py
import sys, os
sys.path.append(".")
from src.db.connections import get_slices_collection
from pathlib import Path

col = get_slices_collection()

LOCAL_DIR = r"C:\Users\HP\Desktop\PROJET_PFE\CBIR-SYS\data\brats_subset_5k"

# Vider l'ancienne collection et réindexer
col.delete_many({})
print("Ancienne collection MongoDB vidée")

pt_files = sorted([f for f in os.listdir(LOCAL_DIR) if f.endswith(".pt")])
print(f"Fichiers .pt trouvés : {len(pt_files)}")