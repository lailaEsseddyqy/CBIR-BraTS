# clean_mongodb.py
import sys, os
sys.path.append(".")
from src.db.connections import get_slices_collection

col = get_slices_collection()

LOCAL_DIR = r"C:\Users\HP\Desktop\PROJET_PFE\CBIR-SYS\Data\brats_subset_5k"

# Garder uniquement les documents dont le fichier existe localement
pt_files = set(os.listdir(LOCAL_DIR))

all_docs  = list(col.find({}, {"_id": 1, "file_path": 1}))
to_delete = []

for doc in all_docs:
    fname = os.path.basename(doc["file_path"])
    if fname not in pt_files:
        to_delete.append(doc["_id"])

print(f"Documents total      : {len(all_docs)}")
print(f"Documents à supprimer: {len(to_delete)}")

if to_delete:
    col.delete_many({"_id": {"$in": to_delete}})
    print(f"✅ {len(to_delete)} documents supprimés")

print(f"Documents restants   : {col.count_documents({})}")