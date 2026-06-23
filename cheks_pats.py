# check_paths.py
import sys
sys.path.append(".")
from src.db.connections import get_slices_collection

col  = get_slices_collection()
docs = list(col.find({}, {"file_path": 1}).limit(3))
for d in docs:
    print(d["file_path"])