# check_qdrant.py
import sys
sys.path.append(".")
from src.db.connections import get_qdrant_client

client = get_qdrant_client()
cols   = client.get_collections().collections
print(f"Collections existantes : {[c.name for c in cols]}")