# reset_qdrant.py
import sys
sys.path.append(".")
from src.db.connections import get_qdrant_client, QDRANT_COLLECTION

client = get_qdrant_client()

# Supprimer les deux collections
for col in [QDRANT_COLLECTION, "resnet_embeddings"]:
    try:
        client.delete_collection(col)
        print(f"✅ Collection '{col}' supprimée")
    except Exception:
        print(f"⚠️ Collection '{col}' inexistante")