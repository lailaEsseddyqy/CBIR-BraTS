"""
src/db/connections.py
Connexions Qdrant + MongoDB Atlas avec test de sante.
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pymongo import MongoClient


load_dotenv()

# ── Constantes depuis .env ────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "brats_embeddings")
LATENT_DIM        = int(os.getenv("LATENT_DIM", 256))

MONGO_URI         = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB          = os.getenv("MONGO_DB", "cbmir_db")
MONGO_COLLECTION  = os.getenv("MONGO_COLLECTION", "slices_metadata")


# ── Qdrant ────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    kwargs = {"url": QDRANT_URL,
              "timeout": 120,   
    }
    
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return QdrantClient(**kwargs)


def ensure_qdrant_collection(client=None) -> None:
    """Crée la collection si elle n'existe pas encore."""
    client = client or get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=LATENT_DIM,
                distance=Distance.COSINE
            ),
        )
        print(f"[Qdrant] Collection '{QDRANT_COLLECTION}' créée.")
    else:
        print(f"[Qdrant] Collection '{QDRANT_COLLECTION}' déjà existante.")


# ── MongoDB Atlas ─────────────────────────────────────────
@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    return MongoClient(MONGO_URI,
                       serverSelectionTimeoutMS=7000,
                    connectTimeoutMS=7000
                       
                       )


def get_slices_collection():
    """Retourne la collection MongoDB avec index unique sur slice_id."""
    col = get_mongo_client()[MONGO_DB][MONGO_COLLECTION]
    col.create_index("slice_id", unique=True)
    return col


# ── Test de connexion ─────────────────────────────────────
def test_connections() -> dict:
    results = {}

    try:
        info = get_qdrant_client().get_collections()
        results["qdrant"] = {
            "status": "ok",
            "collections": len(info.collections)
        }
    except Exception as e:
        results["qdrant"] = {"status": "erreur", "detail": str(e)}

    try:
        get_mongo_client().admin.command("ping")
        results["mongodb"] = {"status": "ok"}
    except Exception as e:
        results["mongodb"] = {"status": "erreur", "detail": str(e)}

    return results


if __name__ == "__main__":
    print("Test des connexions...")
    result = test_connections()
    for db, info in result.items():
        print(f"  {db}: {info}")