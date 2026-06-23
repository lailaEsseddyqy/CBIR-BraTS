import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

# Charger les variables (QDRANT_URL, QDRANT_API_KEY)
load_dotenv()

def main():
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_key = os.getenv("QDRANT_API_KEY", "")
    
    print("Connexion à Qdrant...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    
    # Vos 3 collections actuelles
    collections = [
        "brats_embeddings", 
        "radimagenet_embeddings", 
        "brats_supcon_embeddings"
    ]
    
    for coll in collections:
        try:
            print(f"Création de l'index 'patient_id' pour {coll}...")
            client.create_payload_index(
                collection_name=coll,
                field_name="patient_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"✅ Index créé avec succès !")
        except Exception as e:
            # Si l'index existe déjà ou si la collection n'existe pas, on passe
            print(f"⚠️ Info sur {coll} : {e}")

if __name__ == "__main__":
    main()