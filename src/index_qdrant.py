
from src.db.connections import get_qdrant_client, QDRANT_COLLECTION
from qdrant_client.models import PayloadSchemaType

client = get_qdrant_client()

# Index sur modalite pour les deux collections
for coll in [QDRANT_COLLECTION, "radimagenet_embeddings"]:
    client.create_payload_index(
        collection_name = coll,
        field_name      = "modalite",
        field_schema    = PayloadSchemaType.KEYWORD,
    )
    print(f"Index 'modalite' créé sur {coll}")