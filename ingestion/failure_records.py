"""
Embeds historical failure records and upserts them into Qdrant so they can be
retrieved by semantic similarity (RAG) when the Historian agent investigates
a live anomaly.
"""
import os
import sys
import json
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    QDRANT_URL,
    QDRANT_LOCAL_PATH,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
)

_EMBEDDER = None
_QCLIENT = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDER


def get_qdrant_client():
    global _QCLIENT
    if _QCLIENT is None:
        from qdrant_client import QdrantClient

        if QDRANT_URL:
            # Real Qdrant server (Docker or Qdrant Cloud)
            _QCLIENT = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
        else:
            # Embedded mode: runs in-process, persists to a local folder.
            # No server, no Docker, no virtualization needed.
            _QCLIENT = QdrantClient(path=QDRANT_LOCAL_PATH)

    return _QCLIENT


def _record_to_text(record: dict) -> str:
    """Flatten a failure record into a single text blob for embedding."""
    symptoms = ", ".join(record.get("symptoms", []))

    return (
        f"Equipment type: {record['equipment_type']}. "
        f"Title: {record['title']}. "
        f"Symptoms: {symptoms}. "
        f"Description: {record['description']}. "
        f"Root cause: {record['root_cause']}."
    )


def ensure_collection():
    from qdrant_client.models import Distance, VectorParams

    client = get_qdrant_client()

    existing = [
        c.name for c in client.get_collections().collections
    ]

    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            ),
        )


def seed(records_path: str = "data/sample_failure_records.json"):
    from qdrant_client.models import PointStruct

    with open(records_path) as f:
        records = json.load(f)

    ensure_collection()

    embedder = get_embedder()
    client = get_qdrant_client()

    texts = [_record_to_text(r) for r in records]

    vectors = embedder.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    points = [
        PointStruct(
            id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"failure-{r['id']}"
                )
            ),
            vector=vec,
            payload=r,
        )
        for r, vec in zip(records, vectors)
    ]

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points
    )

    print(
        f"Seeded {len(points)} failure records into "
        f"Qdrant collection '{QDRANT_COLLECTION}'"
    )


def search_similar_failures(
    query_text: str,
    equipment_type: str | None = None,
    top_k: int = 3
) -> list[dict]:
    """
    Semantic search over historical failure records.
    Optionally filter by equipment_type.

    Returns a list of payload dicts with an added `similarity` score.
    """
    from qdrant_client.models import (
        Filter,
        FieldCondition,
        MatchValue,
    )

    embedder = get_embedder()
    client = get_qdrant_client()

    query_vec = embedder.encode(
        [query_text],
        normalize_embeddings=True
    ).tolist()[0]

    qfilter = None

    if equipment_type:
        qfilter = Filter(
            must=[
                FieldCondition(
                    key="equipment_type",
                    match=MatchValue(value=equipment_type)
                )
            ]
        )

    # Qdrant newer versions use query_points() instead of search().
    search_result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vec,
        query_filter=qfilter,
        limit=top_k,
    )

    results = []

    for hit in search_result.points:
        payload = dict(hit.payload)

        payload["similarity"] = round(
            float(hit.score),
            4
        )

        results.append(payload)

    return results


if __name__ == "__main__":
    seed()