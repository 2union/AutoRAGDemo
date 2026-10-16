import os
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION = os.getenv("QDRANT_COLLECTION", "chef_recipes")
# В Docker: /data/recipes.jsonl (см. Dockerfile); локально — Data/recipes.jsonl в корне репозитория
DATA_FILE = Path(os.getenv("DATA_FILE", Path(__file__).resolve().parent.parent / "Data" / "recipes.jsonl"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL)

def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]

def main():
    path = DATA_FILE
    docs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    test_vector = embed(["test"])[0]
    vector_size = len(test_vector)

    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    batch_size = 64

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        vectors = embed([d["text"] for d in batch])

        points = []
        for doc, vector in zip(batch, vectors):
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc["id"])),
                    vector=vector,
                    payload={
                        **doc["metadata"],
                        "text": doc["text"],
                        "recipe_id": doc["id"],
                    },
                )
            )

        qdrant.upsert(
            collection_name=COLLECTION,
            points=points,
        )

        print(f"Indexed {i + len(batch)} / {len(docs)}")

if __name__ == "__main__":
    main()
