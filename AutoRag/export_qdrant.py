"""Шаг 3. Переносит лучший пайплайн AutoRAG в продакшн-коллекцию Qdrant.

Читает best_config.yaml -> пишет rag_profile.json -> индексирует corpus.parquet
в коллекцию с named vectors: "dense" (лучшая модель) и "bm25" (sparse, IDF на стороне Qdrant).

Запуск:  python export_qdrant.py [--recreate]
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from paths import BEST_CONFIG, DATA_FILE, DATASET_DIR, ENV_FILE, PROFILE_FILE, doc_uuid
from retriever import embed_dense, embed_sparse

load_dotenv(ENV_FILE)

# Значения по умолчанию, если best_config.yaml не найден или в нём нет узла
DEFAULT_PROFILE = {
    "collection": "chef_recipes_autorag",
    "embedding": "openai_embed_3_small",
    "mode": "hybrid",          # dense | hybrid
    "fusion": "RRF",           # RRF | DBSF
    "reranker": None,          # например BAAI/bge-reranker-v2-m3
    "candidates": 20,          # сколько кандидатов достаём до реранкера
    "top_k": 5,                # сколько рецептов уходит в контекст gpt-5-nano
    "prompt": "Запрос: {query}\n\nРецепты:\n{retrieved_contents}",
}


def profile_from_best(best_path: Path) -> dict:
    profile = dict(DEFAULT_PROFILE)
    if not best_path.exists():
        print(f"{best_path} не найден — используется профиль по умолчанию")
        return profile
    cfg = yaml.safe_load(best_path.read_text(encoding="utf-8"))
    vdb = {v["name"]: v["embedding_model"] for v in cfg.get("vectordb", [])}
    for line in cfg.get("node_lines", []):
        for node in line.get("nodes", []):
            mod = node["modules"][0]
            t, mtype = node["node_type"], mod["module_type"]
            if t == "semantic_retrieval":
                profile["embedding"] = vdb.get(mod.get("vectordb"), profile["embedding"])
                profile["mode"] = "dense"
                profile["candidates"] = node.get("top_k", profile["candidates"])
            elif t == "lexical_retrieval" and profile["mode"] != "dense":
                profile["mode"] = "hybrid"  # чистый BM25 в проде заменяем гибридом
            elif t == "hybrid_retrieval":
                profile["mode"] = "hybrid"
                # hybrid_cc в Qdrant ближе всего к DBSF (нормализация + взвешенная сумма)
                profile["fusion"] = "RRF" if mtype == "hybrid_rrf" else "DBSF"
                profile["candidates"] = node.get("top_k", profile["candidates"])
            elif t == "passage_reranker":
                profile["reranker"] = None if mtype == "pass_reranker" else mod.get("model_name")
                profile["top_k"] = node.get("top_k", profile["top_k"])
            elif t == "prompt_maker":
                profile["prompt"] = mod.get("prompt", profile["prompt"])
    return profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default=str(BEST_CONFIG))
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    profile = profile_from_best(Path(args.best))
    profile["collection"] = os.getenv("QDRANT_COLLECTION", profile["collection"])
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, indent=2))

    corpus = pd.read_parquet(DATASET_DIR / "corpus.parquet")
    meta = {}
    with DATA_FILE.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta[doc_uuid(d["id"])] = d

    qdrant = QdrantClient(url=os.getenv("QDRANT_URL"))
    name = profile["collection"]
    dim = len(embed_dense(["test"], profile["embedding"])[0])
    if args.recreate and qdrant.collection_exists(name):
        qdrant.delete_collection(name)
    if not qdrant.collection_exists(name):
        qdrant.create_collection(
            name,
            vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field, schema in [("approved", "bool"), ("dish_type", "keyword"), ("recipe_id", "keyword"),
                              ("allergens_lc", "keyword"), ("kcal_per_100g", "float")]:
            qdrant.create_payload_index(name, field, field_schema=schema)

    rows = corpus.to_dict("records")
    for i in range(0, len(rows), args.batch):
        batch = rows[i:i + args.batch]
        texts = [r["contents"] for r in batch]
        dense = embed_dense(texts, profile["embedding"])
        sparse = embed_sparse(texts)
        points = []
        for r, dv, sv in zip(batch, dense, sparse):
            src = meta[r["doc_id"]]
            m = src["metadata"]
            points.append(models.PointStruct(
                id=r["doc_id"],  # уже uuid5(recipe_id), см. paths.doc_uuid
                vector={"dense": dv, "bm25": sv},
                payload={
                    **m,
                    "recipe_id": src["id"],
                    "contents": r["contents"],   # то, что уходит в контекст LLM
                    "text": src["text"],         # полный рецепт для показа пользователю
                    "allergens_lc": [a.lower().strip() for a in m.get("allergens") or []],
                },
            ))
        qdrant.upsert(name, points=points)
        print(f"Indexed {i + len(batch)} / {len(rows)}")


if __name__ == "__main__":
    main()
