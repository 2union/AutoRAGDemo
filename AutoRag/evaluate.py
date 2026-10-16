"""Шаг 4. Проверяет продакшн-коллекцию на отложенной выборке data/qa_test.parquet.

Запуск:  python evaluate.py
"""
import math

import pandas as pd

from paths import DATASET_DIR, doc_uuid
from retriever import build_filter, load_profile, search


def main():
    profile = load_profile()
    qa = pd.read_parquet(DATASET_DIR / "qa_test.parquet")
    recall = mrr = ndcg = 0.0
    for row in qa.itertuples():
        gt = set(row.retrieval_gt[0])
        ids = [doc_uuid(h["recipe_id"]) for h in search(row.query, build_filter(), profile=profile)]
        rank = next((i for i, rid in enumerate(ids) if rid in gt), None)
        if rank is not None:
            recall += 1
            mrr += 1 / (rank + 1)
            ndcg += 1 / math.log2(rank + 2)
    n = len(qa)
    print(f"queries={n} top_k={profile['top_k']}")
    print(f"recall@k={recall / n:.3f}  mrr={mrr / n:.3f}  ndcg={ndcg / n:.3f}")


if __name__ == "__main__":
    main()
