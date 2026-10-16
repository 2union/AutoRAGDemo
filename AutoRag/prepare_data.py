"""Шаг 1. Готовит corpus.parquet и qa.parquet для AutoRAG из Data/recipes.jsonl.

- corpus: один рецепт = один документ (рецепты короткие, агент выбирает рецепты целиком),
  текст в виде компактной «карточки» — экономит контекст gpt-5-nano.
- qa: синтетические русскоязычные запросы, сгенерированные LLM (у AutoRAG нет
  встроенных промптов на русском). Каждый запрос привязан к одному рецепту.

Запуск:  python prepare_data.py --sample 300
"""
import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from paths import DATA_FILE, DATASET_DIR as OUT_DIR, ENV_FILE, doc_uuid

load_dotenv(ENV_FILE)


def load_recipes() -> list[dict]:
    with DATA_FILE.open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    return [d for d in docs if d["metadata"].get("approved")]


def render_card(doc: dict) -> str:
    """Компактная карточка рецепта: всё, что нужно для выбора блюда, без шагов готовки."""
    m = doc["metadata"]
    allergens = ", ".join(m.get("allergens") or []) or "нет"
    return (
        f"ID: {doc['id']}\n"
        f"Название: {m.get('title')}\n"
        f"Тип блюда: {m.get('dish_type') or 'не указан'}\n"
        f"Ингредиенты: {', '.join(m.get('ingredients') or [])}\n"
        f"На 100 г: {m.get('kcal_per_100g')} ккал, белки {m.get('protein_per_100g')}, "
        f"жиры {m.get('fat_per_100g')}, углеводы {m.get('carbs_per_100g')}\n"
        f"Аллергены: {allergens}"
    )


def build_corpus(docs: list[dict], variant: str) -> pd.DataFrame:
    now = datetime.now()
    rows = []
    for d in docs:
        m = d["metadata"]
        rows.append({
            "doc_id": doc_uuid(d["id"]),  # Qdrant не принимает строковые ID
            "contents": render_card(d) if variant == "card" else d["text"],
            "path": m.get("file", ""),
            "metadata": {
                "last_modified_datetime": now,
                "recipe_id": d["id"],
                "title": m.get("title") or "",
                "dish_type": m.get("dish_type") or "",
            },
        })
    return pd.DataFrame(rows)


QA_PROMPT = """Ты помогаешь построить тестовый набор для поиска рецептов.
Пользователь — ИИ-агент шеф-повар, который ищет рецепты для меню в базе из ~2000 блюд.
По карточке рецепта придумай 3 разных поисковых запроса на русском, на которые
ЭТОТ рецепт — лучший ответ:
1. "ingredients": запрос по 2–4 характерным ингредиентам («что приготовить из ...»).
2. "description": описание желаемого блюда своими словами, БЕЗ названия рецепта.
3. "constraints": тип блюда + ключевой ингредиент + ограничение (аллерген, калорийность).
Не копируй название рецепта. Запросы должны быть естественными и короткими (до 20 слов).
Также дай "answer": одна фраза — «Подходит рецепт <название> (ID <id>)».
Верни JSON: {"ingredients": "...", "description": "...", "constraints": "...", "answer": "..."}

Карточка рецепта:
"""


def gen_queries(client: OpenAI, model: str, doc: dict) -> list[dict]:
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": QA_PROMPT + render_card(doc)}],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return [
                {
                    "qid": f"{doc['id']}::{kind}",
                    "query": data[kind].strip(),
                    "retrieval_gt": [[doc_uuid(doc["id"])]],
                    "generation_gt": [data["answer"].strip()],
                }
                for kind in ("ingredients", "description", "constraints")
                if data.get(kind)
            ]
        except Exception as e:  # noqa: BLE001
            print(f"retry {doc['id']}: {e}")
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300, help="сколько рецептов брать для QA")
    ap.add_argument("--test-share", type=float, default=0.3, help="доля отложенной выборки")
    ap.add_argument("--variant", choices=["card", "full"], default="card")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_recipes()
    build_corpus(docs, args.variant).to_parquet(OUT_DIR / "corpus.parquet", index=False)
    print(f"corpus: {len(docs)} docs ({args.variant})")

    # Стратифицированная выборка по типу блюда, чтобы редкие типы тоже попали в тест
    random.seed(42)
    by_type: dict[str, list[dict]] = {}
    for d in docs:
        by_type.setdefault(d["metadata"].get("dish_type") or "", []).append(d)
    sample = []
    for group in by_type.values():
        k = max(3, round(args.sample * len(group) / len(docs)))
        sample += random.sample(group, min(k, len(group)))

    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
    model = os.getenv("QA_MODEL", "gpt-5-mini")
    with ThreadPoolExecutor(args.workers) as pool:
        rows = [r for rs in pool.map(lambda d: gen_queries(client, model, d), sample) for r in rs]

    # Разбиение по рецептам, а не по запросам — нет утечки между dev и test
    ids = sorted({r["qid"].split("::")[0] for r in rows})
    random.shuffle(ids)
    test_ids = set(ids[: int(len(ids) * args.test_share)])
    qa = pd.DataFrame(rows)
    is_test = qa["qid"].str.split("::").str[0].isin(test_ids)
    qa[~is_test].to_parquet(OUT_DIR / "qa.parquet", index=False)
    qa[is_test].to_parquet(OUT_DIR / "qa_test.parquet", index=False)
    print(f"qa: {(~is_test).sum()} dev / {is_test.sum()} test queries")


if __name__ == "__main__":
    main()
