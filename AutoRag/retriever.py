"""Продакшн RAG-сервер (FastAPI) по профилю, найденному AutoRAG (work/rag_profile.json).

Поиск: dense (+ BM25 sparse при hybrid) с фильтрами Qdrant -> реранкер -> компактные карточки для gpt-5-nano.
API совместим с DirectIndex/rag_server.py (/menu, /recipes, /recipes/{id}, /health) и дополнен /search.
В отличие от DirectIndex, меню собирается отдельным поиском на каждый слот (тип блюда).

Сервер:  uvicorn retriever:app --host 0.0.0.0 --port 8001
CLI:     python retriever.py "лёгкий суп без молока" --dish-type Суп --exclude-allergen Молоко
"""
import argparse
import json
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from paths import ENV_FILE, PROFILE_FILE

load_dotenv(ENV_FILE)

DEBUG = os.getenv("DEBUG", "false").strip().lower() in ("true", "1", "yes", "y")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-5-nano")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")  # пусто — не передавать

# Соответствие имён эмбеддингов AutoRAG реальным моделям
EMBEDDINGS = {
    "openai_embed_3_small": ("openai", "text-embedding-3-small"),
    "openai_embed_3_large": ("openai", "text-embedding-3-large"),
    "huggingface_bge_m3": ("hf", "BAAI/bge-m3"),
}
SPARSE_MODEL = "Qdrant/bm25"


# ---------------------------------------------------------------- клиенты и модели

def load_profile() -> dict:
    return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))


@lru_cache
def get_profile() -> dict:
    return load_profile()


def _log_request(request):
    if DEBUG:
        headers = {k: ("***" if k.lower() == "authorization" else v) for k, v in request.headers.items()}
        print(f"--- OpenAI {request.method} {request.url}\n{headers}\n{request.read().decode('utf-8', 'replace')}")


@lru_cache
def _openai():
    from openai import DefaultHttpxClient, OpenAI

    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        http_client=DefaultHttpxClient(event_hooks={"request": [_log_request]}),
    )


@lru_cache
def _qdrant() -> QdrantClient:
    return QdrantClient(url=os.getenv("QDRANT_URL"))


def _collection(profile: dict) -> str:
    return os.getenv("QDRANT_COLLECTION", profile["collection"])


@lru_cache
def _hf(model: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model)


def embed_dense(texts: list[str], embedding: str) -> list[list[float]]:
    kind, model = EMBEDDINGS[embedding]
    if kind == "openai":
        resp = _openai().embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
    return _hf(model).encode(texts, normalize_embeddings=True).tolist()


@lru_cache
def _sparse():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(SPARSE_MODEL, language="russian")


def embed_sparse(texts: list[str], query: bool = False) -> list[models.SparseVector]:
    fn = _sparse().query_embed if query else _sparse().embed
    return [models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in fn(texts)]


@lru_cache
def _reranker(model: str):
    from FlagEmbedding import FlagReranker

    return FlagReranker(model, use_fp16=False)


def chat(messages: list[dict]) -> str:
    kwargs = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
    resp = _openai().chat.completions.create(model=GENERATION_MODEL, messages=messages, **kwargs)
    return resp.choices[0].message.content


def build_filter(
    dish_types: list[str] | None = None,
    exclude_allergens: list[str] | None = None,
    exclude_ids: list[str] | None = None,
    max_kcal: float | None = None,
    recipe_id: str | None = None,
) -> models.Filter:
    must = [models.FieldCondition(key="approved", match=models.MatchValue(value=True))]
    must_not = []
    if dish_types:
        must.append(models.FieldCondition(key="dish_type", match=models.MatchAny(any=dish_types)))
    if max_kcal is not None:
        must.append(models.FieldCondition(key="kcal_per_100g", range=models.Range(lte=max_kcal)))
    if recipe_id:
        must.append(models.FieldCondition(key="recipe_id", match=models.MatchValue(value=recipe_id)))
    if exclude_allergens:
        tags = [a.lower().strip() for a in exclude_allergens]
        must_not.append(models.FieldCondition(key="allergens_lc", match=models.MatchAny(any=tags)))
    if exclude_ids:
        must_not.append(models.FieldCondition(key="recipe_id", match=models.MatchAny(any=exclude_ids)))
    return models.Filter(must=must, must_not=must_not or None)


def search(query: str, query_filter: models.Filter | None = None, top_k: int | None = None,
           profile: dict | None = None, qdrant: QdrantClient | None = None) -> list[dict]:
    p = profile or get_profile()
    qdrant = qdrant or _qdrant()
    top_k = top_k or p["top_k"]
    candidates = max(p["candidates"], top_k)
    dense = embed_dense([query], p["embedding"])[0]

    if p["mode"] == "hybrid":
        res = qdrant.query_points(
            collection_name=_collection(p),
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=candidates, filter=query_filter),
                models.Prefetch(query=embed_sparse([query], query=True)[0], using="bm25",
                                limit=candidates, filter=query_filter),
            ],
            query=models.FusionQuery(fusion=models.Fusion[p["fusion"]]),
            limit=candidates,
            with_payload=True,
        )
    else:
        res = qdrant.query_points(collection_name=_collection(p), query=dense, using="dense",
                                  query_filter=query_filter, limit=candidates, with_payload=True)

    hits = [{**pt.payload, "score": pt.score} for pt in res.points]
    if p.get("reranker") and hits:
        scores = _reranker(p["reranker"]).compute_score(
            [[query, h["contents"]] for h in hits], normalize=True)
        scores = scores if isinstance(scores, list) else [scores]
        for h, s in zip(hits, scores):
            h["score"] = float(s)
        hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


def violates(hit: dict, allergies: list[str], dislikes: list[str]) -> bool:
    """Страховка поверх фильтра Qdrant: аллергены в данных не нормализованы,
    поэтому дополнительно ищем аллергию подстрокой («глютен» в «Глютен (пшеница)»)."""
    recipe_allergens = [a.lower() for a in hit.get("allergens") or []]
    ingredients = " ".join(hit.get("ingredients") or []).lower()
    if any(u.lower().strip() in a for u in allergies for a in recipe_allergens):
        return True
    return any(d.lower().strip() in ingredients for d in dislikes)


def format_context(hits: list[dict]) -> str:
    """Контекст для gpt-5-nano: компактные карточки, разделённые пустой строкой (как в AutoRAG)."""
    return "\n\n".join(h["contents"] for h in hits)


def build_prompt(query: str, hits: list[dict], profile: dict | None = None) -> str:
    p = profile or get_profile()
    return p["prompt"].replace("{query}", query).replace("{retrieved_contents}", format_context(hits))


class RecipeHit(BaseModel):
    recipe_id: str
    title: str
    dish_type: Optional[str]
    kcal_per_100g: Optional[float]
    protein_per_100g: Optional[float]
    fat_per_100g: Optional[float]
    carbs_per_100g: Optional[float]
    allergens: list[str]
    ingredients: list[str]
    source: Optional[str]
    photo: Optional[str]
    score: Optional[float] = None


class MenuRequest(BaseModel):
    kcal_target: int = Field(..., description="Цель по ккал в день")
    days: int = Field(1, ge=1, le=7)
    meals_per_day: int = Field(1, ge=1, le=6)
    allergies: list[str] = []
    dislikes: list[str] = []
    preferred_dish_types: list[str] = Field([], description="Слоты меню; по умолчанию выводятся из meals_per_day")
    recently_dish_ids: list[str] = []
    user_notes: Optional[str] = None
    top_k: Optional[int] = Field(None, description="Рецептов на слот; по умолчанию из rag_profile.json")


class MenuSlot(BaseModel):
    dish_type: str
    recipes: list[RecipeHit]


class MenuResponse(BaseModel):
    menu: str
    recipes: list[RecipeHit]
    slots: list[MenuSlot]


class RecipeListResponse(BaseModel):
    recipes: list[RecipeHit]
    slots: list[MenuSlot]


class SearchRequest(BaseModel):
    query: str
    dish_types: list[str] = []
    exclude_allergens: list[str] = []
    exclude_ids: list[str] = []
    dislikes: list[str] = []
    max_kcal_per_100g: Optional[float] = None
    top_k: Optional[int] = None
    answer: bool = Field(False, description="Сгенерировать ответ gpt-5-nano промптом из rag_profile.json")


class SearchResponse(BaseModel):
    recipes: list[RecipeHit]
    context: str
    answer: Optional[str] = None


def to_recipe(h: dict) -> RecipeHit:
    return RecipeHit(
        recipe_id=str(h.get("recipe_id", "")),
        title=h.get("title") or "",
        dish_type=h.get("dish_type"),
        kcal_per_100g=h.get("kcal_per_100g"),
        protein_per_100g=h.get("protein_per_100g"),
        fat_per_100g=h.get("fat_per_100g"),
        carbs_per_100g=h.get("carbs_per_100g"),
        allergens=h.get("allergens") or [],
        ingredients=h.get("ingredients") or [],
        source=h.get("source"),
        photo=h.get("photo"),
        score=h.get("score"),
    )


MAIN_DISH = "Основное блюдо"
# (тип блюда, минимальное число приёмов пищи, при котором слот нужен)
SLOT_RULES = [("Завтрак", 2), (MAIN_DISH, 1), ("Суп", 3), ("Салат", 3), ("Гарнир", 4),
              ("Закуска", 4), ("Снэк", 4), ("Напиток", 1), ("Десерт", 1)]


def menu_slots(req: MenuRequest) -> list[str]:
    if req.preferred_dish_types:
        return list(dict.fromkeys(req.preferred_dish_types))
    return [t for t, min_meals in SLOT_RULES if req.meals_per_day >= min_meals]


def retrieve_menu(req: MenuRequest) -> list[tuple[str, list[dict]]]:
    """Отдельный поиск на каждый слот: фильтр по dish_type, аллергенам и недавним блюдам."""
    p = get_profile()
    seen: set[str] = set(req.recently_dish_ids)
    result = []
    for slot in menu_slots(req):
        # основное блюдо нужно на обед и ужин каждого дня — берём больше вариантов
        need = req.days * (2 if slot == MAIN_DISH else 1)
        limit = req.top_k or min(p["candidates"], max(p["top_k"], need))
        query = f"{slot}. {req.user_notes}" if req.user_notes else slot
        flt = build_filter([slot], req.allergies, list(seen))
        hits = [h for h in search(query, flt, top_k=limit * 2, profile=p)
                if not violates(h, req.allergies, req.dislikes)][:limit]
        seen.update(h["recipe_id"] for h in hits)
        result.append((slot, hits))
    return result


MENU_SYSTEM = """
Ты AI-агент шеф-повар.

Жесткие правила:
1. Используй только рецепты из переданного списка, ссылайся на них по ID.
2. Не придумывай новые блюда.
3. Не используй блюда с аллергенами пользователя.
4. В обед и ужин всегда добавляй основное блюдо.
5. Если планируется три приёма пищи, в обед также добавляй салат и суп; если больше — салат и суп можно добавить в другие приёмы.
6. В любой приём пищи добавляй напиток.
7. Не добавляй алкогольный напиток, если пользователь не указал это напрямую.
8. Добавляй десерт не чаще раза в день, но не реже раза в три дня.
9. Подбери граммовки так, чтобы приблизиться к целевой калорийности (КБЖУ указаны на 100 г).
10. Если рецептов недостаточно, честно скажи об этом.
11. Ответ верни на русском языке.
"""


def generate_menu(req: MenuRequest, slots: list[tuple[str, list[dict]]]) -> str:
    context = "\n\n".join(f"### {slot}\n\n{format_context(hits)}" for slot, hits in slots if hits)
    user = f"""
Цель:
- Ккал в день: {req.kcal_target}
- Дней: {req.days}
- Приемов пищи в день: {req.meals_per_day}
- Аллергии: {", ".join(req.allergies) if req.allergies else "нет"}
- Нежелательные продукты: {", ".join(req.dislikes) if req.dislikes else "нет"}
- Комментарий: {req.user_notes or "нет"}

Найденные рецепты по типам блюд:
{context}

Сформируй меню.
Формат:
День N:
- Прием пищи N: название, recipe_id, граммовка, ккал, краткое объяснение
Итого за день: примерно X ккал
"""
    return chat([{"role": "system", "content": MENU_SYSTEM}, {"role": "user", "content": user}])


def _slots_response(slots: list[tuple[str, list[dict]]]) -> tuple[list[RecipeHit], list[MenuSlot]]:
    menu_slots_ = [MenuSlot(dish_type=s, recipes=[to_recipe(h) for h in hits]) for s, hits in slots]
    return [r for s in menu_slots_ for r in s.recipes], menu_slots_


app = FastAPI(title="Chef RAG Server (AutoRAG)")


@app.post("/menu", response_model=MenuResponse)
def menu(req: MenuRequest):
    slots = retrieve_menu(req)
    recipes, menu_slots_ = _slots_response(slots)
    if not recipes:
        return MenuResponse(menu="Не найдено подходящих одобренных рецептов с учетом ограничений.",
                            recipes=[], slots=menu_slots_)
    return MenuResponse(menu=generate_menu(req, slots), recipes=recipes, slots=menu_slots_)


@app.post("/recipes", response_model=RecipeListResponse)
def list_recipes(req: MenuRequest):
    recipes, menu_slots_ = _slots_response(retrieve_menu(req))
    return RecipeListResponse(recipes=recipes, slots=menu_slots_)


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest):
    flt = build_filter(req.dish_types, req.exclude_allergens, req.exclude_ids, req.max_kcal_per_100g)
    top_k = req.top_k or get_profile()["top_k"]
    hits = [h for h in search(req.query, flt, top_k=top_k * 2)
            if not violates(h, req.exclude_allergens, req.dislikes)][:top_k]
    answer = chat([{"role": "user", "content": build_prompt(req.query, hits)}]) if req.answer and hits else None
    return SearchResponse(recipes=[to_recipe(h) for h in hits], context=format_context(hits), answer=answer)


@app.get("/recipes/{recipe_id}", response_model=RecipeHit)
def get_recipe(recipe_id: str):
    records, _ = _qdrant().scroll(
        collection_name=_collection(get_profile()),
        scroll_filter=build_filter(recipe_id=recipe_id),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        raise HTTPException(status_code=404, detail=f"Рецепт с id '{recipe_id}' не найден")
    return to_recipe(records[0].payload or {})


@app.get("/health")
def health():
    try:
        p = get_profile()
    except FileNotFoundError:
        return {"status": "no_profile", "detail": f"{PROFILE_FILE} не найден — выполните make export"}
    return {"status": "ok", "profile": {k: p[k] for k in ("mode", "embedding", "fusion", "reranker", "top_k")}}


def main():
    """
    CLI
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--dish-type", action="append")
    ap.add_argument("--exclude-allergen", action="append")
    ap.add_argument("--max-kcal", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--answer", action="store_true", help="сгенерировать ответ gpt-5-nano")
    args = ap.parse_args()

    flt = build_filter(args.dish_type, args.exclude_allergen, max_kcal=args.max_kcal)
    hits = search(args.query, flt, args.top_k)
    for h in hits:
        print(f"{h['score']:.3f}  {h['recipe_id']}  {h['title']}")
    if args.answer:
        print("\n" + chat([{"role": "user", "content": build_prompt(args.query, hits)}]))


if __name__ == "__main__":
    main()
