import os
import httpx
from typing import Optional, Union
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").strip().lower() in ("true", "1", "yes", "y")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION = os.getenv("QDRANT_COLLECTION", "chef_recipes")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-5-nano")

def log_request(request: httpx.Request):
    if DEBUG:
        print(f"--- ПЛАН ЗАПРОСА ---")
        print(f"Метод: {request.method}")
        print(f"URL: {request.url}")
        print(f"Заголовки: {dict(request.headers)}")
        print(f"Тело (байты): {request.read().decode('utf-8')}")
        print(f"--------------------")

http_client = httpx.Client(event_hooks={'request': [log_request]})
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, http_client=http_client)
qdrant = QdrantClient(url=QDRANT_URL)

app = FastAPI(title="Chef RAG Server")

class MenuRequest(BaseModel):
    kcal_target: int = Field(..., description="Цель по ккал в день")
    days: int = Field(1, ge=1, le=7)
    meals_per_day: int = Field(1, ge=1, le=6)
    allergies: list[str] = []
    dislikes: list[str] = []
    preferred_dish_types: list[str] = []
    recently_dish_ids: list[str] = []
    user_notes: Optional[str] = None
    top_k: int = 40

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

class MenuResponse(BaseModel):
    menu: str
    recipes: list[RecipeHit]

class RecipeListResponse(BaseModel):
    recipes: list[RecipeHit]
    next_offset: Optional[Union[int, str]] = None

def embed_query(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding

def approved_condition() -> FieldCondition:
    return FieldCondition(
        key="approved",
        match=MatchValue(value=True),
    )


def build_recipes_filter(
    dish_type: Optional[str] = None,
    recipe_id: Optional[str] = None,
) -> Filter:
    must = [approved_condition()]

    if dish_type:
        must.append(
            FieldCondition(
                key="dish_type",
                match=MatchValue(value=dish_type),
            )
        )

    if recipe_id:
        must.append(
            FieldCondition(
                key="recipe_id",
                match=MatchValue(value=recipe_id),
            )
        )

    return Filter(must=must)


def payload_to_recipe(
    payload: dict,
    score: Optional[float] = None,
) -> RecipeHit:
    return RecipeHit(
        recipe_id=str(payload.get("recipe_id", "")),
        title=payload.get("title", ""),
        dish_type=payload.get("dish_type"),
        kcal_per_100g=payload.get("kcal_per_100g"),
        protein_per_100g=payload.get("protein_per_100g"),
        fat_per_100g=payload.get("fat_per_100g"),
        carbs_per_100g=payload.get("carbs_per_100g"),
        allergens=payload.get("allergens") or [],
        ingredients=payload.get("ingredients") or [],
        source=payload.get("source"),
        photo=payload.get("photo"),
        score=score,
    )

def build_filter(req: MenuRequest):
    must = [
        FieldCondition(
            key="approved",
            match=MatchValue(value=True),
        )
    ]

    if req.preferred_dish_types:
        must.append(
            FieldCondition(
                key="dish_type",
                match=MatchAny(any=req.preferred_dish_types),
            )
        )

    # Важно: Qdrant-фильтр по "не содержит аллерген" зависит от схемы.
    # Надежнее сначала retrieve, потом серверно исключить аллергены.
    return Filter(must=must)

def has_forbidden_allergen(payload: dict, allergies: list[str]) -> bool:
    recipe_allergens = [a.lower().strip() for a in payload.get("allergens", [])]
    user_allergies = [a.lower().strip() for a in allergies]

    return any(a in recipe_allergens for a in user_allergies)

def has_disliked_ingredient(payload: dict, dislikes: list[str]) -> bool:
    ingredients_text = " ".join(payload.get("ingredients", [])).lower()
    return any(d.lower().strip() in ingredients_text for d in dislikes)

def retrieve_recipes(req: MenuRequest) -> list[RecipeHit]:
    query = f"""
    Подбери рецепты для меню.
    Ккал в день: {req.kcal_target}
    Дней: {req.days}
    Приемов пищи в день: {req.meals_per_day}
    Аллергии: {", ".join(req.allergies) if req.allergies else "нет"}
    Нежелательные продукты: {", ".join(req.dislikes) if req.dislikes else "нет"}
    Типы блюд: {", ".join(req.preferred_dish_types) if req.preferred_dish_types else "любые"}
    Идентификаторы рецептов которые желательно не включать в меню: {", ".join(req.recently_dish_ids) if req.recently_dish_ids else "нет"}
    Комментарий пользователя: {req.user_notes or ""}
    """

    vector = embed_query(query)

    result = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=build_filter(req),
        limit=req.top_k * 2,
        with_payload=True,
    )

    hits = []

    for point in result.points:
        p = point.payload

        if has_forbidden_allergen(p, req.allergies):
            continue

        if has_disliked_ingredient(p, req.dislikes):
            continue

        hits.append(
            RecipeHit(
                recipe_id=p.get("recipe_id"),
                title=p.get("title"),
                dish_type=p.get("dish_type"),
                kcal_per_100g=p.get("kcal_per_100g"),
                protein_per_100g=p.get("protein_per_100g"),
                fat_per_100g=p.get("fat_per_100g"),
                carbs_per_100g=p.get("carbs_per_100g"),
                allergens=p.get("allergens") or [],
                ingredients=p.get("ingredients") or [],
                source=p.get("source"),
                photo=p.get("photo"),
                score=point.score,
            )
        )

        if len(hits) >= req.top_k:
            break

    return hits

def generate_menu(req: MenuRequest, recipes: list[RecipeHit]) -> str:
    recipes_context = "\n\n".join(
        f"""
ID: {r.recipe_id}
Название: {r.title}
Тип: {r.dish_type}
Ккал на 100 г: {r.kcal_per_100g}
БЖУ на 100 г: белки {r.protein_per_100g}, жиры {r.fat_per_100g}, углеводы {r.carbs_per_100g}
Аллергены: {", ".join(r.allergens) if r.allergens else "нет"}
Ингредиенты: {", ".join(r.ingredients)}
Источник: {r.source}
Фото: {r.photo}
""".strip()
        for r in recipes
    )

    system = """
Ты AI-агент шеф-повар.

Жесткие правила:
1. Используй только рецепты из переданного списка.
2. Не придумывай новые блюда.
3. Не используй блюда с аллергенами пользователя.
4. В обед и ужин всегда добавляй основное блюдо.
5. Если планируется три приёма пищи в обед так же добавляй салат и суп, если больше, то можно добавить салат и суп в другие приёмы.
6. В любой приём пищи добавляй напиток.
7. Не добаляй алкогольный напиток если пользователь не указал это напрямую.
8. Добавляй дисерт не чаще раза в день, но не реже раза в три дня.
9. Для каждого приема пищи укажи recipe_id.
10. Подбери граммовки так, чтобы приблизиться к целевой калорийности.
11. Если рецептов недостаточно, честно скажи об этом.
12. Ответ верни на русском языке.
"""

    user = f"""
Цель:
- Ккал в день: {req.kcal_target}
- Дней: {req.days}
- Приемов пищи в день: {req.meals_per_day}
- Аллергии: {", ".join(req.allergies) if req.allergies else "нет"}
- Нежелательные продукты: {", ".join(req.dislikes) if req.dislikes else "нет"}
- Комментарий: {req.user_notes or "нет"}

Одобренные найденные рецепты:
{recipes_context}

Сформируй меню.
Формат:
День N:
- Прием пищи N: название, recipe_id, граммовка, ккал, краткое объяснение
Итого за день: примерно X ккал
"""

    response = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    return response.choices[0].message.content


@app.post("/menu", response_model=MenuResponse)
def menu(req: MenuRequest):
    recipes = retrieve_recipes(req)

    if not recipes:
        return MenuResponse(
            menu="Не найдено подходящих одобренных рецептов с учетом ограничений.",
            recipes=[],
        )

    menu_text = generate_menu(req, recipes)

    return MenuResponse(
        menu=menu_text,
        recipes=recipes,
    )

@app.post("/recipes", response_model=RecipeListResponse)
def list_recipes(req: MenuRequest):
    recipes = retrieve_recipes(req)

    return RecipeListResponse(
        recipes=recipes,
        next_offset=req.top_k * 2,
    )


@app.get("/recipes/{recipe_id}", response_model=RecipeHit)
def get_recipe(recipe_id: str):
    records, _ = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=build_recipes_filter(recipe_id=recipe_id),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"Рецепт с id '{recipe_id}' не найден",
        )

    return payload_to_recipe(records[0].payload or {})


@app.get("/health")
def health():
    return {"status": "ok"}
