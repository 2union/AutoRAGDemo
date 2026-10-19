# Пример использования AutoRAG
**для построения качественного RAGа на базе Qdrant**

Данный репозиторий является приложением к статье [RAGу для программиста](https://habr.com/ru/companies/skbkontur/articles/1095364/)

## Суть задачи

В папке `Data` находится файл `recipes.jsonl` содержащий список рецептов. Наша задача построить RAG для ИИ-агента, на базе которого последний будет выбирать подходящие рецепты.



## Простое построение индекса

Папка `DirectIndex`: полный текст рецепта индексируется через `text-embedding-3-small` в Qdrant, поиск только dense. `rag_server.py` — FastAPI-сервер, который строит меню с помощью gpt-5-nano.

Для проведения индексации рецептов из корня проекта вызовите команду:
```bash
make -C DirectIndex index
```

Либо запустите сервис `direct-index-index` (профиль `index`). Он использует образ, который собирает `direct-index`:

```bash
docker compose build direct-index
docker compose run --rm direct-index-index
```

## Построение RAG с помощью AutoRAG

Папка `AutoRag`: AutoRAG сравнивает модели эмбеддингов, BM25, гибридный поиск, реранкер и промпты на синтетическом русскоязычном QA-датасете. Лучший пайплайн экспортируется в Qdrant (dense + sparse BM25 + payload-фильтры).

Для проведения индексации рецептов через AutoRAG из корня проекта вызовите команду:
```bash
make -C AutoRag pipeline
# или: docker compose up --build autorag-pipeline
```

Запуск RAG-сервера AutoRAG на :8001 (профиль `autorag`, без него `docker compose up` запустит только Qdrant и DirectIndex):
```bash
make -C AutoRag up
# или: docker compose --profile autorag up -d
```


**Документация:**
- [Docs/Docker.md](Docs/Docker.md) — подготовка Docker к работе с AutoRAG
- [Docs/DirectIndex.md](Docs/DirectIndex.md) — простая индексация и сервер поиска/генерации меню;
- [Docs/AutoRAG.md](Docs/AutoRAG.md) — подбор пайплайна AutoRAG, экспорт в Qdrant, интеграция с gpt-5-nano.

Все сервисы описаны в `docker-compose.yml`, каждый шаг запускается через `make` в соответствующей папке. У каждого решения один образ, который собирается один раз:

| Образ | Собирает | Использует |
|---|---|---|
| `rag-direct-index` | `direct-index` (сервер :8000) | `direct-index-index` (индексация) |
| `rag-autorag` | `autorag-pipeline` (пайплайн, шаги, дашборд :7690) | `autorag-server` (сервер :8001) |

! Docker Desktop на Mac не даёт контейнеру доступ к GPU, поэтому bge-m3 и реранкер в autorag будут работать на CPU заметно медленнее, чем при запуске без Docker. Это не должно быть проблемой, если вы запускаете Docker-контейнеры через [Colima](https://github.com/abiosoft/colima#features)
