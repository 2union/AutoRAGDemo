# DirectIndex: простой dense-индекс + FastAPI-сервер.
# На хосте цели запускаются в контейнере через docker compose,
# внутри контейнера (IN_DOCKER=1) — выполняются напрямую.

COMPOSE  := docker compose -f ../docker-compose.yml
# direct-index (сервер) собирает образ rag-direct-index; direct-index-index — индексация в том же образе
SERVICE  := direct-index
INDEXER  := direct-index-index
PORT     ?= 8000

ifeq ($(IN_DOCKER),1)
RUN :=
else
RUN := $(COMPOSE) run --rm $(INDEXER)
endif

.PHONY: help install build qdrant index serve up down logs

help:
	@echo "make build   - собрать образ"
	@echo "make qdrant  - поднять Qdrant"
	@echo "make index   - проиндексировать /data/recipes.jsonl в Qdrant"
	@echo "make up      - запустить сервер (http://localhost:$(PORT))"
	@echo "make logs / make down"

install:
	pip install -r requirements.txt

build:
	$(COMPOSE) build $(SERVICE)

qdrant:
	$(COMPOSE) up -d qdrant

index:
ifeq ($(IN_DOCKER),1)
	python index_qdrant.py
else
	$(COMPOSE) build $(SERVICE)
	$(RUN)
endif

serve:
	uvicorn rag_server:app --host 0.0.0.0 --port $(PORT)

up:
	$(COMPOSE) up -d --build $(SERVICE)

logs:
	$(COMPOSE) logs -f $(SERVICE)

down:
	$(COMPOSE) stop $(SERVICE)
