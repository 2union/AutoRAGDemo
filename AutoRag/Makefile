# AutoRAG: подбор пайплайна и экспорт в Qdrant.
# На хосте каждая цель выполняется в контейнере (docker compose run),
# внутри контейнера (IN_DOCKER=1) — напрямую.
# Данные: /data/recipes.jsonl (read-only), артефакты: /work (= AutoRag/work на хосте).

COMPOSE := docker compose -f ../docker-compose.yml
# autorag-pipeline собирает образ rag-autorag и выполняет шаги; autorag-server — тот же образ
SERVICE := autorag-pipeline
SERVER  := autorag-server
PORT    ?= 8001

SAMPLE  ?= 300
VARIANT ?= card
QUERY   ?= что приготовить из курицы и брокколи
ARGS    ?=

ifeq ($(IN_DOCKER),1)
step = $(1)
else
step = $(COMPOSE) run --rm --build $(SERVICE) make $@ SAMPLE="$(SAMPLE)" VARIANT="$(VARIANT)" QUERY="$(QUERY)" ARGS="$(ARGS)"
endif

.PHONY: help install build qdrant prepare optimize export evaluate search serve up logs down dashboard pipeline shell clean

help:
	@echo "make build                 - собрать образ"
	@echo "make qdrant                - поднять Qdrant"
	@echo "make prepare [SAMPLE=300 VARIANT=card] - corpus + QA-датасет"
	@echo "make optimize              - запуск AutoRAG -> work/best_config.yaml"
	@echo "make export                - коллекция в Qdrant + work/rag_profile.json"
	@echo "make evaluate              - метрики на отложенной выборке"
	@echo "make search QUERY='...' [ARGS='--exclude-allergen Молоко --answer'] - поиск из CLI"
	@echo "make up                    - RAG-сервер FastAPI в фоне (http://localhost:$(PORT)/docs)"
	@echo "make serve                 - RAG-сервер в текущем терминале"
	@echo "make logs / make down       - логи / остановка сервера"
	@echo "make dashboard             - отчёт AutoRAG на http://localhost:7690"
	@echo "make pipeline              - prepare -> optimize -> export -> evaluate"
	@echo "make shell                 - bash в контейнере"

# Установка зависимостей (вызывается в Dockerfile). torch — CPU-сборка, чтобы не тянуть CUDA.
install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

build:
	$(COMPOSE) build $(SERVICE)

qdrant:
	$(COMPOSE) up -d qdrant

prepare:
	$(call step,python prepare_data.py --sample $(SAMPLE) --variant $(VARIANT) $(ARGS))

optimize:
	$(call step,python run_autorag.py $(ARGS))

export:
	$(call step,python export_qdrant.py --recreate $(ARGS))

evaluate:
	$(call step,python evaluate.py)

search:
	$(call step,python retriever.py "$(QUERY)" $(ARGS))

serve:
ifeq ($(IN_DOCKER),1)
	uvicorn retriever:app --host 0.0.0.0 --port $(PORT)
else
	$(COMPOSE) build $(SERVICE)
	$(COMPOSE) up $(SERVER)
endif

up: build
	$(COMPOSE) up -d $(SERVER)

logs:
	$(COMPOSE) logs -f $(SERVER)

down:
	$(COMPOSE) stop $(SERVER)

dashboard:
ifeq ($(IN_DOCKER),1)
	autorag dashboard --trial_dir $$(ls -d /work/project/[0-9]* | sort -V | tail -1) --port 7690
else
	$(COMPOSE) run --rm --build --service-ports $(SERVICE) make dashboard
endif

# Полный пайплайн сборки RAG в Qdrant в одном контейнере
ifeq ($(IN_DOCKER),1)
pipeline: prepare optimize export evaluate
else
pipeline: qdrant
	$(COMPOSE) run --rm --build $(SERVICE) make pipeline SAMPLE="$(SAMPLE)" VARIANT="$(VARIANT)"
endif

shell:
	$(COMPOSE) run --rm --build $(SERVICE) bash

clean:
	rm -rf work
