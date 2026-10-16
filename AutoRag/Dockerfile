# AutoRAG поддерживает Python 3.10–3.12
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    IN_DOCKER=1 \
    DATA_FILE=/data/recipes.jsonl \
    WORK_DIR=/work \
    HF_HOME=/cache/huggingface \
    QDRANT_URL=http://qdrant:6333

RUN apt-get update && apt-get install -y --no-install-recommends make build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt Makefile ./
RUN make install
COPY . .

VOLUME ["/work", "/cache"]
# 8001 — RAG-сервер (make serve), 7690 — дашборд AutoRAG
EXPOSE 8001 7690
# По умолчанию — полный пайплайн: prepare -> optimize -> export -> evaluate
CMD ["make", "pipeline"]
