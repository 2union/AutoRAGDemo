FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    IN_DOCKER=1 \
    DATA_FILE=/data/recipes.jsonl \
    QDRANT_URL=http://qdrant:6333

RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt Makefile ./
RUN make install
COPY . .

EXPOSE 8000
CMD ["make", "serve"]
