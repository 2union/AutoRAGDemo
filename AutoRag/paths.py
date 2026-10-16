"""Общие пути. В Docker задаются переменными окружения (см. Dockerfile), локально — относительно репозитория."""
import os
import uuid
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(os.getenv("DATA_FILE", APP_DIR.parent / "Data" / "recipes.jsonl"))
WORK_DIR = Path(os.getenv("WORK_DIR", APP_DIR / "work"))

DATASET_DIR = WORK_DIR / "data"            # corpus.parquet, qa.parquet, qa_test.parquet
PROJECT_DIR = WORK_DIR / "project"         # трайлы AutoRAG
BEST_CONFIG = WORK_DIR / "best_config.yaml"
PROFILE_FILE = WORK_DIR / "rag_profile.json"
CONFIG_FILE = Path(os.getenv("AUTORAG_CONFIG", APP_DIR / "config.yaml"))
ENV_FILE = APP_DIR / ".env"


def doc_uuid(recipe_id: str) -> str:
    """ID документа для AutoRAG и Qdrant: Qdrant принимает только UUID или целые числа,
    а recipe_id вида "0679-recipe-details". Так же строятся ID точек в DirectIndex."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, recipe_id))
