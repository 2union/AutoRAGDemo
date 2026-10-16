"""Шаг 2. Запускает оптимизацию AutoRAG и извлекает лучший пайплайн.

Запуск:  python run_autorag.py [--config config.yaml]
Результат: $WORK_DIR/project/<N>/ (отчёты AutoRAG) и $WORK_DIR/best_config.yaml
"""
import argparse
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv


from paths import BEST_CONFIG, CONFIG_FILE, DATASET_DIR, ENV_FILE, PROJECT_DIR

load_dotenv(ENV_FILE)

# AutoRAG работает с OpenAI через несколько клиентов: официальный SDK читает OPENAI_BASE_URL,
# llama-index (эмбеддинги) — OPENAI_API_BASE. Синхронизируем, чтобы все шли на один адрес.
if os.getenv("OPENAI_BASE_URL"):
    os.environ.setdefault("OPENAI_API_BASE", os.environ["OPENAI_BASE_URL"])


def patch_gpt5():
    """gpt-5-nano может отсутствовать в таблице лимитов AutoRAG/tiktoken — добавляем."""
    try:
        from autorag.nodes.generator import openai_llm

        openai_llm.MAX_TOKEN_DICT.setdefault("gpt-5-nano", 400_000)
        openai_llm.MAX_TOKEN_DICT.setdefault("gpt-5-mini", 400_000)
    except Exception as e:  # noqa: BLE001
        print(f"gpt-5 patch skipped: {e}")
    try:
        import tiktoken.model

        tiktoken.model.MODEL_TO_ENCODING.setdefault("gpt-5-nano", "o200k_base")
        tiktoken.model.MODEL_TO_ENCODING.setdefault("gpt-5-mini", "o200k_base")
    except Exception:  # noqa: BLE001
        pass


def check_corpus_ids():
    """Qdrant принимает только UUID/int ID. Строковые doc_id (старый corpus.parquet)
    дают 400 "Expected some form of condition" глубоко внутри AutoRAG — проверяем заранее."""
    import pandas as pd

    for name in ("corpus.parquet", "qa.parquet"):
        df = pd.read_parquet(DATASET_DIR / name)
        ids = df["doc_id"] if "doc_id" in df else [i for gt in df["retrieval_gt"] for g in gt for i in g]
        bad = []
        for i in ids:
            try:
                uuid.UUID(str(i))
            except ValueError:
                bad.append(i)
        if bad:
            sys.exit(f"{name}: {len(bad)} ID не в формате UUID (например {bad[0]!r}).\n"
                     "Датасет собран старой версией prepare_data.py: пересоберите образ (make build) "
                     "и данные (rm -rf work/data work/project && make prepare).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_FILE))
    ap.add_argument("--project", default=str(PROJECT_DIR))
    args = ap.parse_args()

    # config.yaml ссылается на localhost; в Docker Qdrant доступен по QDRANT_URL (http://qdrant:6333)
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    resolved = PROJECT_DIR.parent / "config.resolved.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        Path(args.config).read_text(encoding="utf-8").replace("http://localhost:6333", qdrant_url),
        encoding="utf-8",
    )

    check_corpus_ids()
    patch_gpt5()
    from autorag.evaluator import Evaluator
    from autorag.deploy import extract_best_config

    Path(args.project).mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(
        qa_data_path=str(DATASET_DIR / "qa.parquet"),
        corpus_data_path=str(DATASET_DIR / "corpus.parquet"),
        project_dir=args.project,
    )
    evaluator.start_trial(str(resolved))

    trials = sorted((p for p in Path(args.project).iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    best = BEST_CONFIG
    extract_best_config(trial_path=str(trials[-1]), output_path=str(best))
    print(f"Лучший пайплайн: {best}\nОтчёт: autorag dashboard --trial_dir {trials[-1]}")


if __name__ == "__main__":
    main()
