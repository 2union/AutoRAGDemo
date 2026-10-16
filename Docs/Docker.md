# Запуск Docker с увеличенными ресурсами

Ресурсов, которые Docker выделяет по умолчанию, контейнеру `autorag-pipeline` не хватит. Ниже объясняется почему и как увеличить лимиты в каждой ОС. Сервисам `qdrant`, `direct-index` и `autorag-server` без реранкера настройки по умолчанию подходят.

## Почему ресурсов по умолчанию не хватает

На macOS и Windows Docker работает внутри виртуальной машины. Контейнеры видят только CPU, память и диск этой ВМ, а не ресурсы всего компьютера. У Colima по умолчанию это **2 CPU и 2 GiB RAM**. У Docker Desktop лимиты зависят от версии и обычно тоже меньше, чем нужно `autorag-pipeline`.

Что потребляет ресурсы во время `make pipeline`:

| Что | Память | Почему |
|---|---|---|
| `BAAI/bge-m3` (эмбеддинги, узел `semantic_retrieval`, vectordb `bge_m3`) | ~2,5–4 GB | 568 млн параметров в fp32, это ~2,3 GB весов плюс активации. Рецепты длинные (до ~4900 символов), а память на attention растёт с длиной текста и размером батча |
| `BAAI/bge-reranker-v2-m3` (узел `passage_reranker`) | ~2,5–3 GB | Модель того же размера. Она оценивает каждую пару «запрос + 20 кандидатов» для всех запросов из `qa.parquet` |
| Токенизатор bge-m3 для BM25 (`lexical_retrieval`) | ~0,5 GB | Токенизирует весь корпус и строит индекс в памяти |
| Сам процесс AutoRAG: torch, transformers, llama-index, pandas | ~1–1,5 GB | Тяжёлые зависимости загружаются в память ещё до работы моделей |
| Qdrant в той же ВМ | ~0,5–1 GB | 3 коллекции AutoRAG (в том числе `text-embedding-3-large`, 3072 измерения) и продовая коллекция |

Модели загружаются в одном Python-процессе. Пиковое потребление получается **6–8 GB**. Если памяти не хватает, ядро ВМ убивает процесс (OOM killer). Контейнер завершается с кодом `137` или сообщением `Killed`, обычно без понятной ошибки в логе. Это самый частый признак нехватки ресурсов.

**CPU.** Внутри контейнера нет GPU и MPS, поэтому torch считает bge-m3 и реранкер на процессоре. Инференс распараллеливается по ядрам. На 2 ядрах эмбеддинг корпуса и реранкинг займут в несколько раз больше времени, чем на 4–8, и прогон может растянуться на часы.

**Диск.** Образ `rag-autorag` весит несколько GB (torch, transformers, зависимости AutoRAG). Модели в volume `hf_cache` занимают ещё ~4,5 GB. К этому добавляются данные Qdrant и `work/project`. Вместе это 15–20 GB, а с запасом под сборку слоёв и старые образы нужно 60+ GB.

**Рекомендуемый минимум для ВМ Docker:** 4 CPU, 10 GB RAM, 60 GB диска. Если ресурсов больше, выделяйте до 6–8 CPU: прогон пройдёт быстрее.

> Если увеличить ресурсы нельзя, уберите локальные модели из `AutoRag/config.yaml`: vectordb `bge_m3` с соответствующим модулем `semantic_retrieval` и `flag_embedding_reranker`. Тогда остаются только эмбеддинги OpenAI и BM25, и хватит 4 GB RAM. Но при этом теряются кандидаты, которые лучше всего работают с русским языком.

## macOS

### Colima

```bash
colima stop
colima start --cpu 4 --memory 10 --disk 100 --vm-type=vz --mount-type virtiofs
```

- `--vm-type=vz`: Apple Virtualization.framework, на Apple Silicon работает быстрее QEMU.
- `--mount-type virtiofs`: быстрое монтирование `./Data` и `./AutoRag/work`.
- `--disk` задаётся в GiB. Уменьшить размер диска существующей ВМ нельзя. Если тип ВМ не меняется, сделайте `colima delete` и создайте ВМ заново.

Чтобы параметры сохранились, их можно прописать в `~/.colima/default/colima.yaml` (`cpu`, `memory`, `disk`) или открыть этот файл командой `colima start --edit`.

### Docker Desktop

**Settings → Resources → Advanced:** CPU limit 4+, Memory limit 10 GB+, Swap 2–4 GB, Virtual disk limit 60 GB+. Затем нажмите **Apply & restart**. Для скорости монтирования в **Settings → General** выберите VirtioFS.

## Windows

### Docker Desktop с WSL 2 (по умолчанию)

В режиме WSL 2 ползунков памяти и CPU в Docker Desktop нет: лимиты задаёт сама WSL. Создайте или отредактируйте файл `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=10GB
processors=4
swap=4GB
```

Затем перезапустите WSL и Docker Desktop:

```powershell
wsl --shutdown
```

Этот лимит общий для всех дистрибутивов WSL. Проект лучше держать внутри файловой системы WSL (`\\wsl$\...`, например `~/Projects/RAG` в Ubuntu), а не на `C:\`. Монтирование из Windows-диска медленное, и `make` с bash работают там без проблем.

### Docker Desktop с Hyper-V

**Settings → Resources → Advanced:** CPUs 4+, Memory 10 GB+, Swap 2–4 GB, Disk image size 60 GB+. Затем нажмите **Apply & restart**.

## Linux

### Docker Engine (нативно)

ВМ нет: контейнеры работают на ядре хоста и по умолчанию видят все CPU и всю память. Настраивать ничего не нужно, достаточно, чтобы у машины было 10+ GB свободной RAM. Если памяти мало, добавьте swap. Модели будут работать медленнее, зато процесс не упадёт по OOM:

```bash
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

Ограничить `autorag-pipeline`, чтобы он не забрал всю машину, можно в `docker-compose.yml`:

```yaml
  autorag-pipeline:
    cpus: 4
    mem_limit: 10g
```

### Docker Desktop для Linux

Docker Desktop для Linux работает в своей ВМ, как на macOS. Настройки те же: **Settings → Resources → Advanced**.

## Проверка

Сколько ресурсов видит Docker:

```bash
docker info --format '{{.NCPU}} CPU, {{.MemTotal}} байт RAM'
```

Потребление во время прогона и причина падения:

```bash
docker stats                                                    # память/CPU контейнеров в реальном времени
docker compose ps -a autorag-pipeline                           # код выхода (137 = убит по памяти)
docker inspect $(docker compose ps -aq autorag-pipeline) --format '{{.State.OOMKilled}}'
```

Если `OOMKilled` равно `true` или код выхода `137`, увеличьте память ВМ или уберите локальные модели из `config.yaml`, как описано выше.
