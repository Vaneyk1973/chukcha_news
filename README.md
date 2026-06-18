# Chukchi News Voice

Локальный сервер для генерации коротких радионовостей с чукотской озвучкой.

Пайплайн работает полностью локально:

1. Пользователь вводит промпт на русском.
2. Qwen 2.5 пишет короткую русскую новость.
3. MT-модель переводит новость `RU -> CKT`.
4. Обратный переводчик `CKT -> RU` показывает контрольный смысл.
5. MMS/VITS TTS синтезирует чукотскую аудиоверсию.

Второй режим сервера позволяет экспериментально генерировать чукотский текст напрямую через Qwen + LoRA, без русского MT-шага.

## Быстрый запуск

Нужны Linux, Python 3.12, CUDA GPU и достаточно места под runtime-артефакты. Полный runtime-бандл занимает десятки гигабайт, потому что включает Qwen, TTS, MT-модели и локальные адаптеры.

```bash
git clone https://github.com/Vaneyk1973/chukcha_news.git
cd chukcha_news
python3 -m pip install -U huggingface_hub
hf download iakhmura/chukchi-news-runtime --repo-type dataset --local-dir .
make serve
```

После старта открой:

```text
http://127.0.0.1:9999
```

Если порт занят:

```bash
.venv/bin/python scripts/server/app.py --port 10099
```

## Что скачивает runtime-бандл

Runtime-бандл должен распаковываться прямо в корень репозитория. Он приносит только тяжелые артефакты, необходимые для локального сервера:

- `models/checkpoints/mt_clean/ru_ckt/final` - основной переводчик `RU -> CKT`;
- `models/checkpoints/mt_clean/ckt_ru/final` - обратный переводчик `CKT -> RU`;
- `models/checkpoints/mt/...` - предыдущая версия MT, оставлена в UI для сравнения;
- `models/checkpoints/llm_chukchi_lora` - LoRA для прямой генерации чукотского текста;
- `models/checkpoints/tts` и `models/checkpoints/tts_top1h` - экспериментальные TTS-варианты;
- `models/hf/Qwen--Qwen2.5-7B-Instruct` - локальная копия Qwen;
- `models/hf/facebook--mms-tts-ckt` - локальная копия MMS TTS baseline;
- `data/processed/mt*` - подготовленные translation-memory и MT splits;
- `configs/server.yaml` - runtime-конфиг сервера.

Проверка восстановленного бандла:

```bash
make verify-runtime-artifacts
```

## Сборка runtime-бандла

На машине, где уже обучены модели и скачаны HF-веса:

```bash
make package-runtime-artifacts
```

Результат будет в:

```text
artifacts/runtime_bundle
```

Загрузка в Hugging Face:

```bash
python3 -m pip install -U huggingface_hub
hf auth login
hf repo create chukchi-news-runtime --type dataset --private
hf upload-large-folder YOUR_USER/chukchi-news-runtime artifacts/runtime_bundle --repo-type dataset
```

## Исходные данные

Исходные данные и внешние репозитории не хранятся в Git. Их можно скачать заново.

| Что | URL | Куда скачивать |
| --- | --- | --- |
| Русско-чукотский параллельный корпус HSE | `https://huggingface.co/datasets/HSE-Chukchi-NLP/russian-chukchi-parallel-corpora` | `data/raw/hse_parallel_corpus/data.csv` |
| FieldASR Chukchi corpus | `https://github.com/ftyers/fieldasr` | `external/fieldasr` |
| Chukchi translator reference notebooks | `https://github.com/hse-chukchi-nlp/chukchi-translator` | `external/chukchi-translator` |
| VITS fine-tuning recipe | `https://github.com/ylacombe/finetune-hf-vits` | `external/finetune-hf-vits` |
| Fairseq / MMS reference code | `https://github.com/facebookresearch/fairseq` | `external/fairseq` |
| Qwen 2.5 7B Instruct | `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct` | HF cache или `models/hf/Qwen--Qwen2.5-7B-Instruct` |
| MMS TTS Chukchi | `https://huggingface.co/facebook/mms-tts-ckt` | HF cache или `models/hf/facebook--mms-tts-ckt` |
| MMS ASR | `https://huggingface.co/facebook/mms-1b-all` | HF cache |
| NLLB base MT | `https://huggingface.co/facebook/nllb-200-distilled-600M` | HF cache |
| HSE baseline `RU -> CKT` | `https://huggingface.co/HSE-Chukchi-NLP/mbart50-rus-ckt` | HF cache |
| HSE baseline `CKT -> RU` | `https://huggingface.co/HSE-Chukchi-NLP/mbart50-ckt-rus` | HF cache |

Автоматический bootstrap для открытых источников:

```bash
make bootstrap
```

FieldASR-аудио и радиозаписи могут требовать ручной загрузки в зависимости от источника. Для ASR-пайплайна ожидаемые директории:

```text
audio/
data/raw/fieldasr/audio/
data/raw/fieldasr/downloads/
```

## Что хранится в Git

В Git должны оставаться только исходники и конфиги:

- `scripts/` - пайплайны подготовки данных, обучения, оценки и локального сервера;
- `src/` - библиотечный код проекта;
- `configs/` - воспроизводимые настройки;
- `web/server/` - интерфейс локального сервера;
- `tests/` - тесты для критичных пайплайнов;
- `README.md`, `Makefile`, `pyproject.toml`.

В Git не должны попадать:

- `data/` - скачанные и сгенерированные датасеты;
- `models/` - веса, adapter-ы, checkpoint-и;
- `reports/` - отчеты экспериментов;
- `outputs/` - результаты запусков сервера;
- `audio/` - сырой аудиоархив;
- `external/` - сторонние репозитории;
- `artifacts/` - собранные runtime-бандлы;
- `.venv*` - локальные окружения.

## Основные команды разработки

```bash
make serve                 # локальный продуктовый сервер
make serve-mock            # сервер без загрузки тяжелой LLM
make showcase              # HTML/PNG/WAV-артефакты для быстрой демонстрации
make package-runtime-artifacts
make verify-runtime-artifacts
make translation           # полный MT-пайплайн
make mt-clean              # clean MT обучение и диагностика
make rerun-asr-filtered    # ASR с фильтрами
make tts                   # TTS-пайплайн
make train-llm-chukchi     # LoRA для прямого чукотского режима
make test
```

Готовые демонстрационные снапшоты лежат в `showcase/index.html`.
