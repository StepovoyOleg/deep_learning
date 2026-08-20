# Нефтегазовый аналитик

«Нефтегазовый аналитик» — MVP AI-агента для анализа нефтегазового рынка. Он отвечает по локальному корпусу OPEC/EIA, при необходимости использует разрешённые web-источники, строит статистический прогноз Brent и показывает provenance фактических источников.

## Возможности

- RAG по загруженным OPEC MOMR и EIA STEO PDF;
- hybrid retrieval: dense Qdrant + BM25 + RRF + reranker;
- фильтрация по организации, месяцу/году и типу отчёта;
- WEB через Tavily и совместный RAG + WEB анализ;
- статистический прогноз Brent;
- локальная `qwen3:4b` через Ollama;
- Pydantic structured output, quality gate и controlled fallback;
- canonical sources из реального tool state;
- управляемый OUT_OF_SCOPE route.

## Архитектура

```text
User
  ↓
Streamlit
  ↓
LangGraph / Query Analysis
  ↓
┌────────────────────────────────────┐
│ RAG                                │
│ metadata filtering                 │
│ Dense (MiniLM) → Qdrant            │
│ BM25 → RRF → BGE reranker → top-k  │
└────────────────────────────────────┘
        │
        ├── WEB → Tavily
        ├── FORECAST → ETS / SARIMA
        └── OUT_OF_SCOPE
        ↓
Qwen3:4b / Ollama
        ↓
Structured output / validation / quality gate
        ↓
Backend canonical sources
        ↓
Final answer
```

LangGraph маршрутизирует запрос в RAG, WEB, RAG + WEB, FORECAST либо OUT_OF_SCOPE. Источники формирует backend; модель не определяет URL, страницы или имена документов.

## RAG pipeline

```text
OPEC/EIA PDF
  → pypdf text layer
  → page-bounded character chunks
  → contextual headers + structured metadata
  → multilingual MiniLM embeddings → Qdrant cosine search
  + BM25 sparse search
  → RRF
  → BAAI/bge-reranker-v2-m3
  → final context
  → qwen3:4b
```

Фактические параметры:

- `chunk_size = 900`, `chunk_overlap = 120`;
- embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
- normalized vectors, dimension 384;
- dense top-12 + BM25 top-12 → RRF → final top-6;
- Qdrant collection `oil_reports`, cosine distance;
- metadata: organization, report name/date, document, page, deterministic `chunk_id`, `section_title`, `table_title`, `content_type`.

В индексируемый текст добавляется короткий contextual header: organization, report, date, section, page и table title, если они надёжно извлечены. Полный JSON metadata в embedding не добавляется; structured metadata отдельно сохраняются в Qdrant payload.

Metadata filtering применяется до dense и BM25 поиска. Router извлекает OPEC/EIA, месяц+год и MOMR/STEO из запроса.

## Generation и provenance

Qwen получает читаемые блоки `SOURCE [REPORT]`, `SOURCE [WEB]` и `SOURCE [BACKEND MODEL]` и генерирует только `answer` и `uncertainty`. Pydantic проверяет JSON. Quality gate отклоняет одно число вместо аналитического ответа, ложное полное отрицание при наличии частичного контекста и смешение временных слоёв report/web. Допускается одна repair-попытка, затем возвращается grounded fallback.

## WEB

Tavily используется при наличии `TAVILY_API_KEY`. Результаты ограничены allowlist из `WEB_PREFERRED_DOMAINS`. Для current-запросов свежие датированные результаты располагаются выше, старые датированные факты не выдаются как текущие. После ограниченных retry применяется graceful degradation.

## Forecasting

Источник — EIA Europe Brent Spot Price FOB (`data/prices/RBRTEd.xls`). Loader валидирует значения, использует последние 10 лет и агрегирует daily observations до monthly mean.

- Holt-Winters Exponential Smoothing;
- SARIMA `(1,1,1)(0,1,1,12)`;
- point forecast и ориентировочный 80% prediction interval;
- RMSE/MAPE, assumptions и interpretation.

Это статистический baseline: он не моделирует неизвестные будущие геополитические шоки или решения OPEC+.

## Данные

- `data/reports/opec/` — OPEC MOMR, May–July 2026;
- `data/reports/eia/` — EIA STEO, May–July 2026;
- `data/prices/RBRTEd.xls` — исторические дневные цены Brent.

PDF обнаруживаются рекурсивно внутри `data/reports`. `starter_corpus` не используется.

## Stack

Python 3.12, Streamlit, LangGraph, Pydantic, pypdf, sentence-transformers, Qdrant, BM25/RRF, BGE reranker, Ollama/Qwen3, Tavily, pandas, statsmodels, xlrd и pytest.

## Структура проекта

```text
src/oil_analyst/       application, orchestration, retrieval, forecasting
tests/                 unit и integration regression tests
scripts/               ingestion и reusable demo check
data/reports/          OPEC/EIA PDF corpus
data/prices/           Brent source data
eval/                  demo questions и результаты smoke-check
app.py                 Streamlit UI
docker-compose.yml     Qdrant и опциональный app container
```

## Установка и запуск

Из корня проекта:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
docker compose up -d qdrant
ollama pull qwen3:4b
python scripts/ingest.py
streamlit run app.py
```

- Streamlit: <http://localhost:8501>
- Qdrant dashboard: <http://localhost:6333/dashboard>

Первая загрузка embedding и reranker требует доступа к Hugging Face. Повторный ingestion idempotent: UUID строится из стабильного `chunk_id`.

## Environment

Настройки читаются из локального `.env` в корне проекта. `.env` исключён из репозитория; `.env.example` содержит только пустые placeholders и безопасные defaults.

- `TAVILY_API_KEY` включает WEB;
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL` задают локальный генератор;
- `QDRANT_URL`, `QDRANT_COLLECTION` задают vector store;
- `EMBEDDING_MODEL`, `RERANKER_MODEL` задают retrieval-модели;
- `WEB_PREFERRED_DOMAINS` задаёт allowlist.

После изменения `.env` перезапустите Streamlit: settings и production agent кэшируются процессом.

## Demo

1. RAG — `Как OPEC оценивает мировой спрос на нефть в 2026 году?`
2. RAG + metadata — `Что OPEC писал о мировом предложении нефти в июльском отчёте 2026 года?`
3. WEB — `Какие последние значимые новости по OPEC+?`
4. RAG + WEB — `Сравни прогноз OPEC/EIA из загруженных отчётов с текущей ситуацией на нефтяном рынке.`
5. FORECAST — `Спрогнозируй цену Brent на следующие 3 месяца.`
6. OUT_OF_SCOPE — `Напиши рецепт борща.`

Вопросы находятся в `eval/demo_questions.json`. Smoke-check использует один production agent и сохраняет raw output без автоматического PASS/FAIL:

```powershell
python scripts/demo_check.py
```

Результат записывается в `eval/demo_results.md`. WEB требует Tavily, generation — запущенный Ollama с `qwen3:4b`.

## Тесты

```powershell
python -m pytest -v
```

Актуальный локальный результат: **46 passed / 0 failed**.

## Known limitations

- графики PDF используются через доступный text layer/captions, без visual understanding;
- извлечение структуры зависит от text layer документа; EIA section headers менее стабильны;
- продолжение таблицы без повторного title может остаться обычным text chunk;
- компактная `qwen3:4b` допускает отдельные synthesis edge cases;
- web freshness зависит от доступных date metadata внешних страниц;
- forecast не учитывает неизвестные будущие события;
- первая локальная инициализация embedding/reranker может быть долгой;
- MVP не production-ready: нет auth, rate limiting и полноценного observability backend.
