# PDF Atlas

Локальный учебный PDF QA-агент: извлекает текст и структуру документа, находит релевантные фрагменты и отвечает через Ollama со ссылками на страницы.

Проект предназначен для локальной демонстрации и проверки исходного кода. Публичный сервер и production-развёртывание не входят в scope.

## Что реализовано

- текстовые PDF и сканы до 50 МБ;
- Docling extraction и OCR через macOS Vision (`ocrmac`, `ru-RU` + `en-US`);
- сохранение структуры страниц и координат текстовых областей;
- chunking с overlap и BM25-подобный лексический retrieval;
- bounded LLM query rewrite с повторной попыткой на языке документа;
- генерация ответа локальной моделью Ollama;
- обязательные ссылки вида `[стр. N]` / `[p. N]` и проверка, что процитированные страницы присутствовали в retrieved-контексте;
- канонический отказ, если документ не даёт достаточного контекста;
- RU/EN Streamlit UI: split-view документа и диалога, навигация к источникам, подсветка связанных областей, история текущей сессии;
- локальный дисковый JSON-кэш результатов extraction/chunking для 10 последних PDF;
- CLI: extraction, одноразовый вопрос и интерактивный диалог;
- анимированный индикатор состояния, не заменяющий нативную кнопку остановки Streamlit.

Это завершённый учебный MVP, а не production-ready RAG-система. Проверки groundedness здесь лексические и структурные — они снижают риск неподтверждённого ответа, но не доказывают его фактическую истинность.

## Архитектура

```text
PDF path / Streamlit upload
            │
            ▼
validation (PDF, non-empty, ≤ 50 MB)
            │
            ▼
Docling + macOS Vision OCR
            │
            ▼
ExtractedDocument (Markdown, pages, text regions)
            │
            ├──────────────► local JSON cache (UI only)
            │
            ▼
chunking (≤ 1,800 chars, overlap 250)
            │
            ▼
bounded query rewrite ──► BM25-like retrieval + coverage gate
            │                         │
            │                  no relevant context
            │                         ▼
            │              detect PDF language → rewrite retry
            │                         │
            │                  original-question fallback
            │
            ▼
local Ollama (`qwen3:14b` by default)
            │
            ▼
refusal / citation-page validation
            │
            ▼
answer + cited source regions
```

Документный контекст в prompt ограничен тегами `<document-context>` и помечен как недоверенные данные. Предыдущие вопросы используются только для разрешения ссылок и продолжения темы, а не как источник фактов.

## Требования

- macOS 10.15 или новее;
- Python 3.13;
- [uv](https://docs.astral.sh/uv/);
- локальный [Ollama](https://ollama.com/);
- модель `qwen3:14b` либо другая совместимая модель Ollama.

Проект ориентирован на macOS из-за Vision OCR. При первом extraction Docling может скачать локальные модели анализа документа.

## Быстрый запуск UI

Из корня репозитория:

```bash
uv sync --locked
ollama pull qwen3:14b
uv run streamlit run src/pdf_document_agent/app.py
```

Если Ollama не запущен автоматически, отдельно выполни:

```bash
ollama serve
```

После старта открой URL, который напечатает Streamlit (обычно `http://localhost:8501`). Загрузи PDF, дождись extraction и задай вопрос по документу.

В настройках UI можно сменить модель, ограничить историю и очистить локальный кэш.

## CLI

### Только extraction

```bash
uv run pdf-document-agent /путь/к/document.pdf
```

Эквивалентный модульный запуск:

```bash
uv run python -m pdf_document_agent /путь/к/document.pdf
```

### Один вопрос

```bash
uv run pdf-document-agent /путь/к/document.pdf \
  --ask "В чём основная идея документа?"
```

### Интерактивный режим

```bash
uv run pdf-document-agent /путь/к/document.pdf --interactive
```

Для выхода введи `выход`, `exit` или `quit`.

### Другая модель

```bash
uv run pdf-document-agent /путь/к/document.pdf \
  --ask "Что сказано о методе?" \
  --model qwen3:14b
```

CLI также читает имя модели из `PDF_AGENT_MODEL`.

## Локальные данные и границы безопасности

- Клиент Ollama по умолчанию обращается только к loopback-адресу `http://127.0.0.1:11434`.
- Исходный файл из UI записывается только во временный каталог на время extraction и затем удаляется; его байты остаются в памяти текущей Streamlit-сессии для просмотра страниц.
- UI-кэш не хранит исходный PDF, но хранит извлечённый текст, страницы и координаты областей. По умолчанию он расположен в `~/.cache/pdf-document-agent/`.
- Каталог кэша можно переопределить через `PDF_DOCUMENT_AGENT_CACHE_DIR`; очистить кэш можно из настроек UI.
- История вопросов живёт только в текущей Streamlit-сессии и сбрасывается при смене документа или закрытии сессии.
- Проект не требует API-ключей и не содержит встроенной авторизации. Поэтому его нельзя безопасно выставлять как открытый интернет-сервис без отдельного auth/resource-control слоя.

## Проверка проекта

```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q src tests
uv build
```

Тесты покрывают extraction seams, chunking/retrieval, query rewrite и fallback-сценарии, проверки ответа, Ollama API boundary, UI state/markup, кэш и PDF viewer. Реальные OCR/Docling/Ollama потоки частично заменены контролируемыми test doubles.

## Структура

```text
src/pdf_document_agent/
├── __init__.py       # публичная CLI-точка входа
├── __main__.py       # запуск через python -m
├── cli.py            # extraction, --ask, --interactive, --model
├── app.py            # Streamlit UI и orchestration загрузки
├── extractor.py      # Docling и macOS Vision OCR
├── retrieval.py      # chunking, токенизация, coverage gate, ranking
├── answering.py      # rewrite, prompt, отказ, citations, source selection
├── ollama.py         # локальный Ollama HTTP boundary
├── cache.py          # версионированный атомарный JSON-кэш
├── viewer.py         # рендер страницы и source-region overlays
└── localization.py   # RU/EN UI-копирайт

tests/
├── test_app.py
├── test_answering.py
├── test_cache.py
├── test_cli.py
├── test_extractor.py
├── test_ollama.py
├── test_retrieval.py
└── test_viewer.py
```

## Ограничения MVP

- Retrieval основан на точных лексических токенах: нет embeddings, морфологии и лемматизации.
- Coverage gate проверяет совпадение терминов, а не семантическую релевантность.
- Проверка citations подтверждает формат и принадлежность страницы retrieved-контексту, но не factual groundedness каждого утверждения.
- Prompt delimiters и инструкции о недоверенном контексте снижают, но не устраняют prompt injection.
- OCR backend не переносим на Linux/Windows без замены macOS Vision.
- Поиск выполняет O(N)-проход по фрагментам и рассчитан на локальный учебный масштаб.
- Нет agent tool loop, долговременной памяти, multi-user isolation, авторизации, rate limiting и resource quotas.
- Качество OCR падает на маленьком, размытом и низкоконтрастном тексте.
