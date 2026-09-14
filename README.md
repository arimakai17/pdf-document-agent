# PDF Atlas V2

Локальный учебный инструмент для вопросов по PDF: извлекает текст и структуру, находит фрагменты и отвечает через Ollama со ссылками на страницы.

Статус A→B — **HOLD B EXTRACTION**: исправленное OCR-измерение не прошло
предобъявленный hard gate. Внутри этой candidate branch B остаётся техническим
fixed default только потому, что A — замороженный внешний baseline, а не режим
приложения. **C — bounded read-only agent поверх артефакта B**, видимый
экспериментальный opt-in со статусом HOLD; merge/release для B и C не разрешены.
C не включается автоматически и не имеет скрытого fallback в B.

Проект предназначен для локальной демонстрации и проверки исходного кода. Публичный сервер и production-развёртывание не входят в scope.

## Что реализовано

- текстовые PDF и сканы до 50 МБ;
- первый проход Docling без OCR и адаптивный постраничный выбор маршрута;
- selective full-page OCR через macOS Vision (`ocrmac`, `ru-RU` + `en-US`);
- сохранение структуры страниц и координат текстовых областей;
- chunking с overlap и BM25-подобный лексический retrieval;
- bounded LLM query rewrite на языке документа с проверкой словаря и одной явной повторной попыткой в fixed-ответах;
- генерация ответа локальной моделью Ollama;
- обязательные ссылки вида `[стр. N]` / `[p. N]` и проверка, что процитированные страницы присутствовали в retrieved-контексте;
- канонический отказ, если документ не даёт достаточного контекста;
- RU/EN Streamlit UI: split-view документа и диалога, навигация к источникам, подсветка связанных областей, история текущей сессии;
- локальный дисковый JSON-кэш результатов extraction/chunking для 10 последних PDF;
- CLI для extraction, одноразового вопроса и интерактивного диалога;
- видимый per-page route/status, warning и компактный trace bounded-agent.

Проверка citation membership подтверждает формат и наличие страницы в evidence, но не доказывает семантическую или фактическую истинность каждого утверждения.

## Постраничное извлечение

`route` и `status` — разные свойства. Для каждой страницы сохраняется фактическая причина маршрута:

```text
Docling first pass (без OCR)
        │
        ├─ text gate passed ───────── route=docling_text, status=ok
        │
        └─ text insufficient/unusable
              └─ selective OCR(page N)
                   ├─ route=docling_ocr, status=ok
                   ├─ route=docling_ocr, status=empty
                   └─ route=docling_ocr, status=failed + diagnostic
```

Смешанная страница с достаточным цифровым текстом может пройти gate без автоматического OCR, даже если на ней есть значимое растровое содержимое. Используй ручной override, когда вопрос зависит от текста на графике, скане или другом raster-слое, который не попал в цифровой слой. Override заменяет результат страницы атомарно; номера страниц и provenance сохраняются.

В UI это поле **«Страницы для ручного OCR»** в настройках. Введи номера и inclusive ranges через запятую, например:

```text
1, 3-5
```

В CLI синтаксис тот же:

```bash
uv run pdf-document-agent document.pdf --ocr-pages "1, 3-5"
```

Если OCR не удался, документ не маскирует проблему: страница получает `status=failed`, diagnostic и warning. Допустимая пустая страница получает `status=empty`.

## Архитектура

```text
PDF path / Streamlit upload
            │
            ▼
validation (PDF, non-empty, ≤ 50 MB)
            │
            ▼
Docling first pass (без OCR)
            │
            ▼
adaptive per-page text gate
       ┌────┴────┐
       ▼         ▼
docling_text   selective OCR(page N)
  status=ok      │
              docling_ocr: ok | empty | failed
                    │
                    ▼
cached ExtractedDocument
(Markdown, pages, route/status, regions, provenance)
            │
       ┌────┴──────────────────────────────┐
       ▼                                   ▼
fixed B (candidate/HOLD;             bounded C (opt-in/HOLD)
       technical default in branch)
chunking → lexical retrieval          outline / deterministic search /
→ answer + citation validation        read_page → bounded evidence
       │                                   │
       └───────────────┬───────────────────┘
                       ▼
             answer, refusal или explicit status
```

Документный контекст в prompt ограничен тегами `<document-context>` и помечен как недоверенные данные. Предыдущие вопросы используются только для разрешения ссылок и продолжения темы, а не как источник фактов.

### Многоязычный поиск и продолжение диалога

В `fixed` вопрос сначала преобразуется в короткий лексический запрос на языке PDF. В результат допускаются только слова, которые встречаются в извлечённом тексте документа. Если первый поиск ничего не нашёл, приложение отдельно определяет основной язык PDF и один раз повторяет rewrite с этим языком как явной целью. После этого остаётся детерминированный поиск по исходному вопросу и, если совпадений всё равно нет, канонический отказ.

Штатный контракт rewrite — одна строка из 2–6 поисковых слов. Для совместимости с реальным выводом Qwen приложение принимает первый полезный компактный сегмент до лишних тем, разделённых запятыми. Near-miss из 7–8 уникальных слов словаря сокращается до шести наиболее информативных с сохранением исходного порядка. Более восьми слов, многострочный или слишком длинный вывод отклоняются; разбиение чрезмерного первого сегмента запятыми не обходит этот лимит.

При follow-up предыдущие вопросы помогают разрешить местоимения и текущую тему, но не добавляют фактов. Уже найденные ranked-фрагменты сохраняются; соседние фрагменты раздела занимают только свободные места в `top_k` и не могут вытеснить удалённый, но релевантный результат.

Например, русский вопрос к англоязычной книге:

```text
что автор имеет ввиду под «мыслить как компьютерный ученый», раскрой важные пункты
```

может быть переписан в английские термины из словаря PDF, а ответ останется на выбранном в UI языке и будет принят только с допустимыми ссылками на страницы.

### Режимы ответа

`fixed` — режим-кандидат B со статусом HOLD; в этой candidate branch он
остаётся техническим default для B extraction и fixed retrieval/answering.
Это не является одобрением merge/release.

`agent` — экспериментальный C. Planner может выбирать только read-only tools `outline`, детерминированный `search` и `read_page`; `outline` не даёт citation authority. Host проверяет typed JSON actions, собирает единый bounded final evidence packet и передаёт его прежнему answer/citation слою.

Лимиты принадлежат host и не могут быть увеличены моделью: до 5 tool executions, не более 2 одинаковых normalized actions, до 6 LLM calls вместе с финальной генерацией, до 256 output tokens для planner и 600 для финального ответа, `180 s` на один вызов и `180 s` на весь run, bounded planner history (`4,000` символов) и final evidence (`8,000` символов). Search ограничен `top_k=5`, trace — компактными action/status/pages/truncated и counters `tool_calls`/`llm_calls`/`elapsed_s` без hidden reasoning.

Malformed action, ошибка tool/planner/answer или исчерпание budget возвращаются как явный fail-closed status (`invalid_action`, `tool_error`, `planner_error`, `answer_error`, `budget_exhausted`) и не переключают запуск незаметно на fixed mode.

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

В настройках UI можно сменить модель, выбрать `fixed` или `agent`, задать страницы ручного OCR, ограничить историю и очистить локальный кэш. UI явно показывает B candidate/HOLD и C experimental/HOLD; автоматического fallback нет.

## CLI

### Только extraction

```bash
uv run pdf-document-agent document.pdf
```

Эквивалентный модульный запуск:

```bash
uv run python -m pdf_document_agent document.pdf
```

### Один вопрос

```bash
uv run pdf-document-agent document.pdf \
  --ask "В чём основная идея документа?"
```

### Интерактивный режим

```bash
uv run pdf-document-agent document.pdf --interactive
```

Для выхода введи `выход`, `exit` или `quit`.

### Другая модель

```bash
uv run pdf-document-agent document.pdf \
  --ask "Что сказано о методе?" \
  --model qwen3:14b
```

По умолчанию используется `qwen3:14b`; его можно заменить переменной окружения `PDF_AGENT_MODEL` или флагом `--model`.

### Режимы и ручной OCR

```bash
# B: candidate/HOLD; technical default only in this candidate branch
uv run pdf-document-agent document.pdf \
  --ask "Что описано в документе?" \
  --mode fixed

# C: visible opt-in, experimental/HOLD; hidden fallback отсутствует
uv run pdf-document-agent document.pdf \
  --ask "Какие шаги описаны?" \
  --mode agent

# Ручной OCR страниц 1 и 3–5; диапазоны inclusive
uv run pdf-document-agent document.pdf \
  --ask "Что указано на схеме?" \
  --ocr-pages "1, 3-5"
```

`--mode` принимает только `fixed` и `agent`; parser default — `fixed`, но это
только технический default candidate branch со статусом HOLD. `--ocr-pages` по
умолчанию пустой, принимает 1-based номера и возрастающие диапазоны через
запятую. `--ask` и `--interactive` взаимоисключающие.

## Локальные данные и границы безопасности

- Клиент Ollama по умолчанию обращается только к loopback-адресу `http://127.0.0.1:11434`.
- PDF и ответы обрабатываются локально; проект не отправляет их в облако и не требует API-ключей.
- Исходный файл из UI записывается только во временный каталог на время extraction и затем удаляется; его байты остаются в памяти текущей Streamlit-сессии для просмотра страниц.
- Дисковый кэш не хранит исходный PDF, но хранит извлечённый текст, страницы, route/status и координаты областей для 10 последних PDF. Это важно учитывать на общей машине. По умолчанию кэш находится в `~/.cache/pdf-document-agent/`.
- Каталог кэша можно переопределить через `PDF_DOCUMENT_AGENT_CACHE_DIR`; очистить кэш можно из настроек UI.
- История вопросов живёт только в текущей Streamlit-сессии и сбрасывается при смене документа или закрытии сессии.
- Нет встроенной авторизации, публикации или resource control; приложение нельзя безопасно выставлять как открытый интернет-сервис.

## Оценка

A→B HOLD: corrected actual recognition pages are 15→9 (`−40%`, `B/A = 60%`),
при predeclared gate `B ≤ 7.5` / `≤50%`. Diagnostic stage visits were 21→9;
case-weighted runs were 22→9, but neither overrides the gate. B is only the
technical fixed default inside this candidate branch; C is experimental/HOLD.
Отчёты — сравнительная регрессионная проверка на локальном запуске, а не
production benchmark; они не заявляют pristine held-out evidence или
превосходство по скорости.

- [PLAN_V2.md](PLAN_V2.md) — контракт и границы V2;
- [A → B report](evals/ab-report.md) и [машинный receipt](evals/results/ab-2026-09-14.json);
- [corrected OCR work receipt](evals/results/ab-ocr-work-2026-09-14.json);
- [B → C report](evals/bc-report.md) и [машинный receipt](evals/results/bc-2026-09-14.json);
- [контракт evaluation](evals/README.md) и [manifest](evals/manifest.json).

## Проверка проекта

```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q src tests
uv build
git diff --check
```

Тесты покрывают extraction seams, chunking/retrieval, многоязычный query rewrite и его лимиты, сохранение ranked-контекста в follow-up, bounded-agent limits/actions, проверки ответа, evaluation contract, Ollama API boundary, UI state/markup, кэш и PDF viewer. Реальные OCR/Docling/Ollama потоки частично заменены контролируемыми test doubles.

Для полного acceptance также нужны локальные extraction/OCR/viewer smoke-проверки, partial extraction flow, B/C report, malformed-action и budget-exhaustion cases, Streamlit startup и clean tree с receipt. Их результаты не подменяются одним happy-path trace.

## Структура

```text
src/pdf_document_agent/
├── __init__.py       # публичная CLI-точка входа
├── __main__.py       # запуск через python -m
├── cli.py            # extraction, --ask, --interactive, --model, --mode, --ocr-pages
├── app.py            # Streamlit UI и orchestration загрузки
├── agent.py          # bounded read-only agent C и limits/trace
├── extractor.py      # adaptive gate, Docling и macOS Vision OCR
├── retrieval.py      # chunking, токенизация, coverage gate, ranking
├── answering.py      # rewrite, prompt, отказ, citations, source selection
├── evaluation.py     # typed evaluation contract и reports
├── eval_fixtures.py  # synthetic evaluation fixtures
├── ollama.py         # локальный Ollama HTTP boundary
├── cache.py          # версионированный атомарный JSON-кэш
├── viewer.py         # рендер страницы и source-region overlays
└── localization.py   # RU/EN UI-копирайт

tests/
├── test_agent.py
├── test_answering.py
├── test_app.py
├── test_cache.py
├── test_cli.py
├── test_evaluation.py
├── test_extractor.py
├── test_ollama.py
├── test_retrieval.py
└── test_viewer.py

evals/
├── README.md
├── manifest.json
├── ab-report.md
├── bc-report.md
└── results/
    ├── ab-2026-09-14.json
    ├── ab-ocr-work-2026-09-14.json
    └── bc-2026-09-14.json
```

## Ограничения

- Retrieval после bounded rewrite основан на точных лексических токенах: нет embeddings, морфологии и лемматизации.
- Coverage gate проверяет совпадение терминов, а не семантическую релевантность.
- Citation page membership подтверждает включение страницы в evidence, но не является семантическим доказательством правильности факта.
- C не восстановил страницу 7 в multi-page case; в измеренном запуске был один сбой output contract и malformed action.
- Подписи графика отсутствуют в доступном OCR; chart case должен оставаться отказом без отдельно одобренного backend.
- Prompt delimiters и инструкции о недоверенном контексте снижают, но не устраняют prompt injection.
- OCR backend — только macOS Vision/`ocrmac`; на Linux/Windows нужна отдельная реализация.
- Поиск выполняет O(N)-проход по фрагментам и рассчитан на локальный учебный масштаб.
- Это локальный образовательный scope без production-гарантии factuality, multi-user isolation, авторизации, rate limiting и resource quotas.
- Качество OCR падает на маленьком, размытом и низкоконтрастном тексте.
