# PDF Atlas V2 — adaptive extraction, bounded agent, eval

## Цель

Доработать оценённый PDF Atlas по трём независимо измеряемым ступеням:

```text
A — замороженная V1: always-OCR extraction + fixed retrieval
B — adaptive per-page extraction + тот же fixed retrieval
C — тот же artifact B + bounded read-only tool agent
```

Обязательный результат V2 — B: постраничный выбор OCR без потери provenance, строгих ссылок, подсветки, локальности и существующего UI/CLI. C остаётся экспериментальным до отдельного доказательства пользы относительно B.

Точка отката: tag `pdf-atlas-v1-submission` → commit `1f7dcc0ee3cd652a3d00688e142d7da33fce802a`.
Рабочая ветка: `feat/pdf-atlas-v2`.

## Граница системы

```text
один пользователь → локальный Streamlit/CLI → локальные PDF
                                      ↓
                            локальный Ollama на loopback
```

- PDF, извлечённый текст, история и eval-артефакты не отправляются в облако.
- Инструменты агента только читают уже извлечённое представление документа.
- Нет записи в PDF, сети за пределами `127.0.0.1:11434`, публикации, auth, multi-user режима или production deployment.
- Новая VLM и загрузка модели не входят в обязательную V2. VLM допускается позже как отдельный backend после проверки ресурсов и явного решения пользователя.
- Использование разных библиотек не является целью. Двух маршрутов Docling достаточно, если corpus/eval показывают пользу.

## Ключевые состояния и инварианты

### Extraction page state

`route` и `status` — разные свойства:

```text
unclassified
    ↓ Docling без OCR
text_sufficient ─────────────→ route=docling_text
text_insufficient / unusable ─→ selective OCR(page N)
                                  ├→ route=docling_ocr, status=ok
                                  ├→ status=empty
                                  └→ status=failed + diagnostic
```

Минимальные статусы: `ok | empty | failed`.

Инварианты:

- решение принимается отдельно для каждой страницы;
- OCR не запускается для страницы, прошедшей text quality gate;
- каждая страница хранит фактический route, status и reason;
- номера страниц, порядок и provenance не меняются после selective OCR;
- fallback-результат атомарно заменяет страницу; text/OCR outputs не склеиваются неявно;
- пригодный текст первого прохода не теряется при сбое OCR;
- допустимая пустая страница отличается от страницы, где визуальный материал не удалось извлечь;
- частично доступный документ остаётся читаемым с явным warning; недоступная страница не интерпретируется как доказательство отсутствия факта;
- mixed PDF поддерживает разные routes, включая same-page digital text + значимое растровое содержимое;
- если same-page mixed content нельзя надёжно определить, существует ручной per-page OCR override и ограничение явно документируется.

### Agent run state

```text
planning → tool_selected → tool_result → planning
    │                              │
    ├──────── answer_ready ◀───────┘
    ├──────── budget_exhausted
    └──────── invalid_action / tool_error
```

Инварианты:

- один conversation owner;
- разрешены только `outline`, `search`, `read_page`;
- аргументы проходят typed validation;
- outline не даёт citation authority;
- search/read_page возвращают source-bearing excerpts: page, text, provenance/boxes и truncation marker;
- host собирает единый bounded final evidence packet;
- citation allowlist строится только из фрагментов, реально попавших в final evidence packet после truncation;
- PDF считается недоверенными данными и не изменяет правила tool loop;
- max steps, повторные typed actions, число всех LLM calls, output tokens, per-call timeout, общий deadline, history/context size и резерв финальной генерации принадлежат host;
- search в первом agent slice детерминированный; скрытый LLM rewrite запрещён либо явно учитывается в call budget и trace;
- модель не может увеличить лимиты, вызвать shell/Python или прочитать файл по пути;
- финальный ответ проходит прежнюю citation membership validation;
- отсутствие пригодного evidence приводит к каноническому отказу;
- trace содержит actions/status/страницы/cost counters без hidden reasoning;
- page membership снижает риск ложной ссылки, но не объявляется доказательством factual groundedness.

## Phase 0 — реальный selective-OCR/provenance spike

До production-кода проверить текущие Docling/ocrmac/pypdfium2 версии на локальных PDF:

- OCR страницы `N > 1` через `page_range=(N, N)`;
- два несмежных fallback-листа;
- страницу с отличающимся размером или поворотом;
- same-page digital text + значимое raster content;
- сохранение исходных page IDs, порядка и page count;
- boxes в координатах исходного PDF;
- отсутствие дубликатов после замены результата страницы;
- реальную подсветку viewer по OCR-source.

Результат: короткий spike report с версиями, PDF hashes, фактическими outputs и verdict `VALIDATED | PARTIAL | INVALIDATED`.

Стоп: при потере page numbering, geometry или связи chunk→source Wave 1 получает `HOLD`; scope не расширяется догадками.

## Wave 1 — Eval contract до настройки алгоритмов

Файлы:

- новый `evals/manifest.json`;
- новый `evals/README.md`;
- новый `src/pdf_document_agent/evaluation.py`;
- новый `tests/test_evaluation.py`;
- минимальный генератор synthetic fixtures без тяжёлого runtime dependency, если его реалистичность подтверждена.

Corpus:

- text page;
- scan page;
- alternating mixed pages;
- same-page text + raster;
- blank/sparse page;
- OCR noise;
- multi-page answer;
- ambiguous lexical matches;
- answerable и unanswerable с похожими словами.

Данные делятся на tuning cases и удержанный acceptance set. Synthetic PDFs проверяют механику; финальная приёмка требует отдельного локального реального corpus.

Рубрика и метрики:

- route correctness per page;
- expected source recall с альтернативными допустимыми наборами страниц;
- citation validity;
- ручная correctness 0–2;
- поддержка ключевых утверждений источниками 0–2;
- false refusal на answerable cases;
- правильный refusal на unanswerable cases;
- latency раздельно для uncached ingestion и warmed QA;
- OCR pages processed;
- LLM calls, tool steps, timeouts/errors;
- числители, знаменатели, skips и failures публикуются явно;
- обязательный skipped/unknown check означает incomplete gate, а не pass.

Для ответа используется небольшая ручная blind A/B/C-оценка замороженных outputs; новая judge-модель не добавляется.

Reproducibility manifest фиксирует:

- PDF SHA-256;
- git commit/lockfile;
- Docling/ocrmac/Ollama versions;
- model name и digest, options/prompts;
- extraction config fingerprint и gate version;
- warm/cold protocol;
- seed как ограничение вариативности, не обещание полной детерминированности.

## Wave 2 — Adaptive extraction (B)

Предполагаемые файлы:

- `src/pdf_document_agent/extractor.py`;
- `src/pdf_document_agent/cache.py`;
- `tests/test_extractor.py`;
- `tests/test_cache.py`;
- UI warning/override seam только если необходим для честной mixed-page поддержки.

Вертикальный результат:

- Docling first pass с `do_ocr=False`;
- явная версия deterministic page gate, его признаки, пороги и reason;
- selective OCR только нужных страниц через `page_range=(N, N)`;
- `ExtractedPage.route`, `status`, `reason`, `diagnostic`;
- атомарная page replacement с исходной нумерацией/provenance;
- manual per-page OCR override, если spike не даёт надёжного same-page raster signal;
- частичный документ + warning вместо полного ложного success/failure;
- cache schema bump и cache identity: PDF hash + schema + extraction-config fingerprint.

Acceptance:

- text-only fixture: OCR converter не вызывается;
- scan-only: OCR вызывается только для нужной страницы;
- alternating mixed и same-page mixed cases;
- blank, sparse, failed OCR и partial document имеют разные результаты;
- real rotated/different-size source сохраняет boxes/highlight;
- существующие extraction, cache, retrieval, answering и viewer tests зелёные.

## Wave 3 — Измерение A → B

A запускается из frozen commit/tag в отдельном окружении с его lockfile. B использует candidate environment. Одинаковыми остаются PDF, вопросы, модель, prompts/options и warm-state protocol настолько, насколько это технически возможно; различия фиксируются.

Сначала сравниваются uncached ingestion и fixed QA. B принимается как основной режим только если:

- citation/refusal hard gates не регрессировали;
- реальные text/scan/mixed документы не потеряли доступный контент или highlights;
- selective OCR действительно уменьшил число OCR pages либо дал другое заранее заявленное преимущество;
- correctness не хуже допустимого порога;
- все обязательные checks complete.

Если B не проходит, C не реализуется поверх неподтверждённого extraction.

## Wave 4 — Bounded tool-using agent (C)

Предполагаемые файлы:

- новый `src/pdf_document_agent/agent.py`;
- минимальный публичный seam в `src/pdf_document_agent/answering.py`;
- новый `tests/test_agent.py`;
- соседние tests answering/retrieval при необходимости.

Вертикальный результат:

- LLM выбирает typed JSON action;
- host исполняет только `outline`, `search`, `read_page` над сохранённым extraction artifact B;
- malformed action не ремонтируется вторым LLM-вызовом в первом slice;
- после host gate `answer_ready` тот же слой generation/citation validation создаёт `GroundedAnswer`;
- AgentRun возвращает ответ, final evidence packet и компактный trace.

Начальные hard budgets уточняются spike/eval, но обязаны включать:

- до 5 tool executions;
- не более 2 одинаковых normalized typed actions; третий блокируется;
- общий лимит всех LLM calls, включая финальную generation/rewrite;
- max output tokens per call;
- per-call timeout и run deadline;
- общий context/history cap;
- заранее зарезервированный budget финального ответа.

Acceptance:

- модель выбирает разные tools в многошаговых cases;
- invalid JSON/tool/args fail closed;
- page out of range отклоняется;
- tool/run budgets нельзя превысить;
- truncation не расширяет citation allowlist;
- prompt injection из PDF не получает tool authority;
- canonical refusal и V1 citation guards сохраняются;
- повторяющийся фиксированный outline→search→read_page pattern считается сигналом, что agent loop не приносит пользы.

## Wave 5 — Измерение B → C

B и C читают один и тот же сохранённый extraction artifact. Это исключает extraction как confounder.

C становится основным только если на удержанном set:

- исправляет реальные промахи fixed retrieval: query refinement, соседняя страница, несколько источников;
- снижает false refusals или повышает blind correctness/support;
- не нарушает citation/refusal hard gates;
- остаётся в заранее допустимых границах latency, LLM calls и deadline.

Один happy-path trace доказывает только работоспособность. Если измеримого выигрыша нет, V2 считается завершённой с B по умолчанию, C остаётся opt-in экспериментом, а отрицательный результат публикуется честно. Автоматического скрытого fallback C→B нет.

## Wave 6 — UI/CLI и документация

Файлы:

- `src/pdf_document_agent/app.py`;
- `src/pdf_document_agent/cli.py`;
- `src/pdf_document_agent/localization.py`;
- `README.md`;
- соответствующие tests.

Результат:

- B — основной фиксированный режим;
- C — видимый opt-in до прохождения B→C acceptance;
- пользователь видит per-page routes/status/warnings и компактный trace;
- manual OCR override доступен только если он требуется по результатам spike;
- frozen V1 называется benchmark/rollback, а B — adaptive fixed mode;
- UI, история, подсветка, cache и CLI сохраняют прежнее поведение.

## Финальный quality gate

```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q src tests
uv build
git diff --check
```

Дополнительно:

- real text/scan/alternating-mixed/same-page-mixed extraction smokes;
- real OCR-source viewer highlight;
- partial extraction UI/CLI flow;
- A/B report;
- live bounded planner pilot, malformed action и budget exhaustion;
- B/C report;
- Streamlit startup и все затронутые состояния без runtime exceptions;
- независимый audit замороженного candidate commit;
- clean tree и evidence receipt.

## Стоп-условия

- Потеря page numbering, geometry, highlights или provenance → HOLD Wave 2.
- Неподтверждённый B → C не строится.
- Нестабильный typed action contract qwen3:14b → C остаётся failed/negative experiment; новый framework/model не добавляется автоматически.
- Synthetic fixtures, не представляющие реальный extraction, остаются только unit tests.
- Две разные неудачные гипотезы по одному blocker → остановка и отдельное архитектурное решение.

## Не входит сейчас

- облачная LLM/VLM;
- скачивание новой модели;
- pypdf/VLM только ради совпадения с чужой архитектурой;
- embeddings/vector DB/reranker;
- LangChain/LlamaIndex;
- редактирование PDF;
- multi-document knowledge base;
- auth, публичный сервер, Docker или deployment;
- production factuality guarantee.

## Оценка времени после Astra-review

- selective OCR/provenance spike: 3–5 часов;
- eval contract/corpus/frozen baseline: 8–12 часов;
- adaptive extraction/cache/tests: 8–12 часов;
- bounded loop/evidence/budgets/tests: 10–16 часов;
- UI/CLI, реальные прогоны, audit/docs: 8–12 часов.

Итого: **37–57 часов непосредственной инженерной работы**. С учебным разбором для начинающего: **50–75 часов**, ориентировочно 2–4 недели при 20–30 часах в неделю. Длительные локальные OCR/Ollama-прогоны могут увеличить календарный срок.
