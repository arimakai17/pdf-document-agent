# PDF Document Agent

Учебный агент, который извлекает содержимое PDF и отвечает на вопросы по найденным фрагментам документа.

## Текущий статус

Готов полный учебный вертикальный срез:

- принимает локальный PDF и проверяет расширение, существование и результат обработки;
- извлекает структуру и текст в Markdown через Docling;
- включает OCR для сканов через macOS Vision (`ocrmac`, языки `ru-RU` и `en-US`);
- сохраняет текст по страницам и разбивает его на перекрывающиеся фрагменты;
- ищет контекст BM25-подобным лексическим ранжированием с coverage gate;
- делает bounded LLM query rewrite на языке документа перед лексическим retrieval; если первая попытка не находит контекст, отдельно определяет язык PDF и повторяет rewrite с явным target language;
- проверяет наличие ссылок вида `[стр. N]` и принадлежность страниц retrieved-контексту;
- нормализует чистый отказ модели к каноническому тексту: `В документе недостаточно информации для ответа.`;
- предоставляет старый режим вывода извлечённого Markdown, одноразовый вопрос и интерактивный CLI;
- предоставляет Streamlit UI с загрузкой PDF, выбором модели и диалогом.

Это не production-ready система и не обещает гарантированную groundedness: проверки ограничены тем, что можно проверить лексически и по страницам.

## Поток данных

```text
PDF-путь или загрузка в UI
          │
          ▼
валидация PDF → Docling + macOS Vision OCR
          │
          ▼
Markdown документа + страницы
          │
          ▼
chunking (до 1 800 символов, overlap 250)
          │
          ▼
LLM query rewrite на языке PDF
          │
          ▼
BM25-подобный retrieval → coverage gate → top-k
          │
          ├─ пусто → определить язык PDF → rewrite retry → retrieval
          │                                             │
          │                                             └─ пусто → поиск исходного вопроса
          │
          ├─ нет релевантного контекста → канонический отказ
          │
          └─ есть контекст → локальный Ollama (qwen3:14b)
                                      │
                                      ▼
                         citation membership check
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
   GroundedAnswer со страницами            ошибка проверки ответа
```

В prompt контекст помещается между `<document-context>` и `</document-context>` и помечается как недоверенные данные. Это снижает риск prompt injection, но не устраняет его.

## Требования

- macOS 10.15 или новее;
- Python 3.13;
- [uv](https://docs.astral.sh/uv/);
- работающий локальный Ollama на `http://127.0.0.1:11434`;
- загруженная в Ollama модель `qwen3:14b` — модель по умолчанию.

Проект использует macOS Vision OCR и потому ориентирован на macOS. Docling при первом запуске может скачать свои модели анализа документа. Для подготовки окружения в проекте проверяется команда `uv sync`; отдельные команды установки Ollama здесь не приводятся.

## Запуск

Подготовить окружение:

```bash
uv sync
```

### Старый режим extraction

Без `--ask` и `--interactive` CLI извлекает PDF и печатает имя документа, число страниц и Markdown:

```bash
uv run pdf-document-agent /путь/к/document.pdf
```

Эквивалентный модульный запуск:

```bash
uv run python -m pdf_document_agent /путь/к/document.pdf
```

### Одноразовый вопрос

```bash
uv run pdf-document-agent /путь/к/document.pdf --ask "В чём основная идея документа?"
```

### Интерактивный режим

```bash
uv run pdf-document-agent /путь/к/document.pdf --interactive
```

Для выхода введите `выход`, `exit` или `quit`.

### Выбор модели

```bash
uv run pdf-document-agent /путь/к/document.pdf --ask "Что сказано о методе?" --model qwen3:14b
```

По умолчанию используется `qwen3:14b`. CLI также учитывает переменную окружения `PDF_AGENT_MODEL`.

### Streamlit UI

```bash
uv run streamlit run src/pdf_document_agent/app.py
```

UI принимает один PDF до 50 МБ, обрабатывает его локально и показывает split-view: документ слева, диалог справа. Нажатие на источник переключает страницу и подсвечивает связанные области; поддельная подсветка не используется. Прямоугольники — это Docling layout/OCR-регионы, лексически связанные с процитированным фрагментом, а не точные spans токенов. История вопросов хранится только в текущей сессии. Модель можно изменить в разделе «Настройки модели».

## Тесты

```bash
uv run pytest -q
```

## Структура

```text
src/pdf_document_agent/
├── __init__.py       # публичная точка входа для CLI
├── __main__.py       # запуск через python -m
├── cli.py            # extraction, --ask, --interactive, --model
├── app.py            # Streamlit UI и загрузка PDF
├── extractor.py      # Docling, страницы и macOS Vision OCR
├── retrieval.py      # chunking, токенизация и BM25-подобный поиск
├── answering.py      # rewrite, prompt, citations и отказ
└── ollama.py         # локальный Ollama API и модель по умолчанию

tests/
├── test_answering.py
├── test_cli.py
├── test_extractor.py
├── test_ollama.py
└── test_retrieval.py
```

## Ограничения учебного MVP

- Retrieval работает по точным лексическим токенам: нет морфологии, лемматизации и embeddings.
- Coverage gate проверяет лексическое совпадение для retrieval, но не доказывает семантическую истинность ответа.
- Citation membership проверяет формат и принадлежность страниц retrieved-контексту, а не factual groundedness утверждения; cross-language paraphrase допустим.
- Prompt delimiters и инструкции о недоверенном контексте снижают, но не устраняют prompt injection.
- OCR backend использует macOS Vision и не является переносимым на Linux/Windows без замены backend.
- Retrieval хранит данные в памяти и имеет O(N)-проход по фрагментам при поиске.
- Нет agent tool loop, долговременной памяти или внешних инструментов.
- Реальные Docling/OCR/Ollama/UI-upload flows не полностью покрыты unit tests; часть тестов проверяет seams без запуска всех внешних систем.
- OCR может ошибаться на маленьком, размытом или низкоконтрастном тексте.
