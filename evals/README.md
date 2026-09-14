# PDF Atlas V2 evaluation contract

Wave 1 фиксирует договор оценки до настройки extraction или agent. Это
детерминированный Python-контракт: LLM-as-judge, judge-модель и новые
зависимости не используются. `manifest.json` содержит только expected data и
идентичность документов; фактический локальный путь к PDF передаётся запуску
вне manifest.

## Что зафиксировано

Manifest имеет `schema_version: 1`, baseline A и два split: `tuning` и
`acceptance`. Каждый документ описан logical id, `kind`, basename и SHA-256.
`synthetic` записи — воспроизводимые mechanics-only fixtures, а не evidence
реального корпуса. Они генерируются без сохранения бинарных PDF в git:

```bash
uv run python -m pdf_document_agent.eval_fixtures --output /tmp/pdf-atlas-eval
```

Генератор строит text-only PDF и пятистраничный mixed PDF (scan, noisy scan,
same-page digital+raster, scan, blank); их SHA-256 зафиксированы в manifest и
проверяются тестом. Реальный acceptance corpus использует два неперсональных
course PDF, которые не копируются в git: manifest хранит только basename и
SHA-256, а локальный запуск связывает их с путями вне manifest.

Каждый case задаёт категории, полный текст вопросов, reference answer для
ручной blind-разметки, expected per-page `route` и `status`, а также явный
`ocr_pages` для manual overrides этого запуска. Вопросы, требующие только
цифровой текст mixed PDF, и вопросы к raster-графику находятся в разных cases:
это не позволяет ненужному full-page OCR менять text-only retrieval и отдельно
измеряет полезность override.
Допустимые routes: `docling_text`, `docling_ocr`; статусы:
`ok`, `empty`, `failed`. Для answerable вопроса `source_page_sets` содержит
один или несколько непустых допустимых наборов страниц. Evaluator выбирает
набор с лучшим recall. Для unanswerable вопроса наборы обязаны быть пустыми.

Покрыты категории `text_page`, `scan_page`, `alternating_mixed_pages`,
`same_page_text_raster`, `blank_sparse_page`, `ocr_noise`,
`multi_page_answer`, `ambiguous_lexical_matches`,
`answerable_similar_words`, `unanswerable_similar_words`.

## Метрики и gate

`evaluate_run(manifest, observations)` возвращает typed report. Для каждой
метрики отдельно публикуются `numerator`, `denominator`, `skipped` и
`failures`; общий score не используется. Отчёт включает:

- route correctness per page;
- expected source recall, citation validity и readiness реального acceptance;
- human correctness и claim support по шкале 0–2 (сумма в `numerator`,
  число размеченных/ожидаемых вопросов в `denominator`);
- false refusal для answerable (`numerator` — число ошибочных отказов, меньше
  лучше) и correct refusal для unanswerable (`numerator` — число чистых
  корректных отказов, больше лучше);
- uncached ingestion latency и warmed QA latency раздельно;
- OCR pages processed, LLM calls, tool steps, timeouts и errors.

`None`/unknown ручные labels и отсутствующие mandatory observations — это
`skipped`, значит gate `incomplete`, а не pass или zero. Runtime timeout/error
попадает в `failures`; malformed observed data и malformed manifest отвергаются
через `ValueError`. Citation считается valid только если её page входит в
реальный final evidence и находится в диапазоне PDF. Неполный source recall,
ошибочный отказ, отсутствие корректного отказа, неверный route/status и
invalid citation увеличивают `failures`; поэтому полный, но нарушивший hard
gate запуск получает status `fail`.

Synthetic cases могут дать зелёные механические метрики, но не заменяют
локальный real corpus. Acceptance case становится ready только при наличии
real document SHA-256 и совпадении SHA-256 в observed run. Hash mismatch —
ошибка контракта; незаполненный binding — incomplete.

## Blind A/B/C workflow

Frozen A запускается из tag `pdf-atlas-v1-submission`, commit
`1f7dcc0ee3cd652a3d00688e142d7da33fce802a`. B — adaptive extraction на той же
зафиксированной eval-выборке. C — тот же artifact B с bounded read-only agent.
Outputs A/B/C замораживаются, перемешиваются и получают небольшие ручные
blind labels. Эти labels вводятся в observations; evaluator их не выводит,
не угадывает и не заменяет judge-моделью. Acceptance cases
не используются для настройки: выборку, вопросы, expected pages и rubric не
меняют после начала сравнения. Если acceptance corpus или обязательные
observations неполны, результат нельзя объявлять pass.

Сначала сравниваются холодная ingestion и тёплая QA отдельно. Cold protocol:
новый/очищенный cache и полный ingestion; warm protocol: уже построенный
artifact и затем QA без повторной ingestion. Конкретные cache/model/runtime
условия фиксируются в reproducibility-полях.

## Reproducibility

Перед каждым сравнением заполняются поля manifest/run для PDF SHA-256, git
commit, lockfile, версий Docling/ocrmac/Ollama, model name и digest,
options/prompts, extraction-config fingerprint, gate version, warm/cold
protocol и seed. Seed ограничивает вариативность, но не обещает полной
детерминированности. Этот Wave 1 contract не утверждает, что baseline уже
измерен.

## Минимальный Python example

```python
from pdf_document_agent.evaluation import evaluate_run, load_manifest

manifest = load_manifest("evals/manifest.json")
report = evaluate_run(manifest, observations)  # typed CaseObservation values
print(report.status)
print(report.route_correctness)
print(report.source_recall)
```

`observations` строятся из локального запуска через публичные dataclasses
`CaseObservation`, `PageObservation`, `QuestionObservation` и
`CitationObservation`; путь к PDF при этом не записывается в manifest.
