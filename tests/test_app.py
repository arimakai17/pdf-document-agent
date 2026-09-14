from pathlib import Path
from xml.etree import ElementTree
import ast

from streamlit.testing.v1 import AppTest

from pdf_document_agent import answering
from pdf_document_agent import cache
from pdf_document_agent.agent import AgentTraceEvent
from pdf_document_agent.extractor import (
    ExtractionConfig,
    ExtractedDocument,
    ExtractedPage,
    TextRegion,
)
from pdf_document_agent.retrieval import TextChunk
from pdf_document_agent import app as app_module


APP_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "pdf_document_agent"
    / "app.py"
)


def test_history_current_uses_same_two_column_grid_as_history_rows() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    current = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and any(
            isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "history_current"
            for keyword in node.keywords
        )
    )
    assert isinstance(current.func.value, ast.Name)
    assert current.func.value.id == "history_item"
    grid = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "columns"
        and isinstance(node.args[0], ast.List)
        and [elt.value for elt in node.args[0].elts] == [9, 1]
    )
    keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in grid.keywords}
    assert keywords == {"gap": "small", "vertical_alignment": "center"}


def test_app_starts_in_russian_with_pdf_atlas_branding() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "PDF Atlas"
    assert app.button_group[0].value == "RU"
    assert len(app.get("file_uploader")) == 1
    assert app.selectbox[0].label == "Хранить вопросов в истории"
    assert app.selectbox[0].value == 10
    assert "Документ" in [item.value for item in app.subheader]
    assert "Диалог" in [item.value for item in app.subheader]
    assert "Загрузи PDF" in app.info[0].value


def test_pdf_uploader_styles_keep_native_controls_visible(monkeypatch) -> None:
    markup = []
    monkeypatch.setattr(app_module.st, "markdown", lambda value, **kwargs: markup.append(value))

    app_module._apply_styles(app_module.DEFAULT_LOCALE)

    css = "".join(markup)
    assert '[data-testid="stFileUploaderDropzone"]>*{visibility:hidden;' not in "".join(css.split())
    assert 'stFileUploaderDropzone::before' not in css
    assert 'stFileUploaderDropzone::after' not in css


def test_language_switch_changes_entire_empty_state_ui_to_english() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    app.button_group[0].set_value("EN").run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "PDF Atlas"
    assert app.button_group[0].value == "EN"
    assert app.selectbox[0].label == "Questions kept in history"
    assert "Document" in [item.value for item in app.subheader]
    assert "Conversation" in [item.value for item in app.subheader]
    assert "Upload a PDF" in app.info[0].value


def test_cache_controls_localized_in_russian(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    captions = [item.value for item in app.caption]
    assert any("кэш" in caption for caption in captions)
    assert any(button.label == "Очистить кэш" for button in app.button)


def test_cache_controls_localized_in_english(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.button_group[0].set_value("EN").run(timeout=30)

    assert not app.exception
    captions = [item.value for item in app.caption]
    assert any("cache" in caption.lower() for caption in captions)
    assert any(button.label == "Clear cache" for button in app.button)


def test_clear_cache_button_removes_disk_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    (tmp_path / "deadbeef.json").write_text('{"schema_version": 1}', encoding="utf-8")
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    clear_button = next(button for button in app.button if button.label == "Очистить кэш")
    clear_button.click().run(timeout=30)

    assert not app.exception
    assert list(tmp_path.glob("*.json")) == []
    assert any(
        "кэш" in item.value.lower() and "очищен" in item.value.lower()
        for item in app.success
    )


# --- collapsible upload panel -------------------------------------------------


def _seed_cached_pdf(file_bytes: bytes) -> None:
    """Prime the disk cache so a fake PDF hits the cache without real Docling."""
    region = TextRegion(page_number=1, text="Hello world", box=(0.0, 0.0, 1.0, 1.0))
    page = ExtractedPage(number=1, markdown="# Test\n\nHello world", regions=(region,))
    document = ExtractedDocument(
        source_name="sample.pdf",
        markdown="# Test\n\nHello world",
        page_count=1,
        pages=(page,),
    )
    chunk = TextChunk(
        index=0,
        page_number=1,
        text="Hello world",
        boxes=((0.0, 0.0, 1.0, 1.0),),
        regions=(region,),
    )
    cache.put(file_bytes, document, [chunk])


def _uploader_expander(app, label: str):
    return next(expander for expander in app.get("expander") if expander.label == label)


def test_uploader_expander_open_on_fresh_session() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    expander = _uploader_expander(app, "PDF-документ")
    assert expander.proto.expanded is True
    assert app.session_state["uploader_expander"] is True


def test_uploader_expander_collapses_after_successful_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4\n1 0 obj\nfake\n"
    _seed_cached_pdf(file_bytes)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("sample.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)

    assert not app.exception
    expander = _uploader_expander(app, "PDF-документ")
    assert expander.proto.expanded is False
    assert app.session_state["uploader_expander"] is False


def test_uploader_expander_reopen_persists_across_reruns(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4\n1 0 obj\nfake\n"
    _seed_cached_pdf(file_bytes)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("sample.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)
    assert app.session_state["uploader_expander"] is False

    # AppTest cannot synthesize a frontend expander click; emulate the documented
    # keyed session-state transition the on_change callback would perform.
    app.session_state["uploader_expander"] = True
    app.run(timeout=30)
    assert app.session_state["uploader_expander"] is True

    # A plain rerun (answering, etc.) must not re-collapse the panel.
    app.run(timeout=30)
    assert app.session_state["uploader_expander"] is True


def test_failed_upload_keeps_expander_open(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(("empty.pdf", b"", "application/pdf"))
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["uploader_expander"] is True
    expander = _uploader_expander(app, "PDF-документ")
    assert expander.proto.expanded is True
    assert app.error


def test_language_switch_preserves_expander_state(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4\n1 0 obj\nfake\n"
    _seed_cached_pdf(file_bytes)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("sample.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)
    assert app.session_state["uploader_expander"] is False

    app.button_group[0].set_value("EN").run(timeout=30)

    assert not app.exception
    assert app.session_state["uploader_expander"] is False
    expander = _uploader_expander(app, "PDF document")
    assert expander.proto.expanded is False


def test_no_duplicate_uploader_label_inside_expander() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert [e.label for e in app.get("expander")].count("PDF-документ") == 1
    uploader = app.get("file_uploader")[0]
    assert "COLLAPSED" in str(uploader.proto.label_visibility)
    assert uploader.help == "Поддерживаются текстовые PDF и сканы до 50 МБ."
    assert uploader.allowed_type == [".pdf"]


def test_history_is_visible_and_can_switch_conversation(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4\n1 0 obj\nfake\n"
    _seed_cached_pdf(file_bytes)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("sample.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)
    app.session_state["messages"] = [
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Первый ответ", "sources": ()},
        {"role": "user", "content": "Второй вопрос"},
        {"role": "assistant", "content": "Второй ответ", "sources": ()},
    ]
    app.run(timeout=30)

    assert not app.exception
    assert [item.proto.popover.label for item in app.get("popover")] == [
        "История вопросов"
    ]

    next(button for button in app.button if button.label == "Первый вопрос").click()
    app.run(timeout=30)

    assert app.session_state["selected_history_index"] == 0
    assert [message.markdown[0].value for message in app.chat_message] == [
        "Первый вопрос",
        "Первый ответ",
    ]
    next(
        button
        for button in app.button
        if button.label == "Вернуться к текущему диалогу"
    ).click()
    app.run(timeout=30)
    assert len(app.chat_message) == 4


def test_delete_controls_remove_whole_exchange_from_chat_and_history(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4\n1 0 obj\nfake\n"
    _seed_cached_pdf(file_bytes)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("sample.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)
    app.session_state["messages"] = [
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Первый ответ", "sources": ()},
        {"role": "user", "content": "Второй вопрос"},
        {"role": "assistant", "content": "Второй ответ", "sources": ()},
        {"role": "user", "content": "Ошибочный вопрос без ответа"},
    ]
    app.run(timeout=30)

    app.get_by_key("delete_message_1").click().run(timeout=30)
    assert [message["content"] for message in app.session_state["messages"]] == [
        "Второй вопрос",
        "Второй ответ",
        "Ошибочный вопрос без ответа",
    ]
    assert not any(button.label == "Первый вопрос" for button in app.button)

    app.get_by_key("delete_history_0").click().run(timeout=30)
    assert [message["content"] for message in app.session_state["messages"]] == [
        "Ошибочный вопрос без ответа"
    ]
    assert not any(button.label == "Второй вопрос" for button in app.button)

    app.get_by_key("delete_message_0").click().run(timeout=30)
    assert app.session_state["messages"] == []
    assert not app.exception


def test_failed_answer_rerenders_question_with_delete_controls(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        answering,
        "answer_question",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            answering.AnswerGenerationError("model failure")
        ),
    )
    file_bytes = b"%PDF-1.4\n1 0 obj\nfake\n"
    _seed_cached_pdf(file_bytes)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("sample.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)
    next(
        item for item in app.text_input if item.label == "Вопрос по документу"
    ).set_value("Вопрос со сбоем")
    next(button for button in app.button if button.label == "Получить ответ").click()
    app.run(timeout=30)

    assert not app.exception
    assert app.get_by_key("delete_message_0").label == "×"
    assert app.get_by_key("delete_history_0").label == "×"
    assert any(button.label == "Вопрос со сбоем" for button in app.button)
    assert any("Попробуй ещё раз" in error.value for error in app.error)


# --- animated status mascot ---------------------------------------------------


def _all_markup(app) -> str:
    """Join every st.markdown body emitted by the app (CSS + mascot HTML)."""
    return "\n".join(m.value for m in app.markdown)


def _normalized_css(app) -> str:
    """Whitespace-collapsed markup for deterministic CSS substring assertions."""
    return "".join(_all_markup(app).split())


def test_busy_mascot_primitives_have_stroke_padding_inside_viewbox() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    markup = _all_markup(app)
    svg = markup.split('<svg class="mascot-cat mascot-cat-busy"', 1)[1].split('</svg>', 1)[0]
    root = ElementTree.fromstring('<svg' + svg + '</svg>')
    for element in root.iter():
        stroke = float(element.attrib.get("stroke-width", "0")) / 2
        if element.tag.endswith("circle"):
            assert float(element.attrib["cx"]) + float(element.attrib["r"]) + stroke <= 46
        elif element.tag.endswith("line"):
            assert float(element.attrib["x2"]) + stroke <= 46


def test_mascot_renders_idle_and_busy_poses() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    markup = _all_markup(app)
    assert "pdf-atlas-mascot" in markup
    assert "mascot-idle" in markup
    assert "mascot-busy" in markup
    # both poses are exposed as images for screen readers
    assert markup.count('role="img"') >= 2


def test_mascot_switches_on_real_streamlit_running_icon_selector() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    css = _normalized_css(app)
    # anchored to Streamlit's real running indicator test-id
    assert 'data-testid="stStatusWidgetRunningIcon"' in css
    assert ":has(" in css
    # graceful fallback gate so browsers without :has keep native behavior
    assert "@supportsselector(body:has(*))" in css
    # under the running icon, idle hides and busy shows
    assert "body:has([data-testid=\"stStatusWidgetRunningIcon\"]).mascot-idle{display:none;}" in css
    assert "body:has([data-testid=\"stStatusWidgetRunningIcon\"]).mascot-busy{display:block;}" in css


def test_mascot_hides_only_running_icon_not_status_widget_or_stop() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    css = _normalized_css(app)
    # hide the running-person graphic specifically, by keeping its layout
    # (visibility, not display) so the Stop button stays put
    assert '[data-testid="stStatusWidgetRunningIcon"]{visibility:hidden;}' in css
    # never target the whole status widget (which wraps the Stop button)
    assert 'data-testid="stStatusWidget"' not in css
    assert "stStatusWidgetStopButton" not in css


def test_mascot_busy_pose_hidden_by_default_for_no_has_fallback() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    css = _normalized_css(app)
    # default: idle visible, busy hidden, native indicator untouched
    assert ".mascot-busy{display:none;}" in css


def test_mascot_supports_prefers_reduced_motion() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    css = _normalized_css(app)
    assert "prefers-reduced-motion:reduce" in css
    assert "animation:none" in css


def test_mascot_labels_localized_russian_and_english() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    ru = _all_markup(app)
    assert "PDF Atlas: ожидание" in ru
    assert "PDF Atlas: обработка" in ru

    app.button_group[0].set_value("EN").run(timeout=30)
    en = _all_markup(app)
    assert "PDF Atlas: idle" in en
    assert "PDF Atlas: working" in en
    assert "PDF Atlas: ожидание" not in en


def test_mascot_preserves_app_startup() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "PDF Atlas"
    assert len(app.get("file_uploader")) == 1
    assert "pdf-atlas-mascot" in _all_markup(app)


# --- Wave 6B answer mode and extraction instrumentation --------------------


def _seed_document(file_bytes: bytes, document: ExtractedDocument, chunks) -> None:
    cache.put(
        file_bytes,
        document,
        chunks,
        config_fingerprint=document.config_fingerprint,
    )


def test_empty_state_exposes_fixed_default_and_localized_ocr_control() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert app.radio[0].value == "fixed"
    assert app.radio[0].options == [
        "Фиксированный · кандидат B (HOLD; технический default в этой ветке)",
        "Ограниченный агент · экспериментальный C",
    ]
    assert next(item for item in app.text_input if item.label == "Страницы для ручного OCR")

    app.button_group[0].set_value("EN").run(timeout=30)

    assert app.radio[0].value == "fixed"
    assert app.radio[0].options == [
        "Fixed · B candidate (HOLD; technical default in this branch)",
        "Bounded-agent · experimental C",
    ]
    assert next(item for item in app.text_input if item.label == "Pages for manual OCR")


def test_bounded_chat_maps_agent_budget_and_forwards_model_timeout(monkeypatch) -> None:
    calls = []

    def fake_chat(system_prompt, user_prompt, **kwargs):
        calls.append((system_prompt, user_prompt, kwargs))
        return "ok"

    monkeypatch.setattr(app_module, "chat_with_ollama", fake_chat)

    assert app_module._bounded_chat(
        "system", "user", model="qwen", max_output_tokens=37, timeout=4.5
    ) == "ok"
    assert calls == [
        (
            "system",
            "user",
            {
                "model": "qwen",
                "timeout": 4.5,
                "num_predict": 37,
                "num_ctx": 8192,
            },
        )
    ]


def test_extraction_summary_keeps_routes_status_diagnostics_and_warnings_visible(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4 summary"
    document = ExtractedDocument(
        source_name="summary.pdf",
        markdown="usable text",
        page_count=3,
        pages=(
            ExtractedPage(
                number=1,
                markdown="usable text",
                route="docling_text",
                status="ok",
                reason="meaningful text gate passed",
            ),
            ExtractedPage(
                number=2,
                markdown="",
                route="docling_ocr",
                status="failed",
                reason="selective OCR failed",
                diagnostic="OCR unavailable",
            ),
            ExtractedPage(
                number=3,
                markdown="",
                route="docling_text",
                status="empty",
                reason="meaningful text gate failed; no significant raster",
            ),
        ),
        warnings=("Page 2: OCR unavailable",),
    )
    region = TextRegion(page_number=1, text="usable text", box=(0, 0, 1, 1))
    chunks = [
        TextChunk(
            index=0,
            page_number=1,
            text="usable text",
            boxes=(region.box,),
            regions=(region,),
        )
    ]
    _seed_document(file_bytes, document, chunks)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("summary.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)

    assert not app.exception
    assert any("Извлечение частичное" in item.value for item in app.warning)
    details = "\n".join(item.value for item in app.caption)
    assert "docling_text" in details
    assert "docling_ocr" in details
    assert "OCR unavailable" in details
    assert "Page 2: OCR unavailable" in details


def test_all_ok_extraction_is_not_reported_as_partial(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4 all ok"
    _seed_cached_pdf(file_bytes)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("all-ok.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)

    assert not app.exception
    assert not any("частич" in item.value.casefold() for item in app.warning)


def test_mode_change_keeps_artifact_but_ocr_override_changes_it(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4 config identity"
    region = TextRegion(page_number=1, text="usable text", box=(0, 0, 1, 1))
    chunks = [
        TextChunk(
            index=0,
            page_number=1,
            text="usable text",
            boxes=(region.box,),
            regions=(region,),
        )
    ]
    default_document = ExtractedDocument(
        source_name="identity.pdf",
        markdown="usable text",
        page_count=1,
        pages=(
            ExtractedPage(
                number=1,
                markdown="usable text",
                regions=(region,),
                route="docling_text",
            ),
        ),
    )
    override_config = ExtractionConfig(ocr_pages=(1,))
    override_document = ExtractedDocument(
        source_name="identity.pdf",
        markdown="usable text",
        page_count=1,
        pages=(
            ExtractedPage(
                number=1,
                markdown="usable text",
                regions=(region,),
                route="docling_ocr",
                reason="manual override",
            ),
        ),
        config_fingerprint=override_config.fingerprint,
    )
    _seed_document(file_bytes, default_document, chunks)
    _seed_document(file_bytes, override_document, chunks)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("identity.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)
    default_identity = app.session_state["artifact_identity"]
    assert app.session_state["document"].pages[0].route == "docling_text"

    app.radio[0].set_value("agent").run(timeout=30)
    assert app.session_state["artifact_identity"] == default_identity

    next(
        item for item in app.text_input if item.label == "Страницы для ручного OCR"
    ).set_value("1").run(timeout=30)

    assert app.session_state["artifact_identity"] != default_identity
    assert app.session_state["artifact_identity"][1] == override_config.fingerprint
    assert app.session_state["document"].pages[0].route == "docling_ocr"


def test_agent_dispatch_stores_compact_trace_and_does_not_call_fixed(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4 agent"
    _seed_cached_pdf(file_bytes)
    fixed_calls = []
    agent_calls = []

    def fake_fixed(*args, **kwargs):
        fixed_calls.append((args, kwargs))
        raise AssertionError("fixed mode must not run for an agent question")

    def fake_agent(question, document, chunks, **kwargs):
        agent_calls.append((question, document, chunks, kwargs))
        return type(
            "FakeAgentRun",
            (),
            {
                "answer": type("Answer", (), {"text": "agent answer", "sources": ()})(),
                "status": "ok",
                "tool_calls": 1,
                "llm_calls": 2,
                "trace": type(
                    "Trace",
                    (),
                    {
                        "events": (
                            AgentTraceEvent(
                                "search", "ok", pages=(1,), truncated=True
                            ),
                        )
                    },
                )(),
            },
        )()

    monkeypatch.setattr(answering, "answer_question", fake_fixed)
    monkeypatch.setattr("pdf_document_agent.agent.run_agent", fake_agent)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("agent.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)
    app.radio[0].set_value("agent").run(timeout=30)
    next(item for item in app.text_input if item.label == "Вопрос по документу").set_value(
        "Что известно?"
    )
    next(button for button in app.button if button.label == "Получить ответ").click().run(
        timeout=30
    )

    assert not app.exception
    assert not fixed_calls
    assert len(agent_calls) == 1
    metadata = app.session_state["messages"][-1]["metadata"]
    assert metadata == {
        "mode": "agent",
        "status": "ok",
        "tool_calls": 1,
        "llm_calls": 2,
        "trace": (
            {
                "action": "search",
                "status": "ok",
                "pages": (1,),
                "truncated": True,
            },
        ),
    }
    rendered = "\n".join(item.value for item in app.caption)
    assert "search" in rendered
    assert "truncated: yes" in rendered


def test_agent_non_ok_renders_canonical_answer_without_fixed_fallback(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4 agent failure"
    _seed_cached_pdf(file_bytes)
    fixed_calls = []

    def fake_fixed(*args, **kwargs):
        fixed_calls.append(True)
        raise AssertionError("fixed fallback is forbidden")

    def fake_agent(*args, **kwargs):
        return type(
            "FakeAgentRun",
            (),
            {
                "answer": type(
                    "Answer",
                    (),
                    {"text": "The document does not contain enough information to answer.", "sources": ()},
                )(),
                "status": "invalid_action",
                "tool_calls": 1,
                "llm_calls": 1,
                "trace": type(
                    "Trace",
                    (),
                    {"events": (AgentTraceEvent("invalid_action", "invalid_action"),)},
                )(),
            },
        )()

    monkeypatch.setattr(answering, "answer_question", fake_fixed)
    monkeypatch.setattr("pdf_document_agent.agent.run_agent", fake_agent)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("agent-failure.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)
    app.radio[0].set_value("agent").run(timeout=30)
    next(item for item in app.text_input if item.label == "Вопрос по документу").set_value(
        "What is known?"
    )
    next(button for button in app.button if button.label == "Получить ответ").click().run(
        timeout=30
    )

    assert not app.exception
    assert fixed_calls == []
    assert app.session_state["messages"][-1]["content"].startswith(
        "The document does not contain enough information"
    )
    assert any("invalid_action" in item.value for item in app.warning)
