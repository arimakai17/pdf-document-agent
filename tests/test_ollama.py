import json
from urllib.error import URLError

import pytest

from pdf_document_agent import ollama


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_chat_with_ollama_sends_non_streaming_grounded_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_urlopen(request, *, timeout: float):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse({"message": {"content": "Ответ [стр. 1]."}})

    monkeypatch.setattr(ollama, "urlopen", fake_urlopen)

    result = ollama.chat_with_ollama(
        "Системная инструкция",
        "Вопрос",
        model="qwen3:14b",
        base_url="http://127.0.0.1:11434/",
    )

    assert result == "Ответ [стр. 1]."
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is False
    assert captured["payload"]["model"] == "qwen3:14b"


def test_chat_with_ollama_maps_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(_request, *, timeout: float):
        raise URLError("connection refused")

    monkeypatch.setattr(ollama, "urlopen", fail_urlopen)

    with pytest.raises(ollama.OllamaError, match="Ollama"):
        ollama.chat_with_ollama("system", "user")


def test_chat_with_ollama_rejects_invalid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama,
        "urlopen",
        lambda _request, *, timeout: FakeResponse({"unexpected": True}),
    )

    with pytest.raises(ollama.OllamaError, match="некорректный ответ"):
        ollama.chat_with_ollama("system", "user")