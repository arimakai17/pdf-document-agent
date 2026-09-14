import json
import threading
import time
from urllib.error import URLError

import pytest

from pdf_document_agent import ollama


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = 200
        self.closed = False

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    instances: list["FakeConnection"] = []

    def __init__(self, host: str, port: int | None = None, *, timeout: float):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.request_args = None
        self.response = FakeResponse({"message": {"content": "Ответ [стр. 1]."}})
        self.closed = False
        type(self).instances.append(self)

    def request(self, method, path, body, headers):
        self.request_args = (method, path, body, headers)

    def getresponse(self):
        return self.response

    def close(self) -> None:
        self.closed = True
        self.response.close()


class BlockingResponse:
    status = 200

    def __init__(self) -> None:
        self.closed = False
        self.aborted = threading.Event()

    def read(self) -> bytes:
        if not self.aborted.wait(1.0):
            raise AssertionError("test transport was not aborted")
        raise OSError("transport closed")

    def close(self) -> None:
        self.closed = True
        self.aborted.set()


class BlockingConnection(FakeConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response = BlockingResponse()


def test_chat_with_ollama_sends_non_streaming_grounded_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeConnection.instances.clear()
    monkeypatch.setattr(ollama, "HTTPConnection", FakeConnection)

    result = ollama.chat_with_ollama(
        "Системная инструкция",
        "Вопрос",
        model="qwen3:14b",
        base_url="http://127.0.0.1:11434/",
    )

    assert result == "Ответ [стр. 1]."
    connection = FakeConnection.instances[0]
    method, path, body, _headers = connection.request_args
    payload = json.loads(body)
    assert connection.host == "127.0.0.1"
    assert connection.port == 11434
    assert method == "POST"
    assert path == "/api/chat"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["model"] == "qwen3:14b"


def test_chat_with_ollama_accepts_bounded_generation_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeConnection.instances.clear()
    monkeypatch.setattr(ollama, "HTTPConnection", FakeConnection)

    ollama.chat_with_ollama(
        "system",
        "user",
        num_predict=17,
        num_ctx=321,
    )

    payload = json.loads(FakeConnection.instances[0].request_args[2])
    assert payload["options"]["num_predict"] == 17
    assert payload["options"]["num_ctx"] == 321


@pytest.mark.parametrize("field", ["num_predict", "num_ctx"])
def test_chat_with_ollama_rejects_non_positive_or_bool_budget(
    field: str,
) -> None:
    kwargs = {field: True}
    with pytest.raises(ValueError):
        ollama.chat_with_ollama("system", "user", **kwargs)


def test_chat_with_ollama_maps_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingConnection:
        def __init__(self, *_args, **_kwargs):
            raise URLError("connection refused")

    monkeypatch.setattr(ollama, "HTTPConnection", FailingConnection)

    with pytest.raises(ollama.OllamaError, match="Ollama"):
        ollama.chat_with_ollama("system", "user")


def test_chat_with_ollama_rejects_invalid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class InvalidResponseConnection(FakeConnection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.response = FakeResponse({"unexpected": True})

    monkeypatch.setattr(ollama, "HTTPConnection", InvalidResponseConnection)

    with pytest.raises(ollama.OllamaError, match="некорректный ответ"):
        ollama.chat_with_ollama("system", "user")


def test_chat_with_ollama_aborts_slow_body_at_total_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    BlockingConnection.instances.clear()
    monkeypatch.setattr(ollama, "HTTPConnection", BlockingConnection)

    started = time.monotonic()
    with pytest.raises(ollama.OllamaTimeoutError):
        ollama.chat_with_ollama("system", "user", timeout=0.03)
    elapsed = time.monotonic() - started

    connection = BlockingConnection.instances[0]
    assert elapsed < 0.5
    assert connection.closed is True
    assert connection.response.closed is True
    assert connection.response.aborted.is_set()


def test_chat_with_ollama_cleans_watchdog_and_closes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeConnection.instances.clear()
    monkeypatch.setattr(ollama, "HTTPConnection", FakeConnection)
    real_thread = threading.Thread
    joined: list[threading.Thread] = []

    class TrackingThread(real_thread):
        def join(self, timeout=None):
            joined.append(self)
            return super().join(timeout)

    monkeypatch.setattr(ollama, "Thread", TrackingThread)

    assert ollama.chat_with_ollama("system", "user", timeout=0.5) == "Ответ [стр. 1]."

    assert FakeConnection.instances[0].response.closed is True
    assert len(joined) == 1
    assert joined[0].is_alive() is False
