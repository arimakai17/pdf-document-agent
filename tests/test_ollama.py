import json
import socket
import socketserver
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

    def read(self, amount: int | None = None) -> bytes:
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

    def read(self, amount: int | None = None) -> bytes:
        if not self.aborted.wait(1.0):
            raise AssertionError("test transport was not aborted")
        raise OSError("transport closed")

    def close(self) -> None:
        self.closed = True


class BlockingSocket:
    def __init__(self, response: BlockingResponse) -> None:
        self.response = response
        self.shutdown_called = False

    def shutdown(self, how: int) -> None:
        assert how == socket.SHUT_RDWR
        self.shutdown_called = True
        self.response.aborted.set()


class BlockingConnection(FakeConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response = BlockingResponse()
        self.sock = BlockingSocket(self.response)


def _start_trickle_server(
    *,
    trickle_headers: bool,
) -> tuple[socketserver.ThreadingTCPServer, threading.Thread, threading.Event]:
    done = threading.Event()
    response_headers = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 39\r\n"
        b"Connection: close\r\n"
        b"\r\n"
    )
    response_body = b'{"message":{"content":"slow response"}}'

    class TrickleHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = self.request.recv(4096)
                    if not chunk:
                        return
                    request += chunk
                if trickle_headers:
                    for byte in response_headers:
                        self.request.sendall(bytes((byte,)))
                        time.sleep(0.05)
                    self.request.sendall(response_body)
                else:
                    self.request.sendall(response_headers)
                    for byte in response_body:
                        self.request.sendall(bytes((byte,)))
                        time.sleep(0.05)
            except OSError:
                pass
            finally:
                done.set()

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), TrickleHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    return server, server_thread, done


def _track_watchdog_threads(monkeypatch: pytest.MonkeyPatch) -> list[threading.Thread]:
    real_thread = threading.Thread
    watchdogs: list[threading.Thread] = []

    class TrackingThread(real_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.name == "ollama-deadline-watchdog":
                watchdogs.append(self)

    monkeypatch.setattr(ollama, "Thread", TrackingThread)
    return watchdogs


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


def test_chat_with_ollama_rejects_oversized_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedResponse(FakeResponse):
        def read(self, amount: int | None = None) -> bytes:
            return b"x" * (ollama._MAX_RESPONSE_BYTES + 1)

    class OversizedConnection(FakeConnection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.response = OversizedResponse({})

    monkeypatch.setattr(ollama, "HTTPConnection", OversizedConnection)

    with pytest.raises(ollama.OllamaError, match="слишком большой"):
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
    assert connection.sock.shutdown_called is True


@pytest.mark.parametrize("trickle_headers", [True, False])
def test_chat_with_ollama_aborts_real_trickle_exchange_at_total_deadline(
    monkeypatch: pytest.MonkeyPatch,
    trickle_headers: bool,
) -> None:
    server, server_thread, handler_done = _start_trickle_server(
        trickle_headers=trickle_headers,
    )
    watchdogs = _track_watchdog_threads(monkeypatch)
    try:
        started = time.monotonic()
        with pytest.raises(ollama.OllamaTimeoutError):
            ollama.chat_with_ollama(
                "system",
                "user",
                base_url=f"http://127.0.0.1:{server.server_address[1]}",
                timeout=0.2,
            )
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)

    assert elapsed < 0.8
    assert handler_done.wait(1.0)
    assert server_thread.is_alive() is False
    assert len(watchdogs) == 1
    assert watchdogs[0].is_alive() is False


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
