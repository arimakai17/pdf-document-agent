import http.client
import json
import socket
from json import JSONDecodeError
from threading import Event, Lock, Thread
from time import monotonic
from urllib.error import URLError
from urllib.parse import urlsplit


DEFAULT_MODEL = "qwen3:14b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
HTTPConnection = http.client.HTTPConnection
HTTPSConnection = http.client.HTTPSConnection
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class OllamaError(RuntimeError):
    """Локальный Ollama API не смог сгенерировать ответ."""


class OllamaTimeoutError(OllamaError):
    """Обмен с Ollama превысил общий отведённый срок."""


class _ExchangeWatchdog:
    def __init__(self, deadline: float) -> None:
        self._deadline = deadline
        self._stop = Event()
        self._expired = Event()
        self._socket = None
        self._lock = Lock()
        self._thread = Thread(
            target=self._run,
            name="ollama-deadline-watchdog",
        )

    @property
    def expired(self) -> bool:
        return self._expired.is_set()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def register_socket(self, sock) -> None:
        if sock is None:
            return
        with self._lock:
            if self._expired.is_set():
                abort = True
            else:
                self._socket = sock
                abort = False
        if abort:
            self._shutdown_socket(sock)

    def _run(self) -> None:
        if self._stop.wait(max(0.0, self._deadline - monotonic())):
            return
        with self._lock:
            self._expired.set()
            sock = self._socket
        if sock is not None:
            self._shutdown_socket(sock)

    @staticmethod
    def _shutdown_socket(sock) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _connect_with_deadline(
    connection,
    deadline: float,
    watchdog: _ExchangeWatchdog,
    *,
    server_hostname: str | None,
):
    connection.timeout = max(0.001, deadline - monotonic())
    if server_hostname is None or not isinstance(
        connection,
        http.client.HTTPSConnection,
    ):
        connection.connect()
        return getattr(connection, "sock", None)

    # HTTPSConnection owns _context; keep staged socket ownership here.
    http.client.HTTPConnection.connect(connection)
    raw_socket = connection.sock
    watchdog.register_socket(raw_socket)
    raw_socket.settimeout(max(0.001, deadline - monotonic()))
    ssl_socket = connection._context.wrap_socket(
        raw_socket,
        server_hostname=server_hostname,
        do_handshake_on_connect=False,
    )
    connection.sock = ssl_socket
    watchdog.register_socket(ssl_socket)
    ssl_socket.settimeout(max(0.001, deadline - monotonic()))
    ssl_socket.do_handshake()
    return ssl_socket


def chat_with_ollama(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 180.0,
    num_predict: int = 600,
    num_ctx: int = 8_192,
) -> str:
    """Получить один нестриминговый ответ от локального Ollama API."""
    if type(num_predict) is not int or num_predict <= 0:
        raise ValueError("num_predict должен быть положительным int.")
    if type(num_ctx) is not int or num_ctx <= 0:
        raise ValueError("num_ctx должен быть положительным int.")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise ValueError("timeout должен быть положительным числом.")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "seed": 42,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    request_body = json.dumps(payload).encode("utf-8")
    try:
        parsed = urlsplit(f"{base_url.rstrip('/')}/api/chat")
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("unsupported Ollama URL")
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        deadline = monotonic() + timeout
        connection_type = (
            HTTPSConnection
            if parsed.scheme == "https"
            else HTTPConnection
        )
        connection = connection_type(
            parsed.hostname,
            parsed.port,
            timeout=max(0.001, deadline - monotonic()),
        )
    except (OSError, URLError, ValueError) as error:
        raise OllamaError(
            "Не удалось подключиться к Ollama. Запусти приложение Ollama."
        ) from error

    watchdog = _ExchangeWatchdog(deadline)
    response = None
    exchange_socket = None
    try:
        watchdog.start()
        if getattr(connection, "sock", None) is None:
            exchange_socket = _connect_with_deadline(
                connection,
                deadline,
                watchdog,
                server_hostname=(
                    parsed.hostname if parsed.scheme == "https" else None
                ),
            )
        else:
            exchange_socket = getattr(connection, "sock", None)
        watchdog.register_socket(exchange_socket)
        if exchange_socket is not None:
            exchange_socket.settimeout(max(0.001, deadline - monotonic()))
        if watchdog.expired or monotonic() >= deadline:
            raise OllamaTimeoutError("Обмен с Ollama превысил отведённый срок.")
        connection.request(
            "POST",
            path,
            body=request_body,
            headers={"Content-Type": "application/json"},
        )
        if watchdog.expired or monotonic() >= deadline:
            raise OllamaTimeoutError("Обмен с Ollama превысил отведённый срок.")
        if exchange_socket is not None:
            exchange_socket.settimeout(max(0.001, deadline - monotonic()))
        response = connection.getresponse()
        if watchdog.expired or monotonic() >= deadline:
            raise OllamaTimeoutError("Обмен с Ollama превысил отведённый срок.")
        if response.status >= 400:
            raise OllamaError(f"Ollama вернул HTTP-ошибку {response.status}.")
        if exchange_socket is not None:
            exchange_socket.settimeout(max(0.001, deadline - monotonic()))
        response_body = response.read(_MAX_RESPONSE_BYTES + 1)
        if watchdog.expired or monotonic() >= deadline:
            raise OllamaTimeoutError("Обмен с Ollama превысил отведённый срок.")
        if len(response_body) > _MAX_RESPONSE_BYTES:
            raise OllamaError("Ollama вернул слишком большой ответ.")
        try:
            body = json.loads(response_body)
        except (JSONDecodeError, UnicodeDecodeError, TypeError) as error:
            if watchdog.expired or monotonic() >= deadline:
                raise OllamaTimeoutError(
                    "Обмен с Ollama превысил отведённый срок."
                ) from error
            raise OllamaError("Ollama вернул некорректный JSON-ответ.") from error
        if watchdog.expired or monotonic() >= deadline:
            raise OllamaTimeoutError("Обмен с Ollama превысил отведённый срок.")
    except OllamaError:
        raise
    except (OSError, http.client.HTTPException, URLError) as error:
        if watchdog.expired or monotonic() >= deadline:
            raise OllamaTimeoutError(
                "Обмен с Ollama превысил отведённый срок."
            ) from error
        raise OllamaError(
            "Не удалось подключиться к Ollama. Запусти приложение Ollama."
        ) from error
    finally:
        watchdog.stop()
        if response is not None:
            response.close()
        connection.close()

    try:
        content = body["message"]["content"]
    except (KeyError, TypeError) as error:
        raise OllamaError("Ollama вернул некорректный ответ.") from error
    if not isinstance(content, str):
        raise OllamaError("Ollama вернул некорректный ответ.")
    return content
