import http.client
import json
from json import JSONDecodeError
from threading import Event, Lock, Thread
from time import monotonic
from urllib.error import URLError
from urllib.parse import urlsplit


DEFAULT_MODEL = "qwen3:14b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
HTTPConnection = http.client.HTTPConnection
HTTPSConnection = http.client.HTTPSConnection


class OllamaError(RuntimeError):
    """Локальный Ollama API не смог сгенерировать ответ."""


class OllamaTimeoutError(OllamaError):
    """Обмен с Ollama превысил общий отведённый срок."""


class _ExchangeWatchdog:
    def __init__(self, connection: http.client.HTTPConnection, deadline: float) -> None:
        self._connection = connection
        self._deadline = deadline
        self._response = None
        self._lock = Lock()
        self._stop = Event()
        self.expired = False
        self._thread = Thread(
            target=self._run,
            name="ollama-deadline-watchdog",
        )

    def start(self) -> None:
        self._thread.start()

    def set_response(self, response) -> None:
        with self._lock:
            self._response = response
            expired = self.expired
        if expired:
            response.close()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        if self._stop.wait(max(0.0, self._deadline - monotonic())):
            return
        self.expired = True
        with self._lock:
            response = self._response
        if response is not None:
            response.close()
        self._connection.close()


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

    watchdog = _ExchangeWatchdog(connection, deadline)
    response = None
    try:
        watchdog.start()
        connection.request(
            "POST",
            path,
            body=request_body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        watchdog.set_response(response)
        if watchdog.expired or monotonic() >= deadline:
            raise OllamaTimeoutError("Обмен с Ollama превысил отведённый срок.")
        if response.status >= 400:
            raise OllamaError(f"Ollama вернул HTTP-ошибку {response.status}.")
        response_body = response.read()
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
        if response is not None:
            response.close()
        watchdog.stop()
        connection.close()

    try:
        body = json.loads(response_body)
    except (JSONDecodeError, UnicodeDecodeError) as error:
        raise OllamaError("Ollama вернул некорректный JSON-ответ.") from error

    try:
        content = body["message"]["content"]
    except (KeyError, TypeError) as error:
        raise OllamaError("Ollama вернул некорректный ответ.") from error
    if not isinstance(content, str):
        raise OllamaError("Ollama вернул некорректный ответ.")
    return content
