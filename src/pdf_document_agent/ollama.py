import json
from json import JSONDecodeError
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "qwen3:14b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    """Локальный Ollama API не смог сгенерировать ответ."""


def chat_with_ollama(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 180.0,
) -> str:
    """Получить один нестриминговый ответ от локального Ollama API."""
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
            "num_ctx": 8_192,
            "num_predict": 600,
        },
    }
    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except HTTPError as error:
        raise OllamaError(
            f"Ollama вернул HTTP-ошибку {error.code}."
        ) from error
    except URLError as error:
        raise OllamaError(
            "Не удалось подключиться к Ollama. Запусти приложение Ollama."
        ) from error
    except (JSONDecodeError, UnicodeDecodeError) as error:
        raise OllamaError("Ollama вернул некорректный JSON-ответ.") from error

    try:
        content = body["message"]["content"]
    except (KeyError, TypeError) as error:
        raise OllamaError("Ollama вернул некорректный ответ.") from error
    if not isinstance(content, str):
        raise OllamaError("Ollama вернул некорректный ответ.")
    return content