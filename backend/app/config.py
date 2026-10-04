import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPOSITORY_ROOT / ".env")
_BACKEND_ENV = dotenv_values(_REPOSITORY_ROOT / "backend" / ".env")


def _string_setting(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _integer_setting(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        value = _BACKEND_ENV.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _supadata_api_key_setting() -> str:
    return _string_setting(
        "SUPADATA_API_KEY",
        _BACKEND_ENV.get("SUPADATA_API_KEY") or "",
    )


def _backend_setting(name: str, default: str = "") -> str:
    return _string_setting(name, _BACKEND_ENV.get(name) or default)


GROQ_API_KEY: str = _backend_setting("GROQ_API_KEY")
HF_TOKEN: str = _backend_setting("HF_TOKEN")
SUPADATA_API_KEY: str = _supadata_api_key_setting()

LLM_MODEL: str = _backend_setting("LLM_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL: str = _backend_setting("EMBED_MODEL", "intfloat/multilingual-e5-small")
RETRIEVER_K: int = _integer_setting("RETRIEVER_K", 4)
FULL_CONTEXT_CHAR_LIMIT: int = _integer_setting("FULL_CONTEXT_CHAR_LIMIT", 24000)

FRONTEND_ORIGIN: str = _backend_setting("FRONTEND_ORIGIN", "http://localhost:3000")


def check_config() -> list[str]:
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if not SUPADATA_API_KEY:
        missing.append("SUPADATA_API_KEY")
    if missing:
        print(f"Missing required configuration keys: {', '.join(missing)}")
    else:
        print("All required configuration keys are set.")
    return missing
