"""api_config.py — Centralised API endpoint / key resolution.

RedEdit talks to an OpenAI-compatible chat-completions endpoint. By default it
uses SiliconFlow; set ``OPENAI_BASE_URL`` to point at any other
OpenAI-compatible server (e.g. a local vLLM instance), and ``OPENAI_API_KEY``
as the key (``SILICONFLOW_API_KEY`` is used otherwise).

These helpers keep the endpoint handling in one place so the README's claim
("you can also use OpenAI-compatible endpoints") is actually true in code.
"""

import os

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"


def resolve_base_url() -> str:
    """Return the OpenAI-compatible base URL (env override or SiliconFlow)."""
    return os.getenv("OPENAI_BASE_URL", "").rstrip("/") or DEFAULT_BASE_URL


def resolve_api_key() -> str:
    """Return the API key: SILICONFLOW_API_KEY first, then OPENAI_API_KEY."""
    return os.getenv("SILICONFLOW_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")


def require_api_key() -> str:
    """Return the API key or raise a helpful error."""
    key = resolve_api_key()
    if not key:
        raise EnvironmentError(
            "No API key set. Export SILICONFLOW_API_KEY (default) or "
            "OPENAI_API_KEY when using a custom OPENAI_BASE_URL endpoint.\n"
            "Get a SiliconFlow key at https://cloud.siliconflow.cn/account/ak"
        )
    return key
