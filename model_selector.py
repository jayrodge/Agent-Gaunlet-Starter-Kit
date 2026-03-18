"""Proxy model roster discovery helpers for starter-kit agents."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from arena_clients.config import get_llm_api_key, get_proxy_host


def resolve_proxy_api_key(api_key: str = "") -> str:
    """Resolve proxy auth key for competitor access."""
    explicit = (api_key or "").strip()
    if explicit:
        return explicit
    return get_llm_api_key()


def _build_proxy_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    agent_id = str(os.getenv("AGENT_ID") or "").strip()
    usage_scope = str(os.getenv("ARENA_USAGE_SCOPE") or "").strip()
    if agent_id:
        headers["X-Agent-ID"] = agent_id
    if usage_scope:
        headers["X-Round-ID"] = usage_scope
    return headers


def _parse_proxy_model_ids(payload: Any) -> list[str]:
    model_ids: list[str] = []
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    model_ids.append(model_id.strip())
        elif isinstance(payload.get("models"), list):
            for item in payload.get("models", []):
                if isinstance(item, str) and item.strip():
                    model_ids.append(item.strip())
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, str) and item.strip():
                model_ids.append(item.strip())

    seen: set[str] = set()
    ordered: list[str] = []
    for model_id in model_ids:
        if model_id in seen:
            continue
        seen.add(model_id)
        ordered.append(model_id)
    return ordered


def fetch_available_models(proxy_host: str | None = None, api_key: str = "") -> list[str]:
    """Fetch available model IDs from the proxy `/models` endpoint."""
    resolved_proxy_host = get_proxy_host(proxy_host)
    url = f"{resolved_proxy_host.rstrip('/')}/models"
    headers = {"Accept": "application/json"}
    resolved_key = resolve_proxy_api_key(api_key)
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"
    headers.update(_build_proxy_headers())
    request = Request(url, headers=headers, method="GET")

    try:
        with urlopen(request, timeout=3.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, HTTPError):
        return []

    return _parse_proxy_model_ids(payload)


__all__ = ["fetch_available_models", "resolve_proxy_api_key"]
