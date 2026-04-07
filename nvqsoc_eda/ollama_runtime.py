from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any


_OLLAMA_SERVE_PROCESS: subprocess.Popen[bytes] | None = None
_OAUTH_TOKEN_CACHE: dict[str, dict[str, Any]] = {}


@dataclass(slots=True)
class OllamaAuthContext:
    auth_header: str | None = None
    mode: str = "none"
    oauth_token_acquired: bool = False
    detail: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OllamaRuntimeStatus:
    endpoint: str
    available: bool = False
    started_local_service: bool = False
    model_available: bool = False
    model_pulled: bool = False
    model_loaded: bool = False
    models: list[str] = field(default_factory=list)
    auth_mode: str = "none"
    detail: str = "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def is_local_ollama_endpoint(endpoint: str) -> bool:
    host = (urllib.parse.urlsplit(endpoint).hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def prefer_local_preloaded_model(auth_mode: str, endpoint: str) -> bool:
    mode = (auth_mode or "").strip().lower()
    return mode.startswith("oauth") and is_local_ollama_endpoint(endpoint)


def _endpoint_with_path(endpoint: str, path: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def _tags_endpoint(endpoint: str) -> str:
    return _endpoint_with_path(endpoint, "/api/tags")


def _pull_endpoint(endpoint: str) -> str:
    return _endpoint_with_path(endpoint, "/api/pull")


def _generate_endpoint(endpoint: str) -> str:
    return _endpoint_with_path(endpoint, "/api/generate")


def _oauth_cache_key(token_url: str, payload: dict[str, str]) -> str:
    ordered = "&".join(f"{key}={payload[key]}" for key in sorted(payload))
    return f"{token_url}|{ordered}"


def _cached_oauth_token(cache_key: str) -> str | None:
    entry = _OAUTH_TOKEN_CACHE.get(cache_key)
    if not entry:
        return None
    if float(entry.get("expires_at", 0.0)) <= time.time() + 30.0:
        _OAUTH_TOKEN_CACHE.pop(cache_key, None)
        return None
    token = str(entry.get("access_token", "")).strip()
    return token or None


def _fetch_oauth_token(
    token_url: str,
    *,
    client_id: str,
    client_secret: str,
    scope: str = "",
    audience: str = "",
    refresh_token: str = "",
    timeout_s: int = 20,
) -> str | None:
    payload = {
        "grant_type": "refresh_token" if refresh_token.strip() else "client_credentials",
        "client_id": client_id.strip(),
        "client_secret": client_secret.strip(),
    }
    if refresh_token.strip():
        payload["refresh_token"] = refresh_token.strip()
    if scope.strip():
        payload["scope"] = scope.strip()
    if audience.strip():
        payload["audience"] = audience.strip()
    cache_key = _oauth_cache_key(token_url, payload)
    cached = _cached_oauth_token(cache_key)
    if cached:
        return cached

    body = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        token_url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result = json.loads(response.read().decode("utf-8", "ignore"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None

    token = str(result.get("access_token", "")).strip()
    if not token:
        return None
    expires_in = max(float(result.get("expires_in", 3600)), 60.0)
    _OAUTH_TOKEN_CACHE[cache_key] = {
        "access_token": token,
        "expires_at": time.time() + expires_in,
    }
    return token


def resolve_ollama_auth(
    ollama_auth_header: str | None = None,
    ollama_auth_token: str | None = None,
    ollama_basic_auth: str | None = None,
    *,
    ollama_oauth_token_url: str | None = None,
    ollama_oauth_client_id: str | None = None,
    ollama_oauth_client_secret: str | None = None,
    ollama_oauth_scope: str | None = None,
    ollama_oauth_audience: str | None = None,
    ollama_oauth_refresh_token: str | None = None,
    timeout_s: int = 20,
) -> OllamaAuthContext:
    raw_header = (ollama_auth_header or os.getenv("NVQSOC_OLLAMA_AUTH_HEADER") or "").strip()
    if raw_header:
        return OllamaAuthContext(auth_header=raw_header, mode="header", detail="explicit_header")

    token = (
        ollama_auth_token
        or os.getenv("NVQSOC_OLLAMA_AUTH_TOKEN")
        or os.getenv("OLLAMA_AUTH_TOKEN")
        or os.getenv("OLLAMA_API_KEY")
        or ""
    ).strip()
    if token:
        lowered = token.lower()
        if lowered.startswith("bearer ") or lowered.startswith("basic "):
            return OllamaAuthContext(auth_header=token, mode="token", detail="preformatted_token")
        return OllamaAuthContext(auth_header=f"Bearer {token}", mode="token", detail="bearer_token")

    basic_auth = (ollama_basic_auth or os.getenv("NVQSOC_OLLAMA_BASIC_AUTH") or "").strip()
    if basic_auth:
        encoded = base64.b64encode(basic_auth.encode("utf-8")).decode("ascii")
        return OllamaAuthContext(auth_header=f"Basic {encoded}", mode="basic", detail="basic_auth")

    oauth_token_url = (ollama_oauth_token_url or os.getenv("NVQSOC_OLLAMA_OAUTH_TOKEN_URL") or "").strip()
    oauth_client_id = (ollama_oauth_client_id or os.getenv("NVQSOC_OLLAMA_OAUTH_CLIENT_ID") or "").strip()
    oauth_client_secret = (ollama_oauth_client_secret or os.getenv("NVQSOC_OLLAMA_OAUTH_CLIENT_SECRET") or "").strip()
    oauth_scope = (ollama_oauth_scope or os.getenv("NVQSOC_OLLAMA_OAUTH_SCOPE") or "").strip()
    oauth_audience = (ollama_oauth_audience or os.getenv("NVQSOC_OLLAMA_OAUTH_AUDIENCE") or "").strip()
    oauth_refresh_token = (ollama_oauth_refresh_token or os.getenv("NVQSOC_OLLAMA_OAUTH_REFRESH_TOKEN") or "").strip()
    if oauth_token_url and oauth_client_id and oauth_client_secret:
        token_value = _fetch_oauth_token(
            oauth_token_url,
            client_id=oauth_client_id,
            client_secret=oauth_client_secret,
            scope=oauth_scope,
            audience=oauth_audience,
            refresh_token=oauth_refresh_token,
            timeout_s=timeout_s,
        )
        if token_value:
            mode = "oauth_refresh_token" if oauth_refresh_token else "oauth_client_credentials"
            return OllamaAuthContext(auth_header=f"Bearer {token_value}", mode=mode, oauth_token_acquired=True, detail="oauth_access_token")
        return OllamaAuthContext(auth_header=None, mode="oauth_error", detail="oauth_token_unavailable")

    return OllamaAuthContext(auth_header=None, mode="none", detail="no_auth")


def _json_request(
    url: str,
    *,
    timeout_s: int,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    auth_header: str | None = None,
) -> dict[str, Any] | None:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    if auth_header:
        headers["Authorization"] = auth_header
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8", "ignore"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def query_ollama_models(endpoint: str, auth_header: str | None = None, timeout_s: int = 10) -> list[str] | None:
    payload = _json_request(_tags_endpoint(endpoint), timeout_s=timeout_s, auth_header=auth_header)
    if payload is None:
        return None
    models = payload.get("models", [])
    names = [str(item.get("name", "")).strip() for item in models if isinstance(item, dict)]
    return [name for name in names if name]


def warm_ollama_model(
    endpoint: str,
    model: str,
    *,
    auth_header: str | None = None,
    timeout_s: int = 60,
    keep_alive: str | None = None,
) -> bool:
    payload = {
        "model": model,
        "prompt": "warm",
        "stream": False,
        "keep_alive": keep_alive or os.getenv("NVQSOC_OLLAMA_KEEP_ALIVE", "24h"),
        "options": {"temperature": 0.0, "num_predict": 1},
    }
    result = _json_request(_generate_endpoint(endpoint), timeout_s=timeout_s, method="POST", payload=payload, auth_header=auth_header)
    return result is not None


def _model_matches(model: str, models: list[str]) -> bool:
    wanted = model.strip().lower()
    if not wanted:
        return True
    for existing in models:
        lowered = existing.lower()
        if lowered == wanted or lowered.startswith(f"{wanted}:") or wanted.startswith(f"{lowered}:"):
            return True
    return False


def _start_local_ollama_service() -> bool:
    global _OLLAMA_SERVE_PROCESS
    if _OLLAMA_SERVE_PROCESS is not None and _OLLAMA_SERVE_PROCESS.poll() is None:
        return True
    kwargs: dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        _OLLAMA_SERVE_PROCESS = subprocess.Popen(["ollama", "serve"], **kwargs)
    except OSError:
        return False
    return True


def ensure_ollama_ready(
    model: str,
    endpoint: str,
    *,
    timeout_s: int = 45,
    auth_header: str | None = None,
    auth_mode: str = "none",
    auto_start: bool | None = None,
    auto_pull: bool | None = None,
    preload_model: bool | None = None,
) -> OllamaRuntimeStatus:
    status = OllamaRuntimeStatus(endpoint=endpoint, auth_mode=auth_mode)
    local_endpoint = is_local_ollama_endpoint(endpoint)
    allow_start = _env_flag("NVQSOC_OLLAMA_AUTOSTART", True) if auto_start is None else auto_start
    allow_pull = _env_flag("NVQSOC_OLLAMA_AUTOPULL", True) if auto_pull is None else auto_pull
    allow_preload = _env_flag("NVQSOC_OLLAMA_AUTOLOAD", True) if preload_model is None else preload_model

    models = query_ollama_models(endpoint, auth_header=auth_header, timeout_s=min(timeout_s, 10))
    if models is None and local_endpoint and allow_start:
        if _start_local_ollama_service():
            status.started_local_service = True
            deadline = time.monotonic() + max(3.0, min(float(timeout_s), 20.0))
            while time.monotonic() < deadline:
                models = query_ollama_models(endpoint, auth_header=auth_header, timeout_s=4)
                if models is not None:
                    break
                time.sleep(0.5)

    if models is None:
        status.detail = "endpoint_unavailable"
        return status

    status.available = True
    status.models = models
    status.model_available = _model_matches(model, models)
    if status.model_available:
        if allow_preload:
            status.model_loaded = warm_ollama_model(endpoint, model, auth_header=auth_header, timeout_s=max(timeout_s, 20))
        status.detail = "ready"
        return status

    if not local_endpoint or not allow_pull:
        status.detail = "model_missing_local_preload_required" if local_endpoint and not allow_pull else "model_missing"
        return status

    pull_payload = _json_request(
        _pull_endpoint(endpoint),
        timeout_s=max(timeout_s, 30),
        method="POST",
        payload={"name": model, "stream": False},
        auth_header=auth_header,
    )
    if pull_payload is not None:
        refreshed = query_ollama_models(endpoint, auth_header=auth_header, timeout_s=min(timeout_s, 10)) or []
        status.models = refreshed
        status.model_available = _model_matches(model, refreshed)
        status.model_pulled = status.model_available
        if status.model_available and allow_preload:
            status.model_loaded = warm_ollama_model(endpoint, model, auth_header=auth_header, timeout_s=max(timeout_s, 20))

    status.detail = "ready" if status.model_available else "model_missing"
    return status
