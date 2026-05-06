"""Reusable ASGI middleware for Energy-Oracle."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Per-IP sliding-window request store
_rate_store: dict[str, list[float]] = {}


class CorrelationIDMiddleware:
    """Attach a correlation ID to every request and response."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            from starlette.types import Message

            headers = dict(scope["headers"])
            corr_id = headers.get(b"x-correlation-id", b"").decode() or str(uuid.uuid4())
            scope["state"] = getattr(scope.get("app"), "state", None) or type("State", (), {})()
            scope.setdefault("_correlation_id", corr_id)

            async def send_with_header(message: Message) -> None:
                if message["type"] == "http.response.start":
                    headers_list = list(message.get("headers", []))
                    headers_list.append((b"x-correlation-id", corr_id.encode()))
                    message = {**message, "headers": headers_list}
                await send(message)

            await self.app(scope, receive, send_with_header)
        else:
            await self.app(scope, receive, send)


async def rate_limit_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
    *,
    limit: int = 60,
) -> Response:
    """
    Sliding-window rate limiter (per client IP).

    Rejects requests exceeding `limit` per 60-second window with HTTP 429.
    """
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _rate_store.setdefault(client_ip, [])
    _rate_store[client_ip] = [t for t in window if now - t < 60]
    if len(_rate_store[client_ip]) >= limit:
        logger.warning("Rate limit exceeded for IP %s", client_ip)
        return JSONResponse(
            {"detail": f"Rate limit exceeded. Max {limit} requests/minute."},
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )
    _rate_store[client_ip].append(now)
    return await call_next(request)
