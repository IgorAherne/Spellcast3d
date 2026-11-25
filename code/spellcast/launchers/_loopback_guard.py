"""Loopback-only guard for the tool servers Spellcast3D starts. STDLIB ONLY: it runs inside the
tool venvs, which never have the hub package.

The tools listen on 127.0.0.1 only, but a web page the artist has open can still reach them with
DNS rebinding: the page's own host name is switched to resolve to 127.0.0.1, and the browser then
treats the tool as part of that site (no CORS check applies). Such requests carry the page's host
name in their Host header. So every HTTP request and WebSocket whose Host is not a loopback name
is refused, and so is one whose Origin names another site - the same rule as the hub's own
local_only middleware (server.py). The hub, the tools' own pages and the hub's iframes all use
127.0.0.1 or localhost.

LICENSE OF THIS FILE: MIT (Copyright (c) Spellcast3D's authors; the MIT terms are in "README - LICENSE.txt"),
not AGPL like most of Spellcast3D: it is loaded into the TriFlow process (launchers/triflow_server.py), next to
TriFlow's ADPNCL code.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
REFUSED = ("This tool only accepts requests from this computer. Open Spellcast3D at "
           "http://127.0.0.1 instead.")

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


def host_name(value: str) -> str:
    """'127.0.0.1:7773' -> '127.0.0.1', '[::1]:7773' -> '::1', 'http://localhost:7773' -> 'localhost'."""
    value = value.split("://", 1)[-1].split("/", 1)[0].strip().lower()
    if value.startswith("["):
        return value[1:value.find("]")] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def is_loopback(host: str) -> bool:
    """For the launchers' --host: the guard is only used when the server listens on this computer only."""
    return host.strip("[]").lower() in LOOPBACK_NAMES


def allowed(headers: list[tuple[bytes, bytes]]) -> bool:
    """True when the request is addressed to a loopback name and sent by a page of this computer."""
    found = {name.decode("latin-1").lower(): value.decode("latin-1") for name, value in headers}
    if host_name(found.get("host", "")) not in LOOPBACK_NAMES:   # a missing Host is refused too
        return False
    origin = found.get("origin")
    return origin is None or host_name(origin) in LOOPBACK_NAMES   # "null" (sandboxed pages) is refused


class LoopbackGuard:
    """ASGI middleware: 403 for HTTP, a refused handshake for WebSockets (see the module doc)."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") not in ("http", "websocket") or allowed(scope.get("headers") or []):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            message = await receive()
            if message.get("type") == "websocket.connect":
                await send({"type": "websocket.close", "code": 4403})   # before accept: the handshake fails
            return
        body = json.dumps({"detail": REFUSED}).encode("utf-8")
        await send({"type": "http.response.start", "status": 403,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode("ascii"))]})
        await send({"type": "http.response.body", "body": body})
