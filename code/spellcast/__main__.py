"""`python -m spellcast`: run the Spellcast3D hub (web UI + API) until Ctrl+C or the window closes."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import logging
import logging.handlers
import os
import signal
import sys
import webbrowser
from typing import IO

import httpx
import uvicorn

from . import __version__, paths
from .installer_bg import BackgroundInstall, console_echo, console_touched, installer_argv
from .procman import ToolError, find_free_port, hub_has_console, port_is_free
from .server import LOOPBACK_NAMES, create_app
from .settings import SettingsStore

log = logging.getLogger("spellcast")


class _ConsoleFormatter(logging.Formatter):
    """Short lines without tracebacks: those go to hub.log only."""

    def format(self, record: logging.LogRecord) -> str:
        record = logging.makeLogRecord(record.__dict__)
        record.exc_info = record.exc_text = record.stack_info = None
        return super().format(record)


def _setup_logging() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    paths.LOGS.mkdir(parents=True, exist_ok=True)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_ConsoleFormatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    # The console shows Spellcast3D's own messages plus anyone's warnings; hub.log gets everything. Each line shown
    # is noted (installer_bg.console_touched): the "ready" banner waits for a quiet console.
    console.addFilter(lambda r: r.name.startswith("spellcast") or r.levelno >= logging.WARNING)
    console.addFilter(lambda r: (console_touched(), True)[1])
    file = logging.handlers.RotatingFileHandler(paths.log_file("hub"), maxBytes=5_000_000, backupCount=2,
                                                encoding="utf-8")
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file)
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _accept_ctrl_c() -> None:
    """Make Ctrl+C in the console window stop the hub even when it was started from a program that
    ignores Ctrl+C: Windows passes that "ignore" flag on to every child process."""
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
        except (AttributeError, OSError):
            pass


class _AlreadyRunning(Exception):
    pass


def _hold_hub_lock() -> IO[bytes] | None:
    """Lock code/venvs/.hub.lock for as long as this hub runs (Windows releases it when the process
    dies). tools/update.bat and the installer check it: they must not replace files the hub and its
    tools are using. Returns the open file (keep it), or None when it cannot be created (the hub then
    runs without it). Raises _AlreadyRunning when another hub holds it."""
    import msvcrt

    try:
        paths.VENVS.mkdir(parents=True, exist_ok=True)
        f = open(paths.VENVS / ".hub.lock", "a+b")
    except OSError:
        return None
    f.seek(0)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        raise _AlreadyRunning() from None
    return f


def _is_spellcast_hub(port: int) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/status", timeout=3.0, trust_env=False)
        return r.status_code == 200 and "tools" in r.json()
    except (httpx.HTTPError, ValueError):
        return False


def _banner(url: str, open_browser: bool) -> str:
    line = "=" * 68
    how = ("Your web browser should open it by itself; if not, open that address."
           if open_browser else "Open that address in your web browser.")
    return (f"\n{line}\n"
            f"  Spellcast3D {__version__} is running at:  {url}\n"
            f"  {how}\n\n"
            f"  Keep this window open while you work.\n"
            f"  To quit: close this window or press Ctrl+C.\n"
            f"{line}\n")


async def _announce_when_listening(server: uvicorn.Server, url: str, open_browser: bool) -> None:
    while not server.started:
        if server.should_exit:
            return
        await asyncio.sleep(0.1)
    print(_banner(url, open_browser), flush=True)
    if open_browser:
        await asyncio.to_thread(webbrowser.open, url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m spellcast", description="Start the Spellcast3D hub.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on (default 127.0.0.1 = this computer only)")
    parser.add_argument("--port", type=int, default=None,
                        help="port to listen on (default: ports.hub from the settings, 7770); "
                             "if it is busy the next free port is used")
    parser.add_argument("--open-browser", action="store_true",
                        help="open the web UI in the default browser once the server is listening")
    parser.add_argument("--no-install", action="store_true",
                        help="do not install missing or changed parts in the background (for developers)")
    args = parser.parse_args(argv)

    _accept_ctrl_c()
    _setup_logging()
    settings = SettingsStore()
    merged = settings.get()
    # SPELLCAST_NO_BROWSER=1: never open a browser tab (tests and automation restart the hub often).
    open_browser = (args.open_browser and bool(merged.get("open_browser", True))
                    and os.environ.get("SPELLCAST_NO_BROWSER") != "1")
    wanted = args.port or int(merged["ports"]["hub"])
    port = wanted
    if not port_is_free(wanted):
        if _is_spellcast_hub(wanted):
            url = f"http://127.0.0.1:{wanted}"
            print(f"\nSpellcast3D is already running at {url} - use that window/tab.\n"
                  f"(This second window can be closed.)\n", flush=True)
            if open_browser:
                webbrowser.open(url)
            return 0
        try:
            port = find_free_port(wanted + 1, {int(merged["ports"][t]) for t in paths.TOOL_IDS})
        except ToolError as e:
            log.error("%s", e)
            return 1
        log.warning("Port %s is used by another program, so Spellcast3D uses port %s instead.", wanted, port)
    try:
        lock = _hold_hub_lock()   # held until this process ends
    except _AlreadyRunning:
        print("\nSpellcast3D is already running in another window - use that window/tab.\n"
              "(This second window can be closed.)\n", flush=True)
        return 0

    allow_remote = args.host.strip("[]").lower() not in LOOPBACK_NAMES
    local_host = "127.0.0.1" if not allow_remote or args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{local_host}:{port}"
    log.info("Spellcast3D %s starting (hub log: %s)", __version__, paths.log_file("hub"))

    # The .bat installed only the hub: the tools are installed (or updated) now, while the page is open.
    installer = None if args.no_install else BackgroundInstall(
        installer_argv(), echo=console_echo if hub_has_console() else None, page_url=url)
    app = create_app(settings=settings, hub_port=port, allow_remote=allow_remote, installer=installer)
    config = uvicorn.Config(app, host=args.host, port=port, log_config=None, access_log=False,
                            timeout_graceful_shutdown=5)
    server = uvicorn.Server(config)

    async def run() -> None:
        announce = asyncio.create_task(_announce_when_listening(server, url, open_browser))
        try:
            await server.serve()
        finally:
            announce.cancel()

    # uvicorn shuts down gracefully on Ctrl+C / Ctrl+Break / window close, then re-raises the
    # signal; make that a KeyboardInterrupt instead of an abrupt exit code 3.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        if lock is not None:
            lock.close()   # the tools have stopped: update.bat may run now
    print("\nSpellcast3D has stopped. You can close this window.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
