"""One HTTPS/WSS front door for every local CloudXR teleoperation slot."""

from __future__ import annotations

import argparse
import asyncio
import signal
from datetime import UTC, datetime
from pathlib import Path

_ROUTE_PREFIX = "/_clothdojo/"


def main() -> None:
    parser = argparse.ArgumentParser(description="run the shared ClothDojo CloudXR gateway")
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--signaling-port", type=int, action="append", required=True)
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            run_gateway(
                install_dir=args.install_dir,
                proxy_port=args.proxy_port,
                signaling_ports=tuple(args.signaling_port),
            )
        )
    )


async def run_gateway(
    *,
    install_dir: Path,
    proxy_port: int,
    signaling_ports: tuple[int, ...],
) -> int:
    from isaacteleop.cloudxr import wss as cloudxr_wss
    from isaacteleop.cloudxr.env_config import EnvConfig

    install_dir = install_dir.expanduser().resolve()
    EnvConfig.from_args(str(install_dir))
    _install_router(cloudxr_wss, signaling_ports)
    loop = asyncio.get_running_loop()
    stop_future = loop.create_future()

    def stop() -> None:
        if not stop_future.done():
            stop_future.set_result(None)

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_number, stop)
    logs = install_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    await cloudxr_wss.run(
        log_file_path=logs / f"gateway.{stamp}.log",
        stop_future=stop_future,
        backend_host="127.0.0.1",
        backend_port=min(signaling_ports),
        proxy_port=proxy_port,
        host_client=True,
    )
    return 0


def _install_router(cloudxr_wss, signaling_ports: tuple[int, ...]) -> None:
    direct_proxy = cloudxr_wss.proxy_handler
    allowed = frozenset(int(port) for port in signaling_ports)

    async def route(client, backend_host: str, backend_port: int):
        routed = gateway_backend(client.request.path or "/", allowed)
        if routed is None:
            return await direct_proxy(client, backend_host, backend_port)
        target_port, backend_path = routed
        original_path = client.request.path
        client.request.path = backend_path
        try:
            return await direct_proxy(client, "127.0.0.1", target_port)
        finally:
            client.request.path = original_path

    cloudxr_wss.proxy_handler = route


def gateway_backend(raw_path: str, allowed_ports: frozenset[int]) -> tuple[int, str] | None:
    """Parse ``/_clothdojo/<port>/<host>/<path>`` without becoming an open proxy."""
    path, separator, query = raw_path.partition("?")
    if not path.startswith(_ROUTE_PREFIX):
        return None
    port_text, slash, remainder = path[len(_ROUTE_PREFIX) :].partition("/")
    if not slash or not port_text.isdecimal():
        return None
    port = int(port_text)
    if port not in allowed_ports:
        return None
    _target_host, slash, backend_tail = remainder.partition("/")
    if not slash or not backend_tail:
        return None
    backend_path = f"/{backend_tail}"
    if separator:
        backend_path += f"?{query}"
    return port, backend_path


if __name__ == "__main__":
    main()
