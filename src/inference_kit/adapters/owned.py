"""Owned POSIX subprocess lifecycle extracted from serving_lab.server."""
import asyncio
import os
from pathlib import Path
import signal
import subprocess
import time
from ..core.io import require
from .nvml import proc_identity
import weakref
_OWNED = weakref.WeakKeyDictionary()
async def ensure_port_free(url):
    # Do not accidentally time a pre-existing server, even when it uses the same alias.
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    try:
        reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
    except OSError:
        return
    writer.close()
    await writer.wait_closed()
    raise ValueError("Port already occupied; choose another port, do not kill the existing server")


def start(launch_plan, directory):
    if os.name != "posix":
        raise ValueError("Managed launch requires Linux/WSL; CPU client tests remain portable")
    log = (Path(directory) / "server.log").open("x", encoding="utf-8")
    try:
        process = subprocess.Popen(launch_plan["argv"], env=os.environ | launch_plan["environment"],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        _OWNED[process] = {"pid":process.pid,"identity":proc_identity(process.pid)}
    except BaseException:
        log.close()
        raise
    return process, log


async def ready(process, url, timeout=180, observations=None):
    import aiohttp  # Optional only when readiness HTTP is requested.
    deadline = time.monotonic() + timeout
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Server exited before readiness; see server.log")
            try:
                health = {"method": "GET", "path": "/health", "started_monotonic": time.monotonic(), "status": None}
                if observations is not None: observations.append(health)
                async with session.get(url + "/health") as response:
                    health["status"] = response.status
                    if response.status == 200:
                        discovery = {"method": "GET", "path": "/v1/models", "started_monotonic": time.monotonic(), "status": None}
                        if observations is not None: observations.append(discovery)
                        async with session.get(url + "/v1/models") as models:
                            discovery["status"] = models.status
                            if models.status == 200 and "serving-lab" in [m["id"] for m in (await models.json())["data"]]:
                                return
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError):
                pass
            await asyncio.sleep(.5)
    raise TimeoutError("Server readiness budget exceeded; no automatic restart")


def group_members(group):
    """Observe live members of a process group created by start_new_session.

    Zombies have already exited; their native exit codes remain unavailable here.
    """
    members = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                members.append(int(directory.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(members)


def stop(process, log, grace=5):
    require(process in _OWNED, 'Refuse cleanup of unowned process')
    require(process.pid != os.getpid() and process.pid != os.getpgrp(), 'Refuse self process/group cleanup')
    recorded=_OWNED[process]['identity'];current=proc_identity(process.pid)
    require(current is None or (recorded is not None and current['start_ticks']==recorded['start_ticks'] and current['group']==process.pid), 'PID reuse/foreign group; refuse signal')
    signals = []
    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
                signals.append(sig.name)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + (min(grace, 2) if sig == signal.SIGKILL else grace)
            while time.monotonic() < deadline:
                process.poll()
                if not group_members(process.pid):
                    break
                time.sleep(min(.025, max(0, deadline - time.monotonic())))
            if not group_members(process.pid):
                break
        process.wait(timeout=max(grace, .1))
        remaining = group_members(process.pid)
        return {"returncode": process.returncode, "signals_sent": signals,
                "remaining_owned_pids": remaining,
                "owned_process_cleanup": "CLEANED" if not remaining else "RESIDUAL",
                "worker_native_exit": {"status": "UNKNOWN", "returncode": None},
                "cleanup_grace_per_signal_seconds": grace}
    finally:
        log.close()

