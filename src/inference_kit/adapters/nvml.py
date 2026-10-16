"""NVML CLI observations and ownership; no query at import."""
import asyncio
import json
from pathlib import Path
import subprocess
import time
from ..core.io import read, require
def proc_identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'ppid': int(fields[1]), 'group': int(fields[2]), 'start_ticks': int(fields[19]), 'state': fields[0]}
    except (OSError, ValueError, IndexError):
        return None


def owned(pid, group, tracked):
    current = proc_identity(pid)
    if not current:
        return False
    if current['group'] == group:
        tracked[pid] = current['start_ticks']
        return True
    if tracked.get(pid) == current['start_ticks']:
        return True
    seen = set()
    while current and current['pid'] not in seen:
        if current['pid'] == group or current['group'] == group:
            identity = proc_identity(pid)
            if identity:
                tracked[pid] = identity['start_ticks']
            return True
        seen.add(current['pid'])
        current = proc_identity(current['ppid'])
    return False


def gpu_query(gpu=0, applications=False):
    option = '--query-compute-apps=pid,process_name,used_gpu_memory' if applications else '--query-gpu=name,uuid,driver_version,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem'
    argv = ['nvidia-smi', '-i', str(gpu), option, '--format=csv,noheader,nounits']
    try:
        r = subprocess.run(argv, text=True, capture_output=True, timeout=3)
        return {'returncode': r.returncode, 'values': r.stdout.strip(), 'error': r.stderr.strip(), 'monotonic': time.perf_counter()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'returncode': None, 'error': type(exc).__name__ + ': ' + str(exc), 'monotonic': time.perf_counter()}


async def gpu_monitor(stop, abort, directory, spec, owner):
    tracked = {}
    with (directory / 'memory.jsonl').open('x',encoding='utf-8') as memory, (directory / 'ownership.jsonl').open('x',encoding='utf-8') as ownership:
        last_owner = -float('inf')
        while not stop.is_set():
            sample = await asyncio.to_thread(gpu_query, spec['runtime']['gpu'])
            memory.write(json.dumps(sample) + '\n'); memory.flush()
            if sample['monotonic'] - last_owner >= spec['instrumentation']['ownership_poll_seconds']:
                apps = await asyncio.to_thread(gpu_query, spec['runtime']['gpu'], True)
                group = owner.get('parent_pid')
                pids = [int(line.split(',')[0]) for line in apps.get('values', '').splitlines() if line.strip()]
                unknown = [p for p in pids if not group or not owned(p, group, tracked)]
                apps.update(pids=pids, owned_pid_start_ticks=tracked, unverified_pids=unknown,
                            observation_level='NVML + local /proc ownership; WSL/brief/graphics activity not fully excluded')
                ownership.write(json.dumps(apps) + '\n'); ownership.flush()
                if unknown or apps['returncode'] != 0:
                    abort.set()
                last_owner = apps['monotonic']
            try:
                await asyncio.wait_for(stop.wait(), spec['instrumentation']['memory_poll_seconds'])
            except asyncio.TimeoutError:
                pass

