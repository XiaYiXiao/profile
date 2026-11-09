"""Local process ownership, metrics and Nsight helpers for aggre_or_not.

Kept independent from profiler/profile so its experiments need no changes.
"""
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

def is_vllm_server(args):
    # Recognize python -m vllm; retain detection of existing console-script servers.
    return any(Path(os.fsdecode(arg)).name == 'vllm'
               and args[index + 1:index + 2] == [b'serve']
               for index, arg in enumerate(args))


def trace_command(output):
    return [os.environ.get('NSYS_BIN', 'nsys'), 'profile', '--sample=none', '--cpuctxsw=none',
            '--trace=' + os.environ['NSYS_TRACE'],
            '--nccl-trace=' + os.environ['NCCL_TRACE'],
            '--cuda-graph-trace=' + os.environ['CUDA_GRAPH_TRACE'],
            '--trace-fork-before-exec=true', '--capture-range=cudaProfilerApi',
            '--capture-range-end=stop', '--output=' + str(output)]


def cleanup(proc):
    if proc is None:
        return
    # nsys starts the application in its own process group.
    groups = {proc.pid}
    for child in descendants(proc.pid):
        try:
            args = Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
            if is_vllm_server(args):
                groups.add(os.getpgid(int(child)))
        except (ProcessLookupError, FileNotFoundError):
            pass
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        proc.poll()
        alive = []
        for group in groups:
            try:
                os.killpg(group, 0)
                alive.append(group)
            except ProcessLookupError:
                pass
        if not alive:
            return
        time.sleep(1)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()


def descendants(pid):
    """nsys-launcher inserts another process between nsys and the API server."""
    try:
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except FileNotFoundError:
        return []
    return children + [p for child in children for p in descendants(child)]


def snapshot(url, path):
    with urllib.request.urlopen(url + '/metrics', timeout=15) as r:
        path.write_bytes(r.read())


def launch(command, log, env=None):
    with log.open('w') as out:
        return subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)


def wait_jobs(jobs, timeout, servers=(), check_returncodes=True):
    """Poll all jobs together, so either side failing stops the experiment."""
    deadline = time.monotonic() + timeout
    pending = set(range(len(jobs)))
    while pending:
        for server in servers:
            if server.poll() is not None:
                raise RuntimeError(f'Profile server exited early: pid={server.pid}')
        for index in list(pending):
            rc = jobs[index].poll()
            if rc is not None:
                pending.remove(index)
                if check_returncodes and rc:
                    raise RuntimeError(f'Job {index} failed with exit {rc}; inspect its log')
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Jobs {sorted(pending)} exceeded {timeout}s')
            time.sleep(1)


def terminate_all(jobs, app_groups=()):
    # Notify selected services before waiting for any to exit. Only owned groups.
    for group in set(app_groups) | {p.pid for p in jobs}:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for proc in jobs:
        cleanup(proc)
    # A target can become reparented after its nsys launcher exits.
    for group in app_groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def server_children(proc):
    found = []
    for child in descendants(proc.pid):
        try:
            args = Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
            if is_vllm_server(args):
                found.append(int(child))
        except FileNotFoundError:
            pass
    return found
