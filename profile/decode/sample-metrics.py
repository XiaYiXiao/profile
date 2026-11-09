#!/usr/bin/env python3
"""Sample lightweight scheduler/cache counters; no GPU instrumentation."""
import json
import math
from pathlib import Path
import re
import signal
import sys
import time
import urllib.request

NAMES = {
    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total", "vllm:generation_tokens_total", "vllm:prompt_tokens_total",
}
LINE = re.compile(r"([^\s{]+)(\{.*\})?\s+(\S+)")


def main():
    url, filename, interval, bench_log = sys.argv[1:]
    interval = float(interval)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    phase = "warmup_or_setup"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with Path(filename).open("a", buffering=1) as stream:
        while running:
            tick = time.monotonic()
            if phase != "benchmark":
                try:
                    if "Starting main benchmark run..." in Path(bench_log).read_text(errors="replace"):
                        phase = "benchmark"
                except FileNotFoundError:
                    pass
            record = dict(time_unix_s=time.time(), elapsed_s=tick - started, phase=phase)
            try:
                with opener.open(url, timeout=5) as response:
                    body = response.read().decode()
                metrics = []
                for line in body.splitlines():
                    match = LINE.fullmatch(line)
                    if match and match[1] in NAMES:
                        value = float(match[3])
                        if math.isfinite(value):
                            metrics.append(dict(name=match[1], labels=match[2] or "", value=value))
                record["metrics"] = metrics
            except (OSError, ValueError) as error:
                record["error"] = str(error)
            stream.write(json.dumps(record) + "\n")
            deadline = tick + interval
            while running and time.monotonic() < deadline:
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
