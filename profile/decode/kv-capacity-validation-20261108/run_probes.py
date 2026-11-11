"""Finite bursts with direct token IDs: no tokenizer bottleneck or replenishment.

The target server and all results belong to this directory. Long/short/long
changes KV demand without restarting or changing any server capacity setting.
"""
import asyncio
import codecs
import json
import os
from pathlib import Path
import random
import time

import aiohttp

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
URL = "http://127.0.0.1:18180"
MODEL = "MiMo-V2.6-Flash-MOPD"
CASES = [
    ("sanity_n512_kv10240", 512, 10240),
    ("long_n4096_kv10240", 4096, 10240),
    ("short_n4096_kv1024", 4096, 1024),
    ("repeat_n4096_kv10240", 4096, 10240),
]
METRIC_NAMES = (
    "vllm:num_requests_running", "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total",
    "vllm:generation_tokens_total", "vllm:prompt_tokens_total",
)


async def wait_ready():
    deadline = time.monotonic() + 1200
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(URL + "/health") as response:
                    if response.status == 200:
                        engine_files = list((DATA / "engine").glob("engine-*.jsonl"))
                        if len(engine_files) != 4:
                            raise RuntimeError(f"Expected 4 instrumented DP engines, found {engine_files}")
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(2)
    raise RuntimeError("Server readiness timeout")


async def sample_metrics(folder, done):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        with (folder / "metrics.jsonl").open("w", buffering=1) as stream:
            while not done.is_set():
                row = dict(time_unix_s=time.time())
                try:
                    async with session.get(URL + "/metrics") as response:
                        response.raise_for_status()
                        body = await response.text()
                    row["lines"] = [line for line in body.splitlines()
                                    if line.startswith(METRIC_NAMES)]
                except Exception as error:
                    row["error"] = repr(error)
                stream.write(json.dumps(row)+"\n")
                try:
                    await asyncio.wait_for(done.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass


async def run_case(name, n, kv_len):
    folder = DATA / name
    folder.mkdir(exist_ok=False)
    rng = random.Random(42)
    prompt = [rng.randrange(1000, 10000) for _ in range(kv_len + 1)]
    payload = dict(model=MODEL, prompt=prompt, max_tokens=64, ignore_eos=True,
                   temperature=0, stream=True, stream_options={"include_usage": True})
    (folder / "request-template.json").write_text(json.dumps(payload))
    body = json.dumps(payload).encode()
    configuration = dict(name=name, num_prompts=n, max_concurrency=n,
                         request_rate="inf", kv_len=kv_len, output_tokens=64,
                         input_mode="direct_token_ids", prefix_cache=False,
                         started_unix_s=time.time())
    (folder / "config.json").write_text(json.dumps(configuration, indent=2))
    done = asyncio.Event()
    sampler = asyncio.create_task(sample_metrics(folder, done))
    rows = []
    begin = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=1200)
    connector = aiohttp.TCPConnector(limit=n, limit_per_host=n)
    with (folder / "requests.jsonl").open("w", buffering=1) as stream:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async def one(index):
                record = dict(index=index, start_unix_s=time.time())
                started = time.monotonic()
                buffer = ""
                decoder = codecs.getincrementaldecoder("utf-8")()
                try:
                    async with session.post(URL + "/v1/completions", data=body,
                                            headers={"Content-Type": "application/json",
                                                     "X-Request-Id": f"{name}-{index}"}) as response:
                        record["http_status"] = response.status
                        if response.status != 200:
                            raise RuntimeError(await response.text())
                        async for part in response.content.iter_any():
                            buffer += decoder.decode(part)
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                if not line.startswith("data:"):
                                    continue
                                value = line[5:].strip()
                                if value == "[DONE]":
                                    continue
                                item = json.loads(value)
                                if "error" in item:
                                    raise RuntimeError(str(item))
                                if item.get("usage"):
                                    record["usage"] = item["usage"]
                                for choice in item.get("choices", []):
                                    if choice.get("text") and "ttft_s" not in record:
                                        record["ttft_s"] = time.monotonic()-started
                                        record["first_token_unix_s"] = time.time()
                                    if choice.get("finish_reason"):
                                        record["finish_reason"] = choice["finish_reason"]
                    usage = record.get("usage", {})
                    if usage.get("prompt_tokens") != kv_len+1 or usage.get("completion_tokens") != 64:
                        raise RuntimeError(f"Unexpected usage {usage}")
                except Exception as error:
                    record["error"] = repr(error)
                record["latency_s"] = time.monotonic()-started
                record["end_unix_s"] = time.time()
                stream.write(json.dumps(record)+"\n")
                rows.append(record)
                if len(rows) % 256 == 0:
                    print(f"{name}: {len(rows)}/{n} complete", flush=True)
            try:
                await asyncio.gather(*(one(i) for i in range(n)))
            finally:
                done.set()
                await sampler
    summary = dict(configuration, finished_unix_s=time.time(), duration_s=time.monotonic()-begin,
                   completed=len(rows), errors=sum("error" in row for row in rows),
                   client_start_spread_s=max(r["start_unix_s"] for r in rows)-min(r["start_unix_s"] for r in rows))
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if summary["errors"]:
        raise RuntimeError(f"Case failed: {name}")
    await asyncio.sleep(3)


async def main():
    print(f"Waiting for {URL}", flush=True)
    await wait_ready()
    for case in CASES:
        await run_case(*case)


if __name__ == "__main__":
    asyncio.run(main())
