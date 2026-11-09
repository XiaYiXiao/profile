#!/usr/bin/env python3
"""Summarize completed and pending decode points; safe to run during a sweep."""
import csv
import json
import os
from pathlib import Path
import sys

from decode_state import make_row, name, read_config, rows, valid_result


def sampled_peaks(path):
    count, peak_running, peak_waiting, peak_kv, preemptions = 0, 0, 0, 0, []
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("phase") != "benchmark" or "metrics" not in record:
                    continue
                metrics = record["metrics"]
                count += 1
                for metric_name, kind in (("vllm:num_requests_running", "running"), ("vllm:num_requests_waiting", "waiting")):
                    value = sum(m["value"] for m in metrics if m["name"] == metric_name)
                    if kind == "running":
                        peak_running = max(peak_running, value)
                    else:
                        peak_waiting = max(peak_waiting, value)
                peak_kv = max([peak_kv] + [m["value"] for m in metrics if m["name"] == "vllm:kv_cache_usage_perc"])
                counters = [m["value"] for m in metrics if m["name"] == "vllm:num_preemptions_total"]
                if counters:
                    preemptions.append(sum(counters))
    return dict(metrics_samples=count, peak_running_requests=peak_running if count else "",
                peak_waiting_requests=peak_waiting if count else "", peak_kv_cache_usage_pct=peak_kv * 100 if count else "",
                sampled_preemptions_delta=max(preemptions) - min(preemptions) if preemptions else "")


def main():
    root = Path(sys.argv[1])
    config = read_config(root)
    fields = ["max_num_seqs", "batch_tokens", "concurrency", "num_prompts", "status", "exit_code",
              "kv_len", "input_len", "max_output", "completed", "failed", "duration_s", "request_throughput",
              "output_tokens_per_s", "total_output_tokens"]
    for metric in ("tpot", "itl", "ttft", "e2el"):
        fields += [f"{prefix}_{metric}_ms" for prefix in ("mean", "p50", "p90", "p95", "p99")]
    fields += ["metrics_samples", "peak_running_requests", "peak_waiting_requests", "peak_kv_cache_usage_pct",
               "sampled_preemptions_delta", "result_file", "validation_error"]
    values = rows(root)
    summary = []
    for seq in sorted(map(int, config["seq_counts"].split())):
        row = dict(values.get(seq, make_row(seq, config)))
        row.update({key: config[key] for key in ("kv_len", "input_len", "max_output")})
        if row["status"] == "finished":
            try:
                data = valid_result(root, seq, config)
                row.update({key: data[key] for key in fields if key in data})
                row.update(duration_s=data["duration"], output_tokens_per_s=data["total_output_tokens"] / data["duration"])
                row["status"] = "ok" if row["exit_code"] == "0" else "bench_failed"
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(status="incomplete_or_mismatched", validation_error=str(error))
        row.update(sampled_peaks(root / (name(seq, config) + ".metrics-series.jsonl")))
        summary.append(row)
    temporary = root / f"summary.csv.tmp-{os.getpid()}"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    os.replace(temporary, root / "summary.csv")
    print("concurrency batch_tokens status completed output_tok/s mean_TPOT_ms p99_TPOT_ms mean_ITL_ms p99_ITL_ms peak_running")
    for row in summary:
        print(" ".join(str(row.get(key, "-")) for key in (
            "concurrency", "batch_tokens", "status", "completed", "output_tokens_per_s",
            "mean_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "p99_itl_ms", "peak_running_requests")))
    print(f"汇总：{root / 'summary.csv'}")
    return 0 if summary and all(row["status"] == "ok" for row in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
