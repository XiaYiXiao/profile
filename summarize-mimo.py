#!/usr/bin/env python3
"""汇总 sweep 结果，也可在中断后独立运行。仅使用 Python 标准库。"""
import csv
import json
import pathlib
import sys


def main():
    root = pathlib.Path(sys.argv[1])
    expected = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    input_len = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
    fields = [
        "batch_tokens", "concurrency", "status", "exit_code", "completed", "failed",
        "duration_s", "request_throughput", "input_tokens_per_s", "output_tokens_per_s",
        "total_tokens_per_s", "total_input_tokens", "total_output_tokens",
        "mean_ttft_ms", "p50_ttft_ms", "p90_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
        "mean_e2el_ms", "p50_e2el_ms", "p90_e2el_ms", "p95_e2el_ms", "p99_e2el_ms",
        "input_length_matches", "result_file", "profiling", "profile_status", "trace_dir",
    ]
    rows = []
    with (root / "runs.tsv").open() as stream:
        runs = list(csv.DictReader(stream, delimiter="\t"))
    for run in runs:
        row = dict(run)
        path = root / run["result_file"]
        if run["status"] == "finished":
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                row["status"] = "missing_or_invalid_result"
            else:
                completed = data.get("completed", 0)
                row.update({key: data[key] for key in fields if key in data})
                row.update(
                    completed=completed,
                    failed=data.get("failed", expected - completed),
                    duration_s=data.get("duration", ""),
                    output_tokens_per_s=data.get("output_throughput", ""),
                    total_tokens_per_s=data.get("total_token_throughput", ""),
                )
                duration = data.get("duration", 0)
                row["input_tokens_per_s"] = data.get("total_input_tokens", 0) / duration if duration else ""
                lengths = data.get("input_lens", [])
                row["input_length_matches"] = len(lengths) == expected and all(n == input_len for n in lengths)
                row["status"] = "ok" if (
                    run["exit_code"] == "0" and completed == expected
                    and row["failed"] == 0 and row["input_length_matches"]
                    and data.get("total_input_tokens") == expected * input_len
                    and data.get("total_output_tokens") == expected
                ) else "incomplete_or_mismatched"
        if row.get("profiling") == "1" and row.get("profile_status") != "captured" and row["status"] == "ok":
            row["status"] = "profile_failed"
        rows.append(row)
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("batch_tokens concurrency status completed req/s input_tok/s mean_TTFT_ms p99_TTFT_ms")
    for row in rows:
        print(" ".join(str(row.get(key, "-")) for key in (
            "batch_tokens", "concurrency", "status", "completed", "request_throughput",
            "input_tokens_per_s", "mean_ttft_ms", "p99_ttft_ms")))
    valid = [row for row in rows if row["status"] == "ok"]
    if any(row.get("profiling") == "1" for row in rows):
        print("注意：本次 profiling 的性能数据包含采集开销，请与 PROFILE=0 的基线分开比较。")
    for concurrency in sorted({row["concurrency"] for row in valid}, key=int):
        best = max((row for row in valid if row["concurrency"] == concurrency), key=lambda row: row["input_tokens_per_s"])
        print(f"并发 {concurrency} 的最高输入吞吐：batch_tokens={best['batch_tokens']}，{best['input_tokens_per_s']:.2f} token/s")
    print(f"汇总：{root / 'summary.csv'}")
    return 0 if rows and all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
