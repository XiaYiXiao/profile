#!/usr/bin/env python3
"""Plot input-token throughput from completed sweep benchmark JSON files.

Usage:
  python profiling_mimo/plot-prefill-throughput.py
  python profiling_mimo/plot-prefill-throughput.py /path/to/sweep-results

Missing/invalid runs remain NaN, so curves stop at gaps. Re-run to refresh.
Without RESULT_DIR, select the newest experiment config, even before the first
benchmark finishes. An experiment with no completed runs gets an empty chart.
"""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shlex

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def read_config(path):
    config = {}
    if path.exists():
        for line in path.read_text().splitlines():
            parts = shlex.split(line)
            if not parts:
                continue
            if all("=" in part for part in parts):
                config.update(part.split("=", 1) for part in parts)
            elif "=" in line:
                key, value = line.split("=", 1)
                config[key] = value.strip()
    return config


def latest_results():
    parent = Path(__file__).resolve().parent / "results"
    candidates = [p for p in parent.iterdir() if p.is_dir() and (p / "config.txt").is_file()]
    if not candidates:
        raise ValueError("未找到含 config.txt 的实验目录，请显式指定结果目录")
    return max(candidates, key=lambda p: (p / "config.txt").stat().st_mtime_ns)


def load_point(path, expected, input_len, output_len):
    if not path.exists():
        return {"status": "pending", "note": "benchmark JSON not yet available"}
    try:
        data = json.loads(path.read_text())
        duration = data["duration"]
        valid = (
            data.get("num_prompts") == expected
            and data.get("completed") == expected
            and data.get("failed") == 0
            and isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0
            and data.get("total_input_tokens") == expected * input_len
            and data.get("total_output_tokens") == expected * output_len
            and len(data.get("input_lens", [])) == expected
            and all(n == input_len for n in data["input_lens"])
        )
        if not valid:
            return {"status": "invalid", "note": "request count, token counts, input lengths or duration do not match"}
        return {"status": "complete", "note": "", "duration_s": duration,
                "total_input_tokens": data["total_input_tokens"],
                "input_tokens_per_s": data["total_input_tokens"] / duration}
    except (OSError, ValueError, KeyError, TypeError):
        return {"status": "invalid_or_writing", "note": "JSON is unreadable, incomplete or still being written"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", nargs="?", type=Path, help="默认选择配置文件最新的实验目录，尚无 benchmark JSON 时也会生成空图")
    parser.add_argument("--output-dir", type=Path, help="默认写入结果目录的 plots/；再次执行会更新绘图文件")
    parser.add_argument("--batch-tokens", nargs="+", type=int, help="覆盖 config.txt 中的 batch 搜索列表")
    parser.add_argument("--concurrencies", nargs="+", type=int, help="覆盖 config.txt 中的并发列表")
    args = parser.parse_args()
    try:
        root = (args.result_dir or latest_results()).resolve()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not root.is_dir():
        parser.error(f"结果目录不存在：{root}")
    config = read_config(root / "config.txt")
    batches = sorted(set(args.batch_tokens or map(int, config.get("batch_tokens", "4096 8192 16384 32768 65536 131072").split())))
    concurrencies = sorted(set(args.concurrencies or map(int, config.get("concurrencies", "10 30 50").split())))
    if not batches or not concurrencies or min(batches + concurrencies) <= 0:
        parser.error("batch 和并发列表必须为正整数")
    expected = int(config.get("num_prompts", 50))
    input_len = int(config.get("input_len", 50000))
    output_len = int(config.get("max_output", 1))
    output = args.output_dir or root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for concurrency in concurrencies:
        for batch in batches:
            filename = f"batch{batch}-concurrency{concurrency}.json"
            point = load_point(root / filename, expected, input_len, output_len)
            rows.append({"batch_tokens": batch, "concurrency": concurrency, **point, "result_file": filename})
    csv_path = output / "prefill_throughput.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["batch_tokens", "concurrency", "status", "input_tokens_per_s", "total_input_tokens", "duration_s", "result_file", "note"])
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 6.2))
    colors = ["#2563eb", "#d97706", "#059669", "#7c3aed"]
    markers = ["o", "s", "^", "D"]
    for index, concurrency in enumerate(concurrencies):
        points = [row for row in rows if row["concurrency"] == concurrency]
        values = [row.get("input_tokens_per_s", float("nan")) for row in points]
        complete = sum(row["status"] == "complete" for row in points)
        ax.plot(batches, values, marker=markers[index % len(markers)], color=colors[index % len(colors)], linewidth=2,
                markersize=6, markerfacecolor="white", markeredgewidth=1.7,
                label=f"Concurrency {concurrency} ({complete}/{len(batches)} runs)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches, [f"{batch:,}" for batch in batches])
    dense_ticks = any(right / left < 1.5 for left, right in zip(batches, batches[1:]))
    if dense_ticks:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.set_xlim(min(batches) / 1.15, max(batches) * 1.15)
    ax.set_xlabel("Batch token budget per DP engine (max_num_batched_tokens, log2 scale)", labelpad=12)
    ax.set_ylabel("Prefill / input throughput (tokens/s)", labelpad=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    maximum = max((row.get("input_tokens_per_s", 0) for row in rows), default=0)
    ax.set_ylim(0, maximum * 1.18 if maximum else 1)
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e2e8f0")
    model = Path(config.get("model_path", "MiMo-V2.6-Flash-MOPD")).name
    completed = sum(row["status"] == "complete" for row in rows)
    if completed == 0:
        ax.set_yticks([])
        ax.text(0.5, 0.58, "No completed benchmark results yet\nRe-run this script after results are saved",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color="#64748b")
    fig.suptitle(f"{model} | Prefill throughput | {completed}/{len(rows)} complete", x=0.10, y=0.96, ha="left", fontsize=17, fontweight="bold")
    settings = f"TP{config.get('TP', '?')} / DP{config.get('DP', '?')} / EP{config.get('EP', '?')} | {expected} requests | Input {input_len:,} / output {output_len} tokens | Trace {'ON' if config.get('profiling') == '1' else 'OFF' if config.get('profiling') == '0' else 'unknown'}"
    fig.text(0.10, 0.895, settings, fontsize=10, color="#475569")
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    fig.text(0.10, 0.045, f"Input tokens / benchmark duration; warmup excluded. Missing runs left blank.\nSnapshot: {timestamp}", fontsize=9, color="#64748b")
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.25 if dense_ticks else 0.19, top=0.85)
    for extension in ("png", "svg"):
        fig.savefig(output / f"prefill_throughput.{extension}", dpi=180)
    plt.close(fig)
    print(f"结果目录：{root}")
    print(f"完成 {completed}/{len(rows)} 组，其余留空。")
    if completed == 0:
        print("尚无通过校验的完成结果，已生成等待数据的空图；结果落盘后再次执行即可刷新。")
    elif completed == 1:
        print("目前只有一个完成点，因此不会形成曲线；后续结果落盘后再次执行即可刷新。")
    for extension in ("png", "svg", "csv"):
        print(output / f"prefill_throughput.{extension}")


if __name__ == "__main__":
    main()
