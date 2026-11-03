#!/usr/bin/env python3
"""Audit and plot finished Decode sweep points without touching the experiment.

Usage:
  python profiling_mimo/decode/plot-decode-metrics.py [RESULT_DIR]

Re-run to refresh. Pending/running/invalid points stay blank, never zero.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from decode_state import name, read_config, rows, valid_result

PREFIXES = ("mean", "p50", "p90", "p95", "p99")
METRIC_LINE = re.compile(r"([^\s{]+)(\{.*\})?\s+(\S+)")


def latest_results():
    parent = Path(__file__).resolve().parent / "results"
    candidates = []
    if parent.exists():
        for root in parent.iterdir():
            try:
                if root.is_dir() and read_config(root).get("benchmark_mode") == "decode":
                    candidates.append(root)
            except (OSError, ValueError):
                pass
    if not candidates:
        raise ValueError("未找到 Decode 结果目录，请指定 RESULT_DIR")
    return max(candidates, key=lambda root: (root / "config.txt").stat().st_mtime_ns)


def parse_metrics(path):
    values = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        match = METRIC_LINE.fullmatch(line)
        if not match:
            continue
        value = float(match[3])
        if math.isfinite(value):
            labels = dict(re.findall(r'(\w+)="([^"\\]*)"', match[2] or ""))
            values.append((match[1], labels, value))
    return values


def sum_metric(metrics, metric, **labels):
    selected = [value for key, tags, value in metrics if key == metric and all(tags.get(k) == v for k, v in labels.items())]
    return sum(selected) if selected else None


def compare(checks, errors, key, actual, expected):
    ok = isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7)
    checks[key] = dict(actual=actual, expected=expected, passed=ok)
    if not ok:
        errors.append(f"{key}: saved={actual}, expected={expected}")


def audit_counters(root, group, seq, config, audit):
    before_path, after_path = (root / f"{group}.metrics-{suffix}.txt" for suffix in ("before", "after"))
    if not before_path.exists() or not after_path.exists():
        audit["warnings"].append("server metrics snapshots unavailable; KV/count cross-check not performed")
        return
    before, after = parse_metrics(before_path), parse_metrics(after_path)
    count = seq + int(config["num_warmups"])
    expected = [
        ("successful_requests_including_warmup", "vllm:request_success_total", {}, count),
        ("output_tokens_including_warmup", "vllm:generation_tokens_total", {}, count * int(config["max_output"])),
        ("logical_input_tokens_including_warmup", "vllm:prompt_tokens_total", {}, count * int(config["input_len"])),
        ("simulated_kv_tokens_including_warmup", "vllm:prompt_tokens_by_source_total", {"source": "external_kv_transfer"}, count * int(config["kv_len"])),
        ("locally_computed_input_tokens_including_warmup", "vllm:prompt_tokens_by_source_total", {"source": "local_compute"}, count),
    ]
    for key, metric, labels, value in expected:
        start, end = sum_metric(before, metric, **labels), sum_metric(after, metric, **labels)
        if start is None or end is None:
            audit["warnings"].append(f"server metric unavailable: {key}")
        else:
            compare(audit["checks"], audit["errors"], key, end - start, value)
    start, end = (sum_metric(m, "vllm:num_preemptions_total") for m in (before, after))
    if start is not None and end is not None:
        audit["server_preemptions_delta"] = end - start
        if end != start:
            audit["warnings"].append(f"server preemptions: {end - start:g}")


def sampled_activity(path):
    samples, failures, running, waiting, kv = 0, 0, [], [], []
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:
                    sample = json.loads(line)
                except ValueError:
                    # The sampler may currently be appending this last line.
                    continue
                failures += int("error" in sample)
                if sample.get("phase") != "benchmark" or "metrics" not in sample:
                    continue
                samples += 1
                metrics = sample["metrics"]
                for metric, target, aggregate in (
                    ("vllm:num_requests_running", running, sum),
                    ("vllm:num_requests_waiting", waiting, sum),
                    ("vllm:kv_cache_usage_perc", kv, max),
                ):
                    values = [m["value"] for m in metrics if m["name"] == metric]
                    if values:
                        target.append(aggregate(values))
    return dict(metrics_samples=samples, metrics_errors=failures,
                peak_running_requests=max(running) if running else None,
                peak_waiting_requests=max(waiting) if waiting else None,
                peak_kv_cache_usage_pct=100 * max(kv) if kv else None)


def audit_point(root, seq, config, run, summary):
    group = name(seq, config)
    point = dict(max_num_seqs=seq, batch_tokens=seq + int(config["batch_token_extra"]),
                 concurrency=seq, num_prompts=seq, result_file=group + ".json",
                 run_status=run.get("status", "pending"), status="pending", note="")
    audit = dict(max_num_seqs=seq, checks={}, errors=[], warnings=[])
    if run.get("status") != "finished" or str(run.get("exit_code")) != "0":
        point["status"] = run.get("status", "pending")
        point["note"] = "waiting for a successful finished run; metrics left blank"
        audit["status"] = point["status"]
        return point, audit
    try:
        data = valid_result(root, seq, config)
        for key in ("ttfts", "latencies", "start_times", "itls", "errors"):
            if len(data.get(key, [])) != seq:
                raise ValueError(f"invalid detail array: {key}")
        if any(data["errors"]):
            raise ValueError("nonempty request errors")
        starts = np.asarray(data["start_times"], dtype=float)
        ttfts = np.asarray(data["ttfts"], dtype=float)
        latencies = np.asarray(data["latencies"], dtype=float)
        if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ttfts)) or not np.all(np.isfinite(latencies)):
            raise ValueError("nonfinite request timestamps/latencies")
        if np.any(ttfts < 0) or np.any(latencies < ttfts):
            raise ValueError("negative TTFT or E2EL < TTFT")
        output = int(config["max_output"])
        if output <= 1:
            raise ValueError("Decode latency requires output length > 1")
        itls = np.asarray([gap for request in data["itls"] for gap in request], dtype=float)
        if not itls.size or not np.all(np.isfinite(itls)) or np.any(itls < 0):
            raise ValueError("empty/nonfinite/negative ITL samples")
        tpot = (latencies - ttfts) / (output - 1)
        for metric, values in (("tpot", tpot), ("itl", itls), ("ttft", ttfts), ("e2el", latencies)):
            expected = {"mean": float(np.mean(values)) * 1000,
                        **{f"p{p}": float(np.percentile(values, p)) * 1000 for p in (50, 90, 95, 99)}}
            for prefix, value in expected.items():
                key = f"{prefix}_{metric}_ms"
                compare(audit["checks"], audit["errors"], key, data.get(key), value)
                point[key] = value
        throughput = data["total_output_tokens"] / data["duration"]
        compare(audit["checks"], audit["errors"], "output_throughput", data.get("output_throughput"), throughput)
        compare(audit["checks"], audit["errors"], "request_throughput", data.get("request_throughput"), seq / data["duration"])
        wall_span = float(np.max(starts + latencies) - np.min(starts))
        if wall_span > data["duration"] + 0.01:
            raise ValueError("request wall span exceeds benchmark duration")
        audit.update(request_wall_span_s=wall_span, request_start_spread_s=float(np.ptp(starts)),
                     itl_samples=int(itls.size), expected_inter_token_count=seq * (output - 1),
                     itl_samples_per_request=dict(Counter(map(len, data["itls"]))),
                     input_lengths=dict(Counter(data["input_lens"])), output_lengths=dict(Counter(data["output_lens"])))
        if any(len(gaps) != output - 1 for gaps in data["itls"]):
            audit["warnings"].append("ITL sample count differs from output_tokens-1 for some requests; ITL measures client stream gaps, not GPU token steps")
        audit_counters(root, group, seq, config, audit)
        activity = sampled_activity(root / f"{group}.metrics-series.jsonl")
        audit.update(activity)
        if activity["peak_running_requests"] is not None and activity["peak_running_requests"] < seq:
            audit["warnings"].append(f"sampled running-request peak {activity['peak_running_requests']:g} < {seq}; sampling cannot prove every-step saturation")
        existing = summary.get(seq)
        if existing and existing.get("status") == "ok":
            for key in ("output_tokens_per_s",) + tuple(f"{prefix}_{metric}_ms" for metric in ("tpot", "itl") for prefix in PREFIXES):
                compare(audit["checks"], audit["errors"], "summary." + key,
                        float(existing[key]), throughput if key == "output_tokens_per_s" else point[key])
        else:
            audit["warnings"].append("summary.csv not yet refreshed for this point; raw benchmark audited independently")
        if audit["errors"]:
            raise ValueError("; ".join(audit["errors"]))
        point.update(status="complete", completed=seq, failed=0, duration_s=data["duration"],
                     total_output_tokens=data["total_output_tokens"], output_tokens_per_s=throughput,
                     itl_samples=int(itls.size), expected_inter_token_count=seq * (output - 1), **activity)
        point["note"] = "; ".join(audit["warnings"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not audit["errors"]:
            audit["errors"].append(str(error))
        # A bad point must never leak its partially calculated values into plots.
        point = {key: value for key, value in point.items() if key in (
            "max_num_seqs", "batch_tokens", "concurrency", "num_prompts", "result_file", "run_status")}
        point.update(status="invalid_or_writing", note=str(error))
    audit["status"] = point["status"]
    return point, audit


def draw_panel(ax, points, metric):
    seqs = [point["max_num_seqs"] for point in points]
    styles = (("mean", "Mean", "#2563eb", "o"), ("p50", "P50", "#059669", "s"), ("p99", "P99", "#d97706", "^"))
    if metric == "throughput":
        styles = (("output", "Output throughput", "#2563eb", "o"),)
    for prefix, label, color, marker in styles:
        key = "output_tokens_per_s" if metric == "throughput" else f"{prefix}_{metric}_ms"
        values = [point.get(key, float("nan")) if point["status"] == "complete" else float("nan") for point in points]
        ax.plot(seqs, values, color=color, marker=marker, markersize=6, markerfacecolor="white",
                markeredgewidth=1.7, linewidth=2, label=label)
        if metric == "throughput":
            for seq, value in zip(seqs, values):
                if math.isfinite(value):
                    ax.annotate(f"{value:,.0f}", (seq, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(seqs, [str(seq) for seq in seqs])
    ax.set_xlim(min(seqs) / 1.15, max(seqs) * 1.15)
    ax.set_ylim(bottom=0)
    if metric == "throughput":
        ax.set_ylim(top=max([p.get("output_tokens_per_s", 0) for p in points] + [1]) * 1.22)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("max_num_seqs (= requests = concurrency)", fontsize=10, labelpad=10)
    ax.set_ylabel("Output tokens/s" if metric == "throughput" else "Milliseconds", labelpad=8)
    ax.set_title({"throughput": "Decode output throughput", "tpot": "Request TPOT", "itl": "Client stream ITL"}[metric], loc="left", fontsize=12, fontweight="bold", pad=14)
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", nargs="?", type=Path, help="默认选择最新 Decode 实验")
    parser.add_argument("--output-dir", type=Path, help="默认 RESULT_DIR/plots；仅更新图表和审计文件")
    parser.add_argument("--seq-counts", nargs="+", type=int, help="覆盖横坐标列表；未完成点留空")
    args = parser.parse_args()
    try:
        root = (args.result_dir or latest_results()).resolve()
        config = read_config(root)
        if config.get("benchmark_mode") != "decode":
            raise ValueError("RESULT_DIR 不是 Decode 实验")
        seqs = sorted(set(args.seq_counts or map(int, config["seq_counts"].split())))
        if not seqs or min(seqs) <= 0:
            raise ValueError("seq counts 必须为正整数")
        runs = rows(root)
        summary = {}
        if (root / "summary.csv").exists():
            with (root / "summary.csv").open() as stream:
                summary = {int(row["max_num_seqs"]): row for row in csv.DictReader(stream)}
        points, audits = zip(*(audit_point(root, seq, config, runs.get(seq, {}), summary) for seq in seqs))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    output = (args.output_dir or root / "plots").resolve()
    if output == root:
        parser.error("请使用独立的图表输出目录，避免与实验文件混放")
    output.mkdir(parents=True, exist_ok=True)
    fields = ["max_num_seqs", "batch_tokens", "concurrency", "num_prompts", "run_status", "status",
              "completed", "failed", "duration_s", "total_output_tokens", "output_tokens_per_s"]
    fields += [f"{prefix}_{metric}_ms" for metric in ("tpot", "itl", "ttft", "e2el") for prefix in PREFIXES]
    fields += ["itl_samples", "expected_inter_token_count", "metrics_samples", "metrics_errors",
               "peak_running_requests", "peak_waiting_requests", "peak_kv_cache_usage_pct", "result_file", "note"]
    with (output / "decode_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    report = dict(snapshot=timestamp, result_dir=str(root), config=config, points=audits,
                  definitions=dict(throughput="total_output_tokens / benchmark duration; warmup excluded, admission/drain included",
                                   tpot="(request latency - TTFT) / (output_tokens - 1); request-level statistics",
                                   itl="client stream gaps pooled across requests; not exact GPU per-token timings",
                                   server_counters="metrics-after minus metrics-before; includes independent warmup requests"))
    (output / "decode_metrics.validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, metric in zip(axes, ("throughput", "tpot", "itl")):
        draw_panel(ax, points, metric)
    model = config.get("served_model_name", Path(config["model_path"]).name)
    completed = sum(p["status"] == "complete" for p in points)
    fig.suptitle(f"{model} | Decode performance | {completed}/{len(seqs)} complete", x=0.055, y=0.97, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.055, 0.895,
             f"TP{config['TP']} / DP{config['DP']} / EP{config['EP']} | GPU {config['GPU']} | Simulated KV {int(config['kv_len']):,} | Output {config['max_output']} tokens | Batch budget = seqs + {config['batch_token_extra']}",
             fontsize=10, color="#475569")
    fig.text(0.055, 0.055,
             "Throughput: output tokens / full benchmark duration; warmup excluded. TPOT: per request; ITL: observed stream gaps.\n"
             f"Pending/running/invalid points left blank. Snapshot: {timestamp}", fontsize=9, color="#64748b")
    fig.subplots_adjust(left=0.055, right=0.98, bottom=0.23, top=0.78, wspace=0.26)
    for extension in ("png", "svg"):
        fig.savefig(output / f"decode_metrics.{extension}", dpi=180)
    plt.close(fig)
    print(f"结果目录：{root}\n校验通过 {completed}/{len(seqs)} 组，其余留空。")
    for point in points:
        print(f"seqs={point['max_num_seqs']}: {point['status']}; output_tok/s={point.get('output_tokens_per_s', '-')}")
    for extension in ("png", "svg", "csv", "validation.json"):
        print(output / f"decode_metrics.{extension}")
    return 1 if any(p["status"] == "invalid_or_writing" for p in points) else 0


if __name__ == "__main__":
    raise SystemExit(main())
