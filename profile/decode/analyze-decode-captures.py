#!/usr/bin/env python3
"""Merge capture runs, ignoring warmup/eval labels, and compare DFlash to decode.

Usage: python analyze-decode-captures.py CAPTURE_PARENT [--output-dir PATH]
CAPTURE_PARENT may contain a captures/ directory or be that directory itself.
Raw logs are read only. EP configurations are never pooled with each other.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/decode-analysis-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

PREFIXES = ("mean", "p50", "p90", "p95", "p99")
METRICS = ("tpot", "itl", "ttft", "e2el")
COLORS = {"baseline": "#2864a0", "dflash7": "#c66132"}
LABELS = {"baseline": "Standard decode", "dflash7": "DFlash (7 draft tokens)"}
STYLES = {"mean": ("-", "o"), "p50": ("--", "s"), "p99": (":", "^")}
CONFIG_KEYS = ("TP", "DP", "KV_LEN", "MAX_OUTPUT", "MAX_MODEL_LEN",
               "MAX_NUM_BATCHED_TOKENS", "CUDAGRAPH_MODE", "MODEL_PATH",
               "GPU_MEMORY_UTILIZATION", "KV_FILL_MEAN", "KV_FILL_STD",
               "KV_STARTUP_FILL", "SEED", "server_max_num_seqs")


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def statistics(values, metric):
    return {f"mean_{metric}_ms": float(np.mean(values)) * 1000,
            **{f"p{p}_{metric}_ms": float(np.percentile(values, p)) * 1000
               for p in (50, 90, 95, 99)}}


def close(actual, expected, label):
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-7):
        raise ValueError(f"{label}: saved={actual}, recalculated={expected}")


def read_result(path, config, root):
    raw = path.read_bytes()
    data = json.loads(raw)
    seq = int(path.parent.name.removeprefix("seq"))
    n, output = data["completed"], int(config["MAX_OUTPUT"])
    if data["failed"] or n != seq or data["num_prompts"] != seq:
        raise ValueError("incomplete requests or failures")
    if data["max_concurrency"] != seq or data["duration"] <= 0:
        raise ValueError("invalid concurrency or duration")
    for key in ("input_lens", "output_lens", "ttfts", "latencies", "start_times", "itls", "errors"):
        if len(data[key]) != n:
            raise ValueError(f"invalid array length: {key}")
    if any(data["errors"]) or set(data["output_lens"]) != {output}:
        raise ValueError("request errors or unexpected output length")
    expected_input = int(config["KV_LEN"]) + 1
    input_counts = Counter(data["input_lens"])
    if any(abs(length - expected_input) > 1 for length in input_counts):
        raise ValueError("input length differs by more than one token")
    ttft, latency, starts = (np.asarray(data[k], dtype=float)
                             for k in ("ttfts", "latencies", "start_times"))
    gaps = np.asarray([v for request in data["itls"] for v in request], dtype=float)
    arrays = {"ttft": ttft, "e2el": latency,
              "tpot": (latency - ttft) / (output - 1), "itl": gaps}
    for key, arr in {**arrays, "starts": starts}.items():
        if not arr.size or not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"invalid samples: {key}")
    total = sum(data["output_lens"])
    close(data["total_output_tokens"], total, "total_output_tokens")
    close(data["total_input_tokens"], sum(data["input_lens"]), "total_input_tokens")
    close(data["output_throughput"], total / data["duration"], "output_throughput")
    close(data["request_throughput"], n / data["duration"], "request_throughput")
    wall_span = float(np.max(starts + latency) - np.min(starts))
    if wall_span > data["duration"] + 0.01:
        raise ValueError("request span exceeds benchmark duration")
    method = "dflash7" if config["speculative_config"] else "baseline"
    row = dict(method=method, ep=int(path.parent.parent.name[-1]), concurrency=seq,
               run=path.parents[2].name, file=str(path.relative_to(root)),
               original_phase="eval" if path.stem.startswith("batch") else "warmup",
               sha256=hashlib.sha256(raw).hexdigest(), requests=n,
               duration_s=data["duration"], output_tokens=total,
               output_tokens_per_s=total / data["duration"],
               request_start_spread_s=float(np.ptp(starts)),
               request_wall_span_s=wall_span, itl_samples=int(gaps.size),
               expected_inter_token_count=n * (output - 1),
               input_length_counts=json.dumps(dict(sorted(input_counts.items()))),
               input_length_deviation_requests=sum(count for length, count in input_counts.items() if length != expected_input))
    for metric, arr in arrays.items():
        summary = statistics(arr, metric)
        for key, value in summary.items():
            close(data[key], value, key)
        row.update(summary)
    if config["speculative_config"]:
        drafts = data["spec_decode_num_drafts"]
        draft_tokens = data["spec_decode_draft_tokens"]
        accepted = data["spec_decode_accepted_tokens"]
        close(draft_tokens, drafts * 7, "draft tokens per round")
        close(data["spec_decode_acceptance_rate"], 100 * accepted / draft_tokens, "acceptance percent")
        close(data["spec_decode_acceptance_length"], 1 + accepted / drafts, "acceptance length")
        row.update(num_drafts=drafts, draft_tokens=draft_tokens, accepted_tokens=accepted,
                   acceptance_pct=100 * accepted / draft_tokens,
                   acceptance_length=1 + accepted / drafts)
        for pos, rate in enumerate(data["spec_decode_per_position_acceptance_rates"], 1):
            row[f"position_{pos}_acceptance_pct"] = 100 * rate
    return row, arrays


def collect_evidence(root):
    evidence = []
    needles = ("GPU KV cache size", "Available KV cache memory", "Filled", "No tuned config covers",
               "max_num_scheduled_tokens is set", "Initializing a V1 LLM engine")
    for path in sorted(root.glob("*/ep*/server.log")):
        for i, line in enumerate(path.read_bytes().decode(errors="replace").split("\n"), 1):
            if any(n in line for n in needles):
                evidence.append(dict(file=str(path.relative_to(root)), line=i, text=line))
    return evidence


def activity(root):
    rows = []
    for path in sorted(root.glob("*/ep*/seq*/eval.metrics-series.jsonl")):
        values = defaultdict(list)
        for line in path.open():
            try:
                sample = json.loads(line)
            except ValueError:
                continue
            if sample.get("phase") != "benchmark" or "metrics" not in sample:
                continue
            for name in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "num_preemptions_total"):
                selected = [m["value"] for m in sample["metrics"] if m["name"] == "vllm:" + name]
                if selected:
                    values[name].append(max(selected) if name == "kv_cache_usage_perc" else sum(selected))
        row = dict(file=str(path.relative_to(root)), ep=int(path.parent.parent.name[-1]),
                   concurrency=int(path.parent.name[3:]))
        for name, samples in values.items():
            row["peak_" + name] = max(samples)
        if values["num_preemptions_total"]:
            row["sampled_preemptions_delta"] = max(values["num_preemptions_total"]) - min(values["num_preemptions_total"])
        rows.append(row)
    return rows


def axes_style(ax, seqs):
    ax.set_xscale("log", base=2)
    ax.set_xticks(seqs, [str(x) for x in seqs])
    ax.set_xlim(min(seqs) / 1.15, max(seqs) * 1.15)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Concurrency (= requested simultaneous requests)")
    ax.grid(color="#e4e9ef", linewidth=0.7)
    ax.set_axisbelow(True)


def save_figure(fig, output, name):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def plot_metrics(points, runs, output, eps=(0, 1)):
    seqs = sorted({p["concurrency"] for p in points})
    fig, axes = plt.subplots(len(eps), 3, figsize=(18, 11.4 if len(eps) == 2 else 6.8), squeeze=False)
    for rowidx, ep in enumerate(eps):
        for col, metric in enumerate(("throughput", "tpot", "itl")):
            ax = axes[rowidx, col]
            for method in COLORS:
                selected = sorted((p for p in points if p["ep"] == ep and p["method"] == method), key=lambda p: p["concurrency"])
                x = [p["concurrency"] for p in selected]
                prefixes = ("mean",) if metric == "throughput" else ("mean", "p50", "p99")
                for prefix in prefixes:
                    key = "output_tokens_per_s" if metric == "throughput" else f"{prefix}_{metric}_ms"
                    linestyle, marker = STYLES[prefix]
                    ax.plot(x, [p[key] for p in selected], color=COLORS[method], ls=linestyle,
                            marker=marker, mfc="white", ms=5, lw=1.8)
                if metric == "throughput":
                    for p in selected:
                        raw = [r["output_tokens_per_s"] for r in runs if (r["method"], r["ep"], r["concurrency"]) == (method, ep, p["concurrency"])]
                        ax.vlines(p["concurrency"], min(raw), max(raw), color=COLORS[method], alpha=.3, lw=5)
                        ax.scatter([p["concurrency"]] * len(raw), raw, color=COLORS[method], alpha=.3, s=12)
                        ax.annotate(f'{p["output_tokens_per_s"]/1000:.1f}k', (p["concurrency"], p["output_tokens_per_s"]),
                                    xytext=(0, 9 if method == "baseline" else -16), textcoords="offset points",
                                    ha="center", fontsize=8, color=COLORS[method])
            axes_style(ax, seqs)
            title = {"throughput": "Output throughput", "tpot": "Request TPOT", "itl": "Client stream ITL"}[metric]
            ax.set_title(f'{chr(97 + rowidx * 3 + col)}   EP {"off" if ep == 0 else "on"} | {title}', loc="left", weight="bold", pad=12)
            ax.set_ylabel("Output tokens/s" if metric == "throughput" else "Milliseconds")
            if metric == "throughput":
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}k"))
            ax.set_ylim(top=ax.get_ylim()[1] * 1.14)
    for col in range(3):
        ceiling = max(axes[i, col].get_ylim()[1] for i in range(len(eps)))
        for i in range(len(eps)):
            axes[i, col].set_ylim(0, ceiling)
    handles = [Line2D([0], [0], color=COLORS[m], lw=2, label=LABELS[m]) for m in COLORS]
    handles += [Line2D([0], [0], color="#475569", ls=s, marker=m, mfc="white", label=p.upper() if p != "mean" else "Mean") for p, (s, m) in STYLES.items()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.055, .915 if len(eps) == 2 else .87), ncol=5, frameon=False)
    fig.suptitle("MiMo-V2.6-Flash-MOPD | Merged decode experiments", x=.055, y=.98, ha="left", weight="bold", fontsize=19)
    fig.text(.055, .94 if len(eps) == 2 else .90, "TP1 / DP4 | Simulated KV: 8,192 tokens | Output: 500 tokens | Warmup and eval included equally", color="#475569")
    counts = {m: "/".join(str(next((p["n_runs"] for p in points if p["method"] == m and p["ep"] == eps[0] and p["concurrency"] == s), 0)) for s in seqs) for m in COLORS}
    fig.text(.055, .025,
             "Throughput = sum(output tokens) / sum(run durations); faint marks show individual runs and their range.\n"
             "Latency percentiles recomputed from pooled samples. ITL measures stream gaps, not GPU steps. DFlash has no data below 2048.\n"
             f"Runs at {'/'.join(map(str, seqs))}: standard = {counts['baseline']}; DFlash = {counts['dflash7']} (per EP). No confidence intervals.",
             fontsize=9, color="#475569")
    fig.subplots_adjust(left=.055, right=.98, top=.83 if len(eps) == 2 else .73, bottom=.14 if len(eps) == 2 else .23, wspace=.24, hspace=.42)
    save_figure(fig, output, "decode_metrics" if len(eps) == 2 else f"decode_metrics_ep{eps[0]}")


def plot_diagnostics(points, comparisons, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    for ep, color in ((0, "#2864a0"), (1, "#c66132")):
        p = [p for p in points if p["method"] == "dflash7" and p["ep"] == ep]
        x = np.arange(len(p)) + (-.17 if ep == 0 else .17)
        for ax, key in zip(axes[:2], ("acceptance_pct", "acceptance_length")):
            ax.bar(x, [v[key] for v in p], width=.30, color=color, label=f"EP {'off' if ep == 0 else 'on'}")
            for xx, v in zip(x, p):
                ax.annotate(f'{v[key]:.3f}', (xx, v[key]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10)
        c = [p for p in comparisons if p["ep"] == ep]
        axes[2].bar(x, [p["tpot_ratio"] for p in c], width=.30, color=color)
        for xx, v in zip(x, c):
            axes[2].annotate(f'{v["tpot_ratio"]:.2f}x', (xx, v["tpot_ratio"]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10)
    titles = ("a   Almost all draft tokens rejected", "b   Only about one token per round", "c   Decode latency increases")
    for ax, title in zip(axes, titles):
        ax.set_xticks([0, 1], ["2048", "4096"])
        ax.set_xlabel("Concurrency")
        ax.set_title(title, loc="left", weight="bold", fontsize=12)
        ax.grid(axis="y", color="#e4e9ef")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accepted draft tokens / drafted tokens (%)")
    axes[0].set_ylim(0, .20)
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_ylabel("1 + accepted draft tokens / draft rounds")
    axes[1].set_ylim(0, 1.25)
    axes[1].axhline(1, color="#475569", ls="--", lw=1)
    axes[2].set_ylabel("Mean TPOT: DFlash / standard decode")
    axes[2].set_ylim(0, max(c["tpot_ratio"] for c in comparisons) * 1.23)
    axes[2].axhline(1, color="#475569", ls="--", lw=1)
    fig.suptitle("DFlash slowdown | Rejection cost without useful multi-token progress", x=.06, y=.98, ha="left", weight="bold", fontsize=17)
    fig.text(.06, .045, "Acceptance is computed from summed counters, not averaged percentages. The logged acceptance_rate is already in percent.\n"
             "All results use simulated KV; low acceptance is observed, while its upstream cause requires a real-KV control experiment.", fontsize=10, color="#475569")
    fig.subplots_adjust(left=.06, right=.98, bottom=.23, top=.80, wspace=.30)
    save_figure(fig, output, "speculative_diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_parent", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    parent = args.capture_parent.resolve()
    root = parent / "captures" if (parent / "captures").is_dir() else parent
    output = (args.output_dir or root.parent / "analysis").resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside the raw captures directory")
    configs = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/config.json"))}
    if not configs:
        parser.error("No capture configs found")
    reference = next(iter(configs.values()))
    for run, config in configs.items():
        if any(config.get(k) != reference.get(k) for k in CONFIG_KEYS):
            parser.error(f"Incompatible benchmark configuration: {run}")
        spec = config.get("speculative_config")
        if spec and (spec.get("method") != "dflash" or spec.get("num_speculative_tokens") != 7):
            parser.error("This analysis expects DFlash with seven draft tokens")
    groups, runs, excluded = defaultdict(list), [], []
    for path in sorted(root.glob("*/ep*/seq*/*.json")):
        try:
            row, arrays = read_result(path, configs[path.parents[2].name], root)
        except (ValueError, KeyError, TypeError) as error:
            excluded.append(dict(file=str(path.relative_to(root)), reason=str(error)))
            continue
        runs.append(row)
        groups[(row["method"], row["ep"], row["concurrency"])].append((row, arrays))
    missing = [str(p.with_suffix(".json").relative_to(root)) for p in sorted(root.glob("*/ep*/seq*/warmup-*.log")) if not p.with_suffix(".json").exists()]
    points = []
    for (method, ep, seq), records in sorted(groups.items()):
        rows = [r for r, _ in records]
        total = sum(r["output_tokens"] for r in rows)
        duration = sum(r["duration_s"] for r in rows)
        point = dict(method=method, ep=ep, concurrency=seq, n_runs=len(rows),
                     requests=sum(r["requests"] for r in rows), duration_s=duration,
                     output_tokens=total, output_tokens_per_s=total / duration,
                     throughput_run_min=min(r["output_tokens_per_s"] for r in rows),
                     throughput_run_max=max(r["output_tokens_per_s"] for r in rows),
                     itl_samples=sum(r["itl_samples"] for r in rows))
        for metric in METRICS:
            point.update(statistics(np.concatenate([a[metric] for _, a in records]), metric))
        if method == "dflash7":
            for key in ("num_drafts", "draft_tokens", "accepted_tokens"):
                point[key] = sum(r[key] for r in rows)
            point["acceptance_pct"] = 100 * point["accepted_tokens"] / point["draft_tokens"]
            point["acceptance_length"] = 1 + point["accepted_tokens"] / point["num_drafts"]
            point["nominal_verification_positions_per_output"] = 8 / point["acceptance_length"]
        points.append(point)
    index = {(p["method"], p["ep"], p["concurrency"]): p for p in points}
    comparisons = []
    for s in points:
        if s["method"] != "dflash7":
            continue
        b = index[("baseline", s["ep"], s["concurrency"])]
        comparisons.append(dict(ep=s["ep"], concurrency=s["concurrency"],
                                baseline_tokens_per_s=b["output_tokens_per_s"],
                                dflash_tokens_per_s=s["output_tokens_per_s"],
                                throughput_change_pct=100 * (s["output_tokens_per_s"] / b["output_tokens_per_s"] - 1),
                                baseline_tpot_ms=b["mean_tpot_ms"], dflash_tpot_ms=s["mean_tpot_ms"],
                                tpot_ratio=s["mean_tpot_ms"] / b["mean_tpot_ms"],
                                acceptance_pct=s["acceptance_pct"], acceptance_length=s["acceptance_length"]))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "decode_runs.csv", runs)
    write_csv(output / "decode_metrics.csv", points)
    write_csv(output / "speculative_comparison.csv", comparisons)
    write_json(output / "server_evidence.json", collect_evidence(root))
    write_json(output / "decode_metrics.validation.json", dict(
        raw_root=str(root), included_runs=len(runs), pooled_points=len(points),
        validated_statistics_per_run=len(METRICS) * len(PREFIXES), excluded=excluded,
        missing_result_files=missing, config_checks=list(CONFIG_KEYS),
        common_config={k: reference[k] for k in CONFIG_KEYS},
        definitions={"throughput": "sum(output tokens) / sum(benchmark duration), separately per method/EP/concurrency",
                     "latencies": "recompute percentiles from pooled request samples; ITL pools all client gaps",
                     "phases": "warmup and eval treated equally; successful JSON overrides stale load-event status",
                     "seq": "client concurrency, not context length or actual GPU batch size",
                     "uncertainty": "run min/max shown for throughput; no inferential statistics"},
        activity=activity(root),
        input_length_deviation_requests=sum(r["input_length_deviation_requests"] for r in runs),
        input_length_policy="Retain +/-1-token input deviations and record their count; reject larger deviations",
        itl_count_mismatch_runs=sum(r["itl_samples"] != r["expected_inter_token_count"] for r in runs)))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    plot_metrics(points, runs, output)
    for ep in (0, 1):
        plot_metrics(points, runs, output, (ep,))
    plot_diagnostics(points, comparisons, output)
    print(json.dumps(dict(included=len(runs), points=len(points), excluded=excluded,
                          missing=missing, comparisons=comparisons), ensure_ascii=False, indent=2))
    return 1 if excluded else 0


if __name__ == "__main__":
    raise SystemExit(main())
