"""Read-only capture analysis; interval overlap is NOT scheduler batch size.

Run with a Python environment containing matplotlib. Writes CSV and PNG/PDF
beside this script. Incomplete results are excluded; warmup and eval stay separate.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "captures"


def timeline(starts, ends, origin):
    changes = Counter()
    for start, end in zip(starts, ends):
        changes[start - origin] += 1
        changes[end - origin] -= 1
    times, counts = [0.0], [0]
    count = peak = 0
    for timestamp, delta in sorted(changes.items()):
        count += delta
        peak = max(peak, count)
        times.append(timestamp)
        counts.append(count)
    assert count == 0
    return times, counts, peak


rows = []
plot_data = {}
for run in sorted(ROOT.iterdir()):
    if not (run / "config.json").exists():
        continue
    config = json.loads((run / "config.json").read_text())
    spec = config["speculative_config"]
    method = spec["method"] if spec else "baseline"
    for path in sorted(run.glob("ep*/seq*/*.json")):
        if not path.name.startswith(("batch", "warmup")):
            continue
        data = json.loads(path.read_text())
        n = int(path.parent.name[3:])
        assert data["completed"] == data["num_prompts"] == n
        assert data["failed"] == 0 and not any(data["errors"])
        assert data["max_concurrency"] == n
        assert set(data["output_lens"]) == {int(config["MAX_OUTPUT"])}
        starts = data["start_times"]
        assert len(starts) == len(data["ttfts"]) == len(data["latencies"]) == n
        first = [s + t for s, t in zip(starts, data["ttfts"])]
        ends = [s + t for s, t in zip(starts, data["latencies"])]
        assert all(s <= f <= e for s, f, e in zip(starts, first, ends))
        origin = min(starts)
        outstanding = timeline(starts, ends, origin)
        emitting = timeline(first, ends, origin)
        duration = data["duration"]
        phase = "eval" if path.name.startswith("batch") else "warmup"
        row = dict(
            method=method,
            ep=int(path.parent.parent.name[2:]),
            phase=phase,
            seq=n,
            kv_len=int(config["KV_LEN"]),
            output_len=int(config["MAX_OUTPUT"]),
            duration_s=duration,
            output_tokens_per_s=data["output_throughput"],
            mean_ttft_s=data["mean_ttft_ms"] / 1000,
            mean_tpot_ms=data["mean_tpot_ms"],
            client_start_spread_s=max(starts) - origin,
            client_outstanding_peak=outstanding[2],
            first_token_to_end_overlap_peak=emitting[2],
            first_token_to_end_overlap_time_mean=sum(e - f for f, e in zip(first, ends)) / duration,
            requests_with_first_token_before_first_completion=sum(f < min(ends) for f in first),
            first_completion_s=min(ends) - origin,
            source=str(path.relative_to(ROOT)),
        )
        rows.append(row)
        if phase == "eval" and n == 4096 and method in ("baseline", "dflash"):
            plot_data[method, row["ep"]] = outstanding, emitting, row

with (HERE / "request-interval-summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
for i, method in enumerate(("baseline", "dflash")):
    for ep in (0, 1):
        ax = axes[i, ep]
        outstanding, emitting, row = plot_data[method, ep]
        ax.step(outstanding[0], outstanding[1], where="post", color="#8594a8", label="Client outstanding (includes waiting)")
        ax.step(emitting[0], emitting[1], where="post", color="#216cb5", label="First-token-to-end interval overlap")
        ax.axhline(4096, color="#999999", linestyle=":", linewidth=0.8)
        ax.set_title(f"{method.capitalize()} | EP {'on' if ep else 'off'} | overlap peak {emitting[2]}", loc="left")
        ax.set_xlabel("Seconds since first client request started")
        ax.set_ylabel("Requests, summed across DP=4")
        ax.set_ylim(0, 4350)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.18)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False)
fig.suptitle("4096-request finite bursts | TP1 / DP4 | simulated KV 10240 | output 1000", y=0.99)
fig.text(0.065, 0.02, "Interval overlap is a client-side proxy, not per-step engine batch size; it includes stalls and possible preemption.", fontsize=9)
fig.tight_layout(rect=(0, 0.05, 1, 0.90))
for suffix in ("png", "pdf"):
    fig.savefig(HERE / f"request-overlap-4096.{suffix}", dpi=160)
print(f"Validated and summarized {len(rows)} completed runs; outputs in {HERE}")
