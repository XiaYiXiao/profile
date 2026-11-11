"""Plot per-step engine observations, not client-side batch proxies."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
os.environ["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

all_rows = []
for path in sorted((DATA / "engine").glob("*.jsonl")):
    for line in path.read_text().splitlines():
        try:
            all_rows.append(json.loads(line))
        except ValueError:
            pass
cases = ["long_n4096_kv10240", "short_n4096_kv1024", "repeat_n4096_kv10240"]
fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharey="row", squeeze=False)
colors = ["#2269a5", "#d5752b", "#3e9561", "#955fa3"]
for col, name in enumerate(cases):
    case = json.loads((DATA / name / "summary.json").read_text())
    t0, t1 = case["started_unix_s"], case["finished_unix_s"]
    for rank in range(4):
        rows = [r for r in all_rows if r["kind"] == "step" and r["dp_rank"] == rank
                and t0 <= r["time_unix_s"] <= t1]
        x = [r["time_unix_s"]-t0 for r in rows]
        for idx, key in enumerate(("scheduled_requests", "waiting", "kv_usage")):
            y = [r[key]*(100 if key == "kv_usage" else 1) for r in rows]
            axes[idx, col].step(x, y, where="post", color=colors[rank], label=f"DP {rank}", lw=1.1, alpha=0.85)
    axes[0, col].set_title(f"{'Repeat: ' if col == 2 else ''}KV = {case['kv_len']:,}", loc="left")
    axes[2, col].set_xlabel("Seconds since burst started")
    for ax in axes[:, col]:
        ax.grid(alpha=0.2)
        ax.set_xlim(0, t1-t0)
        ax.set_ylim(bottom=0)
axes[0, 0].set_ylabel("Scheduled requests per step / DP")
axes[1, 0].set_ylabel("Waiting requests / DP")
axes[2, 0].set_ylabel("KV block pool usage (%) / DP")
for index, key in enumerate(("scheduled_requests", "waiting")):
    maximum = max(r[key] for r in all_rows if r["kind"] == "step")
    axes[index, 0].set_ylim(0, maximum * 1.08)
axes[2, 0].set_ylim(0, 103)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4, frameon=False)
fig.suptitle("Same server and 4096-request bursts: change only simulated KV length", y=0.99)
fig.text(0.065, 0.02, "TP1 / DP4, EP off, no speculation, output 64, token budget 8120 per DP. Observational runtime hooks; source unchanged.", fontsize=10)
fig.tight_layout(rect=(0, 0.05, 1, 0.92))
for ext in ("png", "pdf"):
    fig.savefig(DATA / f"kv-capacity-evidence.{ext}", dpi=170)
print(DATA / "kv-capacity-evidence.png")
