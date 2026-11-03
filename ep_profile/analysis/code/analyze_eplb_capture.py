"""Compare EPLB captures using read-only CUPTI queries and server logs.

Usage: python analyze_eplb_capture.py OLD_CAPTURE NEW_CAPTURE
The 47-layer Humming layout is validated against each checkpoint config.
"""
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_trace(root, ep):
    config = json.loads((root / "config.json").read_text())
    layers = sum(bool(v) for v in config["model_config"]["moe_layer_freq"])
    folder = root / f"ep{ep}"
    with sqlite3.connect(f"file:{folder / 'trace.sqlite'}?mode=ro", uri=True) as db:
        rows = db.execute("""
            SELECT k.start,k.end,k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id=k.demangledName
            WHERE s.value LIKE '%void humming<%' ORDER BY k.start
        """).fetchall()
        ranks = [[r for r in rows if r[2] == d] for d in range(4)]
        assert len({len(r) for r in ranks}) == 1
        assert len(ranks[0]) % (2 * layers) == 0
        iterations = []
        heatmap = []
        for offset in range(0, len(ranks[0]), 2 * layers):
            values = [[sum(ranks[d][offset+2*layer+j][1] -
                           ranks[d][offset+2*layer+j][0] for j in range(2)) / 1e9
                       for d in range(4)] for layer in range(layers)]
            mean = sum(statistics.mean(v) for v in values)
            maximum = sum(max(v) for v in values)
            # Exclude tiny dummy/decode passes, retain the same 24 bulk prefills.
            if mean <= .1:
                continue
            iterations.append(dict(trace_forward=offset//(2*layers),
                                   expert_rank_mean_s=mean,
                                   expert_rank_max_s=maximum,
                                   weighted_skew=maximum/mean))
            heatmap.append([max(v)/statistics.mean(v) for v in values])
        copies = [dict(kind=k, source_kind=s, destination_kind=d, calls=n,
                       bytes=b, gpu_seconds=t) for k,s,d,n,b,t in db.execute("""
            SELECT copyKind,srcKind,dstKind,COUNT(*),SUM(bytes),SUM(end-start)/1e9
            FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY 1,2,3
        """)]
        # _pad_out_tensor commits a [256,1024] int64 logical map per migrated
        # layer. Its source is pageable CPU memory; weight scales are pinned.
        map_bytes = config["model_config"]["n_routed_experts"] * 1024 * 8
        commits = list(db.execute("""
            SELECT deviceId,start/1e9,end/1e9 FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind=1 AND srcKind=0 AND bytes=? ORDER BY start
        """, (map_bytes,)))
    return dict(bulk_iterations=iterations, heatmap=heatmap,
                memcpy=copies, inferred_map_commits=commits)


def read_logs(root):
    lines = (root / "ep1/server.log").read_text(errors="replace").splitlines()
    stats = []
    triggers = []
    for line_number,line in enumerate(lines, 1):
        match = re.search(r"(\d\d:\d\d:\d\d).*EPLB step: (\d+).*"
                          r"avg_tokens=([\d.]+), max_tokens=(\d+), "
                          r"balancedness=([\d.]+).*rearrangement: (-?\d+)", line)
        if match:
            clock,step,avg,maximum,balance,remaining = match.groups()
            stats.append(dict(line=line_number,clock=clock,step=int(step),
                              avg_tokens=float(avg),max_tokens=int(maximum),
                              balancedness=float(balance),remaining=int(remaining)))
        if "Rearranging experts (async mode) ..." in line:
            triggers.append(dict(line=line_number,text=line))
    return dict(stats=stats,non_profile_rebalance_triggers=triggers)


def main():
    old,new = (Path(p).resolve() for p in sys.argv[1:3])
    output = Path(__file__).resolve().parents[1] / f"eplb-{new.name}"
    output.mkdir(exist_ok=True)
    traces = {"old_ep1":read_trace(old,1), "new_ep1":read_trace(new,1),
              "new_ep0":read_trace(new,0)}
    counts = [len(t["bulk_iterations"]) for t in traces.values()]
    assert counts == [24,24,24], counts
    logs = read_logs(new)
    comparison = []
    for label,root,ep in [("old_ep0",old,0),("old_ep1",old,1),
                          ("new_ep0",new,0),("new_ep1",new,1)]:
        result = json.loads((root/f"ep{ep}/repeat1.json").read_text())
        comparison.append(dict(label=label,duration_s=result["duration"],
                               input_tps=result["total_input_tokens"]/result["duration"],
                               completed=result["completed"],failed=result["failed"],
                               mean_ttft_ms=result["mean_ttft_ms"]))
    evidence = dict(old_capture=str(old),new_capture=str(new),benchmarks=comparison,
                    traces=traces,logs=logs,
                    caveats=["Cross-run comparison, one repetition; not a controlled confidence interval.",
                             "Kernel sums and CPU/GPU copy sums are not elapsed wall time.",
                             "Layer map commits inferred from source-backed memcpy size/type signature.",
                             "No hardware bandwidth counters or expert token histograms captured."])
    (output/"evidence.json").write_text(json.dumps(evidence,indent=2))
    with (output/"eplb_steps.csv").open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(logs["stats"][0]))
        writer.writeheader();writer.writerows(logs["stats"])
    with (output/"prefill_iterations.csv").open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=["run","prefill"]+list(traces["old_ep1"]["bulk_iterations"][0]))
        writer.writeheader()
        for label,trace in traces.items():
            writer.writerows(dict(run=label,prefill=i+1,**row)
                             for i,row in enumerate(trace["bulk_iterations"]))
    fig,axes=plt.subplots(3,1,figsize=(12,10),layout="constrained")
    for ax,label,title in zip(axes[:2],["old_ep1","new_ep1"],
                              ["EP1 without EPLB (previous capture)","EP1 with asynchronous EPLB (this capture)"]):
        im=ax.imshow(np.array(traces[label]["heatmap"]),origin="upper",aspect="auto",
                     extent=(.5,47.5,24.5,.5),vmin=1,vmax=2.2,cmap="YlOrRd")
        ax.set(title=title,xlabel="Sparse layer ordinal (1..47)",ylabel="Bulk prefill pass")
    fig.colorbar(im,ax=axes[:2],label="Expert GEMM max(rank time) / mean(rank time)",shrink=.85)
    for label,title in [("old_ep1","EP1 without EPLB"),("new_ep1","EP1 + EPLB"),("new_ep0","EP0")]:
        rows=traces[label]["bulk_iterations"]
        axes[2].plot(range(1,25),[r["weighted_skew"] for r in rows],label=title)
    axes[2].set(xlabel="Bulk prefill pass",ylabel="Sum of layer maxima / sum of layer means",
                title="EPLB improves early layers progressively; later layers remain skewed")
    axes[2].legend();axes[2].grid(alpha=.2)
    fig.savefig(output/"eplb_progress.png",dpi=160)
    fig.savefig(output/"eplb_progress.svg")
    print(output)
    for label,trace in traces.items():
        rows=trace["bulk_iterations"]
        print(label, "expert critical-path proxy",sum(r["expert_rank_max_s"] for r in rows),
              "map commits per GPU",[sum(d==gpu for d,_,_ in trace["inferred_map_commits"]) for gpu in range(4)])


if __name__ == "__main__":
    main()
