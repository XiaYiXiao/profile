"""Join raw allocation failures to exact scheduler steps and finite-burst cases."""
import csv
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def read_lines(path):
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # An active process may not have flushed its last row yet.
            row["source"] = str(path.relative_to(ROOT))
            row["source_line"] = line_number
            yield row


def write_csv(name, rows):
    if not rows:
        return
    with (DATA / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


configs = {}
steps = []
failures = []
for path in sorted((DATA / "engine").glob("*.jsonl")):
    for row in read_lines(path):
        if row["kind"] == "configuration":
            configs[row["dp_rank"]] = row
        elif row["kind"] == "step":
            steps.append(row)
        elif row["kind"] == "allocation_failed":
            failures.append(row)
assert len(configs) == 4, f"Expected 4 DP configurations, got {len(configs)}"
step_index = {(r["pid"], r["step"]): r for r in steps}
summary_rows, rank_rows, examples = [], [], []
case_results = []
for path in sorted(DATA.glob("*/summary.json")):
    case = json.loads(path.read_text())
    t0, t1 = case["started_unix_s"], case["finished_unix_s"]
    selected = [r for r in steps if t0 <= r["time_unix_s"] <= t1]
    denied = [r for r in failures if t0 <= r["time_unix_s"] <= t1]
    requests = list(read_lines(path.parent / "requests.jsonl"))
    assert len(requests) == case["num_prompts"] and case["errors"] == 0
    overlap = []
    for request in requests:
        assert request["usage"]["prompt_tokens"] == case["kv_len"]+1
        assert request["usage"]["completion_tokens"] == case["output_tokens"]
        if "first_token_unix_s" in request:
            overlap.extend([(request["first_token_unix_s"], 1), (request["end_unix_s"], -1)])
    current = peak = 0
    for _, delta in sorted(overlap):
        current += delta
        peak = max(peak, current)
    metrics = []
    for sample in read_lines(path.parent / "metrics.jsonl"):
        parsed = {}
        for line in sample.get("lines", []):
            match = re.fullmatch(r'([^\s{]+)\{.*?engine="(\d+)".*?\}\s+(\S+)', line)
            if match:
                parsed.setdefault(match[1], {})[int(match[2])] = float(match[3])
        metrics.append(dict(time_unix_s=sample["time_unix_s"], values=parsed))
    complete_samples = [m for m in metrics if len(m["values"].get("vllm:num_requests_running", {})) == 4]
    metric_peak_running = max((sum(m["values"]["vllm:num_requests_running"].values()) for m in complete_samples), default=0)
    metric_peak_waiting = max((sum(m["values"].get("vllm:num_requests_waiting", {}).values()) for m in complete_samples), default=0)
    preemptions = [sum(m["values"].get("vllm:num_preemptions_total", {}).values()) for m in metrics]
    local_rows = []
    for rank, cfg in sorted(configs.items()):
        rank_steps = [r for r in selected if r["dp_rank"] == rank and r["scheduled_requests"]]
        rank_denied = [r for r in denied if r["dp_rank"] == rank]
        memory_denied = [r for r in rank_denied if r["checks"] and
                         r["checks"][-1]["required_blocks"]+cfg["watermark_blocks"] >
                         r["checks"][-1]["free_blocks"]-r["reserved_blocks"]]
        assert len(memory_denied) == len(rank_denied), "Allocation failure without observed block deficit"
        row = dict(case=case["name"], dp_rank=rank, kv_len=case["kv_len"],
                   total_blocks=cfg["total_blocks"], steps=len(rank_steps),
                   peak_scheduled_requests=max(r["scheduled_requests"] for r in rank_steps),
                   peak_running=max(r["running"] for r in rank_steps),
                   peak_waiting=max(r["waiting"] for r in rank_steps),
                   peak_scheduled_tokens=max(r["scheduled_tokens"] for r in rank_steps),
                   max_scheduled_tokens=cfg["max_scheduled_tokens"],
                   peak_kv_usage=max(r["kv_usage"] for r in rank_steps),
                   allocation_failures=len(rank_denied),
                   proven_block_deficit_failures=len(memory_denied))
        local_rows.append(row)
        rank_rows.append(row)
        candidates = [r for r in memory_denied if r["request_status"] == "WAITING"
                      and (r["pid"], r["step"]) in step_index]
        pure_decode = [r for r in candidates
                       if step_index[r["pid"],r["step"]]["scheduled_tokens"] ==
                       step_index[r["pid"],r["step"]]["scheduled_requests"]]
        candidates = pure_decode or candidates
        if candidates:
            evidence = max(candidates, key=lambda r: step_index[r["pid"],r["step"]]["waiting"])
            step = step_index[evidence["pid"], evidence["step"]]
            check = evidence["checks"][-1]
            examples.append(dict(
                case=case["name"], dp_rank=rank, step=evidence["step"],
                waiting=step["waiting"], running=step["running"],
                scheduled_requests=step["scheduled_requests"],
                scheduled_tokens=step["scheduled_tokens"],
                token_budget=cfg["max_scheduled_tokens"],
                required_blocks=check["required_blocks"], available_blocks=check["free_blocks"],
                total_blocks=cfg["total_blocks"], kv_usage=step["kv_usage"],
                external_tokens=evidence["external_tokens"],
                new_tokens=evidence["new_tokens"],
                failure_source=evidence["source"], failure_line=evidence["source_line"],
                step_source=step["source"], step_line=step["source_line"],
            ))
    summary = dict(case=case["name"], requests=case["num_prompts"], kv_len=case["kv_len"],
                   output_tokens=case["output_tokens"], completed=case["completed"], errors=case["errors"],
                   client_start_spread_s=case["client_start_spread_s"],
                   client_generation_interval_overlap_peak=peak,
                   metrics_peak_global_running=metric_peak_running,
                   metrics_peak_global_waiting=metric_peak_waiting,
                   metrics_preemptions_delta=max(preemptions, default=0)-min(preemptions, default=0),
                   per_dp_peak_scheduled_requests="/".join(str(r["peak_scheduled_requests"]) for r in local_rows),
                   sum_of_per_dp_peaks_not_simultaneous=sum(r["peak_scheduled_requests"] for r in local_rows),
                   allocation_failures=len(denied),
                   mean_ttft_s=statistics.mean(r["ttft_s"] for r in requests if "ttft_s" in r),
                   duration_s=case["duration_s"])
    summary_rows.append(summary)
    case_results.append(dict(summary=summary, per_rank=local_rows))
write_csv("case-summary.csv", summary_rows)
write_csv("per-dp-summary.csv", rank_rows)
write_csv("allocation-failure-examples.csv", examples)
result = dict(method="Observe unmodified return values; each allocator call and scheduler call executes once.",
              limits="Shortened output (64); direct token IDs; probe adds timing overhead. Per-DP maxima are not a simultaneous global batch.",
              engine_configurations=configs, cases=case_results, failure_examples=examples)
(DATA / "evidence-summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(dict(cases=summary_rows, examples=examples), indent=2))
