#!/usr/bin/env python3
"""Create a Perfetto display copy without changing event timestamps/durations.

Only GPU display tids and matching CPU-to-GPU flow endpoints are reassigned.
args.stream remains the real CUDA stream. Original files are never overwritten.
The complete JSON is loaded into host memory; run one rank at a time for large traces.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path


WORK = {"kernel", "gpu_memcpy", "gpu_memset"}


def timing_digest(events):
    digest = hashlib.sha256()
    for event in events:
        digest.update(repr((event.get("ts"), event.get("dur"))).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def crossings(items):
    stack = []
    count = 0
    for _, event in sorted(items, key=lambda item: (item[1]["ts"], -item[1].get("dur", 0), item[0])):
        start, end = event["ts"], event["ts"] + event.get("dur", 0)
        while stack and stack[-1] <= start:
            stack.pop()
        if stack and end > stack[-1] and end > start:
            count += 1
        if end > start:
            stack.append(end)
    return count


def prepare(trace):
    events = trace["traceEvents"]
    original_count = len(events)
    before_digest = timing_digest(events)
    gpu_tracks = {(e.get("pid"), e.get("tid")) for e in events if e.get("ph") == "X" and e.get("cat") in WORK}
    tracks = defaultdict(list)
    occupied = {(e.get("pid"), e.get("tid")) for e in events}
    for index, event in enumerate(events):
        key = event.get("pid"), event.get("tid")
        if key in gpu_tracks and event.get("ph") == "X":
            tracks[key].append((index, event))
    report = {"input_events": original_count, "gpu_crossings_before": sum(crossings(items) for items in tracks.values()), "moved_gpu_events": 0, "moved_flow_endpoints": 0, "display_lanes": []}
    next_tid = 900_000_000
    flow_targets = defaultdict(set)
    metadata = []
    for (pid, original_tid), items in tracks.items():
        stacks = []
        tids = [original_tid]
        ordered = sorted(items, key=lambda item: (item[1]["ts"], -item[1].get("dur", 0), item[0]))
        for _, event in ordered:
            start, end = event["ts"], event["ts"] + event.get("dur", 0)
            lane = 0
            if end > start:
                for lane, stack in enumerate(stacks):
                    while stack and stack[-1] <= start:
                        stack.pop()
                    if not stack or end <= stack[-1]:
                        break
                else:
                    lane = len(stacks)
                    stacks.append([])
                stacks[lane].append(end)
            while len(tids) <= lane:
                while (pid, next_tid) in occupied:
                    next_tid += 1
                tid = next_tid
                occupied.add((pid, tid))
                next_tid += 1
                tids.append(tid)
                label = f"stream {original_tid} / display lane {len(tids)-1} (synthetic tid)"
                metadata.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name", "args": {"name": label}})
                report["display_lanes"].append({"pid": pid, "original_tid": original_tid, "display_tid": tid, "name": label})
            display_tid = tids[lane]
            if event.get("cat") in WORK:
                flow_targets[pid, original_tid, start].add(display_tid)
            if display_tid != original_tid:
                event["tid"] = display_tid
                report["moved_gpu_events"] += 1
        del ordered
    # Move unambiguous ac2g flow ends with their work event. Keep flow IDs intact.
    for event in events:
        if event.get("cat") == "ac2g" and event.get("ph") == "f":
            candidates = flow_targets.get((event.get("pid"), event.get("tid"), event.get("ts")), ())
            if len(candidates) == 1:
                tid = next(iter(candidates))
                if tid != event.get("tid"):
                    event["tid"] = tid
                    report["moved_flow_endpoints"] += 1
    after = defaultdict(list)
    for items in tracks.values():
        for index, event in items:
            after[event.get("pid"), event.get("tid")].append((index, event))
    report["gpu_crossings_after"] = sum(crossings(items) for items in after.values())
    report["timestamps_and_durations_unchanged"] = timing_digest(events) == before_digest
    if report["gpu_crossings_after"] or not report["timestamps_and_durations_unchanged"]:
        raise ValueError("Failed to preserve timings or resolve GPU display overlaps")
    events.extend(metadata)
    report["added_metadata_events"] = len(metadata)
    report["output_events"] = len(events)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="One rank's .json or .json.gz trace")
    parser.add_argument("-o", "--output", type=Path, help="New output path (.json.gz recommended)")
    args = parser.parse_args()
    suffix = ".json.gz" if args.input.name.endswith(".json.gz") else ".json"
    output = args.output or args.input.with_name(args.input.name.removesuffix(suffix) + ".perfetto" + suffix)
    report_path = output.with_name(output.name + ".report.json")
    if output.resolve() == args.input.resolve() or output.exists() or report_path.exists():
        parser.error("Output/report already exists, or output is the original input; choose a new output path")
    print(f"读取：{args.input}", flush=True)
    opener = gzip.open if args.input.name.endswith(".gz") else open
    with opener(args.input, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    print(f"分析 {len(trace['traceEvents']):,} 个事件…", flush=True)
    report = prepare(trace)
    report.update(input=str(args.input.resolve()), output=str(output.resolve()), note="Display tids are synthetic; args.stream, timestamps, durations and original events are preserved. CPU overlap errors are outside this GPU-only transformation.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    print(f"写入：{output}", flush=True)
    try:
        with temporary.open("xb") as raw:
            if output.name.endswith(".gz"):
                import io
                with gzip.GzipFile(fileobj=raw, mode="wb", filename="", compresslevel=1) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                        stream.write(json.dumps(trace, ensure_ascii=True, separators=(",", ":")))
            else:
                import io
                with io.TextIOWrapper(raw, encoding="utf-8") as stream:
                    stream.write(json.dumps(trace, ensure_ascii=True, separators=(",", ":")))
        # Hard-link publication is atomic and refuses to replace an existing file.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
