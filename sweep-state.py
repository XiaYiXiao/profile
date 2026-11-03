#!/usr/bin/env python3
"""Persistent experiment state for sweep-mimo.sh (standard library only)."""
import argparse
import csv
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import zlib

FIELDS = ["batch_tokens", "concurrency", "status", "exit_code", "result_file", "profiling", "profile_status", "trace_dir"]
DEFAULTS = dict(TP="4", DP="1", EP="1", GPU="0,1,2,3", input_len="50000", num_prompts="50", num_warmups="10", max_model_len="1000000", max_num_seqs="512", max_output="1", seed="42", prefix_caching="0", profiling="0", profile_max_iters="0", profile_delay_iters="0", profile_with_stack="0", profile_record_shapes="0", profile_with_memory="0", gpu_memory_utilization="0.90", dtype="bfloat16", model_path="/host/models/MiMo-V2.6-Flash-MOPD", vllm_source="/host/vllm")
ENV = dict(TP="TP", DP="DP", EP="EP", GPU="CUDA_VISIBLE_DEVICES", input_len="INPUT_LEN", num_prompts="NUM_PROMPTS", num_warmups="NUM_WARMUPS", max_model_len="MAX_MODEL_LEN", max_num_seqs="MAX_NUM_SEQS", max_output="MAX_OUTPUT", seed="SEED", profiling="PROFILE", profile_max_iters="PROFILE_MAX_ITERS", profile_delay_iters="PROFILE_DELAY_ITERS", profile_with_stack="PROFILE_WITH_STACK", profile_record_shapes="PROFILE_RECORD_SHAPES", profile_with_memory="PROFILE_WITH_MEMORY", gpu_memory_utilization="GPU_MEMORY_UTILIZATION", model_path="MODEL_PATH", vllm_source="VLLM_SOURCE")


def read_config(path):
    result = {}
    if path.exists():
        for line in path.read_text().splitlines():
            parts = shlex.split(line)
            if parts and all("=" in part for part in parts):
                result.update(part.split("=", 1) for part in parts)
            elif "=" in line:
                key, value = line.split("=", 1)
                result[key] = value.strip()
    return result


def canonical(config):
    result = {key: str(config.get(key, value)) for key, value in DEFAULTS.items()}
    if config and "num_warmups" not in config:
        result["num_warmups"] = "0"
    for key in ("model_path", "vllm_source"):
        result[key] = str(Path(result[key]).resolve())
    result["served_model_name"] = config.get("served_model_name", Path(result["model_path"]).name)
    result["gpu_memory_utilization"] = str(float(result["gpu_memory_utilization"]))
    if result["profiling"] == "0":
        for key in result:
            if key.startswith("profile_"):
                result[key] = "0"
    return result


def desired():
    result = {key: os.environ.get(env, DEFAULTS[key]) for key, env in ENV.items()}
    result["served_model_name"] = os.environ.get("SERVED_MODEL_NAME", "MiMo-V2.6-Flash-MOPD")
    return canonical(result)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def rows(root):
    path = root / "runs.tsv"
    if not path.exists():
        return {}
    with path.open() as stream:
        return {(row["batch_tokens"], row["concurrency"]): row for row in csv.DictReader(stream, delimiter="\t")}


def save_rows(root, values):
    temporary = root / f"runs.tsv.tmp-{os.getpid()}"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values[key] for key in sorted(values, key=lambda key: tuple(map(int, key))))
    os.replace(temporary, root / "runs.tsv")


def name(batch, concurrency):
    return f"batch{batch}-concurrency{concurrency}"


def trace_files(root, group):
    directory = root / "traces" / group
    return sorted(list(directory.glob("*.pt.trace.json")) + list(directory.glob("*.pt.trace.json.gz")))


def signatures(root, group, config):
    paths = [root / f"{group}.json"]
    if config["profiling"] == "1":
        paths += trace_files(root, group)
    return {str(path.relative_to(root)): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths}


def complete(root, batch, concurrency, config, cache=True, shallow=False):
    group = name(batch, concurrency)
    try:
        data = json.loads((root / f"{group}.json").read_text())
        count, length, output = (int(config[key]) for key in ("num_prompts", "input_len", "max_output"))
        duration = data["duration"]
        if not (data.get("num_prompts") == count and data.get("completed") == count and data.get("failed") == 0
                and isinstance(duration, (float, int)) and math.isfinite(duration) and duration > 0
                and data.get("total_input_tokens") == count * length and data.get("total_output_tokens") == count * output
                and len(data.get("input_lens", [])) == count and all(n == length for n in data["input_lens"])
                and data.get("max_concurrency") == int(concurrency)):
            return False
        old = rows(root).get((str(batch), str(concurrency)))
        if old and (old["exit_code"] not in ("0", "-") or old["status"] in ("interrupted", "startup_failed")):
            return False
        files = trace_files(root, group) if config["profiling"] == "1" else []
        if config["profiling"] == "1":
            ranks = [re.search(r"(?:^|_)rank(\d+)[._]", path.name) for path in files]
            if len(files) != 4 or any(match is None for match in ranks) or {int(match[1]) for match in ranks} != set(range(4)):
                return False
            log = (root / f"{group}.log").read_text(errors="replace").splitlines()
            if not {"Profiler started", "Profiler stopped"} <= set(log):
                return False
        stamp = {"config": config, "files": signatures(root, group, config)}
        marker = root / ".completed" / f"{group}.json"
        if marker.exists() and json.loads(marker.read_text()) == stamp:
            return True
        if not shallow:
            for path in files:
                if path.name.endswith(".gz"):
                    # Reading to EOF validates gzip footer/CRC once; future resumes use the marker.
                    with gzip.open(path, "rb") as stream:
                        while stream.read(1024 * 1024):
                            pass
                else:
                    json.loads(path.read_text())
            if cache:
                atomic_json(marker, stamp)
        return True
    except (OSError, EOFError, ValueError, KeyError, TypeError, zlib.error):
        return False


def legacy_active(root):
    owner = root / ".sweep-owner" / "pid"
    if owner.exists():
        try:
            cmd = Path(f"/proc/{int(owner.read_text())}/cmdline").read_bytes()
            if b"sweep-mimo.sh" in cmd:
                return True
        except (OSError, ValueError):
            pass
    return False


def active(root):
    if legacy_active(root):
        return True
    lock = root / ".sweep.lock"
    if lock.exists():
        with lock.open("r") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    return False


def archive(root, paths, label):
    paths = [path for path in paths if path.exists()]
    if not paths:
        return
    history = root / "history"
    history.mkdir(exist_ok=True)
    index = 1
    while True:
        destination = history / f"{label}-attempt{index:04d}"
        try:
            destination.mkdir()
            break
        except FileExistsError:
            index += 1
    for path in paths:
        relative = path.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    print(f"归档：{destination}", file=sys.stderr)


def copy_file(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def import_group(source, target, group):
    for path in source.glob(group + ".*"):
        if path.is_file():
            copy_file(path, target / path.name)
    directory = source / "traces" / group
    if directory.exists():
        shutil.copytree(directory, target / "traces" / group, copy_function=copy_file)


def pending_row(batch, concurrency, config, status="pending", rc="-", profile_status=None):
    group = name(batch, concurrency)
    return dict(batch_tokens=str(batch), concurrency=str(concurrency), status=status, exit_code=str(rc), result_file=group + ".json", profiling=config["profiling"], profile_status=profile_status or ("pending" if config["profiling"] == "1" else "disabled"), trace_dir="traces/" + group if config["profiling"] == "1" else "-")


def initialize(root, batches, concurrencies, force, dry, search):
    if legacy_active(root):
        raise ValueError("该目录仍由旧版 sweep 写入，请先等待旧实验结束：" + str(root))
    config = desired()
    previous = read_config(root / "config.txt")
    if previous and canonical(previous) != config and not force:
        changed = [key for key in config if canonical(previous).get(key) != config[key]]
        raise ValueError("已有实验参数不同：" + ", ".join(changed) + "；请改用独立 RESULT_DIR 或设置 FORCE_RERUN=1")
    if force and not dry:
        archive(root, [p for p in root.iterdir() if p.name not in (".sweep.lock", "history")], "forced")
        previous = {}
    if not dry:
        root.mkdir(parents=True, exist_ok=True)
        merged = {**config, "batch_tokens": " ".join(map(str, sorted(set(batches) | set(map(int, previous.get("batch_tokens", "").split()))))), "concurrencies": " ".join(map(str, sorted(set(concurrencies) | set(map(int, previous.get("concurrencies", "").split())))))}
        if not (root / "experiment.json").exists():
            atomic_json(root / "experiment.json", {"format_version": 1, "config": config})
        (root / "config.txt").write_text("\n".join(f"{key}={value}" for key, value in merged.items()) + "\n")
    values = {} if force else rows(root)
    sources = []
    if not force and search.exists():
        for source in search.iterdir():
            if source.resolve() != root.resolve() and (source / "config.txt").exists():
                if canonical(read_config(source / "config.txt")) == config and not active(source):
                    sources.append(source)
    sources.sort(key=lambda p: (p / "config.txt").stat().st_mtime_ns, reverse=True)
    for batch in batches:
        for concurrency in concurrencies:
            group = name(batch, concurrency)
            done = not force and complete(root, batch, concurrency, config, cache=not dry, shallow=dry)
            if not done and not force:
                for source in sources:
                    if complete(source, batch, concurrency, config, cache=False, shallow=dry):
                        print(f"{'可复用' if dry else '复用'}：{group} ← {source.name}" + ("（dry-run 未执行 gzip 完整校验）" if dry else ""), file=sys.stderr)
                        if not dry:
                            archive(root, list(root.glob(group + ".*")) + [root / "traces" / group, root / ".completed" / (group + ".json")], group)
                            import_group(source, root, group)
                            atomic_json(root / ".completed" / (group + ".json"), {"config": config, "files": signatures(root, group, config)})
                        done = True
                        break
            if done:
                values[str(batch), str(concurrency)] = pending_row(batch, concurrency, config, "finished", "0", "captured" if config["profiling"] == "1" else "disabled")
            elif (str(batch), str(concurrency)) not in values:
                values[str(batch), str(concurrency)] = pending_row(batch, concurrency, config)
            print(f"{batch}\t{concurrency}\t{'complete' if done else 'pending'}", flush=True)
    if not dry:
        save_rows(root, values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "prepare-run", "archive-staging", "record"])
    parser.add_argument("root", type=Path)
    parser.add_argument("args", nargs="*")
    options = parser.parse_args()
    root, args = options.root.resolve(), options.args
    if options.action == "init":
        initialize(root, list(map(int, args[0].split())), list(map(int, args[1].split())), args[2] == "1", args[3] == "1", Path(args[4]))
    elif options.action == "archive-staging":
        archive(root, [root / "traces" / (".staging-batch" + args[0])], "staging-batch" + args[0])
    elif options.action == "prepare-run":
        group = name(*args)
        archive(root, list(root.glob(group + ".*")) + [root / "traces" / group, root / ".completed" / (group + ".json")], group)
    elif options.action == "record":
        batch, concurrency, status, rc, profile_status = args
        config = canonical(read_config(root / "config.txt"))
        values = rows(root)
        values[str(batch), str(concurrency)] = pending_row(batch, concurrency, config, status, rc, profile_status)
        save_rows(root, values)
        if status == "finished" and rc == "0":
            done = complete(root, batch, concurrency, config)
            if config["profiling"] == "1" and not done:
                values[str(batch), str(concurrency)]["profile_status"] = "missing_or_partial"
                save_rows(root, values)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(str(error))
