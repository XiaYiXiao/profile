#!/usr/bin/env python3
"""Decode sweep configuration, completion checks, and resume state (stdlib)."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys

FIELDS = ["max_num_seqs", "batch_tokens", "concurrency", "num_prompts", "status", "exit_code", "result_file"]
ENV_DEFAULTS = {
    "TP": ("TP", "1"), "DP": ("DP", "4"), "EP": ("EP", "0"),
    "GPU": ("CUDA_VISIBLE_DEVICES", "4,5,6,7"),
    "batch_token_extra": ("BATCH_TOKEN_EXTRA", "48"), "kv_len": ("KV_LEN", "8192"),
    "max_output": ("MAX_OUTPUT", "500"), "warmup_rounds": ("WARMUP_ROUNDS", "2"),
    "max_model_len": ("MAX_MODEL_LEN", "1000000"), "seed": ("SEED", "42"),
    "kv_startup_fill": ("KV_STARTUP_FILL", "1"), "kv_fill_mean": ("KV_FILL_MEAN", "0.015"),
    "kv_fill_std": ("KV_FILL_STD", "0"), "cudagraph_mode": ("CUDAGRAPH_MODE", "FULL_DECODE_ONLY"),
    "gpu_memory_utilization": ("GPU_MEMORY_UTILIZATION", "0.90"),
    "model_path": ("MODEL_PATH", "/models/MiMo-V2.6-Flash-MOPD"),
    "vllm_source": ("VLLM_SOURCE", "/host/vllm"),
    "served_model_name": ("SERVED_MODEL_NAME", "MiMo-V2.6-Flash-MOPD"),
    "metrics_interval": ("METRICS_INTERVAL", "1"),
}


def batch_tokens(seq, config):
    # Preserve read-only support for historical seqs + EXTRA results.
    return int(config['max_num_batched_tokens']) if config.get('max_num_batched_tokens') else seq + int(config['batch_token_extra'])


def desired():
    result = {key: os.environ.get(env, default) for key, (env, default) in ENV_DEFAULTS.items()}
    result['max_num_batched_tokens'] = '8120'
    result['server_max_num_seqs'] = os.environ.get('MAX_NUM_SEQS', '768')
    result['server_lifecycle'] = 'single_start_per_invocation'
    result['metrics_scope'] = 'formal_only'
    for env in ('SPECULATIVE_METHOD', 'NUM_SPECULATIVE_TOKENS', 'DRAFT_MODEL_PATH', 'SPECULATIVE_CONFIG', 'ENABLE_EPLB', 'EPLB_CONFIG', 'MOE_BACKEND', 'ATTENTION_BACKEND', 'ALL2ALL_BACKEND', 'EXPERT_PLACEMENT_STRATEGY', 'EPLB_WINDOW_SIZE', 'EPLB_STEP_INTERVAL', 'EPLB_NUM_REDUNDANT_EXPERTS', 'EPLB_USE_ASYNC', 'EPLB_COMMUNICATOR', 'EPLB_LOG_BALANCEDNESS', 'EPLB_LOG_BALANCEDNESS_INTERVAL', 'PYTHON_BIN'):
        result[env.lower()] = os.environ.get(env, '')
    for key in ("model_path", "vllm_source"):
        if result[key]:
            result[key] = str(Path(result[key]).resolve())
    for key in ("kv_fill_mean", "kv_fill_std", "gpu_memory_utilization", "metrics_interval"):
        result[key] = str(float(result[key]))
    result.update(benchmark_mode="decode", connector="DecodeBenchConnector", dtype="bfloat16",
                  kv_cache_dtype="auto", prefix_caching="0", input_len=str(int(result["kv_len"]) + 1))
    return result


def read_config(root):
    result = {}
    for line in (Path(root) / "config.txt").read_text().splitlines():
        for part in shlex.split(line):
            key, value = part.split("=", 1)
            result[key] = value
    return result


def atomic_text(path, content):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(content)
    os.replace(temporary, path)


def rows(root):
    path = root / "runs.tsv"
    if not path.exists():
        return {}
    with path.open() as stream:
        return {int(row["concurrency"]): row for row in csv.DictReader(stream, delimiter="\t")}


def save_rows(root, values):
    path = root / "runs.tsv"
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n", delimiter="\t")
        writer.writeheader()
        writer.writerows(values[seq] for seq in sorted(values))
    os.replace(temporary, path)


def name(seq, config):
    return f"batch{batch_tokens(seq, config)}-concurrency{seq}"


def make_row(seq, config, status="pending", rc="-"):
    return dict(max_num_seqs=int(config.get('server_max_num_seqs', seq)), batch_tokens=batch_tokens(seq, config),
                concurrency=seq, num_prompts=seq, status=status, exit_code=str(rc),
                result_file=name(seq, config) + ".json")


def valid_result(root, seq, config, filename=None):
    """Check actual lengths and latency metrics, never just file existence."""
    data = json.loads((root / (filename or name(seq, config) + ".json")).read_text())
    output, input_len = int(config["max_output"]), int(config["input_len"])
    for key, expected in (("num_prompts", seq), ("completed", seq), ("failed", 0),
                          ("max_concurrency", seq), ("total_input_tokens", seq * input_len),
                          ("total_output_tokens", seq * output)):
        if data.get(key) != expected:
            raise ValueError(f"{key}: expected {expected}, got {data.get(key)!r}")
    for key, expected in (("input_lens", input_len), ("output_lens", output)):
        lengths = data.get(key, [])
        if len(lengths) != seq or any(length != expected for length in lengths):
            raise ValueError(f"invalid {key}")
    duration = data.get("duration", 0)
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid duration")
    for metric in ("tpot", "itl"):
        for prefix in ("mean", "p50", "p90", "p95", "p99"):
            value = data.get(f"{prefix}_{metric}_ms")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {prefix}_{metric}_ms")
    return data


def complete(root, seq, config):
    try:
        old = rows(root).get(seq)
        if old and (old["status"] != "finished" or old["exit_code"] != "0"):
            return False
        valid_result(root, seq, config)
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def archive(root, paths, label):
    paths = [path for path in paths if path.exists()]
    if not paths:
        return
    history = root / "history"
    history.mkdir(exist_ok=True)
    number = 1
    while True:
        target = history / f"{label}-attempt{number:04d}"
        try:
            target.mkdir()
            break
        except FileExistsError:
            number += 1
    for path in paths:
        destination = target / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
    print(f"归档：{target}", file=sys.stderr)


def initialize(root, seqs, force, dry):
    config = desired()
    config['load_order'] = ' '.join(map(str, seqs))
    old = read_config(root) if (root / "config.txt").exists() else {}
    mismatch = [key for key, value in config.items() if old.get(key) != value] if old else []
    unknown = root.exists() and not old and any(p.name not in (".sweep.lock", "history") for p in root.iterdir())
    if (mismatch or unknown) and not force:
        raise SystemExit("RESULT_DIR 配置不兼容或缺少配置；请换目录，或 FORCE_RERUN=1 归档重跑。差异：" + ", ".join(mismatch))
    if force:
        print("强制重跑：已有结果将归档。" if not dry else "dry-run：强制重跑计划，不归档文件。", file=sys.stderr)
        if not dry and root.exists():
            archive(root, [p for p in root.iterdir() if p.name not in (".sweep.lock", "history")], "forced")
        old = {}
    values = rows(root) if old else {}
    for seq in seqs:
        done = bool(old) and complete(root, seq, config)
        print(f"{seq}\t{'complete' if done else 'pending'}")
        values[seq] = make_row(seq, config, "finished" if done else "pending", "0" if done else "-")
    if not dry:
        root.mkdir(parents=True, exist_ok=True)
        config["seq_counts"] = " ".join(map(str, sorted(values)))
        atomic_text(root / "config.txt", "".join(f"{key}={shlex.quote(value)}\n" for key, value in config.items()))
        save_rows(root, values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "prepare-run", "record", "validate"))
    parser.add_argument("root", type=Path)
    parser.add_argument("args", nargs="+")
    args = parser.parse_args()
    root = args.root
    if args.action == "init":
        seqs, force, dry = args.args
        initialize(root, list(map(int, seqs.split())), force == "1", dry == "1")
    elif args.action == "validate":
        seq = int(args.args[0])
        filename = args.args[1] if len(args.args) > 1 else None
        data = valid_result(root, seq, read_config(root), filename)
        label = f" {Path(filename).stem}" if filename else ""
        print(f"seqs={seq}{label}: output_tokens/s={data['total_output_tokens'] / data['duration']:.3f}", flush=True)
    else:
        seq = int(args.args[0])
        config = read_config(root)
        group = name(seq, config)
        if args.action == "prepare-run":
            files = list(root.glob(group + ".*")) + [root / f"server-{group}.log", root / "server-logs" / (group + ".log")]
            archive(root, files, group)
        else:
            status, rc = args.args[1:]
            values = rows(root)
            values[seq] = make_row(seq, config, status, rc)
            save_rows(root, values)


if __name__ == "__main__":
    main()
