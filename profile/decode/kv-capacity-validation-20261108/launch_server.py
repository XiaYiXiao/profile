"""Start only this experiment's server, using captured baseline settings."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/host/vllm")
DATA = ROOT / "data"


def snapshot():
    return {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SOURCE.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".sh", ".cpp", ".cu", ".h", ".hpp", ".toml")
            and not any(x in p.parts for x in (".git", ".venv", "__pycache__"))}


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    current = snapshot()
    before = DATA / "source-sha256-before.json"
    if "--verify-source" in sys.argv:
        expected = json.loads(before.read_text())
        changed = sorted(k for k in expected.keys() | current.keys()
                         if expected.get(k) != current.get(k))
        result = dict(checked_files=len(current), changed_files=changed, time_unix_s=time.time())
        (DATA / "source-integrity.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        sys.exit(bool(changed))
    if before.exists():
        raise SystemExit("Refusing to overwrite existing experiment; use existing server or a new directory.")
    before.write_text(json.dumps(current, indent=2))
    capture = ROOT.parent / "captures/4dp1tp_decode_base_20261106-121827-18792/config.json"
    config = json.loads(capture.read_text())
    command = config["servers"][0]["command"][2:]
    command[0] = sys.executable
    for key, value in (("--port", "18180"), ("--data-parallel-rpc-port", "29750"),
                       ("--master-port", "29701")):
        command[command.index(key)+1] = value
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", VLLM_USE_RUST_FRONTEND="0", VLLM_USE_RUST_BENCH="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_ALLOW_LONG_MAX_MODEL_LEN="1",
               PYTHONPATH=f"{ROOT / 'hooks'}:{SOURCE}",
               KV_PROBE_DIR=str(DATA / "engine"), NO_PROXY="127.0.0.1,localhost",
               no_proxy="127.0.0.1,localhost", CUDA_HOME="/usr/local/cuda")
    env["PATH"] = "/usr/local/cuda/bin:" + env["PATH"]
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    (DATA / "server.command.txt").write_text(shlex.join(command)+"\n")
    (DATA / "server.environment.json").write_text(json.dumps(
        {k:env[k] for k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "KV_PROBE_DIR", "PYTHONDONTWRITEBYTECODE")}, indent=2))
    with (DATA / "server.log").open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (DATA / "server-process.json").write_text(json.dumps(
        dict(pid=proc.pid, process_group=proc.pid, launched_unix_s=time.time(), command=command), indent=2))
    print(f"Launched isolated server pid={proc.pid}; port=18180; source files hashed={len(current)}")
