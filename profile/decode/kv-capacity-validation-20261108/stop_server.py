"""Stop only the saved process group whose parent matches this experiment."""
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
saved = json.loads((DATA / "server-process.json").read_text())
pid = saved["pid"]
proc_path = Path(f"/proc/{pid}")
if proc_path.exists():
    cmdline = (proc_path / "cmdline").read_bytes().split(b"\0")
    expected = str(ROOT / ".venv/bin/python").encode()
    assert cmdline[0] == expected and b"18180" in cmdline, cmdline
    assert os.getpgid(pid) == saved["process_group"] == pid
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic()+30
    while proc_path.exists() and time.monotonic() < deadline:
        # A reaped engine parent can briefly remain as a zombie under PID 1.
        if ") Z " in (proc_path / "stat").read_text():
            break
        time.sleep(0.2)
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) == pid:
                members.append(int(entry.name))
        except ProcessLookupError:
            pass
    if members:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
else:
    members = []
(DATA / "shutdown.json").write_text(json.dumps(dict(
    parent_pid=pid, process_group=pid, remaining_group_members_before_final_signal=members,
    time_unix_s=time.time()), indent=2))
print(f"Stopped experiment process group {pid}")
