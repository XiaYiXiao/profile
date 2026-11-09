"""Apply the baseline-checked EPLB backport to the local vLLM source checkout.

Default: read-only check. --apply: backup and install. --restore: checked rollback.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_contents(target, data):
    """Replace one file atomically, preserving its existing mode and ownership."""
    stat = target.stat()
    fd, name = tempfile.mkstemp(prefix=".eplb-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
            os.fchmod(output.fileno(), stat.st_mode & 0o7777)
            current = os.fstat(output.fileno())
            if (current.st_uid, current.st_gid) != (stat.st_uid, stat.st_gid):
                os.fchown(output.fileno(), stat.st_uid, stat.st_gid)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def inspect(package, bundle, manifest):
    pending = []
    rejected = []
    for item in manifest["files"]:
        source = bundle / "files" / item["path"]
        if not source.is_file() or digest(source) != item["after_sha256"]:
            raise ValueError(f"安装包文件校验失败：{source}")
        target = package / item["path"]
        actual = digest(target) if target.is_file() else None
        if actual == item["after_sha256"]:
            status = "APPLIED"
        elif actual == item["before_sha256"]:
            status = "BASE"
            pending.append(item)
        else:
            status = "MISSING" if actual is None else "DIFFERENT"
            rejected.append(item["path"])
        print(f"{status:9} {target}")
        if status in {"MISSING", "DIFFERENT"}:
            print(f"          current_sha256={actual}")
    if rejected:
        raise ValueError(
            "目标含非官方基线内容；未修改任何文件。请提供上面的差异列表和对应源文件以便合并。"
        )
    return pending


def apply(package, bundle, manifest, backup):
    pending = inspect(package, bundle, manifest)
    if not pending:
        print("四个文件均已接入，无需安装。")
        return
    backup.mkdir(parents=True, exist_ok=False)
    for item in pending:
        destination = backup / "original" / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / item["path"], destination)
        if digest(destination) != item["before_sha256"]:
            raise ValueError("备份期间目标发生变化；未开始安装。")
    (backup / "restore.json").write_text(
        json.dumps(
            {
                "package": str(package.resolve()),
                "files": pending,
                "python": sys.executable,
                "version": manifest["version"],
            },
            indent=2,
        )
        + "\n"
    )
    written = []
    try:
        for item in pending:
            target = package / item["path"]
            if digest(target) != item["before_sha256"]:
                raise ValueError(f"安装期间目标发生变化：{target}")
            replace_contents(target, (bundle / "files" / item["path"]).read_bytes())
            written.append(item)
            if digest(target) != item["after_sha256"]:
                raise ValueError(f"安装后校验失败：{target}")
    except Exception:
        for item in reversed(written):
            replace_contents(
                package / item["path"],
                (backup / "original" / item["path"]).read_bytes(),
            )
        raise
    print(f"安装完成；原文件备份：{backup}")
    print("请重启 vLLM 服务，并添加 --enable-expert-parallel --enable-eplb。")


def restore(package, backup):
    plan = json.loads((backup / "restore.json").read_text())
    if str(package.resolve()) != plan["package"]:
        raise ValueError("此 Python 导入的 vLLM 路径与备份不一致；拒绝回滚。")
    pending = []
    for item in plan["files"]:
        original = backup / "original" / item["path"]
        if digest(original) != item["before_sha256"]:
            raise ValueError(f"备份校验失败：{original}")
        actual = digest(package / item["path"])
        if actual == item["before_sha256"]:
            continue
        if actual != item["after_sha256"]:
            raise ValueError(f"文件存在后续修改，拒绝覆盖：{item['path']}")
        pending.append(item)
    for item in pending:
        target = package / item["path"]
        replace_contents(target, (backup / "original" / item["path"]).read_bytes())
        assert digest(target) == item["before_sha256"]
    print(f"已还原 {len(pending)} 个文件；服务需重启。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--restore", type=Path, metavar="BACKUP_DIR")
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    if args.backup_dir and not args.apply:
        parser.error("--backup-dir 仅与 --apply 一起使用")
    source = Path(os.environ.get("VLLM_SOURCE") or "/host/vllm").resolve()
    if not (source / "vllm" / "__init__.py").is_file():
        raise ValueError(f"vLLM 源码目录无效：{source}")
    sys.path.insert(0, str(source))
    previous = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(source) + (os.pathsep + previous if previous else "")
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        raise ValueError("此 Python 找不到 vLLM；请使用运行服务的同一 Python。")
    package = Path(spec.origin).resolve().parent
    import vllm
    version = vllm.__version__
    print(f"Python: {sys.executable}\nvLLM: {version}\nPackage: {package}")
    if args.restore:
        restore(package, args.restore.resolve())
        return
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / "manifest.json").read_text())
    if version.split("+", 1)[0] != manifest["version"]:
        raise ValueError("补丁与目标 vLLM 版本不匹配；未修改目标。")
    if args.apply:
        backup = args.backup_dir or Path.cwd() / (
            "vllm-eplb-backup-"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
            + f"-{os.getpid()}"
        )
        apply(package, bundle, manifest, backup.resolve())
    else:
        inspect(package, bundle, manifest)
        print("只读检查完成。停止使用该环境的服务后，使用 --apply 安装。")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        sys.exit(f"错误：{error}")
