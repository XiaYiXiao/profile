#!/usr/bin/env bash
# 复用容器的 torch 等依赖；EvalScope 安装到独立环境，不修改部署 Python。
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-/usr/bin/python3}"
if [[ ! -x "$script_dir/.venv/bin/python" ]]; then
  "$python_bin" -m venv --system-site-packages "$script_dir/.venv"
fi
"$python_bin" -m pip --python "$script_dir/.venv/bin/python" install -r "$script_dir/requirements-load.txt"
"$script_dir/.venv/bin/python" "$script_dir/per_dataset.py" --help >/dev/null
echo "负载环境就绪：bash $script_dir/run_load.sh --help"
