#!/usr/bin/env bash
# 独立负载入口：--help 查看连接/归档参数，--perf-help 查看全部负载参数。
# LOAD_PYTHON_BIN 与部署 PYTHON_BIN 分开，便于使用独立 EvalScope 环境。
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${LOAD_PYTHON_BIN:-}" ]]; then
  if [[ -x "$script_dir/.venv/bin/python" ]]; then
    LOAD_PYTHON_BIN="$script_dir/.venv/bin/python"
  else
    LOAD_PYTHON_BIN=python3
  fi
fi
exec "$LOAD_PYTHON_BIN" "$script_dir/code/load.py" "$@"
