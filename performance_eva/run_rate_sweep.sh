#!/usr/bin/env bash
# 顺序扫描 pd_6p2d 的请求速率；每个 rate 单独清缓存、采集指标并归档。
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
extra_args=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) extra_args+=(--dry-run) ;;
    -h|--help)
      echo '用法: bash run_rate_sweep.sh [--dry-run]'
      echo '依次运行 rate 2、3、4、6、8、10、14，每轮 300 请求；任一轮失败则停止。'
      echo '--dry-run 仅检查并打印各轮配置，不发请求、不创建结果目录。'
      exit 0 ;;
    *) echo "不支持的参数: $arg" >&2; exit 2 ;;
  esac
done
# 非 DFlash 实验，输出长度为 300。
for rate in 2 3 4 6 8 10 14; do
  echo "开始 rate=${rate}，label=prefix8k_new1k_out300_r${rate}"
  bash "$script_dir/run_load.sh" \
    --mode pd --use-chat \
    --tokenizer-path /models/MiMo-V2.6-Flash-MOPD \
    --prepared-dataset-path '' \
    --prefix-length 8000 --input-length 1000 \
    --output-length 300 \
    --number 300 --parallel 512 --rate "$rate" \
    --warmup-num 0 \
    --metrics-interval 1 \
    --reset-prefix-cache \
    --label "prefix8k_new1k_out300_r${rate}" \
    "${extra_args[@]}"
done
