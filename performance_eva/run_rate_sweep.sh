#!/usr/bin/env bash
# 顺序扫描 PD 的请求速率；共享已保存的请求，每个 rate 单独清缓存、采集指标并归档。
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
prefix_length=72000
input_length=8000
output_length=1000
number=3000
parallel=512
seed=42
rates=(2 4 6 8 10 12 14 16)
tokenizer_path=/mnt/pfs/models/MiMo-V2.6-Flash-MOPD
dry_run=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    -h|--help)
      echo '用法: bash run_rate_sweep.sh [--dry-run]'
      echo "依次运行 rate ${rates[*]}，每轮 ${number} 请求、parallel=${parallel}。"
      echo '部分请求失败仅告警并继续；整轮无成功请求或运行出错则停止。'
      echo "共享前缀 ${prefix_length} tokens，后缀 ${input_length} tokens，输出 ${output_length} tokens。"
      echo '首次生成并保存数据集；后续各轮和再次运行均读取同一文件。'
      echo '读取阶段跳过客户端分词和长度检查，token 数采用服务端 usage。'
      echo '--dry-run 仅检查并打印准备及各轮配置，不生成数据集、不发请求、不创建结果目录。'
      exit 0 ;;
    *) echo "不支持的参数: $arg" >&2; exit 2 ;;
  esac
done

dataset_path="$script_dir/datasets/MiMo-V2.6-Flash-MOPD/prefix${prefix_length}_input${input_length}_n${number}_seed${seed}.txt"
dataset_args=(
  --prepared-dataset-path "$dataset_path"
  --prefix-length "$prefix_length" --input-length "$input_length"
  --seed "$seed"
)

run_load() {
  if [[ "$dry_run" -eq 1 ]]; then
    bash "$script_dir/run_load.sh" "$@" --dry-run
  else
    bash "$script_dir/run_load.sh" "$@"
  fi
}

check_dataset_file() {
  if [[ ! -f "$dataset_path" || ! -r "$dataset_path" || ! -s "$dataset_path" ]]; then
    echo "数据集不是可读的非空文件，停止扫描: $dataset_path" >&2
    exit 2
  fi
}

check_dataset() {
  check_dataset_file
  local count
  count="$(awk 'NF {n++} END {print n+0}' "$dataset_path")"
  if [[ "$count" -ne "$number" ]]; then
    echo "数据集包含 ${count} 条非空请求，预期 ${number} 条，停止扫描: $dataset_path" >&2
    exit 2
  fi
}

if [[ -e "$dataset_path" || -L "$dataset_path" ]]; then
  echo "复用已保存的数据集: $dataset_path"
else
  echo "首次生成并保存 ${number} 条请求: $dataset_path"
  run_load \
    --prepare-dataset --prepare-num "$number" \
    --tokenizer-path "$tokenizer_path" \
    "${dataset_args[@]}"
fi

if [[ "$dry_run" -eq 0 || -e "$dataset_path" || -L "$dataset_path" ]]; then
  # Count once per invocation; do not rescan the entire large file for every rate.
  check_dataset
fi

for rate in "${rates[@]}"; do
  # 防止文件缺失时 per_dataset.py 静默回退到随机生成。
  if [[ "$dry_run" -eq 0 ]]; then
    check_dataset_file
  fi
  echo "开始 rate=${rate}，label=prefix72k_new8k_out1000_r${rate}"
  run_load \
    --mode pd --use-chat \
    --tokenizer-path '' \
    "${dataset_args[@]}" \
    --output-length "$output_length" \
    --number "$number" --parallel "$parallel" --rate "$rate" \
    --warmup-num 0 \
    --allow-failed-requests \
    --metrics-interval 1 \
    --reset-prefix-cache \
    --label "prefix72k_new8k_out1000_r${rate}"
done
