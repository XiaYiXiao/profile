#!/usr/bin/env bash
set -Eeuo pipefail

# ==================== 常用调参区 ====================
# 可直接修改默认值，也可用环境变量覆盖，例如：TP=2 DP=2 EP=1 bash profiling_mimo/mimo.sh
# 仅支持单节点 4 卡：TP * DP 必须等于 4，PP 固定为 1。
TP="${TP:-4}"
DP="${DP:-1}"
# EP=0 关闭，EP=1 开启；vLLM 的 EP 大小由 TP * DP 决定，不额外乘卡数。
EP="${EP:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_OUTPUT="${MAX_OUTPUT:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.6-Flash-MOPD}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
VLLM_SOURCE="${VLLM_SOURCE:-/host/vllm}"
LOG_DIR="${LOG_DIR:-/host/logs}"
# 可选 Perfetto 时间线：PyTorch Profiler 导出 Chrome Trace。
PROFILE="${PROFILE:-0}"
PROFILE_DIR="${PROFILE_DIR:-$LOG_DIR/traces-$(date +%Y%m%d-%H%M%S)}"
PROFILE_MAX_ITERS="${PROFILE_MAX_ITERS:-0}" # 0 表示采集至 /stop_profile
PROFILE_DELAY_ITERS="${PROFILE_DELAY_ITERS:-0}"
PROFILE_WITH_STACK="${PROFILE_WITH_STACK:-0}"
PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-0}"
PROFILE_WITH_MEMORY="${PROFILE_WITH_MEMORY:-0}"
# ====================================================

fail() {
  echo "错误：$*" >&2
  exit 1
}

for name in TP DP MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_OUTPUT MAX_MODEL_LEN; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || fail "$name 必须为正整数"
done
[[ "$TP" =~ ^(1|2|4)$ && "$DP" =~ ^(1|2|4)$ ]] || fail "TP、DP 只能为 1、2、4"
(( TP * DP == 4 )) || fail "只允许 4 卡部署，必须满足 TP * DP = 4"
[[ "$EP" == 0 || "$EP" == 1 ]] || fail "EP 只能为 0（关闭）或 1（开启）"
[[ "$ENABLE_PREFIX_CACHING" == 0 || "$ENABLE_PREFIX_CACHING" == 1 ]] || fail "ENABLE_PREFIX_CACHING 只能为 0 或 1"
for name in PROFILE PROFILE_WITH_STACK PROFILE_RECORD_SHAPES PROFILE_WITH_MEMORY; do
  [[ "${!name}" == 0 || "${!name}" == 1 ]] || fail "$name 只能为 0 或 1"
done
for name in PROFILE_MAX_ITERS PROFILE_DELAY_ITERS; do
  [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]] || fail "$name 必须为非负整数"
done
(( MAX_OUTPUT <= MAX_MODEL_LEN )) || fail "MAX_OUTPUT 不能超过 MAX_MODEL_LEN"
(( MAX_NUM_BATCHED_TOKENS >= MAX_NUM_SEQS )) || fail "MAX_NUM_BATCHED_TOKENS 不能小于 MAX_NUM_SEQS"

# 拒绝少于/多于 4 张卡、空设备项以及重复设备。
[[ "$CUDA_VISIBLE_DEVICES" != ,* && "$CUDA_VISIBLE_DEVICES" != *, ]] || fail "GPU 列表不能有空设备项"
IFS=',' read -r -a gpu_devices <<< "$CUDA_VISIBLE_DEVICES"
(( ${#gpu_devices[@]} == 4 )) || fail "CUDA_VISIBLE_DEVICES 必须指定且仅指定 4 张 GPU"
declare -A seen_devices=()
for device in "${gpu_devices[@]}"; do
  [[ -n "$device" && "$device" != *[[:space:]]* ]] || fail "GPU 设备项不能为空或包含空白"
  [[ -z "${seen_devices[$device]:-}" ]] || fail "GPU 设备重复：$device"
  seen_devices[$device]=1
done

# 沿用 mimo.sh 的本地源码、CUDA 和 Python 前端环境。
export PYTHONPATH="$VLLM_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_RUST_FRONTEND=0
export VLLM_USE_RUST_BENCH=0
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# MiMo 原生上限为 1048576；保留模板允许显式设置服务上限的行为。
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

[[ -d "$VLLM_SOURCE" ]] || fail "vLLM 源码目录不存在：$VLLM_SOURCE"
[[ -f "$MODEL_PATH/config.json" ]] || fail "缺少 $MODEL_PATH/config.json，请等待模型下载完成"
cd "$VLLM_SOURCE"

mkdir -p "$LOG_DIR"
LOG="${LOG_FILE:-$LOG_DIR/MiMo-V2.6-Flash-MOPD-tp${TP}-dp${DP}-ep${EP}-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo "日志：$LOG"
echo "GPU=$CUDA_VISIBLE_DEVICES TP=$TP DP=$DP EP=$EP"
echo "max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS max_output=$MAX_OUTPUT max_model_len=$MAX_MODEL_LEN"

# 检查实际可见卡数；仅导入并查询设备，不加载模型。
"$PYTHON_BIN" -c '
import sys, torch, vllm
print("Python:", sys.executable)
print("vLLM:", vllm.__file__)
count = torch.cuda.device_count()
if count != 4:
    sys.exit(f"错误：实际可见 GPU 数为 {count}，仅允许 4 卡部署")
'

ep_args=()
if [[ "$EP" == 1 ]]; then
  ep_args+=(--enable-expert-parallel)
fi
cache_args=(--enable-prefix-caching)
if [[ "$ENABLE_PREFIX_CACHING" == 0 ]]; then
  cache_args=(--no-enable-prefix-caching)
fi
profiler_args=()
if [[ "$PROFILE" == 1 ]]; then
  mkdir -p "$PROFILE_DIR"
  PROFILE_DIR="$(cd -- "$PROFILE_DIR" && pwd)"
  stack=false; shapes=false; memory=false
  if [[ "$PROFILE_WITH_STACK" == 1 ]]; then stack=true; fi
  if [[ "$PROFILE_RECORD_SHAPES" == 1 ]]; then shapes=true; fi
  if [[ "$PROFILE_WITH_MEMORY" == 1 ]]; then memory=true; fi
  profiler_args=(
    --profiler-config.profiler torch
    --profiler-config.torch_profiler_dir "$PROFILE_DIR"
    --profiler-config.torch_profiler_activities '["CPU","CUDA"]'
    --profiler-config.torch_profiler_use_gzip true
    --profiler-config.torch_profiler_with_stack "$stack"
    --profiler-config.torch_profiler_record_shapes "$shapes"
    --profiler-config.torch_profiler_with_memory "$memory"
    --profiler-config.ignore_frontend true
    --profiler-config.detailed_trace_annotation true
    --profiler-config.max_iterations "$PROFILE_MAX_ITERS"
    --profiler-config.delay_iterations "$PROFILE_DELAY_ITERS"
  )
  echo "Profiler：$PROFILE_DIR（max_iterations=$PROFILE_MAX_ITERS，delay_iterations=$PROFILE_DELAY_ITERS）"
fi

# max_new_tokens 是服务端输出 token 上限（包含推理 token）。
exec "$PYTHON_BIN" -m vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --data-parallel-size-local "$DP" \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --data-parallel-backend mp \
  "${ep_args[@]}" \
  "${cache_args[@]}" \
  "${profiler_args[@]}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --enable-chunked-prefill \
  --generation-config vllm \
  --override-generation-config "{\"max_new_tokens\":$MAX_OUTPUT}"
