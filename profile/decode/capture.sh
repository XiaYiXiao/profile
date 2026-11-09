#!/usr/bin/env bash
# 0：采集 Nsight / metrics；1：仅运行并保留日志、benchmark JSON 和配置，不需要 nsys。
export LOG_ONLY="${LOG_ONLY:-1}"
# 实验 / 设备：0-3 运行 EP=0，4-7 运行 EP=1；每组 TP=1、DP=4。
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-4dp1tp_decode_mtp}"
export GPU_EP0="${GPU_EP0:-0,1,2,3}" GPU_EP1="${GPU_EP1:-4,5,6,7}"
export TP="${TP:-1}" DP="${DP:-4}"
# ==================== 负载（与 sweep-mimo.sh 一致） ====================
export SEQ_COUNTS="${SEQ_COUNTS:-512 1024 2048 4096}"
# 每组模型只启动一次，max_num_seqs=max(SEQ_COUNTS)；随后按列表顺序发送负载。
export MAX_NUM_BATCHED_TOKENS=8120 # 所有负载档位固定使用此预算
export KV_LEN="${KV_LEN:-10240}" MAX_OUTPUT="${MAX_OUTPUT:-1000}"
export WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}" SEED="${SEED:-42}" # 每档完整预热轮数；0 跳过
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}" GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export KV_STARTUP_FILL="${KV_STARTUP_FILL:-1}" KV_FILL_MEAN="${KV_FILL_MEAN:-0.015}" KV_FILL_STD="${KV_FILL_STD:-0}"
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}" METRICS_INTERVAL="${METRICS_INTERVAL:-1}"
# ==================== 投机推理：none / mtp / dflash ====================
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-mtp}" # dflash mtp none
#mtp : /models/MiMo-V2.6-Flash-MOPD  dflash:/models/MiMo-V2.6-Flash-MOPD/dflash
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}" DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD}"
# Preserve an explicit JSON override or empty value. none must not inherit the
# default synthetic sampling configuration used by this MTP experiment.
if [[ ! -v SPECULATIVE_CONFIG ]]; then
  if [[ "${SPECULATIVE_METHOD,,}" == none ]]; then
    SPECULATIVE_CONFIG=''
  else
    SPECULATIVE_CONFIG='{"rejection_sample_method":"synthetic","synthetic_acceptance_length":4.2}'
  fi
fi
export SPECULATIVE_CONFIG
# ==================== EPLB：只作用于 EP1，其他 EPLB / 后端参数见 settings.sh ====================
export ENABLE_EPLB="${ENABLE_EPLB:-0}"
export EPLB_WINDOW_SIZE="${EPLB_WINDOW_SIZE:-8}" EPLB_STEP_INTERVAL="${EPLB_STEP_INTERVAL:-16}"
export EPLB_NUM_REDUNDANT_EXPERTS="${EPLB_NUM_REDUNDANT_EXPERTS:-0}"
export EPLB_USE_ASYNC="${EPLB_USE_ASYNC:-true}" EPLB_CONFIG="${EPLB_CONFIG:-}"
# ==================== 模型 / Python / Nsight ====================
export MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.6-Flash-MOPD}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
export VLLM_SOURCE="${VLLM_SOURCE:-/host/vllm}" # 使用本地源码
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/settings.sh"
export NSYS_BIN="${NSYS_BIN:-/usr/local/bin/nsys}"
export NSYS_TRACE="${NSYS_TRACE:-cuda-sw,nvtx,nccl,osrt}" NCCL_TRACE="${NCCL_TRACE:-api,coll,p2p,group,gpu}"
export CUDA_GRAPH_TRACE="${CUDA_GRAPH_TRACE:-node}"
export PORT_EP0="${PORT_EP0:-18000}" PORT_EP1="${PORT_EP1:-18001}"
export DP_RPC_PORT_EP0="${DP_RPC_PORT_EP0:-29550}" DP_RPC_PORT_EP1="${DP_RPC_PORT_EP1:-29650}"
export MASTER_PORT_EP0="${MASTER_PORT_EP0:-29501}" MASTER_PORT_EP1="${MASTER_PORT_EP1:-29601}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}" BENCH_TIMEOUT="${BENCH_TIMEOUT:-3600}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$STARTUP_TIMEOUT}"
export DRY_RUN="${DRY_RUN:-0}" SMOKE_ONLY="${SMOKE_ONLY:-0}"
# 不自动绘图或分析；LOG_ONLY=1 时跳过 trace、SQLite 和 metrics。
export RESULT_DIR="${RESULT_DIR:-$script_dir/captures/${EXPERIMENT_NAME}_$(date +%Y%m%d-%H%M%S)-$$}"
exec "$PYTHON_BIN" "$script_dir/collect.py"
