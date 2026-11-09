#!/usr/bin/env bash
# 非 PD 分离：8 GPU，DP2 × TP4；默认开启 EP8，ENABLE_EP=0 关闭 EP。
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PROFILE_MODE=aggre
export ENABLE_EP="${ENABLE_EP:-1}" # 1=开启 EP，0=关闭；不改变 TP/DP 拓扑
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-aggre_dp2_tp4_enable_ep${ENABLE_EP}}"

# ==================== 设备 / 并行 / 调度 ====================
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}" TP="${TP:-4}" DP="${DP:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export KV_CACHE_LAYOUT="${KV_CACHE_LAYOUT:-LBHNC}" KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-16}"
export PORT="${PORT:-18000}" DP_RPC_PORT="${DP_RPC_PORT:-29550}" MASTER_PORT="${MASTER_PORT:-29501}"

# ==================== 负载：每档 50 × (50K 输入 + 500 输出) ====================
export CONCURRENCIES="${CONCURRENCIES:-2 4 8 16}"
export NUM_PROMPTS="${NUM_PROMPTS:-50}" INPUT_LEN="${INPUT_LEN:-50000}" OUTPUT_LEN="${OUTPUT_LEN:-500}"
export WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}" REPEATS="${REPEATS:-1}" SEED="${SEED:-42}"
# 全档完整预热结束后再评估；关闭 prefix caching，ignore_eos 保证输出长度。

# ==================== EP 后端 / EPLB（EPLB 默认关闭，要求 ENABLE_EP=1） ====================
export MOE_BACKEND="${MOE_BACKEND:-}" ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
export ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-allgather_reducescatter}"
export EXPERT_PLACEMENT_STRATEGY="${EXPERT_PLACEMENT_STRATEGY:-linear}"
export ENABLE_EPLB="${ENABLE_EPLB:-0}"
export EPLB_WINDOW_SIZE="${EPLB_WINDOW_SIZE:-8}" EPLB_STEP_INTERVAL="${EPLB_STEP_INTERVAL:-16}"
export EPLB_NUM_REDUNDANT_EXPERTS="${EPLB_NUM_REDUNDANT_EXPERTS:-0}"
export EPLB_USE_ASYNC="${EPLB_USE_ASYNC:-true}" EPLB_CONFIG="${EPLB_CONFIG:-}"

# ==================== Nsight Trace（默认关闭） ====================
export ENABLE_NSYS="${ENABLE_NSYS:-0}" NSYS_BIN="${NSYS_BIN:-/usr/local/bin/nsys}"
export NSYS_TRACE="${NSYS_TRACE:-cuda-sw,nvtx,nccl,osrt}" NCCL_TRACE="${NCCL_TRACE:-api,coll,p2p,group,gpu}"
export CUDA_GRAPH_TRACE="${CUDA_GRAPH_TRACE:-node}"

# ==================== 模型 / Python / 超时 ====================
export MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.6-Flash-MOPD}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}" VLLM_SOURCE="${VLLM_SOURCE:-/host/vllm}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}" BENCH_TIMEOUT="${BENCH_TIMEOUT:-7200}"
export DRY_RUN="${DRY_RUN:-0}" PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

# ==================== 文件管理：所有运行产物集中到 aggre_or_not ====================
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$script_dir}"
export RESULT_DIR="${RESULT_DIR:-$OUTPUT_ROOT/aggre/${EXPERIMENT_NAME}_$(date +%Y%m%d-%H%M%S)-$$}"
exec "$PYTHON_BIN" "$script_dir/code/collect_pd.py"
