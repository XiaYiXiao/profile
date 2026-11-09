#!/usr/bin/env bash
# PD 分离：GPU、DP、TP 和 EP 按下面的配置设置。
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PROFILE_MODE=pd
export ENABLE_EP="${ENABLE_EP:-0}" # 1=开启 P/D 的 EP，0=同时关闭 P/D 的 EP

# ==================== 设备 / 并行 / 调度 ====================
export P_GPUS="${P_GPUS:-0,1,2,3}" P_TP="${P_TP:-2}" P_DP="${P_DP:-2}"
export D_GPUS="${D_GPUS:-4,5,6,7}" D_TP="${D_TP:-1}" D_DP="${D_DP:-4}"
# 在设备与并行参数确定后生成名称；仍可通过 EXPERIMENT_NAME 手动覆盖。
p_gpu_ids="${P_GPUS//[[:space:]]/}"
d_gpu_ids="${D_GPUS//[[:space:]]/}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-pd_p_dp${P_DP}_tp${P_TP}_gpus${p_gpu_ids//,/-}_d_dp${D_DP}_tp${D_TP}_gpus${d_gpu_ids//,/-}_enable_ep${ENABLE_EP}}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# P、D 共用 token budget，默认 16384，与非分离组一致。
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export P_PORT="${P_PORT:-18100}" D_PORT="${D_PORT:-18200}" PROXY_PORT="${PROXY_PORT:-18000}"
export P_DP_RPC_PORT="${P_DP_RPC_PORT:-29750}" P_MASTER_PORT="${P_MASTER_PORT:-29701}"
export D_DP_RPC_PORT="${D_DP_RPC_PORT:-29850}" D_MASTER_PORT="${D_MASTER_PORT:-29801}"

# ==================== KV 传输：NIXL pull，使用上面的 P/D 并行配置 ====================
export P_NIXL_PORT="${P_NIXL_PORT:-5557}" D_NIXL_PORT="${D_NIXL_PORT:-5657}"
# D 的 side-channel 占用 D_NIXL_PORT + dp_rank；P/D 使用同一 KV layout / dtype。
export KV_CACHE_LAYOUT="${KV_CACHE_LAYOUT:-LBHNC}" KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-16}"
export KV_EXTRA_CONFIG="${KV_EXTRA_CONFIG:-}" # 可选 JSON，例如 {"backends":["UCX"]}
export UCX_TLS="${UCX_TLS:-all}" UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"

# ==================== 负载：每档 50 × (50K 输入 + 500 输出) ====================
export CONCURRENCIES="${CONCURRENCIES:-2 4 8 16 32}"
export NUM_PROMPTS="${NUM_PROMPTS:-50}" INPUT_LEN="${INPUT_LEN:-50000}" OUTPUT_LEN="${OUTPUT_LEN:-500}"
export WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}" REPEATS="${REPEATS:-1}" SEED="${SEED:-42}"
# 请求经代理先 P 后 D，端到端 TTFT 包含 P 排队、计算及 KV 传输。

# ==================== EP 后端 / EPLB（EPLB 默认关闭，要求 ENABLE_EP=1） ====================
export MOE_BACKEND="${MOE_BACKEND:-}" ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
export ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-allgather_reducescatter}"
export EXPERT_PLACEMENT_STRATEGY="${EXPERT_PLACEMENT_STRATEGY:-linear}"
export ENABLE_EPLB="${ENABLE_EPLB:-0}"
export EPLB_WINDOW_SIZE="${EPLB_WINDOW_SIZE:-8}" EPLB_STEP_INTERVAL="${EPLB_STEP_INTERVAL:-16}"
export EPLB_NUM_REDUNDANT_EXPERTS="${EPLB_NUM_REDUNDANT_EXPERTS:-0}"
export EPLB_USE_ASYNC="${EPLB_USE_ASYNC:-true}" EPLB_CONFIG="${EPLB_CONFIG:-}"

# ==================== Nsight Trace（默认关闭，P/D 分开保存） ====================
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
export DEPS_DIR="${DEPS_DIR:-$OUTPUT_ROOT/.deps}" # 可选的隔离 NIXL 安装目录
export RESULT_DIR="${RESULT_DIR:-$OUTPUT_ROOT/pd/${EXPERIMENT_NAME}_$(date +%Y%m%d-%H%M%S)-$$}"
exec "$PYTHON_BIN" "$script_dir/code/collect_pd.py"
