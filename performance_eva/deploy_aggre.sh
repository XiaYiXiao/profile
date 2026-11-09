#!/usr/bin/env bash
# 非 PD 分离：8 GPU，DP2 × TP4；默认开启 EP8，ENABLE_EP=0 关闭 EP。
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PROFILE_MODE=aggre

# ==================== GPU / TP（张量并行） ====================
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export TP="${TP:-4}"
# ==================== DP（数据并行） ====================
export DP="${DP:-2}" # TP * DP 必须等于 GPU 数量
# ==================== EP（专家并行） ====================
export ENABLE_EP="${ENABLE_EP:-1}" # EP 规模由 TP * DP 决定
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-aggre_dp${DP}_tp${TP}_enable_ep${ENABLE_EP}}"

# ==================== 调度 / 显存 / KV ====================
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export KV_CACHE_LAYOUT="${KV_CACHE_LAYOUT:-LBHNC}" KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-16}"
export PORT="${PORT:-19000}" DP_RPC_PORT="${DP_RPC_PORT:-31050}" MASTER_PORT="${MASTER_PORT:-31001}"

# ==================== Prefix caching（1 开启 / 0 关闭） ====================
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"

# ==================== EP / Attention 后端 ====================
export MOE_BACKEND="${MOE_BACKEND:-}" ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
export ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-allgather_reducescatter}"
export EXPERT_PLACEMENT_STRATEGY="${EXPERT_PLACEMENT_STRATEGY:-linear}"
# ==================== EPLB（专家负载均衡，默认开，要求 EP=1） ====================
export ENABLE_EPLB="${ENABLE_EPLB:-1}"
export EPLB_WINDOW_SIZE="${EPLB_WINDOW_SIZE:-8}" EPLB_STEP_INTERVAL="${EPLB_STEP_INTERVAL:-16}"
export EPLB_NUM_REDUNDANT_EXPERTS="${EPLB_NUM_REDUNDANT_EXPERTS:-0}"
export EPLB_USE_ASYNC="${EPLB_USE_ASYNC:-true}" EPLB_CONFIG="${EPLB_CONFIG:-}"

# ==================== 投机解码：none / mtp / dflash ====================
# 与 profiler/profile/decode/capture.sh 接口一致；默认 none 保持普通推理。
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-none}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}" # mtp 可空（使用目标模型）；dflash 必须提供 draft 模型
export SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}" # JSON 额外配置覆盖同名基础参数
# 合成接受长度实验示例（需先选择 mtp/dflash）：
# export SPECULATIVE_CONFIG='{"rejection_sample_method":"synthetic","synthetic_acceptance_length":4.2}'

# ==================== 对外访问 ====================
export HOST="${HOST:-0.0.0.0}"
export ENABLE_NSYS=0 # 本入口仅常驻服务，不调度采集

# ==================== 模型 / Python / 超时 ====================
export MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.6-Flash-MOPD}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}" VLLM_SOURCE="${VLLM_SOURCE:-/host/vllm}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}" REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-7200}"
export DRY_RUN="${DRY_RUN:-0}" PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

# ==================== 部署日志（实验结果由 run_load.sh 管理） ====================
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$script_dir}"
export DEPS_DIR="${DEPS_DIR:-$script_dir/../aggre_or_not/.deps}"
export DEPLOY_MODE="${DEPLOY_MODE:-aggre}"
exec "$PYTHON_BIN" "$script_dir/code/deploy.py"
