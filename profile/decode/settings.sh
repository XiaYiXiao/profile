#!/usr/bin/env bash
# 公共配置：除固定的 token 预算外，参数均可通过 export 覆盖。
# ==================== 模型 / 运行环境 ====================
export MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.6-Flash-MOPD}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
export VLLM_SOURCE="${VLLM_SOURCE:-/host/vllm}"
if [[ -n "$VLLM_SOURCE" && "${PYTHONPATH:-}" != "$VLLM_SOURCE" && "${PYTHONPATH:-}" != "$VLLM_SOURCE":* ]]; then
  export PYTHONPATH="$VLLM_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1 VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"; export no_proxy="$NO_PROXY"
# ==================== 负载 / 并行 ====================
export TP="${TP:-1}" DP="${DP:-4}" EP="${EP:-0}"
export MAX_NUM_BATCHED_TOKENS=8120 # 所有负载档位固定使用此预算
export KV_LEN="${KV_LEN:-8192}" MAX_OUTPUT="${MAX_OUTPUT:-500}"
export WARMUP_ROUNDS="${WARMUP_ROUNDS:-2}" SEED="${SEED:-42}"
# ==================== 投机推理 ====================
# none / mtp / dflash（不区分大小写）。none 不传 speculative-config。
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-none}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}" SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
# ==================== EP 后端 / EPLB（仅 EP=1 生效） ====================
export MOE_BACKEND="${MOE_BACKEND:-}" ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
export ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-allgather_reducescatter}"
export EXPERT_PLACEMENT_STRATEGY="${EXPERT_PLACEMENT_STRATEGY:-linear}"
export ENABLE_EPLB="${ENABLE_EPLB:-0}"
export EPLB_WINDOW_SIZE="${EPLB_WINDOW_SIZE:-8}" EPLB_STEP_INTERVAL="${EPLB_STEP_INTERVAL:-16}"
export EPLB_NUM_REDUNDANT_EXPERTS="${EPLB_NUM_REDUNDANT_EXPERTS:-0}"
export EPLB_USE_ASYNC="${EPLB_USE_ASYNC:-true}" EPLB_COMMUNICATOR="${EPLB_COMMUNICATOR:-}"
export EPLB_LOG_BALANCEDNESS="${EPLB_LOG_BALANCEDNESS:-true}"
export EPLB_LOG_BALANCEDNESS_INTERVAL="${EPLB_LOG_BALANCEDNESS_INTERVAL:-1}" EPLB_CONFIG="${EPLB_CONFIG:-}"
