#!/usr/bin/env bash
# 6P + 2D：独立配置入口，P 默认 DP3 × TP2，D 默认 DP1 × TP2。
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PROFILE_MODE=pd

# ==================== GPU / TP（张量并行） ====================
export P_GPUS="${P_GPUS:-0,1,2,3,4,5}"
export D_GPUS="${D_GPUS:-6,7}"
export P_TP="${P_TP:-2}"
export D_TP="${D_TP:-2}"
# ==================== DP（数据并行） ====================
export P_DP="${P_DP:-3}"
export D_DP="${D_DP:-1}" # 每端 TP * DP 必须等于该端 GPU 数量
# ==================== EP（专家并行，P / D 可独立设置） ====================
export ENABLE_EP="${ENABLE_EP:-1}" # 兼容旧的统一开关；下面的端侧值优先
export P_ENABLE_EP="${P_ENABLE_EP:-$ENABLE_EP}"
export D_ENABLE_EP="${D_ENABLE_EP:-$ENABLE_EP}"
# 在设备与并行参数确定后生成名称；仍可通过 EXPERIMENT_NAME 手动覆盖。
p_gpu_ids="${P_GPUS//[[:space:]]/}"
d_gpu_ids="${D_GPUS//[[:space:]]/}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-pd_6p2d_p_dp${P_DP}_tp${P_TP}_gpus${p_gpu_ids//,/-}_d_dp${D_DP}_tp${D_TP}_gpus${d_gpu_ids//,/-}_p_ep${P_ENABLE_EP}_d_ep${D_ENABLE_EP}}"

# ==================== 调度 / 显存 ====================
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# P、D 共用 token budget，默认 65536，与非分离组一致。
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export P_PORT="${P_PORT:-19210}" D_PORT="${D_PORT:-19220}" PROXY_PORT="${PROXY_PORT:-19200}"
export P_DP_RPC_PORT="${P_DP_RPC_PORT:-31350}" P_MASTER_PORT="${P_MASTER_PORT:-31301}"
export D_DP_RPC_PORT="${D_DP_RPC_PORT:-31450}" D_MASTER_PORT="${D_MASTER_PORT:-31401}"

# ==================== KV 传输：NixlConnector / MooncakeConnector ====================
export KV_CONNECTOR="${KV_CONNECTOR:-NixlConnector}"
export P_MOONCAKE_BOOTSTRAP_PORT="${P_MOONCAKE_BOOTSTRAP_PORT:-6600}"
export P_NIXL_PORT="${P_NIXL_PORT:-6300}" D_NIXL_PORT="${D_NIXL_PORT:-6400}"
# D 的 side-channel 占用 D_NIXL_PORT + dp_rank；P/D 使用同一 KV layout / dtype。
export KV_CACHE_LAYOUT="${KV_CACHE_LAYOUT:-LBHNC}" KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-16}"
export KV_EXTRA_CONFIG="${KV_EXTRA_CONFIG:-}" # NIXL: {"backends":["UCX"]}；Mooncake: {"mooncake_protocol":"rdma"}
export UCX_TLS="${UCX_TLS:-all}" UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"

# ==================== Prefix caching（1 开启 / 0 关闭） ====================
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}" # PD 模式同时作用于 P / D
export ENABLE_RESET_PREFIX_CACHE="${ENABLE_RESET_PREFIX_CACHE:-1}" # 开启 P/D 的缓存重置接口

# ==================== EP / Attention 后端 ====================
export MOE_BACKEND="${MOE_BACKEND:-}" ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
export ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-allgather_reducescatter}"
export EXPERT_PLACEMENT_STRATEGY="${EXPERT_PLACEMENT_STRATEGY:-linear}"
# ==================== EPLB（专家负载均衡，要求对应端 EP=1） ====================
export ENABLE_EPLB="${ENABLE_EPLB:-1}" # 兼容旧的统一开关；下面的端侧值优先
export P_ENABLE_EPLB="${P_ENABLE_EPLB:-$ENABLE_EPLB}"
export D_ENABLE_EPLB="${D_ENABLE_EPLB:-$ENABLE_EPLB}"
# 下列 EPLB 调优参数由本脚本管理，作用于本次部署中开启 EPLB 的端。
export EPLB_WINDOW_SIZE="${EPLB_WINDOW_SIZE:-8}" EPLB_STEP_INTERVAL="${EPLB_STEP_INTERVAL:-16}"
# MiMo 每层 256 个逻辑专家；增加 2 个副本后，258 可被 P 的 EP6 和 D 的 EP2 整除。
export EPLB_NUM_REDUNDANT_EXPERTS="${EPLB_NUM_REDUNDANT_EXPERTS:-2}"
export EPLB_USE_ASYNC="${EPLB_USE_ASYNC:-true}" EPLB_CONFIG="${EPLB_CONFIG:-}"

# ==================== 投机解码：none / mtp / dflash ====================
# 由本目录 code/speculative_config.py 解析；默认关闭投机，可显式选择 mtp / dflash。
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-none}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH-/models/MiMo-V2.6-Flash-MOPD/dflash}" # 显式空值用于 MTP；DFlash 必须提供模型
if [[ "${SPECULATIVE_CONFIG+x}" != x ]]; then
  SPECULATIVE_CONFIG='{
  "rejection_sample_method": "synthetic",
  "synthetic_acceptance_length": 4.2
}'
  [[ "$SPECULATIVE_METHOD" != none ]] || SPECULATIVE_CONFIG=''
fi
export SPECULATIVE_CONFIG

# 两端使用相同投机配置；P 仍仅做一步 prefill。
# ==================== 对外访问 ====================
export HOST="${HOST:-0.0.0.0}"
export ENABLE_NSYS=0 # 不执行 Nsight；服务指标由下面的独立开关控制

# ==================== 服务指标 / 前缀命中率（READY 后自动采集） ====================
export ENABLE_METRICS="${ENABLE_METRICS:-1}" # 1 开启；0 关闭部署侧采集
export METRICS_INTERVAL="${METRICS_INTERVAL:-5}" # 周期秒数
export METRICS_TIMEOUT="${METRICS_TIMEOUT:-3}" # 单次 HTTP 超时秒数
export METRICS_GPU="${METRICS_GPU:-1}" # 同时采集本次部署 GPU 的利用率/显存

# ==================== 模型 / Python / 超时 ====================
export MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.6-Flash-MOPD}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
# 只使用镜像安装的包；忽略外部源码和额外 Python 依赖目录。
unset VLLM_SOURCE DEPS_DIR PYTHONPATH PYTHONHOME
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}" REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-7200}"
export DRY_RUN="${DRY_RUN:-0}" PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

# ==================== 部署日志（实验结果由 run_load.sh 管理） ====================
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$script_dir}"
export DEPLOY_MODE="${DEPLOY_MODE:-pd_6p2d}"
exec "$PYTHON_BIN" -I "$script_dir/code/deploy.py"
