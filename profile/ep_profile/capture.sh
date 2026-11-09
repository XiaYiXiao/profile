#!/usr/bin/env bash
# EP 单组模式：1=仅运行第 1 组 EP1/EPLB1（使用 GPU_EP1），0=同时运行两组。
export EP_SOLO="${EP_SOLO:-0}"

# 实验记录名称，可用 EXPERIMENT_NAME=名称 bash ep_profile/capture.sh 覆盖。
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-ep}"

# MiMo 多轮 EP 对比；EP_SOLO=1 时仅验证和采集 EP1，否则同时采集 EP0/EP1。
# 启动：bash ep_profile/capture.sh
# 仅预览（不启动 GPU / 模型 / Nsight）：DRY_RUN=1 bash ep_profile/capture.sh
# 仅工具链：SMOKE_ONLY=1 bash ep_profile/capture.sh
# 单组开启 EPLB：EP_SOLO=1 ENABLE_EPLB=1 COMPARISON=eplb bash ep_profile/capture.sh
# EPLB 开关对照：EP_SOLO=0 COMPARISON=eplb bash ep_profile/capture.sh
# 完整预热后，单个连续 trace 覆盖全部评估轮（含轮间间隔）。
# 仅采集原始数据、导出 SQLite 并校验完整性；不执行统计汇总、绘图或 HTML 报告生成。
set -Eeuo pipefail
export COMPARISON="${COMPARISON:-ep}" # ep=EP 实验；eplb=EPLB 实验；EP_SOLO=1 时只运行对应第 1 组

# ==================== Backend / EP 参数（仅 EP1 生效） ====================
# EP0 不传以下覆盖参数，使用 vLLM 默认 backend / 专家放置配置。
# 留空时不传对应 backend 参数，采用 vLLM 默认选择；可选项及限制见 backends/。
# --moe-backend：auto、humming、marlin、flashinfer_cutlass；其他候选见文档。
export MOE_BACKEND="${MOE_BACKEND:-}" # default= -
# --attention-backend：FLASH_ATTN_DIFFKV、TRITON_ATTN_DIFFKV；自动选择请留空。
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-}" # default= -
# --all2all-backend：仅覆盖 EP1；vLLM 默认值为 allgather_reducescatter。
# 其他候选：flashinfer_nvlink_one_sided、flashinfer_nvlink_two_sided 等，需检查组合兼容性。
export ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-allgather_reducescatter}" # default: allgather_reducescatter
# EP0 不启用专家并行；EP1 启用专家并行，TP=2、DP=2 对应 EP size=4。
# --expert-placement-strategy：linear / round_robin，仅传给 EP1。
# 当前 MiMo n_group=1，且此 DP/通信组合不支持 round_robin，指定后 vLLM 会回退 linear。
export EXPERT_PLACEMENT_STRATEGY="${EXPERT_PLACEMENT_STRATEGY:-linear}" #default=linear

# ==================== EPLB 参数（仅 EP1 生效） ====================
export ENABLE_EPLB="${ENABLE_EPLB:-0}"                         # 默认关闭，仅比较 EP；1=启用
export EPLB_WINDOW_SIZE="${EPLB_WINDOW_SIZE:-8}"              # 负载统计窗口，单位：engine step
export EPLB_STEP_INTERVAL="${EPLB_STEP_INTERVAL:-16}"         # 重平衡触发间隔，不是整轮迁移耗时
export EPLB_NUM_REDUNDANT_EXPERTS="${EPLB_NUM_REDUNDANT_EXPERTS:-0}" # 全局冗余专家数；0 仍可重排
export EPLB_USE_ASYNC="${EPLB_USE_ASYNC:-true}"                # true=异步，false=同步
export EPLB_COMMUNICATOR="${EPLB_COMMUNICATOR:-}"              # 留空=自动；torch_gloo/nixl/pynccl/torch_nccl
export EPLB_LOG_BALANCEDNESS="${EPLB_LOG_BALANCEDNESS:-true}"   # 记录负载均衡度；日志有额外开销
export EPLB_LOG_BALANCEDNESS_INTERVAL="${EPLB_LOG_BALANCEDNESS_INTERVAL:-1}" # 日志间隔，单位：step
export EPLB_CONFIG="${EPLB_CONFIG:-}"                        # 可选 JSON：覆盖以上同名配置字段
# 当前版本异步 EPLB 不支持 pynccl/torch_nccl；选择这两者时须设 EPLB_USE_ASYNC=false。
# 8/16 是接入验证值；完整预热轮数不保证 EPLB 收敛，需结合迁移日志确认。

# ==================== 负载 / 并行参数（均可通过环境变量覆盖） ====================
export TP="${TP:-2}" DP="${DP:-2}" REPEATS="${REPEATS:-3}" BENCH_PARALLEL=1
export GPU_EP0="${GPU_EP0:-0,1,2,3}" GPU_EP1="${GPU_EP1:-4,5,6,7}"
export NUM_PROMPTS="${NUM_PROMPTS:-30}"
# 每轮预热运行完整 benchmark，负载与评估一致；REPEATS 仅为评估轮数。
export WARMUP_ROUNDS="${WARMUP_ROUNDS:-${warmup_round:-2}}"
export NUM_WARMUPS=0 # 禁用原来的重复首个 prompt 预热
export CONCURRENCY="${CONCURRENCY:-10}" INPUT_LEN="${INPUT_LEN:-50000}" OUTPUT_LEN="${OUTPUT_LEN:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}" GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export SEED="${SEED:-42}"

# ==================== 模型 / 服务参数 ====================
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.6-Flash-MOPD}"
export MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
# 默认优先使用本地 vLLM 源码，保留已有 PYTHONPATH。
export VLLM_SOURCE="${VLLM_SOURCE:-/host/vllm}"
# 同时延长外层健康检查与 vLLM API 等待引擎就绪的时间。
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}" BENCH_TIMEOUT="${BENCH_TIMEOUT:-900}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$STARTUP_TIMEOUT}"

# ==================== Nsight 采集参数 ====================
export NSYS_BIN="${NSYS_BIN:-/usr/local/bin/nsys}" SMOKE_ONLY="${SMOKE_ONLY:-0}"
export DRY_RUN="${DRY_RUN:-0}"
# cuda-sw 使用 CUPTI 软件 trace，避免新版默认硬件 trace 的驱动限制。
export NSYS_TRACE="${NSYS_TRACE:-cuda-sw,nvtx,nccl,osrt}"
export NCCL_TRACE="${NCCL_TRACE:-api,coll,p2p,group,gpu}"
export CUDA_GRAPH_TRACE="${CUDA_GRAPH_TRACE:-node}"

# ==================== 运行环境 / 输出目录 ====================
if [[ -n "$VLLM_SOURCE" && "${PYTHONPATH:-}" != "$VLLM_SOURCE" && "${PYTHONPATH:-}" != "$VLLM_SOURCE":* ]]; then
  export PYTHONPATH="$VLLM_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
fi
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"; export no_proxy="$NO_PROXY"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RESULT_DIR="${RESULT_DIR:-$script_dir/captures/${EXPERIMENT_NAME}_$(date +%Y%m%d-%H%M%S)}"
exec "$PYTHON_BIN" "$script_dir/analysis/code/collect.py"
