#!/usr/bin/env bash
set -Eeuo pipefail

# ==================== 负载 / 并行 ====================
export TP="${TP:-1}" DP="${DP:-4}" EP="${EP:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export MAX_NUM_BATCHED_TOKENS=8120 # 所有负载档位固定使用此预算
export KV_LEN="${KV_LEN:-8192}" MAX_OUTPUT="${MAX_OUTPUT:-500}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
export KV_STARTUP_FILL="${KV_STARTUP_FILL:-1}"
export KV_FILL_MEAN="${KV_FILL_MEAN:-0.015}" KV_FILL_STD="${KV_FILL_STD:-0}"
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
# ==================== 投机推理 / EPLB ====================
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-none}" # none / mtp / dflash
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-}" DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"
export ENABLE_EPLB="${ENABLE_EPLB:-0}" # 仅 EP=1 生效；详细参数见 settings.sh
# ==================== 模型 / 服务 ====================
export MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.6-Flash-MOPD}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
export VLLM_SOURCE="${VLLM_SOURCE:-/host/vllm}"
HOST="${HOST:-127.0.0.1}" PORT="${PORT:-8001}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/results/standalone}"
# ====================================================

source "$SCRIPT_DIR/settings.sh"

fail() { echo "错误：$*" >&2; exit 1; }
for name in TP DP MAX_NUM_SEQS KV_LEN MAX_OUTPUT MAX_MODEL_LEN PORT; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || fail "$name 必须为正整数"
done
[[ "$MAX_NUM_BATCHED_TOKENS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_NUM_BATCHED_TOKENS 必须为正整数"
[[ "$TP" =~ ^(1|2|4)$ && "$DP" =~ ^(1|2|4)$ ]] || fail "TP、DP 只能为 1、2、4"
(( TP * DP == 4 )) || fail "只允许 4 卡部署，TP * DP 必须等于 4"
[[ "$EP" == 0 || "$EP" == 1 ]] || fail "EP 只能为 0 或 1"
[[ "$KV_STARTUP_FILL" == 0 || "$KV_STARTUP_FILL" == 1 ]] || fail "KV_STARTUP_FILL 只能为 0 或 1"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "DRY_RUN 只能为 0 或 1"
(( MAX_NUM_BATCHED_TOKENS >= MAX_NUM_SEQS )) || fail "batch token 预算不能小于 max_num_seqs"
(( KV_LEN + 1 + MAX_OUTPUT <= MAX_MODEL_LEN )) || fail "MAX_MODEL_LEN 必须至少为 KV_LEN + 1 + MAX_OUTPUT"
(( PORT <= 65535 )) || fail "端口超出范围"
case "$CUDAGRAPH_MODE" in NONE|PIECEWISE|FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE) ;; *) fail "无效 CUDAGRAPH_MODE";; esac
[[ "$CUDA_VISIBLE_DEVICES" != ,* && "$CUDA_VISIBLE_DEVICES" != *, ]] || fail "GPU 列表不能有空项"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
(( ${#devices[@]} == 4 )) || fail "必须且只能指定 4 张 GPU"
seen=" "
for device in "${devices[@]}"; do
  [[ "$device" =~ ^[0-9]+$ ]] || fail "GPU 必须使用数字索引"
  [[ "$seen" != *" $device "* ]] || fail "GPU 重复：$device"
  seen+="$device "
done

kv_config="$("$PYTHON_BIN" - <<'PY'
import json, math, os
mean, std, memory = (float(os.environ[k]) for k in ("KV_FILL_MEAN", "KV_FILL_STD", "GPU_MEMORY_UTILIZATION"))
if not all(math.isfinite(v) for v in (mean, std, memory)) or std < 0 or not 0 < memory < 1:
    raise SystemExit("KV 填充值必须有限，标准差必须非负，显存比例必须在 (0, 1) 内")
print(json.dumps({"kv_connector": "DecodeBenchConnector", "kv_role": "kv_both",
                  "kv_connector_extra_config": {"startup_fill": os.environ["KV_STARTUP_FILL"] == "1",
                                                "fill_mean": mean, "fill_std": std}}))
PY
)"
ep_args=()
if [[ "$EP" == 1 ]]; then
  ep_args+=(--enable-expert-parallel --all2all-backend "$ALL2ALL_BACKEND" --expert-placement-strategy "$EXPERT_PLACEMENT_STRATEGY")
  if [[ -n "$MOE_BACKEND" ]]; then ep_args+=(--moe-backend "$MOE_BACKEND"); fi
  if [[ -n "$ATTENTION_BACKEND" ]]; then ep_args+=(--attention-backend "$ATTENTION_BACKEND"); fi
  if [[ "$ENABLE_EPLB" == 1 ]]; then
    eplb_config="$("$PYTHON_BIN" "$SCRIPT_DIR/../ep_profile/analysis/code/eplb_config.py")"
    ep_args+=(--enable-eplb --eplb-config "$eplb_config")
  fi
fi
[[ "$ENABLE_EPLB" == 0 || "$ENABLE_EPLB" == 1 ]] || fail "ENABLE_EPLB 只能为 0/1"
spec_config="$("$PYTHON_BIN" "$SCRIPT_DIR/speculative_config.py")"
spec_args=()
if [[ "$spec_config" != null ]]; then spec_args+=(--speculative-config "$spec_config"); fi
cmd=("$PYTHON_BIN" -m vllm serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME" --host "$HOST" --port "$PORT"
  --tensor-parallel-size "$TP" --data-parallel-size "$DP" --data-parallel-size-local "$DP"
  --pipeline-parallel-size 1 --distributed-executor-backend mp --data-parallel-backend mp
  --data-parallel-address 127.0.0.1
  --data-parallel-rpc-port "${DP_RPC_PORT:-29650}" --master-port "${MASTER_PORT:-29601}"
  --trust-remote-code --dtype bfloat16 --kv-cache-dtype auto
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --enable-chunked-prefill --no-enable-prefix-caching
  --kv-transfer-config "$kv_config"
  --compilation-config "{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\"}"
  --generation-config vllm --override-generation-config "{\"max_new_tokens\":$MAX_OUTPUT}")

if (( ${#ep_args[@]} )); then cmd+=("${ep_args[@]}"); fi
if (( ${#spec_args[@]} )); then cmd+=("${spec_args[@]}"); fi

echo "GPU=$CUDA_VISIBLE_DEVICES TP=$TP DP=$DP EP=$EP port=$PORT"
echo "max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS KV=$KV_LEN output=$MAX_OUTPUT max_model_len=$MAX_MODEL_LEN"
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
[[ -f "$MODEL_PATH/config.json" ]] || fail "缺少模型 config.json"
"$PYTHON_BIN" -c 'import importlib.util; assert importlib.util.find_spec("vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector"), "当前 Python 中的 vLLM 缺少 DecodeBenchConnector"'
"$PYTHON_BIN" -c 'import torch, vllm; assert torch.cuda.device_count() == 4, "实际可见 GPU 必须为 4 张"; print("vLLM:", vllm.__file__)'
log="${LOG_FILE:-$LOG_DIR/server-seqs${MAX_NUM_SEQS}-batch${MAX_NUM_BATCHED_TOKENS}.log}"
mkdir -p "$(dirname -- "$log")"
exec > >(tee -a "$log") 2>&1
exec "${cmd[@]}"
