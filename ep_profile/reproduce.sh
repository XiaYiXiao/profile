#!/usr/bin/env bash
# TP2/DP2 EP 复现：bash profiling_mimo/ep_profile/reproduce.sh
# 预览命令：DRY_RUN=1 bash profiling_mimo/ep_profile/reproduce.sh
# MiMo：两个服务同时启动，每轮两侧同时压测；固定同时运行。
# COMPARISON=eplb 时两组均 EP=1，以下 backend 参数同时作用于两组，仅组 1 开启 EPLB。
# 每次执行新建目录，绝不复用历史测试。先完整预热 WARMUP_ROUNDS 轮，再评估 REPEATS 轮；全程不重启服务。
set -Eeuo pipefail
export COMPARISON="${COMPARISON:-ep}" # ep=EP0/EP1；eplb=两组 EP1，仅 EPLB 开关不同

# ==================== Backend / EP 参数（仅 EP1 生效） ====================
# EP0 不传以下覆盖参数，使用 vLLM 默认 backend / 专家放置配置。
# 留空时不传对应 backend 参数，采用 vLLM 默认选择；可选项及限制见 backends/。
# --moe-backend：auto、humming、marlin、flashinfer_cutlass；其他候选见文档。
export MOE_BACKEND="${MOE_BACKEND:-}"
# --attention-backend：FLASH_ATTN_DIFFKV、TRITON_ATTN_DIFFKV；自动选择请留空。
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
# --all2all-backend：仅覆盖 EP1；vLLM 默认值为 allgather_reducescatter。
export ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-allgather_reducescatter}"
# --expert-placement-strategy：linear / round_robin，仅传给 EP1。
# 当前 MiMo n_group=1，此 DP/通信组合指定 round_robin 后会回退 linear。
export EXPERT_PLACEMENT_STRATEGY="${EXPERT_PLACEMENT_STRATEGY:-linear}"

# ==================== EPLB 参数（仅 EP1 生效） ====================
export ENABLE_EPLB="${ENABLE_EPLB:-1}"                         # 1=启用，0=关闭，便于做对照
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
export NUM_PROMPTS="${NUM_PROMPTS:-50}"
# WARMUP_ROUNDS 为完整 benchmark 预热轮数；REPEATS 只计评估轮数。
export WARMUP_ROUNDS="${WARMUP_ROUNDS:-${warmup_round:-2}}"
# 禁用 benchmark 内部重复首个 prompt 的旧预热；所有轮次使用相同负载和 seed。
export NUM_WARMUPS=0
# CONCURRENCY 为每个服务的全局并发，不是每 DP rank 并发。
export CONCURRENCY="${CONCURRENCY:-10}" INPUT_LEN="${INPUT_LEN:-50000}" OUTPUT_LEN="${OUTPUT_LEN:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}" GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export SEED="${SEED:-42}"

# ==================== 模型 / 服务参数 ====================
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.6-Flash-MOPD}"
export MODEL_PATH="${MODEL_PATH:-/host/models/MiMo-V2.6-Flash-MOPD}"
export VLLM_SOURCE="${VLLM_SOURCE:-/host/vllm}" PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
export PORT_EP0="${PORT_EP0:-8000}" PORT_EP1="${PORT_EP1:-8001}"
export DP_RPC_PORT_EP0="${DP_RPC_PORT_EP0:-29550}" DP_RPC_PORT_EP1="${DP_RPC_PORT_EP1:-29650}"
export MASTER_PORT_EP0="${MASTER_PORT_EP0:-29501}" MASTER_PORT_EP1="${MASTER_PORT_EP1:-29601}"
# NFS 权重加载、编译及热身可能超过 10 分钟；内外两层默认均等待 1 小时。
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}" BENCH_TIMEOUT="${BENCH_TIMEOUT:-3600}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$STARTUP_TIMEOUT}"
export STOP_TIMEOUT="${STOP_TIMEOUT:-30}" DRY_RUN="${DRY_RUN:-0}"

# ==================== 输出目录 ====================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/results}"
# RESULT_DIR 必须不存在；默认使用时间戳+PID，避免覆盖和自动跳过重复实验。
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/$(date +%Y%m%d-%H%M%S)-$$}"
# ======================================================================

fail() { echo "错误：$*" >&2; exit 1; }
for name in TP DP MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS CONCURRENCY NUM_PROMPTS REPEATS INPUT_LEN OUTPUT_LEN MAX_MODEL_LEN STARTUP_TIMEOUT VLLM_ENGINE_READY_TIMEOUT_S BENCH_TIMEOUT STOP_TIMEOUT; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || fail "$name 必须为正整数"
done
[[ "$TP" == 2 && "$DP" == 2 ]] || fail "此复现脚本固定 TP=2、DP=2"
[[ "$SEED" =~ ^(0|[1-9][0-9]*)$ ]] || fail "SEED 必须为非负整数"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "DRY_RUN 只能为 0/1"
[[ "$BENCH_PARALLEL" == 0 || "$BENCH_PARALLEL" == 1 ]] || fail "BENCH_PARALLEL 只能为 0/1"
[[ "$ENABLE_EPLB" == 0 || "$ENABLE_EPLB" == 1 ]] || fail "ENABLE_EPLB 只能为 0/1"
[[ "$EXPERT_PLACEMENT_STRATEGY" == linear || "$EXPERT_PLACEMENT_STRATEGY" == round_robin ]] || fail "EXPERT_PLACEMENT_STRATEGY 只能为 linear/round_robin"
[[ "$WARMUP_ROUNDS" =~ ^(0|[1-9][0-9]*)$ ]] || fail "WARMUP_ROUNDS 必须为非负整数"
[[ "$COMPARISON" == ep || "$COMPARISON" == eplb ]] || fail "COMPARISON 只能为 ep/eplb"
if [[ "$COMPARISON" == eplb ]]; then export ENABLE_EPLB=1; fi
(( CONCURRENCY <= NUM_PROMPTS )) || fail "并发不能大于正式请求数"
(( MAX_NUM_BATCHED_TOKENS >= MAX_NUM_SEQS )) || fail "token 预算必须 >= max_num_seqs"
(( INPUT_LEN + OUTPUT_LEN <= MAX_MODEL_LEN )) || fail "输入与输出长度之和超过模型长度上限"
declare -A seen_gpus=() seen_ports=()
for name in GPU_EP0 GPU_EP1; do
  [[ "${!name}" =~ ^(0|[1-9][0-9]*),(0|[1-9][0-9]*),(0|[1-9][0-9]*),(0|[1-9][0-9]*)$ ]] || fail "$name 必须指定四个 GPU 数字索引"
  IFS=',' read -r -a devices <<< "${!name}"
  for gpu in "${devices[@]}"; do
    [[ -z "${seen_gpus[$gpu]:-}" ]] || fail "GPU $gpu 重复或两组设备重叠"
    seen_gpus[$gpu]=1
  done
done
for name in PORT_EP0 PORT_EP1 DP_RPC_PORT_EP0 DP_RPC_PORT_EP1 MASTER_PORT_EP0 MASTER_PORT_EP1; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || fail "$name 必须为端口号"
  (( ${!name} <= 65535 )) || fail "$name 超出端口范围"
  [[ -z "${seen_ports[${!name}]:-}" ]] || fail "端口 ${!name} 重复"
  seen_ports[${!name}]=1
done
for cmd in "$PYTHON_BIN" curl setsid timeout; do command -v "$cmd" >/dev/null || fail "找不到命令：$cmd"; done
if [[ "$ENABLE_EPLB" == 1 ]]; then
  EPLB_CONFIG="$("$PYTHON_BIN" "$SCRIPT_DIR/analysis/code/eplb_config.py")" || fail "EPLB 参数无效"
fi

export PYTHONPATH="$VLLM_SOURCE"
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export RESULT_DIR

groups=(ep0 ep1)
if [[ "$COMPARISON" == eplb ]]; then groups=(eplb0 eplb1); fi
gpus=("$GPU_EP0" "$GPU_EP1")
ports=("$PORT_EP0" "$PORT_EP1")
rpc_ports=("$DP_RPC_PORT_EP0" "$DP_RPC_PORT_EP1")
master_ports=("$MASTER_PORT_EP0" "$MASTER_PORT_EP1")
server_pids=("" "")
bench_pids=("" "")

server_command() {
  local ep="$1"
  server_cmd=(env "CUDA_VISIBLE_DEVICES=${gpus[$ep]}" "$PYTHON_BIN" -m vllm serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME" --host 127.0.0.1 --port "${ports[$ep]}"
    --tensor-parallel-size "$TP" --data-parallel-size "$DP" --data-parallel-size-local "$DP"
    --data-parallel-address 127.0.0.1 --data-parallel-rpc-port "${rpc_ports[$ep]}"
    --master-port "${master_ports[$ep]}" --pipeline-parallel-size 1
    --distributed-executor-backend mp --data-parallel-backend mp
    --trust-remote-code --dtype bfloat16
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --enable-chunked-prefill --no-enable-prefix-caching --generation-config vllm
    --override-generation-config "{\"max_new_tokens\":$OUTPUT_LEN}")
  if [[ "$ep" == 1 || "$COMPARISON" == eplb ]]; then
    server_cmd+=(--enable-expert-parallel
      --all2all-backend "$ALL2ALL_BACKEND"
      --expert-placement-strategy "$EXPERT_PLACEMENT_STRATEGY")
    if [[ -n "$MOE_BACKEND" ]]; then server_cmd+=(--moe-backend "$MOE_BACKEND"); fi
    if [[ -n "$ATTENTION_BACKEND" ]]; then server_cmd+=(--attention-backend "$ATTENTION_BACKEND"); fi
    if [[ "$ep" == 1 && "$ENABLE_EPLB" == 1 ]]; then
      server_cmd+=(--enable-eplb --eplb-config "$EPLB_CONFIG")
    fi
  fi
}
bench_command() {
  local ep="$1" repeat="$2" phase="${3:-eval}" stem
  stem="repeat$repeat"
  if [[ "$phase" == warmup ]]; then stem="warmup$repeat"; fi
  bench_cmd=(env "CUDA_VISIBLE_DEVICES=${gpus[$ep]}" "$PYTHON_BIN" -m vllm bench serve
    --backend vllm --base-url "http://127.0.0.1:${ports[$ep]}" --endpoint /v1/completions
    --model "$SERVED_MODEL_NAME" --tokenizer "$MODEL_PATH"
    --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN"
    --random-range-ratio 0 --random-prefix-len 0 --num-prompts "$NUM_PROMPTS"
    --num-warmups "$NUM_WARMUPS" --request-rate inf --max-concurrency "$CONCURRENCY" --seed "$SEED"
    --ready-check-timeout-sec 0 --ignore-eos --percentile-metrics ttft,e2el --metric-percentiles 50,90,95,99
    --save-result --save-detailed --result-dir "$RESULT_DIR/${groups[$ep]}" --result-filename "$stem.json")
}
print_command() { printf '%q ' "$@"; printf '\n'; }

if [[ "$DRY_RUN" == 1 ]]; then
  echo "预览：${groups[0]} GPU=$GPU_EP0 port=$PORT_EP0；${groups[1]} GPU=$GPU_EP1 port=$PORT_EP1；预热 $WARMUP_ROUNDS 轮，评估 $REPEATS 轮"
  echo "BENCH_PARALLEL=$BENCH_PARALLEL；结果=$RESULT_DIR（预览不创建目录）"
  echo "STARTUP_TIMEOUT=$STARTUP_TIMEOUT；VLLM_ENGINE_READY_TIMEOUT_S=$VLLM_ENGINE_READY_TIMEOUT_S"
  for ep in 0 1; do server_command "$ep"; print_command "${server_cmd[@]}"; done
  for phase in warmup eval; do
    rounds="$WARMUP_ROUNDS"
    if [[ "$phase" == eval ]]; then rounds="$REPEATS"; fi
    for ((r=1;r<=rounds;r++)); do
      for ep in 0 1; do bench_command "$ep" "$r" "$phase"; print_command "${bench_cmd[@]}"; done
    done
  done
  exit 0
fi

# 在启动任何服务前核对模型、GPU 索引、端口；不终止已有任务。
"$PYTHON_BIN" - <<'PY'
import json, os, pathlib, socket
import torch
root=pathlib.Path(os.environ['MODEL_PATH'])
required=['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']
missing=[n for n in required if not (root/n).is_file()]
if (root/'model.safetensors.index.json').is_file():
    shards=set(json.loads((root/'model.safetensors.index.json').read_text())['weight_map'].values())
    missing.extend(n for n in shards if not (root/n).is_file())
if missing: raise SystemExit('模型文件缺失：'+', '.join(missing))
if not pathlib.Path(os.environ['VLLM_SOURCE'],'vllm').is_dir(): raise SystemExit('VLLM_SOURCE 无效')
u=float(os.environ['GPU_MEMORY_UTILIZATION'])
if not 0<u<=1: raise SystemExit('GPU_MEMORY_UTILIZATION 必须在 (0,1]')
# 两个子服务会分别覆盖 CUDA_VISIBLE_DEVICES，因此当前外部可见性不作为索引依据。
import subprocess
indices={int(s) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).split()}
requested={int(s) for k in ['GPU_EP0','GPU_EP1'] for s in os.environ[k].split(',')}
if not requested<=indices: raise SystemExit(f'不存在的 GPU：{requested-indices}')
for key in ['PORT_EP0','PORT_EP1','DP_RPC_PORT_EP0','DP_RPC_PORT_EP1','MASTER_PORT_EP0','MASTER_PORT_EP1']:
    with socket.socket() as sock:
        try: sock.bind(('127.0.0.1',int(os.environ[key])))
        except OSError as error: raise SystemExit(f'{key}={os.environ[key]} 无法绑定：{error}')
print('预检通过；Torch',torch.__version__)
PY
mkdir -p -- "$(dirname -- "$RESULT_DIR")"
mkdir -- "$RESULT_DIR" || fail "结果目录已存在或无法创建：$RESULT_DIR"
RESULT_DIR="$(cd -- "$RESULT_DIR" && pwd)"
export RESULT_DIR
mkdir -- "$RESULT_DIR/${groups[0]}" "$RESULT_DIR/${groups[1]}"
"$PYTHON_BIN" - <<'PY'
import datetime, json, os, pathlib, subprocess
root=pathlib.Path(os.environ['RESULT_DIR'])
keys='TP DP GPU_EP0 GPU_EP1 PORT_EP0 PORT_EP1 DP_RPC_PORT_EP0 DP_RPC_PORT_EP1 MASTER_PORT_EP0 MASTER_PORT_EP1 MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS CONCURRENCY NUM_PROMPTS REPEATS INPUT_LEN OUTPUT_LEN SEED MAX_MODEL_LEN GPU_MEMORY_UTILIZATION ALL2ALL_BACKEND BENCH_PARALLEL MODEL_PATH SERVED_MODEL_NAME VLLM_SOURCE PYTHON_BIN'.split()
config={k:os.environ[k] for k in keys}
config.update({k:os.environ[k] for k in ['WARMUP_ROUNDS','NUM_WARMUPS','COMPARISON']})
config.update({k:os.environ[k] for k in ['MOE_BACKEND','ATTENTION_BACKEND','EXPERT_PLACEMENT_STRATEGY']})
config['backend_parameter_scope']='both_groups' if config['COMPARISON']=='eplb' else 'ep1_only'
config['group_settings']={f"{config['COMPARISON']}{i}": {'ep_enabled':config['COMPARISON']=='eplb' or i==1, 'eplb_enabled':i==1 and os.environ['ENABLE_EPLB']=='1'} for i in (0,1)}
config['ENABLE_EPLB']=os.environ['ENABLE_EPLB']=='1'
config['EPLB_CONFIG']=json.loads(os.environ['EPLB_CONFIG']) if config['ENABLE_EPLB'] else None
config['eplb_enabled_by_group']={k:v['eplb_enabled'] for k,v in config['group_settings'].items()}
config.update({k:os.environ[k] for k in ['STARTUP_TIMEOUT','VLLM_ENGINE_READY_TIMEOUT_S']})
config['created_at']=datetime.datetime.now().astimezone().isoformat()
config['prefix_caching']=False
try: config['vllm_git_head']=subprocess.check_output(['git','-C',config['VLLM_SOURCE'],'rev-parse','HEAD'],text=True).strip()
except (OSError,subprocess.CalledProcessError): config['vllm_git_head']=None
(root/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
for args,name in [(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv'],'gpus.txt'),(['nvidia-smi','topo','-m'],'topology.txt')]:
    result=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (root/name).write_text(result.stdout)
PY

stop_group() {
  local pid="$1" deadline=$((SECONDS + STOP_TIMEOUT))
  [[ -n "$pid" ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
  while kill -0 -- "-$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
  if kill -0 -- "-$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  local pid
  # 先同时通知两边，随后逐组等待退出，只清理本脚本创建的进程组。
  for pid in "${bench_pids[@]}" "${server_pids[@]}"; do
    if [[ -n "$pid" ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
  done
  for pid in "${bench_pids[@]}" "${server_pids[@]}"; do stop_group "$pid"; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ep in 0 1; do
  server_command "$ep"
  print_command "${server_cmd[@]}" > "$RESULT_DIR/${groups[$ep]}/server-command.txt"
  setsid "${server_cmd[@]}" > "$RESULT_DIR/${groups[$ep]}/server.log" 2>&1 &
  server_pids[$ep]=$!
done
ready=(0 0)
deadline=$((SECONDS + STARTUP_TIMEOUT))
until [[ "${ready[0]}" == 1 && "${ready[1]}" == 1 ]]; do
  for ep in 0 1; do
    kill -0 "${server_pids[$ep]}" 2>/dev/null || fail "${groups[$ep]} 服务退出，查看 $RESULT_DIR/${groups[$ep]}/server.log"
    if curl --silent --fail --max-time 2 "http://127.0.0.1:${ports[$ep]}/health" >/dev/null; then ready[$ep]=1; else ready[$ep]=0; fi
  done
  (( SECONDS < deadline )) || fail "等待两个服务就绪超时：$RESULT_DIR"
  if [[ "${ready[0]}" != 1 || "${ready[1]}" != 1 ]]; then sleep 2; fi
done
echo "两个服务已就绪。结果目录：$RESULT_DIR"

snapshot() {
  local ep="$1" stem="$2" boundary="$3"
  if ! curl --silent --show-error --fail --max-time 10 "http://127.0.0.1:${ports[$ep]}/metrics" > "$RESULT_DIR/${groups[$ep]}/$stem.metrics-$boundary.txt"; then
    echo "${groups[$ep]} $stem $boundary metrics 采集失败" >&2
  fi
}
start_bench() {
  local ep="$1" repeat="$2" phase="$3" stem="$4"
  kill -0 "${server_pids[$ep]}" 2>/dev/null || fail "${groups[$ep]} 服务提前退出"
  snapshot "$ep" "$stem" before
  bench_command "$ep" "$repeat" "$phase"
  print_command "${bench_cmd[@]}" > "$RESULT_DIR/${groups[$ep]}/$stem.command.txt"
  setsid timeout --signal=TERM --kill-after=30s "${BENCH_TIMEOUT}s" "${bench_cmd[@]}" > "$RESULT_DIR/${groups[$ep]}/$stem.log" 2>&1 &
  bench_pids[$ep]=$!
}
finish_bench() {
  local ep="$1" repeat="$2" phase="$3" stem="$4" rc=0
  wait "${bench_pids[$ep]}" || rc=$?
  stop_group "${bench_pids[$ep]}"
  bench_pids[$ep]=""
  snapshot "$ep" "$stem" after
  "$PYTHON_BIN" "$SCRIPT_DIR/analysis/code/experiment.py" validate "$RESULT_DIR" "${groups[$ep]}" "$phase" "$repeat" "$rc"
}
for phase in warmup eval; do
  rounds="$WARMUP_ROUNDS"
  if [[ "$phase" == eval ]]; then rounds="$REPEATS"; fi
  for ((r=1;r<=rounds;r++)); do
    stem="warmup$r"
    if [[ "$phase" == eval ]]; then stem="repeat$r"; fi
    echo "开始 $phase 第 $r/$rounds 轮；每侧 $NUM_PROMPTS 请求；$(date -Is)"
    failed=0
    start_bench 0 "$r" "$phase" "$stem"
    start_bench 1 "$r" "$phase" "$stem"
    finish_bench 0 "$r" "$phase" "$stem" || failed=1
    finish_bench 1 "$r" "$phase" "$stem" || failed=1
    (( failed == 0 )) || fail "$phase 第 $r 轮未通过校验；停止后续轮次"
  done
done
"$PYTHON_BIN" "$SCRIPT_DIR/analysis/code/experiment.py" summarize "$RESULT_DIR"
echo "完成完整预热 $WARMUP_ROUNDS 轮、评估 $REPEATS 轮：$RESULT_DIR/summary.csv（仅评估）"
