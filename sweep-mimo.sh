#!/usr/bin/env bash
set -Eeuo pipefail

# ==================== 常用调参区 ====================
# 50K 按 50,000 个输入 token 解释；每组 100 请求，输出 1 token（沿用当前模板）。
read -r -a BATCH_TOKENS <<< "${BATCH_TOKENS:-256 512 1024 4096 16384 32768 49152 65536}"
read -r -a CONCURRENCIES <<< "${CONCURRENCIES:-10 30 50 100}"
INPUT_LEN="${INPUT_LEN:-50000}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
# 每组先发送至少 10 个同长度热身请求，再开始计时和采集正式负载。
NUM_WARMUPS="${NUM_WARMUPS:-10}"
SEED="${SEED:-42}"
export TP="${TP:-4}" DP="${DP:-1}" EP="${EP:-0}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
export MAX_OUTPUT=1
export ENABLE_PREFIX_CACHING=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MODEL_PATH="${MODEL_PATH:-/host/models/MiMo-V2.6-Flash-MOPD}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.6-Flash-MOPD}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
export VLLM_SOURCE="${VLLM_SOURCE:-/host/vllm}"
export PORT="${PORT:-8000}"
export HOST=127.0.0.1
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
# 默认续跑；1 表示归档该实验已有文件后，从头重跑。
FORCE_RERUN="${FORCE_RERUN:-0}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-3600}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
# 全程 trace 导出可能较慢，中断时先等待刷新，再清理服务。
PROFILE_FLUSH_TIMEOUT="${PROFILE_FLUSH_TIMEOUT:-300}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_TAG="${SERVED_MODEL_NAME//[^a-zA-Z0-9._-]/_}"
BATCH_TAG="${BATCH_TOKENS[*]}"
BATCH_TAG="${BATCH_TAG// /_}"
# DRY_RUN=1 仅打印全部启动/压测命令，不加载模型、不发请求。
DRY_RUN="${DRY_RUN:-0}"

# 默认关闭 trace；显式 PROFILE=1 时每组全程采集 Perfetto 可读的 CPU/CUDA trace。
# PROFILE_MAX_ITERS=0 表示不限制引擎迭代数。
export PROFILE="${PROFILE:-0}"
export PROFILE_MAX_ITERS="${PROFILE_MAX_ITERS:-0}"
export PROFILE_DELAY_ITERS="${PROFILE_DELAY_ITERS:-0}"
export PROFILE_WITH_STACK="${PROFILE_WITH_STACK:-0}"
export PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-0}"
export PROFILE_WITH_MEMORY="${PROFILE_WITH_MEMORY:-0}"
STATE_HELPER="$SCRIPT_DIR/sweep-state.py"
export INPUT_LEN NUM_PROMPTS NUM_WARMUPS SEED
# ====================================================

fail() { echo "错误：$*" >&2; exit 1; }
for name in INPUT_LEN NUM_PROMPTS NUM_WARMUPS MAX_NUM_SEQS MAX_MODEL_LEN STARTUP_TIMEOUT BENCH_TIMEOUT STOP_TIMEOUT PROFILE_FLUSH_TIMEOUT PORT; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || fail "$name 必须为正整数"
done
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "SEED 必须为非负整数"
[[ "$TP" =~ ^(1|2|4)$ && "$DP" =~ ^(1|2|4)$ ]] || fail "TP、DP 只能为 1、2、4"
(( TP * DP == 4 )) || fail "只允许 TP * DP = 4"
[[ "$EP" == 0 || "$EP" == 1 ]] || fail "EP 只能为 0 或 1"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "DRY_RUN 只能为 0 或 1"
[[ "$FORCE_RERUN" == 0 || "$FORCE_RERUN" == 1 ]] || fail "FORCE_RERUN 只能为 0 或 1"
for name in PROFILE PROFILE_WITH_STACK PROFILE_RECORD_SHAPES PROFILE_WITH_MEMORY; do
  [[ "${!name}" == 0 || "${!name}" == 1 ]] || fail "$name 只能为 0 或 1"
done
for name in PROFILE_MAX_ITERS PROFILE_DELAY_ITERS; do
  [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]] || fail "$name 必须为非负整数"
done
(( INPUT_LEN + MAX_OUTPUT <= MAX_MODEL_LEN )) || fail "输入加输出长度超过 MAX_MODEL_LEN"
(( NUM_WARMUPS >= 10 )) || fail "每组至少需要 10 个热身请求，NUM_WARMUPS 不能小于 10"
(( PORT <= 65535 )) || fail "PORT 不能超过 65535"
(( ${#BATCH_TOKENS[@]} > 0 && ${#CONCURRENCIES[@]} > 0 )) || fail "搜索列表不能为空"
declare -A seen_batches=() seen_concurrencies=()
for batch in "${BATCH_TOKENS[@]}"; do
  [[ "$batch" =~ ^[1-9][0-9]*$ ]] || fail "batch token 上限必须为正整数"
  (( batch >= MAX_NUM_SEQS )) || fail "batch token 上限不能小于 MAX_NUM_SEQS"
  [[ -z "${seen_batches[$batch]:-}" ]] || fail "BATCH_TOKENS 包含重复值：$batch"
  seen_batches[$batch]=1
done
for concurrency in "${CONCURRENCIES[@]}"; do
  [[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || fail "并发数必须为正整数"
  (( concurrency <= NUM_PROMPTS )) || fail "并发数不能大于总请求数"
  [[ -z "${seen_concurrencies[$concurrency]:-}" ]] || fail "CONCURRENCIES 包含重复值：$concurrency"
  seen_concurrencies[$concurrency]=1
done
BATCH_TAG="$("$PYTHON_BIN" -c 'import sys; print("_".join(sorted(sys.argv[1:], key=int)))' "${BATCH_TOKENS[@]}")"
# 固定目录用于 resume；trace 开/关始终为不同实验，batch 列表顺序不影响目录名。
RESULT_DIR="${RESULT_DIR:-$SCRIPT_DIR/results/${MODEL_TAG}-TP${TP}-DP${DP}-EP${EP}-batch${BATCH_TAG}-trace${PROFILE}}"
for command in setsid timeout curl flock; do
  command -v "$command" >/dev/null || fail "缺少命令：$command"
done
export PYTHONPATH="$VLLM_SOURCE"
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# 本机请求不经代理。
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

preflight() {
  # 下载未完成时提前退出，避免反复启动；不触碰下载临时目录。
  "$PYTHON_BIN" - "$MODEL_PATH" "$PORT" <<'PY'
import json, pathlib, socket, sys
root = pathlib.Path(sys.argv[1])
required = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]
missing = [name for name in required if not (root / name).is_file()]
index = root / "model.safetensors.index.json"
if index.is_file():
    shards = set(json.loads(index.read_text())["weight_map"].values())
    missing.extend(sorted(name for name in shards if not (root / name).is_file()))
if missing:
    sys.exit("模型尚未下载完整，缺少：" + ", ".join(missing))
with socket.socket() as sock:
    if sock.connect_ex(("127.0.0.1", int(sys.argv[2]))) == 0:
        sys.exit("端口已被占用，请先停止现有服务或修改 PORT；不会停止已有服务。")
PY
}

if [[ "$DRY_RUN" == 0 ]]; then
  mkdir -p "$RESULT_DIR"
  RESULT_DIR="$(cd -- "$RESULT_DIR" && pwd)"
  # flock 在进程退出后自动释放，崩溃不留下阻止 resume 的陈旧锁。
  exec {sweep_lock_fd}>"$RESULT_DIR/.sweep.lock"
  flock -n "$sweep_lock_fd" || fail "已有 sweep 正在写入该实验：$RESULT_DIR"
fi
plan_output="$("$PYTHON_BIN" "$STATE_HELPER" init "$RESULT_DIR" "${BATCH_TOKENS[*]}" "${CONCURRENCIES[*]}" "$FORCE_RERUN" "$DRY_RUN" "$SCRIPT_DIR/results")"
declare -A resume_plan=()
while IFS=$'\t' read -r batch concurrency state; do
  resume_plan["$batch:$concurrency"]="$state"
done <<< "$plan_output"
export LOG_DIR="$RESULT_DIR/server-logs"
printf 'TP=%s DP=%s EP=%s GPU=%s input_len=%s num_prompts=%s max_model_len=%s seed=%s\n' \
  "$TP" "$DP" "$EP" "$CUDA_VISIBLE_DEVICES" "$INPUT_LEN" "$NUM_PROMPTS" "$MAX_MODEL_LEN" "$SEED"
echo "每组先发送 $NUM_WARMUPS 个热身请求，再发送 $NUM_PROMPTS 个正式请求；热身不计入性能统计或 trace。"
if [[ "$PROFILE" == 1 ]]; then
  echo "已启用 CPU/CUDA profiling，采集迭代上限=$PROFILE_MAX_ITERS；性能统计包含采集开销。"
fi
server_pid=""
bench_pid=""

# 仅清理本脚本创建的进程组，包含该服务/负载生成器的子进程。
stop_group() {
  local pid="$1" deadline=$((SECONDS + STOP_TIMEOUT))
  kill -TERM -- "-$pid" 2>/dev/null || true
  while kill -0 -- "-$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
  if kill -0 -- "-$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  if [[ -n "$bench_pid" ]]; then stop_group "$bench_pid"; fi
  if [[ "$PROFILE" == 1 && -n "$server_pid" ]]; then
    # 中断时尽量刷新尚未落盘的 trace，再停止服务。
    curl --silent --fail --max-time "$PROFILE_FLUSH_TIMEOUT" -X POST "http://127.0.0.1:$PORT/stop_profile" >/dev/null 2>&1 || true
  fi
  if [[ -n "$server_pid" ]]; then stop_group "$server_pid"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

record() {
  "$PYTHON_BIN" "$STATE_HELPER" record "$RESULT_DIR" "$1" "$2" "$3" "$4" "${6:-disabled}"
}
collect_profile() {
  # stop_profile 返回后所有 worker 已完成导出；将本组文件移入独立目录。
  "$PYTHON_BIN" - "$PROFILE_DIR" "$1" "$((TP * DP))" <<'PY'
import pathlib, shutil, sys
source, target = map(pathlib.Path, sys.argv[1:3])
target.mkdir(parents=True, exist_ok=False)
for item in source.iterdir():
    shutil.move(str(item), str(target / item.name))
traces = list(target.rglob("*.pt.trace.json")) + list(target.rglob("*.pt.trace.json.gz"))
print(f"Trace：{target}，文件数={len(traces)}")
if len(traces) < int(sys.argv[3]):
    sys.exit("未收齐 4 个 worker 的 trace，请检查压测日志及服务 profiler 日志")
PY
}
wait_ready() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while (( SECONDS < deadline )); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    if curl --silent --fail --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null; then return 0; fi
    sleep 2
  done
  return 1
}

for batch in "${BATCH_TOKENS[@]}"; do
  echo "===== max_num_batched_tokens=$batch ====="
  pending_concurrencies=()
  for concurrency in "${CONCURRENCIES[@]}"; do
    if [[ "${resume_plan[$batch:$concurrency]}" == complete ]]; then
      echo "跳过已完成：batch=$batch concurrency=$concurrency trace=$PROFILE"
    else
      pending_concurrencies+=("$concurrency")
    fi
  done
  if (( ${#pending_concurrencies[@]} == 0 )); then continue; fi
  server_log="$RESULT_DIR/server-batch${batch}.log"
  export PROFILE_DIR="$RESULT_DIR/traces/.staging-batch${batch}"
  if [[ "$DRY_RUN" == 0 ]]; then
    preflight
    "$PYTHON_BIN" "$STATE_HELPER" archive-staging "$RESULT_DIR" "$batch"
    setsid env MAX_NUM_BATCHED_TOKENS="$batch" LOG_FILE="$LOG_DIR/batch${batch}.log" bash "$SCRIPT_DIR/mimo.sh" >> "$server_log" 2>&1 &
    server_pid=$!
    if ! wait_ready; then
      echo "服务启动失败或超时：$server_log"
      for concurrency in "${pending_concurrencies[@]}"; do record "$batch" "$concurrency" startup_failed 1 "-" not_captured; done
      stop_group "$server_pid"
      server_pid=""
      continue
    fi
  else
    printf 'PROFILE=%q PROFILE_DIR=%q MAX_NUM_BATCHED_TOKENS=%q bash %q\n' "$PROFILE" "$PROFILE_DIR" "$batch" "$SCRIPT_DIR/mimo.sh"
  fi

  for concurrency in "${pending_concurrencies[@]}"; do
    name="batch${batch}-concurrency${concurrency}"
    bench_cmd=("$PYTHON_BIN" -m vllm bench serve
      --backend vllm --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions
      --model "$SERVED_MODEL_NAME" --tokenizer "$MODEL_PATH"
      --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$MAX_OUTPUT"
      --random-range-ratio 0 --random-prefix-len 0 --num-prompts "$NUM_PROMPTS"
      --request-rate inf --max-concurrency "$concurrency" --seed "$SEED"
      --num-warmups "$NUM_WARMUPS" --ready-check-timeout-sec 0 --ignore-eos
      --percentile-metrics ttft,e2el --metric-percentiles 50,90,95,99
      --save-result --save-detailed --result-dir "$RESULT_DIR" --result-filename "$name.json")
    if [[ "$PROFILE" == 1 ]]; then bench_cmd+=(--profile); fi
    printf '%q ' "${bench_cmd[@]}"; printf '\n'
    if [[ "$DRY_RUN" == 1 ]]; then continue; fi
    "$PYTHON_BIN" "$STATE_HELPER" prepare-run "$RESULT_DIR" "$batch" "$concurrency"
    # 每组单独保存服务 metrics 快照，观察 KV cache、抢占等计数变化。
    curl --silent --fail --max-time 5 "http://127.0.0.1:$PORT/metrics" > "$RESULT_DIR/$name.metrics-before.txt" || true
    setsid timeout --signal=TERM --kill-after=30s "${BENCH_TIMEOUT}s" "${bench_cmd[@]}" > "$RESULT_DIR/$name.log" 2>&1 &
    bench_pid=$!
    rc=0
    wait "$bench_pid" || rc=$?
    # timeout 可能留下子进程；退出后清理该负载生成器的整个进程组。
    stop_group "$bench_pid"
    bench_pid=""
    curl --silent --fail --max-time 5 "http://127.0.0.1:$PORT/metrics" > "$RESULT_DIR/$name.metrics-after.txt" || true
    profile_status=disabled
    trace_dir="-"
    if [[ "$PROFILE" == 1 ]]; then
      trace_dir="traces/$name"
      profile_status=captured
      # vLLM bench 的 --profile 会在生成数据之后、发送负载之前启用采集，
      # 并在负载完成后调用 /stop_profile。超时/异常时先补一次 stop 以保存 trace。
      if (( rc != 0 )); then
        curl --silent --show-error --fail --max-time "$PROFILE_FLUSH_TIMEOUT" -X POST "http://127.0.0.1:$PORT/stop_profile" >> "$RESULT_DIR/$name.log" 2>&1 || true
      fi
      if ! collect_profile "$RESULT_DIR/$trace_dir"; then profile_status=missing_or_partial; fi
      # 若 bench 静默忽略启停 API 失败，也要标记本组采集不完整。
      # 使用已配置的 Python 检查日志，不依赖运行 shell 中可能不存在的 rg。
      if ! "$PYTHON_BIN" - "$RESULT_DIR/$name.log" <<'PY'
import pathlib, sys
lines = set(pathlib.Path(sys.argv[1]).read_text(errors="replace").splitlines())
sys.exit(0 if {"Profiler started", "Profiler stopped"} <= lines else 1)
PY
      then
        profile_status=missing_or_partial
      fi
    fi
    record "$batch" "$concurrency" finished "$rc" "$name.json" "$profile_status" "$trace_dir"
    echo "$name 完成，退出码=$rc；日志：$RESULT_DIR/$name.log"
    # 仅 bench 异常/超时时恢复服务；trace 归档或状态异常单独记录，
    # 不触发模型重载。同一 batch 下的正常测试始终复用同一个服务。
    if (( rc != 0 )) && [[ "$concurrency" != "${pending_concurrencies[-1]}" ]]; then
      stop_group "$server_pid"
      server_pid=""
      setsid env MAX_NUM_BATCHED_TOKENS="$batch" LOG_FILE="$LOG_DIR/batch${batch}.log" bash "$SCRIPT_DIR/mimo.sh" >> "$server_log" 2>&1 &
      server_pid=$!
      if ! wait_ready; then
        echo "服务恢复失败，停止 sweep：$server_log" >&2
        "$PYTHON_BIN" "$SCRIPT_DIR/summarize-mimo.py" "$RESULT_DIR" "$NUM_PROMPTS" "$INPUT_LEN"
        exit 1
      fi
    fi
  done
  if [[ -n "$server_pid" ]]; then stop_group "$server_pid"; server_pid=""; fi
done

if [[ "$DRY_RUN" == 0 ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/summarize-mimo.py" "$RESULT_DIR" "$NUM_PROMPTS" "$INPUT_LEN"
fi
echo "结果目录：$RESULT_DIR"
