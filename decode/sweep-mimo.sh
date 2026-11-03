#!/usr/bin/env bash
set -Eeuo pipefail

# ==================== 常用调参区 ====================
read -r -a SEQ_COUNTS <<< "${SEQ_COUNTS:-16 32 64 128 256 512 768}"
export TP="${TP:-1}" DP="${DP:-4}" EP="${EP:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export BATCH_TOKEN_EXTRA="${BATCH_TOKEN_EXTRA:-48}"
export KV_LEN="${KV_LEN:-8192}" MAX_OUTPUT="${MAX_OUTPUT:-500}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
export KV_STARTUP_FILL="${KV_STARTUP_FILL:-1}"
export KV_FILL_MEAN="${KV_FILL_MEAN:-0.015}" KV_FILL_STD="${KV_FILL_STD:-0}"
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export NUM_WARMUPS="${NUM_WARMUPS:-10}" SEED="${SEED:-42}"
export MODEL_PATH="${MODEL_PATH:-/host/models/MiMo-V2.6-Flash-MOPD}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.6-Flash-MOPD}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
export VLLM_SOURCE="${VLLM_SOURCE:-/host/vllm}"
export HOST=127.0.0.1 PORT="${PORT:-8001}"
export METRICS_INTERVAL="${METRICS_INTERVAL:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}" DRY_RUN="${DRY_RUN:-0}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}" BENCH_TIMEOUT="${BENCH_TIMEOUT:-3600}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 每组正式请求数、并发数均等于该组 max_num_seqs；batch token 预算 = seqs + EXTRA。
# ====================================================

fail() { echo "错误：$*" >&2; exit 1; }
for name in KV_LEN MAX_OUTPUT MAX_MODEL_LEN NUM_WARMUPS STARTUP_TIMEOUT BENCH_TIMEOUT STOP_TIMEOUT PORT; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || fail "$name 必须为正整数"
done
for name in BATCH_TOKEN_EXTRA SEED; do [[ "${!name}" =~ ^(0|[1-9][0-9]*)$ ]] || fail "$name 必须为非负整数"; done
for name in FORCE_RERUN DRY_RUN; do [[ "${!name}" == 0 || "${!name}" == 1 ]] || fail "$name 只能为 0 或 1"; done
(( NUM_WARMUPS >= 10 )) || fail "至少需要 10 个独立热身请求"
(( ${#SEQ_COUNTS[@]} > 0 )) || fail "SEQ_COUNTS 不能为空"
declare -A seen=()
for seq in "${SEQ_COUNTS[@]}"; do
  [[ "$seq" =~ ^[1-9][0-9]*$ ]] || fail "SEQ_COUNTS 必须为正整数"
  [[ -z "${seen[$seq]:-}" ]] || fail "SEQ_COUNTS 重复：$seq"
  seen[$seq]=1
done
seq_tag="$("$PYTHON_BIN" -c 'import sys; print("_".join(sorted(sys.argv[1:], key=int)))' "${SEQ_COUNTS[@]}")"
model_tag="${SERVED_MODEL_NAME//[^a-zA-Z0-9._-]/_}"
RESULT_DIR="${RESULT_DIR:-$SCRIPT_DIR/results/${model_tag}-TP${TP}-DP${DP}-EP${EP}-seqs${seq_tag}-extra${BATCH_TOKEN_EXTRA}-kv${KV_LEN}-output${MAX_OUTPUT}}"
RESULT_DIR="$("$PYTHON_BIN" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$RESULT_DIR")"
export PYTHONPATH="$VLLM_SOURCE" PYTHONUNBUFFERED=1
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

# 先检查全部启动参数，dry-run 不加载模型、不创建结果目录。
for seq in "${SEQ_COUNTS[@]}"; do
  MAX_NUM_SEQS="$seq" MAX_NUM_BATCHED_TOKENS="$((seq + BATCH_TOKEN_EXTRA))" DRY_RUN=1 bash "$SCRIPT_DIR/mimo.sh" >/dev/null
done
for command in setsid timeout curl flock; do command -v "$command" >/dev/null || fail "缺少命令：$command"; done
"$PYTHON_BIN" - <<'PY'
import math, os
value = float(os.environ["METRICS_INTERVAL"])
if not math.isfinite(value) or value <= 0:
    raise SystemExit("METRICS_INTERVAL 必须为有限正数")
PY
if [[ "$DRY_RUN" == 0 ]]; then
  mkdir -p "$RESULT_DIR"
  exec {sweep_lock_fd}>"$RESULT_DIR/.sweep.lock"
  flock -n "$sweep_lock_fd" || fail "已有 sweep 正在写入 $RESULT_DIR"
fi
plan="$("$PYTHON_BIN" "$SCRIPT_DIR/decode_state.py" init "$RESULT_DIR" "${SEQ_COUNTS[*]}" "$FORCE_RERUN" "$DRY_RUN")"
declare -A states=()
while IFS=$'\t' read -r seq state; do states["$seq"]="$state"; done <<< "$plan"

server_pid="" bench_pid="" sampler_pid="" active_seq=""
stop_group() {
  local pid="$1" deadline=$((SECONDS + STOP_TIMEOUT))
  kill -TERM -- "-$pid" 2>/dev/null || true
  while kill -0 -- "-$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
stop_sampler() {
  if [[ -n "$sampler_pid" ]]; then kill -TERM "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; sampler_pid=""; fi
}
record() { "$PYTHON_BIN" "$SCRIPT_DIR/decode_state.py" record "$RESULT_DIR" "$1" "$2" "$3"; }
cleanup() {
  local exit_code=$?
  if [[ -n "$bench_pid" ]]; then stop_group "$bench_pid"; fi
  stop_sampler
  if [[ -n "$server_pid" ]]; then stop_group "$server_pid"; fi
  if [[ -n "$active_seq" ]]; then
    record "$active_seq" interrupted "$exit_code" || true
    "$PYTHON_BIN" "$SCRIPT_DIR/summarize-mimo.py" "$RESULT_DIR" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
preflight() {
  "$PYTHON_BIN" - "$MODEL_PATH" "$PORT" <<'PY'
import json, pathlib, socket, sys
root, port = pathlib.Path(sys.argv[1]), int(sys.argv[2])
required = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]
missing = [name for name in required if not (root / name).is_file()]
index = root / required[-1]
if index.is_file():
    missing += [name for name in sorted(set(json.loads(index.read_text())["weight_map"].values())) if not (root / name).is_file()]
if missing:
    raise SystemExit("模型尚未下载完整：" + ", ".join(missing))
with socket.socket() as sock:
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit(f"端口 {port} 已有服务；请修改 PORT，不会停止已有服务")
PY
}
wait_ready() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while (( SECONDS < deadline )); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    if curl --silent --fail --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

echo "Decode sweep：GPU=$CUDA_VISIBLE_DEVICES TP=$TP DP=$DP EP=$EP port=$PORT"
echo "seqs=[${SEQ_COUNTS[*]}]；每组请求数=并发=seqs；batch_tokens=seqs+$BATCH_TOKEN_EXTRA"
echo "预制 KV=$KV_LEN，bench 输入=$((KV_LEN + 1))，每条输出=$MAX_OUTPUT；另有 $NUM_WARMUPS 个热身请求"
echo "结果目录：$RESULT_DIR"
for seq in "${SEQ_COUNTS[@]}"; do
  batch=$((seq + BATCH_TOKEN_EXTRA))
  name="batch${batch}-concurrency${seq}"
  if [[ "${states[$seq]}" == complete ]]; then echo "跳过已完成：seqs=$seq batch_tokens=$batch"; continue; fi
  bench_cmd=("$PYTHON_BIN" -m vllm bench serve
    --backend vllm --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions
    --model "$SERVED_MODEL_NAME" --tokenizer "$MODEL_PATH"
    --dataset-name random --random-input-len "$((KV_LEN + 1))" --random-output-len "$MAX_OUTPUT"
    --random-range-ratio 0 --random-prefix-len 0 --num-prompts "$seq"
    --request-rate inf --max-concurrency "$seq" --seed "$SEED"
    --num-warmups "$NUM_WARMUPS" --ready-check-timeout-sec 0 --ignore-eos
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99
    --save-result --save-detailed --result-dir "$RESULT_DIR" --result-filename "$name.json")
  if [[ "$DRY_RUN" == 1 ]]; then
    MAX_NUM_SEQS="$seq" MAX_NUM_BATCHED_TOKENS="$batch" DRY_RUN=1 bash "$SCRIPT_DIR/mimo.sh"
    printf '%q ' "${bench_cmd[@]}"; printf '\n'
    continue
  fi
  preflight
  "$PYTHON_BIN" "$SCRIPT_DIR/decode_state.py" prepare-run "$RESULT_DIR" "$seq"
  active_seq="$seq"
  record "$seq" starting -
  mkdir -p "$RESULT_DIR/server-logs"
  setsid env MAX_NUM_SEQS="$seq" MAX_NUM_BATCHED_TOKENS="$batch" LOG_FILE="$RESULT_DIR/server-logs/$name.log" \
    bash "$SCRIPT_DIR/mimo.sh" > "$RESULT_DIR/server-$name.log" 2>&1 &
  server_pid=$!
  if ! wait_ready; then
    echo "服务启动失败或超时：$RESULT_DIR/server-$name.log" >&2
    stop_group "$server_pid"; server_pid=""
    record "$seq" startup_failed 1
    active_seq=""
    continue
  fi
  curl --silent --fail --max-time 5 "http://127.0.0.1:$PORT/metrics" > "$RESULT_DIR/$name.metrics-before.txt" || true
  "$PYTHON_BIN" "$SCRIPT_DIR/sample-metrics.py" "http://127.0.0.1:$PORT/metrics" \
    "$RESULT_DIR/$name.metrics-series.jsonl" "$METRICS_INTERVAL" "$RESULT_DIR/$name.log" &
  sampler_pid=$!
  record "$seq" running -
  setsid timeout --signal=TERM --kill-after=30s "${BENCH_TIMEOUT}s" "${bench_cmd[@]}" > "$RESULT_DIR/$name.log" 2>&1 &
  bench_pid=$!
  rc=0
  wait "$bench_pid" || rc=$?
  stop_group "$bench_pid"; bench_pid=""
  stop_sampler
  curl --silent --fail --max-time 5 "http://127.0.0.1:$PORT/metrics" > "$RESULT_DIR/$name.metrics-after.txt" || true
  record "$seq" finished "$rc"
  active_seq=""
  "$PYTHON_BIN" "$SCRIPT_DIR/summarize-mimo.py" "$RESULT_DIR" || true
  stop_group "$server_pid"; server_pid=""
done
if [[ "$DRY_RUN" == 0 ]]; then "$PYTHON_BIN" "$SCRIPT_DIR/summarize-mimo.py" "$RESULT_DIR"; fi
