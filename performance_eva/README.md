# 独立模型部署与性能评估

模型部署和负载发射是两个独立进程。部署脚本常驻运行，`run_load.sh` 调用 `per_dataset.py` 发负载并管理实验结果。以下命令在仓库的 `performance_eva` 下执行。请保留同级 `profile` 与 `aggre_or_not` 目录，以便复用公共模块。

## 1. 部署模型

任选一种，在服务器终端启动；看到 `READY` 后再发负载：

```bash
bash deploy_aggre.sh
bash deploy_pd.sh
bash deploy_pd_6p2d.sh
```

三个脚本默认都占用八张 GPU，必须依次运行；端口隔离不代表可以在同一组 GPU 上同时部署。Ctrl-C 停止本次部署及其子进程，不会终止已有的其他服务。预检发现选定 GPU 已占用会拒绝启动。

| 入口 | 默认拓扑 | EP | EPLB | 对外 HTTP | P / D HTTP | DP RPC | Master | NIXL P / D 起点 |
|---|---|---|---|---|---|---|---|---|
| deploy_aggre.sh | DP2 × TP4 | 开 | 开 | 19000 | — | 31050 | 31001 | — |
| deploy_pd.sh | P DP2 × TP2；D DP4 × TP1 | 开 | 开 | 19100 | 19110 / 19120 | 31150 / 31250 | 31101 / 31201 | 6100 / 6200 |
| deploy_pd_6p2d.sh | P DP3 × TP2；D DP2 × TP1 | 开 | 开 | 19200 | 19210 / 19220 | 31350 / 31450 | 31301 / 31401 | 6300 / 6400 |

NIXL 每端占用 `[起点, 起点 + DP - 1]`。对外监听 `0.0.0.0`，P/D 后端仅监听 `127.0.0.1`。负载客户端使用系统临时源端口，无需启动监听服务。跨机使用服务器实际 IP，不能使用 `0.0.0.0` 作为目标地址。

三个部署脚本均完整、独立列出 GPU、TP、DP、EP、EPLB、投机等参数，支持环境变量覆盖。6P2D 不再调用 `deploy_pd.sh`；修改一个脚本的默认值不会改变其他脚本。默认模型 `/models/MiMo-V2.6-Flash-MOPD`，对外模型名 `MiMo-V2.6-Flash-MOPD`，三种模式各端均默认 `MAX_NUM_BATCHED_TOKENS=65536`、`MAX_NUM_SEQS=512`，EP/EPLB 开启；prefix caching 在 aggre 默认关闭、在 pd / pd_6p2d 默认开启，Nsight 关闭。这里 batch 表示每个引擎调度步的 token 预算，max_num_seqs 表示引擎允许的最大序列数，并非发射器的请求数或并发参数。

```bash
# 仅打印完整配置和命令，不占 GPU、不写部署目录
DRY_RUN=1 bash deploy_pd_6p2d.sh

# 检查模型、依赖、GPU、端口，不启动服务；保留预检记录
PREFLIGHT_ONLY=1 bash deploy_pd.sh

# 修改模型、服务名、对外端口、EP
MODEL_PATH=/models/example SERVED_MODEL_NAME=example \
  PROXY_PORT=19300 ENABLE_EP=1 bash deploy_pd.sh
```

`PYTHON_BIN` 默认 `/usr/bin/python3`；`VLLM_SOURCE` 默认 `/host/vllm`。部署依赖与 `aggre_or_not` 相同，复用其 `code/collect_pd.py` 的命令构造、预检和进程管理，不执行采集入口、不修改原目录文件。`DEPS_DIR` 默认指向 `../aggre_or_not/.deps`，可覆盖以使用已有 NIXL 安装。

接口提供 `/health`、`/v1/models`、`/v1/completions`、`/v1/chat/completions`。PD 请求先 prefill 再 decode，缺失 KV 传输元数据返回错误；支持流式输出并转发会话和请求标识。

### 独立设置 TP / DP / EP / EPLB

- aggre：`TP`、`DP`、`ENABLE_EP`、`ENABLE_EPLB`。
- pd / pd_6p2d：`P_TP`、`P_DP`、`P_ENABLE_EP`、`P_ENABLE_EPLB`；D 端使用对应的 `D_` 参数。
- PD 保留 `ENABLE_EP` / `ENABLE_EPLB` 作为本脚本的统一默认值，显式的 P/D 开关优先。EPLB 要求**对应端** EP 开启。
- `EPLB_WINDOW_SIZE`、`EPLB_STEP_INTERVAL`、`EPLB_NUM_REDUNDANT_EXPERTS`、`EPLB_USE_ASYNC` 和 `EPLB_CONFIG` 在每个脚本中独立列出，作用于本次部署中开启 EPLB 的端。
- EP 组大小由该端 TP×DP 决定，不另设 EP 数值；每端 TP×DP 必须匹配 GPU 数量，全部 GPU 不可重复。

```bash
# 修改 aggre 的 TP / DP；显式关闭 EPLB 和 EP
TP=2 DP=4 ENABLE_EP=0 ENABLE_EPLB=0 bash deploy_aggre.sh

# P 关闭 EP/EPLB，D 开启 EP/EPLB
P_ENABLE_EP=0 P_ENABLE_EPLB=0 D_ENABLE_EP=1 D_ENABLE_EPLB=1 \
  bash deploy_pd_6p2d.sh

# 两端统一开启；pd 与 pd_6p2d 均可使用
ENABLE_EP=1 ENABLE_EPLB=1 bash deploy_pd.sh
```

`deployment.json` 的 `servers[]` 记录各端实际 TP、DP、EP、EPLB 和最终命令；顶层 `ep` / `eplb` 表示是否至少一端启用。

6P2D 默认 `EPLB_NUM_REDUNDANT_EXPERTS=2`：MiMo 的 256 个逻辑专家加 2 个副本得到
258 个物理专家，可被 P 的 EP6 和 D 的 EP2 整除。aggre / pd 默认冗余专家数为 0。
其他模型或拓扑需按实际专家数调整。EPLB 需要已完成 MiMo 适配的 vLLM，见
[补丁说明](../patches/eplb/README.md)。

本机 6P2D 的 D 端曾在 `MAX_MODEL_LEN=65536` 时因 KV 显存不足启动失败。
可先使用 `MAX_MODEL_LEN=32768 bash deploy_pd_6p2d.sh`；batch token 预算仍保持 65536。
这一缩短上下文的配置尚未完成本机完整 GPU 推理验收，应以实际启动日志为准。


### Prefix caching 开关

每个部署脚本独立提供 `ENABLE_PREFIX_CACHING`，`1` 开启、`0` 关闭；aggre 默认 `0`，pd / pd_6p2d 默认 `1`。PD 模式同时应用到 P、D 两端。实际取值会保存到 `deployment.json`，启动命令只包含对应的一种开关。

```bash
ENABLE_PREFIX_CACHING=1 bash deploy_aggre.sh
ENABLE_PREFIX_CACHING=1 bash deploy_pd.sh
ENABLE_PREFIX_CACHING=1 bash deploy_pd_6p2d.sh
```

修改部署脚本或环境变量后，需要重新启动对应服务才生效；运行中的服务不会自动更新参数。

共享前缀负载示例（`--mode` 应与已启动服务一致）：

```bash
bash run_load.sh --mode pd --parallel 1 --number 20 --warmup-num 6 \
  --prefix-length 4096 --input-length 128 --output-length 16 \
  --prepared-dataset-path ''
```

这里总输入长度为 4096 + 128。`--multi-turn` 模式不使用 `--prefix-length`。
在 P / D 各自的 `/metrics` 中分别查看本地 prefix cache 查询与命中，
以及外部 KV 查询与命中；NIXL 外部 KV 命中不等于 D 的本地缓存命中。


### 投机解码接口

接口复用 `../profile/decode/speculative_config.py`，与 `../profile/decode/capture.sh` 的解析规则一致：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `SPECULATIVE_METHOD` | `none` | `none` / `mtp` / `dflash` |
| `NUM_SPECULATIVE_TOKENS` | `7` | 每步草稿 token 数，启用时必须为正整数 |
| `DRAFT_MODEL_PATH` | 空 | MTP 可空，由 vLLM 使用目标模型；DFlash 必须提供草稿模型路径 |
| `SPECULATIVE_CONFIG` | 空 | JSON 额外配置，覆盖同名基础参数；method 必须与 METHOD 一致 |

`none` 不向 vLLM 传 `--speculative-config`；此时不能设置非空 `SPECULATIVE_CONFIG`。默认保持真实采样，参考脚本里的 synthetic 接受长度配置须显式开启。

```bash
# MTP：使用目标模型的 MTP 权重
SPECULATIVE_METHOD=mtp NUM_SPECULATIVE_TOKENS=7 bash deploy_aggre.sh

# DFlash：指定草稿模型。三种部署入口都接受相同参数。
SPECULATIVE_METHOD=dflash NUM_SPECULATIVE_TOKENS=7 \
  DRAFT_MODEL_PATH=/models/MiMo-V2.6-Flash-MOPD/dflash \
  bash deploy_pd_6p2d.sh

# 与参考 capture.sh 一样的 synthetic 接受长度实验；先用 dry-run 检查命令
SPECULATIVE_METHOD=mtp NUM_SPECULATIVE_TOKENS=7 \
  SPECULATIVE_CONFIG='{"rejection_sample_method":"synthetic","synthetic_acceptance_length":4.2}' \
  DRY_RUN=1 bash deploy_pd.sh
```

当前 NIXL 对 MTP/DFlash 的方法、草稿模型等进行兼容校验，所以 PD 的两端使用同一份投机配置；P 请求仍限制为一步 prefill。配置会保存到部署记录及各端命令中。投机是部署参数，负载接口、发射参数和默认端口保持一致。Synthetic 模式用于模拟接受长度的性能实验，不能作为真实接受率或回答质量结果。

## 2. 准备发射器

在发射机器单独安装负载依赖，不需要安装 vLLM，也不需要 GPU：

```bash
python3 -m venv .venv-load
.venv-load/bin/python -m pip install -r requirements-load.txt
export LOAD_PYTHON_BIN="$PWD/.venv-load/bin/python"
```

环境缺少 `ensurepip` 时也可用 `uv venv .venv-load` 和 `uv pip install --python .venv-load/bin/python -r requirements-load.txt`。部署用 `PYTHON_BIN`、负载用 `LOAD_PYTHON_BIN`，二者独立。

## 3. 发负载

```bash
# 与部署分开启动；默认目标为本机该模式的公共端口
bash run_load.sh --mode aggre --parallel '2 4 8 16' --number 50

# 远端服务器；tokenizer 路径属于发射器本机
bash run_load.sh --mode pd_6p2d --ip 192.168.1.100 \
  --model MiMo-V2.6-Flash-MOPD \
  --tokenizer-path /models/MiMo-V2.6-Flash-MOPD \
  --parallel '1 2 4' --number 20 --input-length 1024 --output-length 128

# 读取部署记录里的模式、端口、模型名；跨机器仍需指定 --ip
bash run_load.sh --deployment deployments/pd/部署ID/deployment.json \
  --parallel 1 --number 2 --input-length 128 --output-length 16

# 聊天和多轮；multi-turn 自动使用 chat/completions
bash run_load.sh --mode pd --use-chat --parallel 2 --number 10 \
  --input-length 512 --output-length 64
bash run_load.sh --mode pd --multi-turn --min-turns 2 --max-turns 3 \
  --parallel 2 --number 4 --input-length 128 --output-length 32 \
  --inject-session-id --session-mode per_conv --inject-req-ids

# 长稳；未传 number 时由 duration 控制停止，已发请求仍会完成
bash run_load.sh --mode aggre --parallel 4 --duration 60 \
  --warmup-num 4 --input-length 1024 --output-length 128

# 自定义实验分类、结果位置；预览不发请求、不创建目录
bash run_load.sh --mode pd --label short_chat --results-root /data/eva \
  --use-chat --input-length 128 --output-length 32 --dry-run

bash run_load.sh --help
bash run_load.sh --perf-help
```

默认负载为输入 50000、输出 500、每档 50 请求、1 轮、不预热；aggre 默认并发 `2 4 8 16`，PD 两种模式为 `2 4 8 16 32`。`--number` 单值广播到所有并发档，多值必须与 `--parallel` 等长。`--rate`、`--loop`、长度区间、数据集及 Header 参数沿用 `per_dataset.py`。多轮的 number 代表对话数。超时参数 `--read-timeout` 控制请求读取，duration 为软退出预算，不是强制终止时间。

数据集可以通过 `per_dataset.py --prepare-dataset` 预生成，再用 `--prepared-dataset-path` 或 `--multi-turn-dataset custom_multi_turn --multi-turn-dataset-path` 复用。请显式提供数据集路径，避免工作目录中的默认 `prompts_random.txt` 意外影响实验。单轮自定义文本无需 tokenizer 时可传 `--tokenizer-path ''`，此时依赖服务返回 usage；随机负载必须提供有效 tokenizer。

## 4. 结果与退出状态

```text
deployments/<模式>/<微秒时间戳_UUID>/
  deployment.json           # 状态、拓扑、端口、服务命令
  aggre|prefill|decode/      # 模型服务日志和命令
  proxy.log                 # PD 代理日志
results/<模式>/<模型>/<负载摘要_标签_参数哈希>/<微秒时间戳_UUID>/
  manifest.json             # 有效负载参数、版本、命令、目标、状态、部署记录
  load.log
  perf/
    summary.csv
    summary.xlsx
    request_map.csv         # 启用 Header / 多轮插件时生成
    request_ids.log
    loop_000/sweep_000_p2_n50/.../
      benchmark_summary.json
      benchmark_data.db
      ...                   # EvalScope 原始报告、时间线等
```

每次目录排他创建，同模型同负载重复执行或同时运行不会覆盖。每个 loop/sweep 单独存储，只汇总本次结果；失败也保留日志和已有产物，manifest 标记 `failed` 或 `interrupted`，发射器返回非零。出现零成功请求或失败请求时仍导出可用汇总，然后返回非零。Ctrl-C 仅停止本次发射器。

`run_load.sh` 统一管理输出路径，使用 `--results-root` 修改根目录。直接调用 `per_dataset.py` 可用 `--output-dir` 指定**尚不存在**的独占目录，并使用 `--output-csv`、`--output-xlsx`、`--output-req-map` 自定义文件名；不指定目录时自动生成唯一目录。汇总兼容旧版与 EvalScope 1.14 的吞吐、TPOT 字段，其他原始指标原样保留。

## 5. 验证

```bash
# 在负载环境安装测试依赖
"$LOAD_PYTHON_BIN" -m pip install pytest
"$LOAD_PYTHON_BIN" -m pytest -q tests
```

测试使用模拟 HTTP 上游和实际 EvalScope，覆盖部署 dry-run、端口冲突、PD 聊天/单轮/流式/KV 错误、多轮评估、汇总和输出隔离。模拟结果不能当成模型性能。实际模型验收应在 GPU 空闲时分别启动三种部署，用 `--parallel 1 --number 2 --input-length 128 --output-length 16` 完成单轮、聊天及多轮冒烟。
