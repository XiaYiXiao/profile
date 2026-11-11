# 独立模型部署与性能评估

模型部署和负载发射是两个独立进程。部署脚本常驻运行，`run_load.sh` 调用 `per_dataset.py` 发负载并管理实验结果。以下命令在仓库的 `performance_eva` 下执行。请保留同级 `profile` 与 `aggre_or_not` 目录，以便复用公共模块。

## 本容器快速开始

部署使用 `/usr/bin/python3 -I` 和镜像已安装的 vLLM / Mooncake，负载使用独立 `.venv`。
部署入口会清除 `VLLM_SOURCE`、`DEPS_DIR`、`PYTHONPATH`、`PYTHONHOME`；不加载工作区 vLLM 源码或额外 `.deps`。
当前镜像版本为 vLLM 0.31.0、mooncake-transfer-engine-cuda13 0.3.13.post1；实际导入路径记录在 `deployment.json` 的 `preflight` 中。
`MOE_BACKEND` 默认留空，由镜像 vLLM 自动选择；可通过环境变量显式指定后端。
Mooncake 默认使用 `rdma`，不根据设备检测自动回退到 TCP；显式 `KV_EXTRA_CONFIG` 可覆盖协议。
首次运行先安装负载依赖（不会改动部署环境）：

```bash
cd /host/profiler/performance_eva
bash setup_load.sh
PREFLIGHT_ONLY=1 bash deploy_aggre.sh
bash deploy_aggre.sh
```

看到 `READY` 后，在另一终端运行短负载，确认成功后再扩大并发与长度：

```bash
cd /host/profiler/performance_eva
bash run_load.sh --mode aggre --parallel 1 --number 2 \
  --input-length 128 --output-length 16 --warmup-num 0 \
  --prepared-dataset-path ''
```

`run_load.sh` 自动选择本目录 `.venv/bin/python`，也可通过 `LOAD_PYTHON_BIN` 覆盖。
它实际调用 `per_dataset.py`；也可直接执行 `.venv/bin/python per_dataset.py --help`。
更换模型时建议用 `--deployment <READY 输出的 deployment.json>`，自动读取端口、模型名及 tokenizer 路径。
PD 模式将部署入口和 `--mode` 同时改为 `pd` 或 `pd_6p2d`；三种部署依次执行。

aggre 启动至 READY 后，可用 `bash run_rate_sweep.sh` 顺序运行 rate
**2、3、4、6、8、10、14**。每轮 300 请求、parallel=512、prefix=8000、input=1000、
output=1000、warmup=0、metrics 间隔 1 秒，使用 chat 和内存生成的数据。
每个 rate 单独调用 `run_load.sh` 并清理 Prefix KV cache，结果和 metrics 独立保存，
标签自动使用 `prefix8k_new1k_out1k_r<rate>`。任一轮失败会停止后续扫描。
各轮沿用默认 seed，因此生成相同请求，便于比较速率。
`bash run_rate_sweep.sh --dry-run` 可只检查七轮参数，不发请求或创建结果目录。

### 每轮负载前清理 Prefix KV cache

三个部署入口默认 `ENABLE_RESET_PREFIX_CACHE=1`，映射到镜像 vLLM 的
`VLLM_SERVER_DEV_MODE=1`，使 `/reset_prefix_cache` 可用。这会同时开启该版本的其他开发接口，
适用于本实验环境。设为 `0` 可关闭。已启动的旧服务需要重新部署一次。

`run_load.sh` 默认执行 `--reset-prefix-cache`：先检查所有后端、所有 DP 指标中的
running/waiting 为零且持续至少 6 秒，再 POST 重置接口并要求返回 `success: true`，
然后才建立本轮 metrics 基线并调用 `per_dataset.py`。PD 通过代理分别重置 P、D。
镜像首次启动时 gauge 可能只有声明、没有样本；仅在这些声明存在且已暴露的负载计数器全为零时，
允许按尚未开始处理请求的初始状态检查，并记录 `initial_gauges_unset`。
接口缺失、服务持续忙碌、指标不可用或重置失败都会中止负载，原因保存在本轮
`manifest.json` 的 `prefix_cache_reset` 中。等待空闲默认最多 60 秒，可用
`--reset-prefix-cache-timeout` 调整；显式 `--no-reset-prefix-cache` 可进行热缓存实验。
即使 `--no-server-metrics`，默认重置仍需读取 `/metrics` 来检查空闲状态。

每次调用 `run_load.sh` 清理一次；一次调用内的 parallel 扫描、loop 和 warmup
之间不清理。若要每个 rate/parallel 都从冷缓存开始，请分别调用，且设置 `--warmup-num 0`。
测试期间应独占服务，不要并行启动其他负载。此流程不锁定其他客户端的请求入口。
只清理本地 Prefix KV cache，不清零 Prometheus 计数器，也不重置 EPLB、模型权重或编译缓存，
不请求清理 connector 管理的外部缓存。
镜像 vLLM 的 DP 重置会广播所有引擎，但 HTTP 只返回首个 DP 的结果；
空闲检查覆盖所有可见 DP 指标，验收时还应核对各 DP 的 `Successfully reset prefix cache` 日志。

本容器 aggre 实测（2026-11-10）：保持镜像 vLLM、DP2 × TP4、自动 MoE 后端和 DFlash 配置，
用相同 seed 的 4 条请求测试 prefix=8000、input=1000、output=1000、parallel=1、rate=4、warmup=0。
清缓存、不清缓存重复、再次清缓存三轮命中率分别为 **0%、99.56%、0%**，每轮 4/4 成功。
两个 DP 均有清缓存成功日志；每轮 metrics 的 query 增量均为 36000，累计计数器保持连续，
采集失败、计数器重置、序列缺口均为 0。114 项 CPU 回归测试通过；PD/6P2D 本次未做 GPU 实测。
结果和指标路径见 [Prefix cache 重置验证索引](validation/reset_prefix_cache_smoke.json)。

### 镜像环境验证记录（2026-11-10）

当前部署只使用镜像安装的 vLLM 0.31.0 和 Mooncake 0.3.13.post1，路径均在
`/usr/local/lib/python3.12/dist-packages`。没有修改或安装镜像依赖。
101 项回归测试通过，包含三个入口忽略外部源码/依赖路径的测试。
以下实际推理验证当时使用 Marlin 和 TCP；当前默认已恢复 MoE 自动选择和 Mooncake RDMA，尚未重新进行 GPU 推理验证。
PD 实际部署通过 `per_dataset.py` 完成并发 1 / 2 各 2 请求：总成功 4、失败 0，
输入 128、输出 16，已生成 CSV/XLSX/HTML。使用 Mooncake TCP 与 DFlash synthetic 接受长度 4.2。
aggre 和 6P2D 本次仅做镜像环境预检，未重新执行完整推理测试。
测试服务在验证后停止；实际导入路径和结果见 [镜像环境验证索引](validation/image_runtime_smoke.json)。

### 历史源码环境验证记录（2026-11-10，非当前镜像环境验证）

以下历史结果使用 `/host/vllm` 和额外 `.deps`，不能作为当前仅镜像环境的验收。
8 × H20-3e、MiMo-V2.6-Flash-MOPD，三个旧部署入口均达到 READY。
通过 `run_load.sh` 实际调用 `per_dataset.py`：输入 128、输出 16，并发 1 / 2 各 2 请求，
aggre / pd / pd_6p2d 分别成功 4、失败 0；6P2D 使用 chat 接口，其余使用 completions。
各次运行均生成 CSV、XLSX、HTML 和指标记录；98 项回归测试通过。
PD 使用 TCP 和脚本原有的 DFlash synthetic 接受长度 4.2，这只是运行链路验证，不是 RDMA 或自然接受率性能结论。
结果路径见 [验证索引](validation/container_smoke.json)。测试服务在验证后停止。

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
| deploy_pd.sh | P DP2 × TP2；D DP2 × TP2 | 开 | 开 | 19100 | 19110 / 19120 | 31150 / 31250 | 31101 / 31201 | 6100 / 6200 |
| deploy_pd_6p2d.sh | P DP3 × TP2；D DP1 × TP2 | 开 | 开 | 19200 | 19210 / 19220 | 31350 / 31450 | 31301 / 31401 | 6300 / 6400 |

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

`PYTHON_BIN` 默认 `/usr/bin/python3`。部署仅复用 `aggre_or_not` 的命令构造、GPU/端口预检和进程管理；环境初始化由本目录独立实现，不使用其源码/依赖路径注入逻辑。服务、代理和部署控制进程均使用 `-I` 隔离模式。镜像缺少依赖时直接报错，不回退到工作区依赖，也不修改镜像中的包。

接口提供 `/health`、`/v1/models`、`/v1/completions`、`/v1/chat/completions`。PD 请求先 prefill 再 decode，缺失 KV 传输元数据返回错误；支持流式输出并转发会话和请求标识。

### 选择 Mooncake 进行 P/D KV 传输

`deploy_pd.sh` 和 `deploy_pd_6p2d.sh` 支持 `KV_CONNECTOR=MooncakeConnector`（默认）
或 `KV_CONNECTOR=NixlConnector`。Mooncake 使用 `MooncakeConnector` 直接传输，
无需部署 Mooncake Store。当前 CUDA 13 镜像已安装
`mooncake-transfer-engine-cuda13==0.3.13.post1`，部署脚本直接使用该包。

```bash
# 先检查生成的 P/D 和代理命令；不启动模型
KV_CONNECTOR=MooncakeConnector DRY_RUN=1 bash deploy_pd_6p2d.sh

# 检查 Python 依赖、模型参数、GPU 占用及端口；不启动模型
KV_CONNECTOR=MooncakeConnector PREFLIGHT_ONLY=1 bash deploy_pd_6p2d.sh

# 使用 Mooncake 启动（仍需满足下述模型兼容限制）
KV_CONNECTOR=MooncakeConnector bash deploy_pd_6p2d.sh
```

也可以直接修改对应脚本中的 `KV_CONNECTOR` 默认值。切换后 P 使用
`kv_producer`，D 使用 `kv_consumer`，代理使用 Mooncake 的 `transfer_id` 和
`remote_bootstrap_addr` 协议。代理从 bootstrap 获取完整的 P DP/TP 注册信息，
轮询选择 P 的 DP rank，并将对应 `engine_id` 传给 D；bootstrap 信息不完整时拒绝启动。

`P_MOONCAKE_BOOTSTRAP_PORT` 在 `pd` 中默认 6500，在 `pd_6p2d` 中默认 6600，
每个 P 实例使用一个 bootstrap 端口，内部 worker 通信端口动态分配。
脚本按单机部署设置 `VLLM_HOST_IP=127.0.0.1`；跨机需要另外配置地址和路由。
Mooncake 模式不占用配置中的 NIXL side-channel 端口，也不执行 NIXL 依赖预检。

`KV_EXTRA_CONFIG` 直接传给所选 connector。Mooncake 默认 `mooncake_protocol=rdma`，不自动回退到 TCP。显式配置优先，
例如 `KV_EXTRA_CONFIG='{"mooncake_protocol":"rdma","device_name":"mlx5_0"}'`。
NIXL 的 `{"backends":["UCX"]}` 不能用于 Mooncake。GPU RDMA 注册还取决于主机驱动、
网卡和 peermem/DMA-BUF 环境；`PREFLIGHT_ONLY` 通过仅表示导入/API、GPU 占用及端口检查通过，
不代表实际 GPU KV 传输已验证。

**兼容限制：** Mooncake 实现仍拒绝“带 padding 的 KV 页 + P/D TP 不同”
的 head 重分片。当前 MiMo + DFlash、P TP2 / D TP1 会命中该限制。
这些脚本改动负责接入 Mooncake，不修改 vLLM 的 KV 布局实现，也不自动改 TP 或关闭投机。
保留 DFlash 的一个待验证方案是两端 TP2：
`KV_CONNECTOR=MooncakeConnector D_TP=2 D_DP=1 bash deploy_pd_6p2d.sh`；
它会改变 decode 的并行方式，需重新评估性能。保持原拓扑则需要补齐 connector 的 padded-KV 重分片支持。

当前两个 PD 脚本使用相同 TP2，避开上述异构 TP 限制，并保留已有的 DFlash + synthetic 接受长度 4.2 默认值。
真实采样可显式设置 `SPECULATIVE_CONFIG=''`；关闭投机可设置 `SPECULATIVE_METHOD=none`。
已显式设置的非空 `SPECULATIVE_CONFIG` 仍须与所选方法兼容。

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
其他模型或拓扑需按实际专家数调整。当前镜像已包含 MiMo EPLB 支持；部署不应用工作区补丁。

6P2D 的 D 端显存较紧，当前默认 `MAX_MODEL_LEN=16384`；pd 默认为 32768，aggre 为 65536。
batch token 预算仍保持 65536。调整上下文长度后，应重新确认 KV 容量和启动日志。


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


### 自动采集 P/D 指标与前缀缓存命中率

三个部署入口现在默认 `ENABLE_METRICS=1`，服务健康检查通过后、打印 `READY` 前建立采集基线，
之后每 `METRICS_INTERVAL=5` 秒采集一次。无需修改负载脚本：别人通过服务发请求也会计入部署侧指标。
`METRICS_TIMEOUT=3` 控制单次 HTTP 超时，`METRICS_GPU=1` 同时通过 `nvidia-smi` 记录本次部署
GPU 的利用率、显存和功耗。设置 `ENABLE_METRICS=0` 可关闭部署侧持久采集，
`METRICS_GPU=0` 可单独关闭 GPU 采集。采集故障记录为错误，不终止模型服务。

```bash
# 按原来的方式启动即可；可选择更短的采样周期
ENABLE_PREFIX_CACHING=1 METRICS_INTERVAL=2 bash deploy_pd.sh

# 同样适用于 deploy_pd_6p2d.sh / deploy_aggre.sh
```

每次部署独立保存以下文件，不覆盖旧部署：

| 文件（相对本次 deployment.json 所在目录） | 内容 |
|---|---|
| `metrics/samples.jsonl` | 带时间与完整标签的 P/D 指标、直方图 buckets、采集错误以及 GPU 样本 |
| `metrics/summary.json` | 持续更新的窗口汇总、各 DP 缓存命中率、时延、速率与 PD 排查线索 |
| `metrics/summary.csv` | 各端命中 token、查询 token、命中率、计算吞吐和输出吞吐 |
| `proxy_requests.jsonl`（PD） | 每请求 ID、P 耗时、D 响应头/首内容等待、代理 TTFT、总耗时及错误阶段；不存提示词和输出文本 |

`run_load.sh` 也默认采集，同一次实验的 `results/.../<run_id>/server_metrics/` 下生成
`samples.jsonl`、`summary.json` 和 `summary.csv`；原来的 `perf/summary.csv`、XLSX、DB 保持原格式。
`manifest.json` 的 `server_metrics` 记录采集地址、状态和各端前缀命中率。负载结束时终端打印命中率。
实验 ID 沿用时间戳加 UUID，相同模型、负载参数的结果不会互相覆盖。

```bash
# 原有负载参数可继续使用，无需额外对齐 P/D 端口
LOAD_PYTHON_BIN=/tmp/performance-eva-venv/bin/python bash run_load.sh \
  --mode pd --parallel 1 --number 20 --warmup-num 6 \
  --prefix-length 4096 --input-length 128 --output-length 16 \
  --prepared-dataset-path '' --metrics-interval 2
```

新代理在原有公开端口上增加 `/metrics`（代理阶段统计）、`/metrics/prefill` 和 `/metrics/decode`
（分别读取后端的 `/metrics`）。跨机负载只需原来的 `--ip` 与 `--port`，不必暴露后端端口。
这些接口只读取指标，不重置缓存。aggre 直接使用服务自己的 `/metrics`。
`--no-server-metrics` 关闭实验侧采集，不改变部署侧采集；负载环境新增依赖 `prometheus-client`，
已列入 `requirements-load.txt`。

命中率读取 `summary.json` 的 `roles.prefill.prefix_cache.hit_rate`（取值 0–1）：

```text
P 本地命中率 = Σ各 DP Δprefix_cache_hits_total / Σ各 DP Δprefix_cache_queries_total
```

这是 token 命中率，采用采集后的增量，不混入基线之前的累计值，不平均各 DP 的百分比，
也不使用 EvalScope 的多轮历史长度估算。D 本地缓存放在 `roles.decode.prefix_cache`，
外部 KV 查询与命中放在各端 `external_prefix_cache`，两者不能混为一谈。
没有查询、接口未暴露、缺少基线、计数器重置或 series 变更时，对应命中率为 `null`（终端 `N/A`），
原因与采集错误保存在 `quality`，不会伪装成 0%。

实验窗口从负载子进程启动前到退出后刷数结束，**包含数据准备、warmup 和全部 sweep**。
默认退出后再等待 `--metrics-settle-seconds 6` 秒，给 vLLM 异步统计刷出留出时间；
服务统计周期更长时需增加此值。采集前请让上一批请求完成并等待刷数。
窗口内其他客户端的请求也会计入服务统计；单个实验需要隔离流量。
若比较不同并发的服务端指标，建议每个并发单独运行一次；时间序列也支持后续按窗口分析。

PD 排查可结合以下数据：

- `computed_prefill_tokens_per_second`：`prompt_tokens_by_source{source="local_compute"}` 的增量速率，排除缓存及外部 KV；旧版本不暴露时为 `null`。
- `output_tokens_per_second`：各 DP 输出 token 增量之和 / 观测时长。`inverse_mean_tpot_tokens_per_second` 是平均 TPOT 的倒数，两者口径不同。
- `latencies`：服务端 TTFT、排队、prefill、decode、TPOT 等直方图的增量均值；原始 buckets 保留在样本文件中。
- `gauges`：运行/等待请求数、等待原因、KV 拉取阶段、KV 利用率，保留引擎标签；`preemptions` 记录抢占增量。
- `kv_transfer` / `connector_errors`：connector **实际暴露**的传输时长、字节数与失败计数。
- `roles.proxy.proxy_stages` 和逐请求文件：P HTTP 往返、D 首内容等待和代理 TTFT。SSE 的 role/usage 消息不计为首内容；非流式请求无法观测 TTFT，记为 `null`。

`pd_analysis` 提供带证据的排查线索，不能仅凭一次运行断定唯一瓶颈。
P HTTP 耗时包含 P 排队和执行，D 首内容等待包含传输、排队、首步和网络；不能相减当作纯 KV 传输时间。
**本机 Mooncake 实现尚未接入 Prometheus 传输指标**，因此相关字段为空并标记 `not_exposed`，
仍可结合 D KV 等待阶段、代理时序、GPU 和服务日志排查。NIXL 是否提供对应字段也以实际版本为准。

已经启动的旧服务不会自动加载新代码。本次修改不重启运行中的服务。
下次使用更新的部署脚本启动即可启用全部功能；暂时不重启时可独立附加只读采集：

```bash
python3 code/server_metrics.py \
  --deployment /path/to/deployment.json \
  --output-dir /path/to/new-metrics-directory --interval 5
```

该命令的输出目录必须尚不存在，Ctrl-C 结束采集。旧代理没有 `/metrics` 时会记录该项缺失，
P/D 仍从部署文件中的后端地址直接采集；旧代理无法补采逐请求阶段耗时。
旧服务与本机 `run_load.sh` 配合时，也可显式指定
`--prefill-metrics-url http://127.0.0.1:19110/metrics --decode-metrics-url http://127.0.0.1:19120/metrics`。
独立采集需在能访问这些地址的服务器上运行；不要在远程客户端直接使用复制部署文件内的 loopback 地址。

### 投机解码接口

接口复用 `../profile/decode/speculative_config.py`，与 `../profile/decode/capture.sh` 的解析规则一致：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `SPECULATIVE_METHOD` | `none` | `none` / `mtp` / `dflash` |
| `NUM_SPECULATIVE_TOKENS` | `7` | 每步草稿 token 数，启用时必须为正整数 |
| `DRAFT_MODEL_PATH` | 空 | MTP 可空，由 vLLM 使用目标模型；DFlash 必须提供草稿模型路径 |
| `SPECULATIVE_CONFIG` | 空 | JSON 额外配置，覆盖同名基础参数；method 必须与 METHOD 一致 |

`none` 不向 vLLM 传 `--speculative-config`；此时不能设置非空 `SPECULATIVE_CONFIG`。
上表是通用接口默认值；当前两个 PD 入口已有 DFlash/synthetic 配置，见上面的说明。

```bash
# MTP：使用目标模型的 MTP 权重
SPECULATIVE_METHOD=mtp NUM_SPECULATIVE_TOKENS=7 bash deploy_aggre.sh

# DFlash：指定草稿模型。三种部署入口都接受相同参数。
SPECULATIVE_METHOD=dflash NUM_SPECULATIVE_TOKENS=7 \
  DRAFT_MODEL_PATH=/models/MiMo-V2.6-Flash-MOPD/dflash \
  bash deploy_pd_6p2d.sh

# 与参考 capture.sh 一样的 synthetic 接受长度实验；先用 dry-run 检查命令
SPECULATIVE_METHOD=mtp DRAFT_MODEL_PATH='' NUM_SPECULATIVE_TOKENS=7 \
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

默认负载为输入 1024、输出 500（适配三种部署的默认上下文；长上下文测试请显式设置长度）、每档 50 请求、1 轮、不预热；aggre 默认并发 `2 4 8 16`，PD 两种模式为 `2 4 8 16 32`。`--number` 单值广播到所有并发档，多值必须与 `--parallel` 等长。`--rate`、`--loop`、长度区间、数据集及 Header 参数沿用 `per_dataset.py`。多轮的 number 代表对话数。超时参数 `--read-timeout` 控制请求读取，duration 为软退出预算，不是强制终止时间。

数据集可以通过 `per_dataset.py --prepare-dataset` 预生成，再用 `--prepared-dataset-path` 或 `--multi-turn-dataset custom_multi_turn --multi-turn-dataset-path` 复用。请显式提供数据集路径，避免工作目录中的默认 `prompts_random.txt` 意外影响实验。单轮自定义文本无需 tokenizer 时可传 `--tokenizer-path ''`，此时依赖服务返回 usage；随机负载必须提供有效 tokenizer。

### 保存并复用共享前缀数据集

```bash
# 只生成一次：3000 条完整 prompt，各含共享 8000 token 前缀和不同的 1000 token 后缀。
# 写完才发布文件，拒绝覆盖已有文件。模型服务无需启动。
bash run_load.sh --prepare-dataset --prepare-num 3000 \
  --tokenizer-path /models/MiMo-V2.6-Flash-MOPD \
  --prefix-length 8000 --input-length 1000 \
  --prepared-dataset-path datasets/prefix8000_suffix1000_n3000_seed42.txt

# 后续重复执行：读取上述文件，跳过随机生成；每次评估结果仍单独保存。
bash run_load.sh --mode aggre --use-chat \
  --tokenizer-path /models/MiMo-V2.6-Flash-MOPD \
  --prepared-dataset-path datasets/prefix8000_suffix1000_n3000_seed42.txt \
  --prefix-length 8000 --input-length 1000 --output-length 1000 \
  --number 3000 --parallel 512 --rate 14 --warmup-num 0 \
  --label prefix8k_new1k_out1k_r14
```

先在发射器环境设置 `LOAD_PYTHON_BIN`。单轮带前缀的预生成模式会重新分词校验：
前 8000 个 token 一致、每条总长 9000 个 token，且完整 prompt 不重复；
长度不含服务端聊天模板。使用同一个 tokenizer 和模板时，共享内容保持一致。
没有前缀的旧生成模式及多轮模式仍使用原来的采样方式，重新分词后的长度可能变化。
文件复用时已包含前后缀，传入的 `prefix-length/input-length` 仅用于实验参数记录，
不会再次拼接或截断文件内容。3000 条文件建议配合 `--number 3000 --warmup-num 0`；
超过文件条数（包括 warmup）时 EvalScope 会循环读取，后缀也会重复。

`parallel` 是同时未完成 HTTP 请求的上限，`rate` 是整个发射器的目标调度速率（请求/秒），
并非每个并发槽各发这么多请求。本机 EvalScope 使用随机间隔调度；并发占满时必须等待，
所以设定 14 不保证实际持续达到 14 请求/秒。粗略估计所需并发 ≈ 速率 × 单请求平均耗时：
14 请求/秒、20 秒/请求需要约 280 个并发槽，并为时延波动留余量。

保存数据集避免重复生成；每次启动仍会从磁盘读取一次并构造请求。
这与服务端 prefix caching 无关，服务仍需开启 `ENABLE_PREFIX_CACHING=1`。
重复使用同一文件会复用相同后缀；如果要比较每次都“后缀全新”的实验，应清空服务端缓存，
或使用新 seed 生成另一份文件（新 seed 也会改变前缀）。

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
