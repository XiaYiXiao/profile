# MiMo-V2.6-Flash-MOPD 压测与 Perfetto 时间线

## 仓库范围与快速开始

本仓库保存可复用的压测、采集、分析脚本及使用文档。实验结果保留在运行机器，
不上传 captures、results、报告、trace、日志、缓存、旧备份或 EPLB 安装快照。
`.gitignore` 按核心源码目录设置白名单，后续执行 `git add -A` 仍会排除这些数据。

```bash
git clone https://github.com/XiaYiXiao/profile.git profiler-v2
# v2 将原根目录的采集脚本收纳到 profile/。
DRY_RUN=1 ENABLE_EPLB=0 bash profiler-v2/profile/ep_profile/capture.sh
```

脚本默认模型目录为 `/models/MiMo-V2.6-Flash-MOPD`，vLLM 源码目录为
`/host/vllm`，可通过 MODEL_PATH、VLLM_SOURCE、PYTHON_BIN 覆盖。请先准备模型、
匹配的运行依赖及 GPU；当前自维护运行版本见 [vllm_dev](https://github.com/XiaYiXiao/vllm_dev)。
所有服务和压测入口均通过 `python -m vllm` 调用本地源码；
`PYTHONPATH` 前置 `VLLM_SOURCE` 并保留原有内容，不依赖安装的 vLLM CLI。
Nsight 采集还需要 nsys、可用的 CUDA/CUPTI/NCCL trace 功能。

| 路径 | 用途 |
|---|---|
| 根目录的 `.sh` / `.py` | Prefill 部署、参数扫描、汇总和 Perfetto 处理 |
| `decode/` | Decode 参数扫描、metrics 采样、结果分析 |
| `ep_profile/` | EP0/EP1、EPLB0/EPLB1 对照、Nsight 采集 |
| `ep_profile/analysis/code/` | 采集调度、通用分析和 CPU 回归测试 |
| `ep_profile/backends/` | H20 / B300 的后端选择说明 |

EP/EPLB 使用方法及上传范围详见 [ep_profile/README.md](ep_profile/README.md)。
当前 capture.sh 默认开启 EPLB，需使用已完成 MiMo EPLB 适配的 vLLM；
普通 EP 对照可显式设置 ENABLE_EPLB=0。后端文档是指定源码版本的参考，
不保证适用于任意上游版本。

## Prefill 参数扫描

独立复制 `/host/profiling` 的脚本，结果、resume 状态和图表均位于 `profiling_mimo`，不复用 GPT-OSS 数据。模型改为 `/models/MiMo-V2.6-Flash-MOPD`，部署增加该模型所需的 `--trust-remote-code`。当前 Prefill sweep 默认 TP4/DP1/EP0，其余负载和执行流程沿用模板。

当前 Prefill sweep 默认如下，参数集中在 `sweep-mimo.sh` 顶部，也可用环境变量覆盖：

| 参数 | 默认值 |
|---|---|
| GPU / TP / DP / EP | `0,1,2,3` / `4` / `1` / `0` |
| batch token 预算 | `256 512 1024 4096 16384 32768 49152 65536`，每个 DP 引擎分别生效 |
| 客户端并发 | `10 30 50 100`，整个部署的总并发 |
| 正式请求数 / 输入 / 输出 | `100` / `50000` / `1` |
| max_num_seqs / max_model_len | `256` / `1000000` |
| 热身 / 显存比例 / trace | `10` / `0.90` / 关闭 |

单独运行 `mimo.sh` 时，沿用部署模板的 TP4/DP1/EP0、max_num_seqs=512、batch=16384 默认值；sweep 会覆盖为上表的实验参数。EP=1 表示开启专家并行，实际 EP 组大小为 TP×DP=4。

四卡并行支持、权重及 KV cache 估算见 [PARALLELISM.md](PARALLELISM.md)。TP1/DP4 是可测试的配置，较大的 batch 和高并发仍可能受显存限制；本目录的创建和校验没有启动模型实验。

## TP4/DP1/EP0 的计算与通信后端

启动命令：

```bash
bash /host/profiler/profile/sweep-mimo.sh
# 等价的显式并行配置
TP=4 DP=1 EP=0 bash /host/profiler/profile/sweep-mimo.sh
```

本段依据本机源码以及旧的 `logs/mimo-tp4-20261029-213619.log`；新 sweep 尚未进行完整运行验证。

**MoE 计算**默认 `--moe-backend auto`，当前会选择 `HUMMING / HummingIndexedExperts`，对应 MXFP4 专家权重。计算后端与跨卡通信后端是两类设置。用本机后端选择器检查 MiMo 的 SiLU、256 experts、top-8、hidden4096、intermediate2048、TP4/DP1/EP0，得到：

| `--moe-backend` | 本机配置检查 |
|---|---|
| `auto` / `humming` | HummingIndexedExperts，通过；旧 TP4 日志有实际加载记录 |
| `marlin` | MarlinExperts，通过；尚未验证完整模型运行或性能 |
| `flashinfer_cutlass` | FlashInferExperts，MXFP4/BF16，通过；尚未验证完整模型运行或性能 |
| `triton` | 拒绝：该 MXFP4 内核不支持 MiMo 的 SiLU 激活 |
| `flashinfer_trtllm` | 拒绝：当前 H20 设备不满足该 MXFP4 内核要求 |

**跨卡通信**主要是 TP All-Reduce：各卡计算自己持有的专家张量分片，再归并部分输出；Attention 的 TP 输出也需要归并。当前不是专家 All-to-All 路径，`--all2all-backend` 的默认值虽然是 `allgather_reducescatter`，但不能把它当作这组配置实际执行的 MoE 通信方式。

旧 TP4 日志明确记录 TP 组的常规 All-Reduce 分派链为：

```text
FLASHINFER -> CUSTOM -> SYMM_MEM -> PYNCCL
```

这是候选优先级，不是每次都执行四种通信。每次调用根据 tensor 大小、dtype、布局、拓扑及工作区是否可用选择实现；模型编译还可能融合 All-Reduce 与 RMSNorm，因此仅凭启动日志不能确定每个算子的最终 kernel。

| 通信实现 | 本机默认与控制方式 |
|---|---|
| FlashInfer All-Reduce | 默认启用；`VLLM_ALLREDUCE_USE_FLASHINFER=0/1`。子后端由 `VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto/mnnvl/trtllm` 控制；单机 auto 优先尝试 mnnvl，不可用时可回退 trtllm |
| vLLM Custom All-Reduce | 默认启用；服务参数 `--disable-custom-all-reduce` 可关闭这一项 |
| PyTorch Symmetric Memory All-Reduce | 默认启用；`VLLM_ALLREDUCE_USE_SYMM_MEM=0/1`，受设备、拓扑及大小限制 |
| PyNCCL / NCCL | 默认作为通用回退，适用于优化实现不接受的消息 |
| NCCL Symmetric Memory | 可选，默认关闭；`VLLM_USE_NCCL_SYMM_MEM=1`，受版本、拓扑和消息大小限制；与上一项 PyTorch SYMM_MEM 不同 |

单独设置 `--disable-custom-all-reduce` 不会关闭 FlashInfer 和 PyTorch SYMM_MEM，不能将它解释成“强制所有通信使用 NCCL”。以上 env 可由 shell 传入；当前 `mimo.sh` 尚未暴露计算后端及 disable-custom-all-reduce 的可调变量，也未将这些选项加入 resume 配置身份。后续做后端对比时，应补齐参数管理，并将不同后端实验写入不同结果目录，避免复用另一种后端的结果。

DeepEP、FlashInfer NVLink All-to-All 等属于专家分发/合并的通信路径，需要相应的 EP/DP 配置、软件依赖和兼容内核；当前 TP4/DP1/EP0 不使用。AMD 的 AITER/QuickReduce 也不是本机 H20 的候选。建议本轮保持自动通信选择，先建立 TP4 基线。

本地实现依据：`vllm/distributed/device_communicators/cuda_communicator.py`、`flashinfer_all_reduce.py`、`envs.py`、`model_executor/layers/fused_moe/runner/moe_runner.py`。可对照 [vLLM 通信器文档](https://docs.vllm.ai/en/latest/api/vllm/distributed/device_communicators/cuda_communicator/) 与 [EP 部署说明](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)。

## 热身与实验执行

每组正式测试前先发送 `NUM_WARMUPS=10` 个相同输入/输出长度的热身请求，允许调大，
不允许少于 10。热身遵循本组并发上限，不计入正式请求、性能统计和 trace；
`/metrics` 的前后快照包含热身阶段的计数变化。

```bash
bash /host/profiler/profile/sweep-mimo.sh
```

显式开启时间线采集：

```bash
PROFILE=1 bash /host/profiler/profile/sweep-mimo.sh
```

脚本默认 resume：再次执行相同命令会跳过完成的组，只补跑缺失、失败或不完整的组；
一个 batch 的所有组都已完成时，不再启动该 batch 的模型。`PROFILE=0` 和 `PROFILE=1`
分别保存在 `trace0`、`trace1` 目录，不会相互复用或混合统计。

强制从头重跑当前实验：

```bash
FORCE_RERUN=1 PROFILE=0 bash /host/profiler/profile/sweep-mimo.sh
```

旧数据会先移入本目录的 `history/forced-attempt0001/` 等编号目录。补跑单组时，
该组的旧 JSON、日志、metrics 和 trace 也会归档后再执行，避免静默覆盖。
`FORCE_RERUN` 未设置或为 `0` 时，始终优先续跑。

完成判定检查配置数量的正式请求全部成功、输入/输出 token 数和长度符合配置；
trace1 还要求 profiler 启停成功、四个 rank 的原始 trace 齐全且 gzip 完整性通过。
首次验证较大 trace 需要时间；随后用 `.completed/` 中的配置和文件签名缓存校验结果。
Perfetto 的显示重叠提示不会使一份完整 trace 被判为采集失败。

脚本也会搜索旧的带时间目录，自动复用参数一致的完整结果；优先硬链接，跨文件系统时复制，
不会修改旧结果。正在写入的旧实验目录不会被复用。`DRY_RUN=1` 只预览计划，不复制文件，
也不进行完整 gzip 解压校验。输入长度、请求数、热身数、显存比例等关键参数与已有目录
不兼容时会拒绝混用；可指定新的 `RESULT_DIR`，或明确设置 `FORCE_RERUN=1` 归档后重跑。

显式设置 `PROFILE=1` 时，每组使用 `vllm bench serve --profile` 自动调用 `/start_profile` 和 `/stop_profile`，
从实际负载开始采集，持续到本组负载结束，不采集模型加载和负载数据生成阶段。
每个 worker 分别导出 CPU/CUDA trace；前端 CPU trace 不采集。
默认关闭调用栈、tensor shape 和内存记录，保留正常部署的 CUDA Graph 行为。

只分析一个配置、采集完整负载：

```bash
PROFILE=1 PROFILE_MAX_ITERS=0 BATCH_TOKENS=16384 CONCURRENCIES=30 \
  bash /host/profiler/profile/sweep-mimo.sh
```

可调参数：

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `NUM_WARMUPS` | `10` | 每组热身请求数，至少为 10 |
| `PROFILE` | `0` | 显式设为 `1` 开启 CPU/CUDA profiling |
| `PROFILE_MAX_ITERS` | `0` | 引擎迭代采集上限，`0` 为不限 |
| `PROFILE_DELAY_ITERS` | `0` | 收到开始请求后跳过的引擎迭代数 |
| `PROFILE_WITH_STACK` | `0` | 记录调用栈 |
| `PROFILE_RECORD_SHAPES` | `0` | 记录 tensor shape |
| `PROFILE_WITH_MEMORY` | `0` | 记录内存分配事件 |

结果位于 `profiling_mimo/results/<模型名>-TP<TP>-DP<DP>-EP<EP>-batch<搜索值列表>-trace<0或1>/`。
例如单个 batch 配置使用 `MiMo-V2.6-Flash-MOPD-TP4-DP1-EP1-batch16384-trace1`；
多个 batch 搜索值用下划线连接，例如 `batch4096_8192_16384_32768_65536_131072`。
目录名和 sweep 服务日志名不再包含时间；vLLM 自动生成的原始 rank trace 文件名保持原样。

- `summary.csv`：吞吐、延迟、请求成功数、采集状态及 trace 路径。
- `traces/batch16384-concurrency30/*.pt.trace.json.gz`：各 worker 的时间线。
- 同目录的 `*.profiler_out.txt`：vLLM 导出的算子耗时统计（如果可用）。

打开 [Perfetto UI](https://ui.perfetto.dev/)，选择 **Open trace file**，加载一个
`.pt.trace.json.gz` 文件即可查看 CPU 算子、CUDA runtime、GPU kernel、stream 和执行间隙。
如果浏览器无法读取 gzip，可先用 `gzip -dk 文件.pt.trace.json.gz` 解压后打开 JSON。
各 worker 文件独立查看；脚本不会将它们直接拼接成一份多卡时间线。
时间线不包含系统级 CPU 调度、网卡和硬件性能计数器等系统采集数据。

目录通过 `flock` 防止多个 sweep 同时写入；正常退出或进程崩溃后锁会自动释放，
不留下阻止 resume 的陈旧标记。搜索列表中的重复值会被拒绝，各 worker 原始文件名保留。
`runs.tsv` 和 `summary.csv` 记录每组参数、退出码、采集状态及 trace 目录。
中断时会尝试刷新 trace，最长等待 `PROFILE_FLUSH_TIMEOUT=300` 秒，再清理服务进程。

采集会增加 CPU、内存和磁盘开销，统计结果包含采集开销。建议先用 `PROFILE=0`
做性能基线，再对感兴趣的配置开启采集；全量 50K 输入的 trace 可能很大。

部署脚本也支持相同参数；单独启动时需设置 `PROFILE=1`，并手动 POST `/start_profile` 和 `/stop_profile`。
在 sweep 中使用 `DRY_RUN=1 PROFILE=1` 可检查命令，不启动服务或发出请求。

## 处理 Perfetto GPU 重叠事件提示

遇到 `SLICE_SPILL_OVERLAPPING_COMPLETE_EVENT` 时，可为单个 rank 生成独立显示副本：

```bash
python /host/profiler/profile/prepare-perfetto-trace.py /完整路径/原始.pt.trace.json.gz
```

默认在原目录生成 `原始.pt.trace.perfetto.json.gz`，以及同名 `.report.json`。
也可用 `-o /新目录/便于识别的文件名.perfetto.json.gz` 指定输出。已有文件拒绝覆盖。
脚本在主机内存中读取一份完整 JSON，处理较大文件时建议一次只转换一个 rank。

该脚本只对 GPU 轨道上的交叉完整事件分配合成的显示 tid，并同步移动能唯一匹配的
`ac2g` flow 终点；原始 `ts`、`dur`、事件及 `args.stream` 保留不变。
新轨道名为 `stream 19 / display lane 1 (synthetic tid)` 等，其中 lane 是显示分层，
不是新增了真实 CUDA stream。报告记录 GPU 交叉数、移动事件数及时间值一致性。
CPU 轨道上的其他解析问题不在该转换的处理范围内。

将生成的 `.perfetto.json.gz` 下载到浏览器所在电脑，在 Perfetto UI 点击
**Open trace file** 打开。通过 **Trace Info / Trace Errors** 和以下 SQL 检查：

```sql
SELECT name, value FROM stats
WHERE name GLOB '*overlapping_complete_event*';
```

观察 GPU 时应包含原 stream 轨道和新增的 display lane。原文件继续作为采集数据保存，
处理后的文件用于可视化，不改变模型或后续推理配置。

## 绘制 prefill 吞吐曲线

```bash
python /host/profiler/profile/plot-prefill-throughput.py
```

默认读取 config.txt 修改时间最新的实验目录；尚无 benchmark JSON 时生成标明等待数据的空图，不回退到旧实验。也可在命令后显式指定结果目录。结果落盘后再次执行即可刷新。
输出 `plots/prefill_throughput.png`、`.svg` 和 `.csv`。重复运行可刷新实验进度。
横轴为 `max_num_batched_tokens`（以 2 为底的对数刻度，显示实际 token 数），
纵轴为 `total_input_tokens / duration`，各并发值分别对应一条曲线。
只有请求数、输入/输出 token 数及输入长度满足配置的完整测试才绘制；
未完成、失败或尚未写完的 JSON 均留空，曲线在缺失位置断开，不插值。
本图使用客户端观测到的正式输入吞吐，包含启用 profiling 时的采集开销。
