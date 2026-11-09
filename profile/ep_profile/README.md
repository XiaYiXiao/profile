# MiMo EP / EPLB 采集脚本

## 入口

服务器上将本目录部署为 `/host/profiler/profile/ep_profile`。
以下命令从 `/host/profiler/profile` 执行。先用 DRY_RUN=1 检查服务和压测命令，
再根据需要去掉该变量运行实验。

```bash
# EP0 / EP1 对照，关闭 EPLB，单独比较专家并行。
DRY_RUN=1 bash ep_profile/capture.sh

# 单组模式：只运行 EP1，EPLB 仍默认关闭。
EP_SOLO=1 DRY_RUN=1 bash ep_profile/capture.sh

# 仍使用 capture.sh：单独采集 EP1 + EPLB。
EP_SOLO=1 ENABLE_EPLB=1 COMPARISON=eplb DRY_RUN=1 bash ep_profile/capture.sh

# 仍使用 capture.sh：两组均 EP，仅组 1 开启 EPLB。
EP_SOLO=0 COMPARISON=eplb DRY_RUN=1 bash ep_profile/capture.sh

# 两侧均开启 EP，仅组 1 开启 EPLB。
DRY_RUN=1 bash ep_profile/eplb_cmp.sh

# 不使用 Nsight 的多轮 benchmark。
DRY_RUN=1 bash ep_profile/reproduce.sh
```

实际实验默认使用两组四卡。先检查 GPU_EP0 / GPU_EP1、端口、模型路径、vLLM 路径及显存预算。
参数集中在 shell 脚本顶部，也支持环境变量覆盖；输出位置可用 RESULT_DIR 指定。

`capture.sh` 最上面的 `EP_SOLO` 默认为 `0`。改为 `1` 或执行
`export EP_SOLO=1` 后，只检查和运行第 1 组的四卡工具链、模型服务、预热及评估，
使用 `GPU_EP1`（默认 `4,5,6,7`）及 EP1 端口。
`COMPARISON=ep` 时结果写入本次目录的 `ep1/`；`COMPARISON=eplb` 时仅运行
开启 EP 和 EPLB 的 EPLB1，结果写入 `eplb1/`。
未使用的第 0 组 GPU 和端口不参与预检。恢复双组实验用 `export EP_SOLO=0`。

服务器默认模型为 `/models/MiMo-V2.6-Flash-MOPD`，解释器为
`/usr/bin/python3`，vLLM 源码为 `/host/vllm`，Nsight 为 `/usr/local/bin/nsys`。
服务和压测统一使用 `"$PYTHON_BIN" -m vllm serve ...` / `bench serve ...`，
不再使用 `VLLM_BIN` 或已安装的 CLI 启动器。

默认设置 `VLLM_SOURCE=/host/vllm`，通过
`export PYTHONPATH="$VLLM_SOURCE${PYTHONPATH:+:$PYTHONPATH}"` 前置源码路径。
版本和模块路径从实际导入的 vLLM 读取；源码 commit 在 Git 信息可用时记录。

两个采集入口默认保存到脚本旁的 `captures/实验名称_时间戳/`，
因此部署后结果位于 `/host/profiler/profile/ep_profile/captures/`。
`reproduce.sh` 单独运行时保存到脚本旁的 `results/`。

`capture.sh` 和 `eplb_cmp.sh` 只进行数据采集，保留 `.nsys-rep`、导出的
`trace.sqlite`、每轮 benchmark JSON、metrics、日志、命令和配置。
每个预热轮及评估轮通过校验后，在终端打印该组的 `input_tokens/s`，
计算方式为该轮输入 token 总数除以该轮 benchmark 耗时；不写入额外汇总文件。
仍校验请求结果及 trace 完整性，但不自动生成统计汇总、`expert_skew.svg`、
`evidence.json`、`reduce_scatter.csv`、`summary.json` 或 HTML 报告。
需要分析时，在采集结束后单独运行下面的分析工具。

## Backend 与 EPLB

COMPARISON=ep 时，MoE、Attention、All-to-All、专家放置策略及 EPLB 的覆盖参数
只用于 EP1；EP0 使用 vLLM 默认后端配置。COMPARISON=eplb 时，两侧均 EP=1，
backend 配置相同，仅 EPLB 开关不同。

`capture.sh` 和 `reproduce.sh` 默认 `ENABLE_EPLB=0`，仅比较 EP 关闭与开启。
可用 `export ENABLE_EPLB=1` 显式开启 EP1 的 EPLB。
`eplb_cmp.sh` 固定两侧均开启 EP，仅 EPLB1 开启 EPLB；入口会固定覆盖
`ENABLE_EPLB=1`，以保留 EPLB 对照实验含义。启用 EPLB 依赖 MiMo 模型侧完整的
EPLB 注册和 MXFP4 权重迁移支持。请使用自己的适配版本，不能仅根据 vLLM
接受 --enable-eplb 参数判断支持情况。源码和运行产物维护于
[vllm_dev](https://github.com/XiaYiXiao/vllm_dev)。

REPEATS 表示评估轮数，WARMUP_ROUNDS 表示完整 benchmark 预热轮数。
每侧先完成预热，再进行评估；Nsight 的连续 trace 包含评估轮及轮间间隔。
8/16 等短 EPLB 窗口和间隔用于接入验证，正式实验需自行选择并验证收敛情况。

## 分析工具

| 脚本 | 用途 |
|---|---|
| `analysis/code/collect.py` | 多轮采集与实验调度，由 shell 入口调用 |
| `analysis/code/experiment.py` | 轮次和实验结果结构的公共逻辑 |
| `analysis/code/eplb_config.py` | EPLB 参数配置 |
| `analysis/code/analyze.py` | 根据本地采集数据生成 HTML 报告 |
| `analysis/code/explain_ep_gap.py` | Humming / ReduceScatter 偏斜分析与 SVG |
| `analysis/code/analyze_eplb_capture.py` | 比较本地 EPLB capture 的日志和 trace |
| `analysis/code/check_ep_backends.py` | 不加载完整模型的依赖、选择器探测；不是完整运行验证 |
| `analysis/code/test_experiment_rounds.py` | CPU 回归测试，模拟服务和采集操作 |

例如，重新分析本地已有采集目录：

```bash
python ep_profile/analysis/code/explain_ep_gap.py CAPTURE_DIR --output-dir CAPTURE_DIR
python ep_profile/analysis/code/analyze.py CAPTURE_DIR
```

偏斜图目前依赖 Humming GEMM 和 NCCL ReduceScatter 的匹配，不能直接适配所有
计算及通信后端。工具可能拒绝不支持的 trace；原始数据保留在本地。
图表脚本按需依赖 matplotlib、numpy；启动模型则使用已安装完整 vLLM 依赖的解释器。

## 哪些文件上传

上传三个入口 shell、通用 Python 分析代码、回归测试和后端指南。
以下内容不上传，但本机原文件仍保留：

- `captures/`、`results/`、`analysis/` 中生成的 JSON、HTML、CSV 和图表。
- Nsight / SQLite trace、服务日志、metrics、benchmark 结果和归档包。
- `backups/`、`eplb_install/`、`eplb_validation/` 中的旧快照及一次性验证材料。
- `compare_batches.py`、`compare_eplb_warmups.py`：依赖固定历史实验目录的一次性对比脚本。

以后新增核心文件时，先用 git status 确认是否在白名单内；不要用 git add -f
把实验结果强制加入仓库。
