# MiMo profiling 与性能评估 · v2

`main` 为 v2 版本，`v2` 标签标记本次更新。仓库包含模型部署、独立负载评估、
Prefill/Decode 采集及结果分析代码；模型权重、实验结果、日志、缓存和安装备份留在运行机器。

## 目录

| 目录 | 用途 |
|---|---|
| [performance_eva/](performance_eva/README.md) | 三种常驻部署、单轮/多轮负载、prefix caching 实验和评估汇总 |
| [aggre_or_not/](aggre_or_not/README.md) | 混部与 PD 分离采集，以及部署复用的进程管理和命令构造 |
| [profile/](profile/README.md) | Prefill、Decode、EP/EPLB 对照与 Perfetto/Nsight 分析 |
| [patches/eplb/](patches/eplb/README.md) | MiMo EPLB 最小源码补丁及适用版本说明 |
| [v0.31.0/](v0.31.0/README.md) | vLLM 0.31.0 EPLB 安装包，以及独立的 0.32 开发版适配包；包含版本/文件哈希检查、安装和回滚工具 |
| [v0.31.0_eplb/](v0.31.0_eplb/README.md) | 原始 vLLM 0.31.0 EPLB 补丁目录，包含安装脚本和四个补丁源码文件 |

v1 根目录中的采集脚本移入 `profile/`，例如 `decode/capture.sh` 现在是
`profile/decode/capture.sh`。Git 历史保留。请完整克隆仓库并保留这三个相邻目录，
`performance_eva` 依赖另外两个目录中的模块。

## 快速开始

```bash
git clone https://github.com/XiaYiXiao/profile.git profiler-v2
cd profiler-v2

# 预览部署命令，不加载模型。
DRY_RUN=1 bash performance_eva/deploy_pd.sh
```

部署默认使用 `/models/MiMo-V2.6-Flash-MOPD`、`/host/vllm` 和 `/usr/bin/python3`，
可通过 `MODEL_PATH`、`VLLM_SOURCE`、`PYTHON_BIN` 覆盖。需要提前准备支持该模型的
vLLM/CUDA 环境；PD 还需与运行版本匹配的 NIXL，安装方式见
[PD 文档](aggre_or_not/README.md)。开启 EPLB 需要相应的 MiMo 适配，见
[补丁说明](patches/eplb/README.md)。

三种部署都默认使用全部八张 GPU，在同一台机器上依次运行。部署终端任选一个入口：

```bash
bash performance_eva/deploy_aggre.sh
bash performance_eva/deploy_pd.sh
# 本机 6P2D 的 D 端无法容纳默认 65536 上下文，显式缩短后再尝试。
MAX_MODEL_LEN=32768 bash performance_eva/deploy_pd_6p2d.sh
```

| 模式 | P / 混部 | D | 对外端口 | Prefix caching 默认值 |
|---|---|---|---:|---|
| `aggre` | TP4 × DP2 | — | 19000 | 关闭 |
| `pd` | TP2 × DP2 | TP1 × DP4 | 19100 | 开启 |
| `pd_6p2d` | TP2 × DP3 | TP1 × DP2 | 19200 | 开启 |

各端默认开启 EP/EPLB，`MAX_NUM_BATCHED_TOKENS=65536`、`MAX_NUM_SEQS=512`。
6P2D 默认冗余专家数为 2，使 MiMo 的 256 个逻辑专家扩展到 258 个物理专家，
满足 P 的 EP6 与 D 的 EP2 整除要求。投机默认 `none`，可通过
`SPECULATIVE_METHOD=mtp/dflash` 等参数配置。三个脚本独立管理参数。

看到 `READY` 后，在另一个终端安装负载依赖并测试；`--mode` 与正在运行的部署一致即可：

```bash
cd profiler-v2
python3 -m venv performance_eva/.venv-load
performance_eva/.venv-load/bin/python -m pip install -r performance_eva/requirements-load.txt
export LOAD_PYTHON_BIN="$PWD/performance_eva/.venv-load/bin/python"

# 一条对话、两轮，随机生成输入；不需要先生成数据集。
bash performance_eva/run_load.sh --mode pd \
  --multi-turn --min-turns 2 --max-turns 2 \
  --parallel 1 --number 1 --warmup-num 0 \
  --input-length 128 --output-length 16 --prepared-dataset-path ''

# 共享前缀测试，使用单轮随机数据；前缀和 input-length 相加为总输入长度。
bash performance_eva/run_load.sh --mode pd \
  --parallel 1 --number 20 --warmup-num 6 \
  --prefix-length 4096 --input-length 128 --output-length 16 \
  --prepared-dataset-path ''
```

若系统缺少 `ensurepip`，可改用 `uv venv`、`uv pip install --python` 创建并安装负载环境。
跨机发负载使用 `--ip <服务器IP>`；客户端仍需可用的 tokenizer。

模型日志写入 `performance_eva/deployments/<模式>/<时间戳_UUID>/`；实验结果由负载脚本写入
`performance_eva/results/<模式>/<模型>/<负载分类>/<时间戳_UUID>/`。
同模型、同负载重复执行不会覆盖。每次保留 manifest、负载日志、CSV/XLSX 汇总和 EvalScope 原始结果。

## 已知运行边界

- 本机已完成混部多轮对话与 4P4D 共享前缀实验。4P4D 的 P 端观察到约 91.6% 累计 prefix 命中率；
  D 端本地命中为 0、通过 NIXL 接收 KV。该比例包含预热，仅用于确认缓存生效。
- 6P2D 已修正 EPLB 专家数整除配置，但本机默认 65536 上下文曾因 D 端 KV 显存不足启动失败；
  此次发布没有完成其完整 GPU 推理验收。缩短上下文的命令是待目标机器验证的配置。
- 多轮模式会携带对话历史，`--prefix-length` 仅用于单轮随机负载。缓存统计应分别读取
  P/D 的 `/metrics`，不能把 NIXL 外部 KV 命中当成本地 prefix cache 命中。

详细参数、端口范围、数据集复用与测试方法见 [performance_eva/README.md](performance_eva/README.md)。
