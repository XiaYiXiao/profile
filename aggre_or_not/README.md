# MiMo PD 分离 / 非分离对照

提供以下入口，参数按设备、负载、EPLB、Nsight、模型和输出路径分区，均可用环境变量覆盖：

| 入口 | GPU / 并行 | max_model_len | max_num_batched_tokens | max_num_seqs |
|---|---|---:|---:|---:|
| `capture_aggre.sh` | 0–7：DP2 × TP4 + EP8 | 65536 | 16384 | 32 |
| `capture_pd.sh` | P=0–3：DP2 × TP2；D=4–7：DP4 × TP1；EP 关闭 | 65536 | 16384（P/D 均为此值） | 32 |
| `capture_pd_6p2d.sh` | P=0–5：DP3 × TP2 + EP6；D=6–7：DP2 × TP1 + EP2 | 65536 | 16384（P/D 均为此值） | 32 |

所有入口均提供 `ENABLE_EP=1/0`，`capture_aggre.sh` 和 `capture_pd_6p2d.sh` 默认 `1`，
原入口 `capture_pd.sh` 默认 `0`。两个 PD 入口的开关同时作用于 P 和 D；关闭 EP 时 TP、DP 和 GPU 分配不变，
不传 `--enable-expert-parallel`、`--all2all-backend` 或 `--expert-placement-strategy`。
`MOE_BACKEND` 和 `ATTENTION_BACKEND` 仍可独立配置。`ENABLE_EP=0 ENABLE_EPLB=1` 会在启动前报错。
默认实验名称包含 `enable_ep0/1`，`config.json` 的 `ep` 和 `summary.csv` 的 `enable_ep` 记录实际开关，方便区分对照组。
PD 默认名称还会读取实际的 `P_GPUS`、`P_DP`、`P_TP`、`D_GPUS`、`D_DP`、`D_TP`，
GPU 列表以连字符连接，例如 `pd_p_dp2_tp2_gpus0-1-2-3_d_dp4_tp1_gpus4-5-6-7_enable_ep0`。
设置 `EXPERIMENT_NAME` 可覆盖自动名称。

## 6P + 2D：P=DP3×TP2+EP6，D=DP2×TP1+EP2

`capture_pd_6p2d.sh` 设置 GPU 分配和并行度后调用 `capture_pd.sh`，沿用其负载、
token budget、NIXL、端口、trace、模型和输出配置，支持同样的环境变量覆盖。
P 使用 GPU `0,1,2,3,4,5`，每个 DP 副本使用 2 张卡；D 使用 GPU `6,7`，
每个 DP 副本使用 1 张卡。采集器按每端 `TP × DP = GPU 数` 校验，要求总共 8 张互不重复的卡。

默认开启 P/D 两端的 EP：P 的专家跨 6 张卡分布，D 的专家跨 2 张卡分布，两端的 EP 组彼此独立。
沿用 `allgather_reducescatter` 通信后端和 `linear` 专家放置方式，EPLB 默认关闭。
可用 `ENABLE_EP=0` 同时关闭两端 EP。

本机 MiMo 的 `num_attention_heads` 和 `swa_num_attention_heads` 均为 64，
可以被当前 P 的 TP2 和 D 的 TP1 整除。原先 TP3 的静态检查错误已消除。
静态检查和命令预览不代表完整的模型加载、后端和 NIXL 运行验证；本次修改未启动 GPU 任务。

```bash
# 从 /host 执行：只生成配置及命令，不启动 GPU 任务。
DRY_RUN=1 bash profiler/aggre_or_not/capture_pd_6p2d.sh

# GPU 空闲后正式运行；默认已开启 EP。
bash profiler/aggre_or_not/capture_pd_6p2d.sh
```

产物仍位于本目录的 `pd/` 下，默认名称包含实际 GPU 分配、DP、TP 和 EP 开关。
`config.json` 记录静态兼容性检查结果；`sources/` 同时保存新入口和原入口的快照。

## PD 引擎参数及启动关系

PD 实验入口为本目录的 `capture_pd.sh` 和 `capture_pd_6p2d.sh`。它们调用 `code/collect_pd.py`，由其中的
`server_command()` 生成参数，`collect()` 分别启动 P、D 两个 `python -m vllm serve`
进程组，再启动 `code/pd_proxy.py`，最后运行 benchmark sweep。无需调用其他目录的脚本。
下表为原入口 `capture_pd.sh` 的当前默认配置。

| 参数 | P / Prefill | D / Decode |
|---|---|---|
| GPU（`CUDA_VISIBLE_DEVICES`） | `0,1,2,3` | `4,5,6,7` |
| Tensor parallel | 2 | 1 |
| Data parallel / local DP | 2 / 2 | 4 / 4 |
| Expert parallel | 默认关闭 | 默认关闭 |
| Pipeline parallel | 1 | 1 |
| `max_model_len` | 65536 | 65536 |
| `max_num_batched_tokens` | 16384 | 16384 |
| `max_num_seqs`（每个 DP 引擎） | 32 | 32 |
| `gpu_memory_utilization` | 0.90 | 0.90 |
| HTTP 地址 | `127.0.0.1:18100` | `127.0.0.1:18200` |
| DP RPC 端口 | 29750 | 29850 |
| master 端口 | 29701 | 29801 |
| NIXL side-channel 端口 | 5557、5558（按 DP rank） | 5657–5660（按 DP rank） |

两端共同使用模型 `/models/MiMo-V2.6-Flash-MOPD`、`dtype=bfloat16`、`kv_cache_dtype=auto`、
`KV_CACHE_LAYOUT=LBHNC`、`--block-size=16`，进程执行后端为 `mp`。开启 EP 时才传入
`all2all_backend=allgather_reducescatter` 和 `expert_placement_strategy=linear`。chunked prefill 开启，prefix caching、
EPLB、Nsight 和投机推理默认关闭。MoE 和 Attention 后端留空，由本地 vLLM 自动选择。

两端的 KV 配置均为 `NixlConnector`、`kv_role=kv_both`、`kv_load_failure_policy=fail`；
请求由代理指定 P 生成 KV、D 拉取 KV。代理监听 `127.0.0.1:18000`，benchmark 只请求代理。
最终展开的 P/D 命令分别保存在本次 `pd/<实验目录>/prefill/server.command.txt` 和
`pd/<实验目录>/decode/server.command.txt`；`DRY_RUN=1` 也会生成这些文件。

## 负载和运行

非分离入口默认并发档位为 `2 4 8 16`，两个 PD 入口为 `2 4 8 16 32`。
每个档位分别发送 **50 个请求，每请求 50,000 输入 token、500 输出 token**。
每组服务只启动一次，先完成所有档位的完整预热（`WARMUP_ROUNDS=1`），再依次评估
（`REPEATS=1`）；设 `WARMUP_ROUNDS=0` 可跳过预热。两组使用同一随机种子，关闭 prefix
caching，不启用投机推理；请求设置 `ignore_eos`，并逐请求校验实际输入/输出 token 数。
非分离组默认评估 200 个请求、预热 200 个请求；PD 组默认评估 250 个请求、预热 250 个请求。

从 `/host` 执行，各实验都使用全部 8 卡，须依次运行：

```bash
# 仅生成配置及命令，不启动服务，不要求安装 NIXL/Nsight。
DRY_RUN=1 bash profiler/aggre_or_not/capture_aggre.sh
DRY_RUN=1 bash profiler/aggre_or_not/capture_pd.sh

# 检查 GPU、端口和依赖，不加载模型。
PREFLIGHT_ONLY=1 bash profiler/aggre_or_not/capture_pd.sh

# 正式实验，显式开启 EP；各实验依次运行，等待前一组释放 GPU。
ENABLE_EP=1 bash profiler/aggre_or_not/capture_aggre.sh
ENABLE_EP=1 bash profiler/aggre_or_not/capture_pd.sh

# 关闭 EP；PD 组的 P、D 同时关闭。
ENABLE_EP=0 bash profiler/aggre_or_not/capture_aggre.sh
ENABLE_EP=0 bash profiler/aggre_or_not/capture_pd.sh

# EPLB、Nsight 均默认关闭；EPLB 需要同时开启 EP。
ENABLE_EP=1 ENABLE_EPLB=1 bash profiler/aggre_or_not/capture_pd.sh
ENABLE_NSYS=1 bash profiler/aggre_or_not/capture_aggre.sh
```

PD 使用 `NixlConnector` pull 协议和本地 `code/pd_proxy.py`。代理先向 P 请求
1 token 以取得 KV 元数据，再将**原始 prompt 和完整 500 token 预算**交给 D；P 的临时输出
不拼接到客户端结果。D 的 DP 副本由同一 vLLM 服务进行内部调度。
缺少 KV 元数据或 KV 加载失败会报错，不回退为 D 完整重算。端到端 TTFT 包含 P 排队、
prefill 计算、代理和 KV 传输时间。P/D 使用相同的 `LBHNC` KV layout、block size 和 dtype。
协议及 TP 支持参考本地 vLLM 的 NIXL 示例及
[官方兼容说明](https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/)。

NIXL 版本应与 `VLLM_SOURCE/requirements/kv_connectors.txt` 匹配；本地要求 1.5.0。
本机 CUDA 13 的隔离依赖放在 `/host/profiler/aggre_or_not/.deps`，PD 脚本自动加入 Python 路径。
在同样已有完整 vLLM/CUDA 13 依赖的环境可按如下方式准备，避免替换已有 PyTorch：

```bash
UV_CACHE_DIR=/host/profiler/aggre_or_not/.cache/uv uv pip install \
  --python /usr/bin/python3 --target /host/profiler/aggre_or_not/.deps --no-deps \
  'nixl==1.5.0' 'nixl-cu13==1.5.0'
```

本实验入口、辅助代码、依赖、缓存和运行产物集中于 `/host/profiler/aggre_or_not/`。运行代码不依赖 `profiler/profile` 下的采集模块：

```text
aggre_or_not/
  capture_aggre.sh, capture_pd.sh, capture_pd_6p2d.sh, README.md, .gitignore
  code/                              # collect_pd.py、pd_proxy.py、runtime.py、eplb_config.py、测试
  aggre/<实验名称_时间戳_PID>/
  pd/<实验名称_时间戳_PID>/
    config.json, runtime.json, status.json, collector.log, summary.csv
    sources/                          # 本次脚本快照
    prefill/, decode/                 # 非分离组使用 aggre/
      server.command.txt, server.log
      trace.nsys-rep, trace.sqlite    # 仅 ENABLE_NSYS=1
    proxy.command.txt, proxy.log      # 仅 PD
    benchmarks/concurrency{2,4,8,16,32}/  # 非分离组默认不含 32
      {warmup,eval}-roundNNN.command.txt
      {warmup,eval}-roundNNN.json, *.log
      *.before.metrics, *.after.metrics
  .deps/, .cache/, .tmp/, validation/
```

`OUTPUT_ROOT` 可更改总目录，`RESULT_DIR` 可指定单次实验目录；拒绝覆盖已有目录。
失败、中断和预览均保留配置及状态；退出时只清理本次启动的进程。Nsight 在预热后启动，
连续覆盖所有评估档位，P/D 分别保存 trace。`summary.csv` 汇总端到端吞吐和 TTFT/TPOT/ITL/E2EL
分位数，完整请求记录保存在 benchmark JSON。两组当前默认 token budget 均为 16384；
PD 的 `MAX_NUM_BATCHED_TOKENS` 同时作用于 P、D，运行时以各次保存的配置和启动命令为准。

历史实验 `pd_p_dp1_tp4_d_dp2_tp2_enable_ep1_20261109-113643-87068` 的 P、D 均使用 1638，
这是原始需求中的笔误，现已更正为 16384。历史结果及脚本快照保持原样；该次对比包含调度预算
差异，需要使用更正后的预算重跑 PD 组，才能评估预期配置下的性能。

辅助代码位于 `code/`，进程管理及 EPLB 配置均使用本目录的独立模块。`.gitignore` 排除运行结果、缓存和隔离依赖。
vLLM 框架继续使用 `/host/vllm`，模型继续使用 `/models/MiMo-V2.6-Flash-MOPD`，不修改这些目录。
