# 当前 H20 环境的后端可选列表

更新日期：2026-11-03。对应 [B300 可选列表](b300.md)。

## 适用范围与验证状态

本文针对当前 MiMo-V2.6-Flash-MOPD 模型、MXFP4 专家权重、SiLU 激活、TP=2、DP=2，以及 capture.sh 的 EP0/EP1 对照实验。它不是所有模型通用的 vLLM 后端支持表。

| 项目 | 当前环境 |
|---|---|
| GPU | 8×NVIDIA H20，compute capability 9.0（SM90） |
| 显存 | 每卡 nvidia-smi 报告 97871 MiB，约 96 GB 级 |
| GPU 互联 | nvidia-smi topo -m 显示任意两卡均为 NV18 |
| 驱动 | 580.126.20 |
| PyTorch / CUDA runtime | 2.14.0+cu130 / 13.0 |
| FlashInfer | 0.7.2 |
| vLLM 源码 | /host/vllm；HEAD 为 db3b0f0fae4bc98492b3863685e541e62bc59576，结论以当前工作区源码为准 |
| Attention 形状 | Q/K head dimension=192，V head dimension=128，需要 DiffKV |
| 当前组合 | MoE 自动选择 HUMMING；Attention 自动选择 FLASH_ATTN_DIFFKV；通信指定 allgather_reducescatter |

状态说明：**已实跑**有当前模型实验记录；**候选**仅有源码支持或已有选择器探测，未证明完整模型正确性、稳定性及性能。2026-11-03 的依赖复核没有启动完整模型或通信测试。

## --moe-backend：专家计算

下表候选默认按 allgather_reducescatter 通信组合理解。

| 参数值 | 当前判断 | 条件或限制 |
|---|---|---|
| `auto` | 已实跑 | 默认值；按优先级和支持条件选择，当前选中 HUMMING，不是全后端现场测速 |
| `humming` | 已实跑对应实现 | 当前日志确认 HUMMING；显式指定可固定后端族 |
| `marlin` | 候选 | 已有选择器探测通过，选中 MARLIN；不支持 flashinfer_nvlink_two_sided |
| `flashinfer_cutlass` | 候选 | 已有探测选中 FLASHINFER_CUTLASS_MXFP4_BF16，即 MXFP4 权重、BF16 激活 |
| `triton_unfused` | 实验性候选 | 源码支持 SM90、MXFP4、SiLU；因 MTP 问题被移出自动候选列表，需单独验证 |
| `triton` | 当前模型不适配 | 对应 MXFP4 专家实现要求 SWIGLUOAI，当前模型是 SILU |
| `flashinfer_trtllm` | 硬件不支持该 MXFP4 路径 | 实现要求 compute capability 10.x；不能在 H20 上使用目标 MXFP4/MXFP8 kernel |
| `deep_gemm` | 硬件不支持该 MXFP4 路径 | Hopper 的 FP8 支持不等于这里的 MXFP4 支持 |
| `flashinfer_moe_ep_cutedsl`、`flashinfer_moe_ep_mega_deep_gemm` | 硬件不支持 | 当前适配器设备白名单为 SM100、SM103；还有独立的模型和并行限制 |
| `emulation` | 缺依赖 | 需要 amd-quark，当前 has_quark=False；属于反量化模拟路径 |
| `batched_triton`、`deep_gemm_mega_moe`、`cutlass`、`flashinfer_cutedsl`、`flashinfer_b12x`、`aiter_mega_moe`、`flydsl`、`hpc`、`rdna3` | 不是当前 MXFP4 路径的直接候选 | 虽在通用 CLI 枚举中，但当前 MXFP4 映射没有相应入口；不可据名称认定兼容 |
| `b12x`、`aiter`、`aiter_triton_mxfp4_bf16`、`aiter_triton_mxfp4_mxfp4` | 其他平台/架构 | 不适用于当前 H20 部署 |

命令行使用小写后端名。`FLASHINFER_TRTLLM_MXFP4_MXFP8` 是内部枚举和日志名称。`flashinfer_trtllm_afp8` 虽存在于内部映射，但不在当前 CLI 的 MoEBackend 枚举中，不应直接用作命令行值。

## --attention-backend：Attention

| 参数值 / 设置 | 当前判断 | 说明 |
|---|---|---|
| 不指定 | 已实跑 | MiMo 自动选择 FLASH_ATTN_DIFFKV；自动模式不写成 `--attention-backend auto` |
| `FLASH_ATTN_DIFFKV` | 已实跑对应实现 | 支持当前不同 K/V head dimension；底层为 SM90 的 FA3/FA4 实现 |
| `TRITON_ATTN_DIFFKV` | 候选 | 模型明确提供的另一条 DiffKV 路径；尚未在本次记录中验证完整运行 |
| `FLASH_ATTN`、`TRITON_ATTN`、`FLASHINFER` 等普通后端 | 不能直接替换该 DiffKV 路径 | MiMo 对非 `_DIFFKV` 指定值仍执行 DiffKV 自动选择；仅看到 CLI 参数不能证明切换成功 |
| MLA、稀疏 MLA、其他模型专用后端 | 不适用当前模型路径 | 不能作为本模型 DiffKV 的直接替代 |
| `FlashAttentionForwardSm100` | 不是合法参数值，且硬件不支持 | 属于底层 kernel 类名；H20 上 FA4 使用 FlashAttentionForwardSm90 |

当前 Attention 可根据 DiffKV 与 sink 配置使用 SM90 的 FA3/FA4 实现。FA 版本可由 attention-config 控制，但不能用版本号跨越硬件架构限制。具体运行状态应检查本地日志和 trace；实验结果不随脚本仓库上传。

## --all2all-backend：EP 通信

本表覆盖当前 All2AllBackend 枚举，以下覆盖参数只应用于 EP1。EP0 虽未启用 expert parallel，但 TP2/DP2 路径仍需要 AllGather/ReduceScatter；它不能搭配 flashinfer_nvlink_one_sided 的普通 dispatch 接口。脚本不给 EP0 传入后端覆盖参数，保留 vLLM 默认配置。

| 参数值 | 当前判断 | 条件或限制 |
|---|---|---|
| `allgather_reducescatter` | 已实跑，当前默认 | 基于 AllGather / ReduceScatter |
| `flashinfer_nvlink_one_sided` | 候选 | 本机依赖检查通过；已有选择器探测允许 Humming、Marlin、FlashInfer CUTLASS |
| `flashinfer_nvlink_two_sided` | 候选 | 本机依赖检查通过；已有探测允许 Humming、FlashInfer CUTLASS，不允许 Marlin |
| `flashinfer_all2allv` | 旧别名 | 指向 flashinfer_nvlink_two_sided，不是独立实验项 |
| `deepep_high_throughput` | 缺依赖 | has_deep_ep=False；H20 架构符合候选条件，仍需安装与链路验证 |
| `deepep_low_latency` | 缺依赖且组合受限 | 还要求相应 RDMA 环境、batched-experts 格式；历史探测 Humming 不通过、Marlin 选中 BATCHED_MARLIN，仅作待复核线索 |
| `deepep_v2` | 缺依赖 | has_deep_ep_v2=False |
| `moonep` | 缺依赖 | has_moonep=False |
| `nixl_ep` | 缺依赖 | has_nixl_ep=False |
| `mori_high_throughput`、`mori_low_latency` | 不适用 | 当前实现面向相应 AMD GPU |
| `naive`、`pplx` | 已移除实现的兼容参数 | 当前配置会警告并回退到 allgather_reducescatter |
| `passthrough` | 不适合当前普通组合 | 为内部完成 dispatch/combine 的特定 MoE 融合后端保留，不能单独用来替代通信 |

DeepEP 官方要求 Hopper 或更新架构，各路径还有独立的通信和依赖要求，不能仅凭 GPU 型号认定可运行：[DeepEP 官方说明](https://github.com/deepseek-ai/DeepEP)。

## 可安排的对照范围

| 维度 | 可比较的候选 | 固定其他条件 |
|---|---|---|
| MoE | humming、marlin、flashinfer_cutlass | FLASH_ATTN_DIFFKV + allgather_reducescatter |
| Attention | FLASH_ATTN_DIFFKV、TRITON_ATTN_DIFFKV | humming + allgather_reducescatter |
| EP 通信 | allgather_reducescatter、flashinfer_nvlink_one_sided、flashinfer_nvlink_two_sided | humming + FLASH_ATTN_DIFFKV；先验证完整运行 |

上述候选不是任意组合的笛卡尔积，也不代表替代后端一定更快。capture.sh 的 Backend / EP 区域提供 MOE_BACKEND、ATTENTION_BACKEND、ALL2ALL_BACKEND 和 EXPERT_PLACEMENT_STRATEGY，全部仅对 EP1 生效，服务命令由 reproduce.sh 的 server_cmd 构建。EP1 的 MoE/Attention 留空表示不传对应参数；EP0 始终不传这些覆盖参数，使用 vLLM 默认选择。config.json 中 backend_parameter_scope=ep1_only 明确这些配置值的作用范围，最终命令可在 commands 或 server-command.txt 复核。

## EP 专家放置策略

`EXPERT_PLACEMENT_STRATEGY` 对应 `--expert-placement-strategy`，默认 `linear`，只传给启用 EP 的 EP1；EP0 不启用专家并行。config.json 记录请求的策略，不能把它当成运行时最终生效策略。

当前模型有 256 个 routed experts，EP1 的 TP=2、DP=2 对应 EP size=4。无冗余专家、未启用 EPLB 时，linear 按 EP rank 连续放置：rank 0 为 0–63，rank 1 为 64–127，rank 2 为 128–191，rank 3 为 192–255。round_robin 的概念是交错放置，例如 rank 0 为 0、4、8……，但当前组合不能直接启用。

| round_robin 的源码条件 | 当前值 | 结论 |
|---|---|---|
| num_expert_group > 1 | MiMo 的 n_group=1，并原样传入专家层 | 不满足，回退 linear |
| num_redundant_experts=0，且未启用 EPLB | 当前默认符合 | 满足，但不能消除其他限制 |
| use_all2all_kernels 时需要 round-robin routing tables | EP1 的 DP=2，因此为 True；allgather_reducescatter 不提供所需表 | 不满足；当前表支持分支为 deepep_low_latency 或 nixl_ep |

因此设置 `EXPERT_PLACEMENT_STRATEGY=round_robin` 可以传递请求，但 vLLM 会警告并回退到 linear；不是 H20 硬件禁止交错放置，而是当前模型和实现的支持条件不满足。只换 B300 或只换通信后端都不能消除 n_group=1 的限制。不要为启用放置策略而直接修改模型的 n_group，因为它参与路由分组语义。

依据：[放置策略判定](https://github.com/XiaYiXiao/vllm_dev/blob/main/vllm/model_executor/layers/fused_moe/expert_map_manager.py)、[通信条件](https://github.com/XiaYiXiao/vllm_dev/blob/main/vllm/model_executor/layers/fused_moe/config.py)、[模型传参](https://github.com/XiaYiXiao/vllm_dev/blob/main/vllm/model_executor/models/mimo_v2.py)。

## 本地依据

- [启动命令](../reproduce.sh)、[实验入口](../capture.sh)。
- [依赖与选择器探测工具](../analysis/code/check_ep_backends.py)：不包含完整模型或多卡通信实测；结果由工具在本地生成。
- 实际服务日志和 CUDA trace 属于本地实验数据，不随仓库上传。
- [CLI MoE 枚举](https://github.com/XiaYiXiao/vllm_dev/blob/main/vllm/config/kernel.py)、[通信枚举与回退](https://github.com/XiaYiXiao/vllm_dev/blob/main/vllm/config/parallel.py)。
- [MXFP4 选择器](https://github.com/XiaYiXiao/vllm_dev/blob/main/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py)、[MiMo Attention 选择](https://github.com/XiaYiXiao/vllm_dev/blob/main/vllm/model_executor/models/mimo_v2.py)。
