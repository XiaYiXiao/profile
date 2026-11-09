# MiMo 四卡并行与显存估算

本分析针对本地 `models/MiMo-V2.6-Flash-MOPD`、`/host/vllm` 和 NVIDIA H20（每卡 97871 MiB，约 95.58 GiB）。显存比例仍为模板的 0.90，每卡预算约 86.02 GiB。未运行新模型服务或压测；下面区分配置检查、旧日志证据和容量估算。

## 并行支持结论

**可以测试 TP1/DP4/EP1，并非只能 TP2/DP2。** 配置声明 `MiMoV2ForCausalLM`；vLLM 检测到 vision_config 后转换为 `MiMoV2OmniForCausalLM`，其语言主干使用通用 FusedMoE。Attention 的 GA KV heads=4、SWA KV heads=8、Q heads=64，都能被 TP=1/2/4 整除；专家数 256 也能按四卡 EP 分为每卡 64 个完整专家。权重索引中的 `tp_size=4` 是检查点保存布局，不要求运行时只能 TP4：本地 `_shard_fp8_qkv_proj` 有 TP1/TP2 重新分片路径。

已用本地真实 MoE 配置类和后端选择器检查 hidden=4096、intermediate=2048、256 experts、top-8、SiLU、DeepSeekV3 分组 sigmoid 路由、BF16 激活以及默认 allgather_reducescatter 通信：

| Attention TP / DP | EP 开关 | MoE 内部 TP / EP | 自动选择的 MXFP4 后端 |
|---|---|---|---|
| 1 / 4 | 0 | 4 / 1 | HummingIndexedExperts |
| 1 / 4 | 1 | 1 / 4 | HummingGroupedExperts |
| 2 / 2 | 0 | 4 / 1 | HummingIndexedExperts |
| 2 / 2 | 1 | 1 / 4 | HummingGroupedExperts |
| 4 / 1 | 0 | 4 / 1 | HummingIndexedExperts |
| 4 / 1 | 1 | 1 / 4 | HummingGroupedExperts |

这只验证后端配置接受上述组合，不代表完整权重加载、CUDA Graph、通信和实际负载均已实测成功。此前 `/host/logs/mimo-tp4-20261029-213619.log` 中已有 TP4 的成功加载记录：Humming、每卡加载约 41.57 GiB，KV pool 约 42.64 GiB；该日志使用 max_model_len=8192、max_num_seqs=8、batch=4096，不能作为本次百万上下文和 DP4 的实测结果。

EP=0 时，当前 vLLM 会把 MoE 的 TP 跨 DP 展开为 4，因此 TP1/DP4/EP0 也不是每卡保存整个模型。EP=0 是每卡保存所有专家的张量分片，EP=1 是每卡保存部分完整专家。Attention、embedding 和输出层仍按外部 TP 切分，并在 DP 引擎间复制。

## 权重精度与大小

逐个读取本地 safetensors 文件头，所有索引中的文件均存在。专家权重及尺度共 **149.8125 GiB**，语言主干其他权重约 **8.2837 GiB**，视觉权重约 **1.3571 GiB**，主索引中的音频权重约 **0.4864 GiB**，MTP 约 **1.1077 GiB**。此外，Omni 的音频编码器还会单独加载 `audio_tokenizer/model.safetensors`；按加载代码转换为 BF16 后约 **1.1587 GiB**。即使发送的是纯文本，也不能默认这些编码器都不占显存。本次沿用模板，没有额外禁用多模态或启用 MTP 推测解码。

`quant_method=fp8` 不代表全模型 FP8：配置同时声明 `store_dtype=mxfp4`，本机 `Fp8Config.get_quant_method` 对专家层选择 `Mxfp4MoEMethod`，保留 MXFP4 专家执行路径。其他矩阵混合 FP8/BF16。`--dtype bfloat16` 不会把这些已量化专家全部展开成 BF16；默认 KV cache 则为 BF16。

令 E=149.8125 GiB，R 为语言主干其他权重与视觉权重之和（9.6408 GiB），A 为音频权重及 BF16 audio tokenizer（1.6451 GiB）。默认视觉按 TP 切分、音频编码器复制，权重估算为 `E/4 + R/TP + A`，忽略少量复制的 norm/bias 和后端布局差异：

| TP / DP | 理想权重/卡 | 说明 |
|---|---:|---|
| 1 / 4 | 48.74 GiB | 非专家权重完整复制，专家按四卡切分 |
| 2 / 2 | 43.92 GiB | 语言/视觉权重按 TP2 切分，音频部分复制 |
| 4 / 1 | 41.51 GiB | 语言/视觉权重按 TP4 切分，音频部分复制 |

这是权重近似值；不含后端重排/填充、通信、激活、CUDA Graph、KV pool 和加载峰值。TP4 估算与旧日志实际加载的 41.57 GiB 接近。

## Prefill 50K 请求

本地配置有 9 层全 Attention、39 层 SWA-128。K head_dim=192，V head_dim=128，GA/SWA KV heads 分别为 4/8。按 BF16 计算，不含 block 对齐和混合分组填充时，一条完成 50K prefill 的请求在 TP1 的 KV 约为：

`9 × 50000 × 4 × (192+128) × 2 + 39 × 128 × 8 × (192+128) × 2 = 1.097 GiB`

TP2/TP4 分别约 0.548/0.274 GiB。以客户端总并发均匀分配到 DP 引擎，估算最繁忙一张卡的“权重 + 请求稳态 KV”下限：

| TP / DP | 并发10 | 并发30 | 并发50 | 并发100 |
|---|---:|---:|---:|---:|
| 1 / 4 | 52.03 | 57.51 | 63.00 | 76.16 |
| 2 / 2 | 46.66 | 52.14 | 57.63 | 71.34 |
| 4 / 1 | 44.25 | 49.73 | 55.22 | 68.93 |

单位 GiB/卡。并发100、DP4 这里按每引擎25条计算，不是每引擎100条；bench 的 NUM_PROMPTS 和并发都是整个部署的总量。DP 负载不均匀会增加最忙引擎的压力。

这些值不包含 chunked prefill 在途 token 的 SWA 缓存、混合 KV 分组空位、激活/工作区和图捕获。因而 TP1/DP4 在中小 batch 下从容量看可行；并发100和 batch65536 的组合需要实际检查，不能用上表保证全部请求同时驻留。显存不足时也可能排队/抢占，HTTP 并发不等于活跃 GPU batch。

## 百万 max_model_len 和大 batch

max_model_len 不会为每个 max_num_seqs 请求预分配一百万 token，但启动检查需要至少容纳一条最长请求以及调度中的在途缓存。不能仅凭实际请求只有50K忽略该检查。

用本地 KV planner 在 CPU 上构造 BF16 DiffKV 的 9 GA + 39 SWA 规格，初始 block_size=16、层优先 LBHNC 布局、混合缓存管理开启、每次一个在途批次，计算 max_model_len=1000000 的单请求池需求，已计入 planner 的分组/对齐：

| TP | batch256 | batch16384 | batch65536 |
|---|---:|---:|---:|
| 1 | 21.54 | 25.00 | 35.55 |
| 2 | 10.77 | 12.50 | 17.78 |
| 4 | 5.39 | 6.25 | 8.89 |

单位 GiB/卡，只是启动检查需要的 KV pool 下限，不是模型实测池容量；最终布局、block_size、异步调度的在途数量和后端工作区可能改变结果。TP1 的 batch65536 加权重估算已约 **84.29 GiB/卡**，距离 86.02 GiB 预算只剩约1.7 GiB，还未包含其他开销，因此这个点有明显的启动/容量压力。TP2 相应值约61.69 GiB，余量大得多。默认 Omni 还可能在启动时进行多模态编码器的显存探测，启动峰值须以新日志确认。

建议保留模板进行对比，先用 TP1/DP4/EP1 的 batch4096 或16384 验证，再扩大搜索。TP2/DP2 提供更大的显存余量；不把它当作模型架构的硬性要求。每个 DP 引擎独立拥有 batch token 预算，因此相同 batch 值在 DP4 和 DP2 下，整节点调度预算不同。

## 执行与数据管理

```bash
# 预览当前模板默认的全部命令，不启动服务、不写实验结果
DRY_RUN=1 bash /host/profiler/profile/sweep-mimo.sh

# 明确测试四卡 DP4 + EP；先缩小到一个 batch
TP=1 DP=4 EP=1 BATCH_TOKENS=4096 bash /host/profiler/profile/sweep-mimo.sh

# TP2 + DP2 + EP 对照
TP=2 DP=2 EP=1 BATCH_TOKENS=4096 bash /host/profiler/profile/sweep-mimo.sh
```

以上实际测试命令供后续使用，本次没有执行。Prefill 默认 GPU0–3、port8000；Decode 沿用 GPU4–7、port8001。与 GPT-OSS 模板的端口和 GPU 选择相同，切换模型时须选择空闲资源，或用环境变量覆盖。结果路径独立，保留 resume、强制归档、trace0/trace1 隔离以及 config/runs/summary/metrics/plots 格式。

## 参考与本地依据

- [官方模型卡](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)：309B/15B 架构及部署示例；精度/大小估算以实际下载的 config 和 safetensors 为准。
- [vLLM EP 部署说明](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)：TP×DP 专家组以及 Attention/专家的分工。
- `/models/MiMo-V2.6-Flash-MOPD/config.json`、`model.safetensors.index.json` 及对应 safetensors 文件头。
- `/host/vllm/vllm/model_executor/models/mimo_v2.py`：MiMoV2MoE、DiffKV Attention 和 `_shard_fp8_qkv_proj`。
- `/host/vllm/vllm/transformers_utils/model_arch_config_convertor.py`、`models/mimo_v2_omni.py`、`models/mimo_audio.py`：Omni 架构转换、视觉 TP 与音频 tokenizer 的额外加载。
- `/host/vllm/vllm/model_executor/layers/quantization/fp8.py`、`mxfp4.py`、`fused_moe/oracle/mxfp4.py`：量化与后端选择。
- `/host/vllm/vllm/v1/kv_cache_interface.py`、`v1/core/kv_cache_planning.py`：缓存规格和启动容量检查。
