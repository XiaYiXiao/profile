# MiMo-V2.6-Flash-MOPD Decode sweep

独立复制当前 Decode 模板，直接执行 `bash profiling_mimo/decode/sweep-mimo.sh` 会启动服务并完成七组测试；每组修改调度上限，需要重新启动一次服务。模型使用 MiMo-V2.6-Flash-MOPD；结果仅写本目录的 `results/`。

| 参数 | 默认值 |
|---|---|
| GPU | 4,5,6,7，仅 4 卡 |
| TP / DP / EP | sweep 默认 1 / 4 / 0，沿用当前模板；单独部署脚本默认 4 / 1 / 1 |
| max_num_seqs | 16 / 32 / 64 / 128 / 256 / 512 / 768 |
| max_num_batched_tokens | max_num_seqs + 48 |
| 每组正式请求数 / 并发 | 均等于该组 max_num_seqs |
| 预制 KV 长度 | 8192 |
| bench 输入长度 | 8193；连接器跳过前 N−1 个输入 token，最后一个 token 执行前向计算 |
| 每条请求输出 | 500 token，忽略 EOS；包含首 token，通常有 499 个后续生成间隔 |
| 热身 | 正式测试前至少 10 个同长度独立请求，不计入正式请求数和 bench 耗时 |
| max_model_len | 1000000，沿用 P 设置，必须至少容纳 KV_LEN + 1 + MAX_OUTPUT |
| 服务端口 / 显存比例 | 8001 / 0.90 |
| KV / 计算精度 | KV auto（当前为 BF16）；计算 BF16，模型原有 MXFP4 权重保持自动识别 |
| CUDA Graph | FULL_DECODE_ONLY |
| KV 填充 | DecodeBenchConnector，startup_fill=true，mean=0.015，std=0（常量） |

主要参数集中在两个 sh 文件顶部，也支持环境变量覆盖。修改 sweep 负载应修改 `sweep-mimo.sh`，直接启动服务则使用 `mimo.sh`。

若改为 DP>1，vLLM 的 max_num_seqs 是每个引擎的上限；本脚本的总正式请求数和客户端并发仍为该组 seqs，不会自动乘以 DP。

```bash
# 打印全部配置及命令；不加载模型、不发请求、不写结果目录
DRY_RUN=1 bash profiling_mimo/decode/sweep-mimo.sh

# 启动测试，默认续跑；已完成且校验通过的组会跳过
bash profiling_mimo/decode/sweep-mimo.sh

# 只测试一组
SEQ_COUNTS="128" bash profiling_mimo/decode/sweep-mimo.sh

# 相同结果目录的旧文件先归档，再完整重跑
FORCE_RERUN=1 bash profiling_mimo/decode/sweep-mimo.sh

# 仅启动服务，便于手动测试
MAX_NUM_SEQS=128 bash profiling_mimo/decode/mimo.sh

# 测试中或结束后重新汇总
/usr/bin/python profiling_mimo/decode/summarize-mimo.py /绝对路径/结果目录

# 绘制 Decode 吞吐量、TPOT、ITL，并独立复核原始统计；可在实验运行中执行
/usr/bin/python profiling_mimo/decode/plot-decode-metrics.py
# 或显式指定结果目录
/usr/bin/python profiling_mimo/decode/plot-decode-metrics.py /绝对路径/结果目录
```

默认结果目录位于本文件夹的 `results/` 内，命名包括模型、TP/DP/EP、seqs 列表、batch 额外预算、KV 和输出长度，不使用时间。列表顺序不影响目录。其他配置变化时用新的 `RESULT_DIR`，或显式强制归档重跑；不会静默复用不兼容结果。锁阻止同一目录同时写入。中断、失败、输出长度不符或结果损坏的组会归档后重试，成功组不重跑。不会导入 P 的实验结果。

结果沿用 P 的文件组织：`config.txt`、`runs.tsv`、原始详细 bench JSON、bench/server 日志、`summary.csv` 和每组 `metrics-before/after.txt`。例如 seqs=64 的结果是 `batch112-concurrency64.json`。额外的 `*.metrics-series.jsonl` 默认每秒采样运行/等待请求数、KV 占用和 token/抢占计数，按标签保留数值并记录 setup/warmup 或 benchmark 阶段。

汇总的核心指标是 `output_tokens_per_s`、Mean/P50/P90/P95/P99 的 TPOT 和 ITL。前者为正式请求总输出 token / bench 总墙钟时间，包含正式请求的接入和排空；ITL 是客户端流式输出间隔，TPOT 是每个请求 `(E2EL−TTFT)/(output_tokens−1)`。完成校验要求每条请求 500 输出、总输出为 seqs×500、输入长度符合配置，并具有 TPOT/ITL 统计。

`peak_running_requests` 是 benchmark 阶段采样到的运行请求数峰值，不保证每个 GPU 步骤都达到这一 batch；`peak_kv_cache_usage_pct` 是各引擎采样峰值的最大值。`sampled_preemptions_delta` 是首末可用正式采样的计数差，可能漏掉采样边界事件。metrics-before/after 在整个 bench（含热身）前后采集，差值包含热身，不能用该差值除以正式 bench 耗时作为吞吐量。运行中间隔采样也有少量 CPU/HTTP 开销。

模拟 KV 使用真实显存和缓存布局，但不经过真实 Prefill，也不包含 P→D 网络传输。常量/随机 KV 不保证 MiMo-V2.6-Flash-MOPD 的专家路由、EP 通信负载和真实数据相同，最终容量应使用真实 KV 交叉验证。设置 `KV_FILL_STD` 为正数可测试随机填充，但随机数据也不等于真实 KV。startup_fill 在启动时填充符合条件的整个缓存池，再按请求长度分配；特殊缓存仍可能逐请求填充。512 组是否可容纳以实际缓存容量、运行请求数和抢占情况为准。

绘图脚本默认读取最新 Decode 实验，在该结果的 `plots/` 下生成 `decode_metrics.png`、`.svg`、`.csv`、`.validation.json`。横坐标为 max_num_seqs；吞吐量使用实际输出 token 计数，不能将模拟 KV 对应的逻辑输入 token 加到 Decode 吞吐量中。TPOT/ITL 展示 Mean/P50/P99，CSV 保留 P90/P95 等统计。pending/running/starting 或校验失败的点留空，不填零、不跨缺口连线；再次执行刷新快照。脚本只写图表目录，不修改 runs、summary 或原始实验文件，也不会请求模型服务、发送负载或操作实验进程。

审计从逐请求 latency/TTFT 和原始 ITL 数组重新计算分位数，核对 JSON 和已有 summary，并通过 before/after 计数交叉检查正式请求加热身的成功数、输出 token 数、模拟 KV 和本地计算输入 token 数。ITL 记录的是客户端相邻流式输出间隔，部分请求的样本数可能少于 output_tokens−1；这不表示模型少生成了 token，逐请求实际输出仍以 usage 的 output_lens 校验。图中的 ITL 不应解释为 GPU 每一步的精确耗时。
