# MiMo Decode benchmark / Nsight 采集

服务器将整个 `profile` 目录放在 `/host/profiler/profile`，保留相邻的
`decode/` 和 `ep_profile/`。decode 复用 prefill 的 Nsight、进程清理、EPLB 配置和完整性校验工具。
所有入口通过 export 配置；公共参数集中在 `settings.sh`，`capture.sh` 顶部保留常用分区。

| 参数 | 默认值 |
|---|---|
| TP / DP | 1 / 4，每组四卡 |
| max_num_batched_tokens | 恒定 8120，入口脚本会覆盖外部同名环境变量 |
| SEQ_COUNTS（负载列表） | 16 / 32 / 64 / 128 / 256 / 512 / 768 |
| max_num_seqs（服务启动参数） | max(SEQ_COUNTS)，默认 768，运行中保持不变 |
| 正式请求数 / 客户端总并发 | 各等于当前 seqs，不乘以 DP |
| KV_LEN / 输入长度 | 8192 / 8193 |
| 每请求输出 | 500 token，ignore EOS |
| WARMUP_ROUNDS | 每档完整预热 2 轮；0 跳过，每轮请求数＝并发数＝seqs |
| CUDA Graph | FULL_DECODE_ONLY |
| 模型 | /models/MiMo-V2.6-Flash-MOPD |
| Python / vLLM 源码 | /usr/bin/python3 /host/vllm（python -m vllm） |
| 投机方法 / EPLB | none / 关闭 |
| LOG_ONLY（capture） | 0；设为 1 时运行负载并保留日志、benchmark JSON，关闭 Nsight 和 metrics 采样 |

`max_num_seqs` 是每个引擎的调度上限；客户端总并发和正式请求数均等于当前负载点的 seqs。
例如默认服务固定以 max_num_seqs=768 启动，依次发送 16 个并发请求、32 个并发请求，
直至 768 个并发请求。按 SEQ_COUNTS 的原始顺序执行，预算不能小于其最大值。

随机文本在解码、重新分词后偶尔会超过指定输入长度。capture 和 sweep 的每轮预热与正式
请求都通过 `--extra-body` 设置 `truncate_prompt_tokens=KV_LEN+1`（默认 8193），
由服务端在推理前截断超长输入。结果仍严格检查每条输入长度、输入总数、成功数和输出长度；
输入不足目标长度、请求失败或输出不足仍会中止，不修改原始结果，也不放宽 token 校验。

## 入口

```bash
cd /host/profiler/profile
# EP0/EP1 Nsight 采集：先预览，再运行。
DRY_RUN=1 bash decode/capture.sh
bash decode/capture.sh

# 跳过 Nsight 和 metrics 采集；仍运行两组模型、全部预热和正式负载。
LOG_ONLY=1 bash decode/capture.sh

# 只采集一个负载点。
SEQ_COUNTS="128" bash decode/capture.sh

# 每个负载档位先完整预热 3 轮；sweep 同样支持。
WARMUP_ROUNDS=3 bash decode/capture.sh
WARMUP_ROUNDS=3 bash decode/sweep-mimo.sh

# 单服务 sweep，保留续跑 / FORCE_RERUN=1 归档机制。
DRY_RUN=1 bash decode/sweep-mimo.sh
bash decode/sweep-mimo.sh

# 仅启动服务；默认 EP=0，可通过 EP=1 覆盖。
MAX_NUM_SEQS=128 EP=1 bash decode/mimo.sh
```

`capture.sh` 默认同时启动 GPU 0,1,2,3 的 EP0 和 GPU 4,5,6,7 的 EP1，
各 TP=1、DP=4；API / DP RPC / master 端口均分开设置。两侧使用相同负载和投机方法。
先同时进行两组四卡 Nsight 工具链检查（不加载模型），然后每组模型各启动一次。
先按负载列表完成所有预热阶段，每档执行 WARMUP_ROUNDS 轮完整 benchmark；随后两侧请求
`/start_profile`，依次运行全部正式负载，最后请求 `/stop_profile`，关闭模型并导出。
每组只产生一份连续 trace，包含全部正式负载、档位间隔、客户端启动及采样开销，
不包含预热。两侧不是精确同步屏障。每轮预热的请求数、并发、输入输出长度及 seed
均与该档正式请求一致；等待本轮完成并校验通过后再进入下一轮。
所有 benchmark 均设置 num_warmups=0，由脚本显式控制轮数。
例如 seqs=128、WARMUP_ROUNDS=3，每组在该档执行 3×128 个预热请求，然后执行 128 个正式请求。
NUM_WARMUPS 不再作为 decode 的预热配置项。

只采集数据，不自动汇总、绘图或生成 HTML；每档结束打印 output_tokens/s。
结果默认保存到 `decode/captures/decode_ep_时间戳-PID/`：

- `ep0/`、`ep1/`：各一份 `trace.nsys-rep`、`trace.sqlite`、服务命令和日志。
- `ep0/seqN/`、`ep1/seqN/`：每档详细 benchmark JSON，以及逐轮 `warmup-round001.json`、metrics、命令和客户端日志。
- `load-events.json`：记录 profiler 边界、预热轮号、各档客户端启动和完成时间，便于定位连续 trace 中的负载。
- `config.json`：参数、服务容量、负载顺序、有效启动命令和投机配置。

`DRY_RUN=1` 不启动 GPU 或创建结果目录；
`SMOKE_ONLY=1` 仅执行工具链检查（仍需模型文件通过预检）。

`LOG_ONLY=1` 不要求安装 nsys，直接启动两组 vLLM；跳过 Nsight 工具链检查、profiler
配置和启停请求、metrics 快照及采样、trace 导出。每组模型仍只启动一次，仍先完成全部
warmup，再逐档运行正式负载。输出目录及日志布局保持一致，保留 `server.log`、
`warmup-roundNNN.log`、`eval.log`、对应 benchmark JSON、命令、配置和负载事件。
吞吐 / TPOT / ITL 等仍由 benchmark 计算并写入日志及 JSON，结果完整性校验继续生效。
该模式不能与 `SMOKE_ONLY=1` 同时使用；恢复 trace 采集时设置 `LOG_ONLY=0`。

单服务 sweep 也只启动一次，按列表顺序执行未完成的负载；每档先执行 WARMUP_ROUNDS
轮预热，再执行一轮正式 benchmark。预热保存为 `batch8120-concurrency64.warmup-round001.*`，
各轮独立校验；正式结果、吞吐汇总和 metrics 快照均不包含预热请求。
完成全部负载后关闭服务；某档失败即停止，不自动重启模型。结果写入 `decode/results/`，
文件如 `batch8120-concurrency64.json`，共享服务日志为 `server-session-时间戳-PID.log`。
固定预算、服务容量、负载顺序、预热轮数、投机、EPLB 等配置纳入续跑校验；旧结果不兼容时需换目录或
FORCE_RERUN=1。续跑会新启动一次服务，仅执行未完成档位；若需要完整连续的服务状态，
使用新目录或 FORCE_RERUN=1 从头执行。跨档复用模型会保留缓存、EPLB 等服务状态。
sweep 保留原有逐档自动汇总和吞吐打印；summarize / plot 也可手动运行，兼容历史 seqs+48
的命名和配置。新结果的 max_num_seqs 列为固定服务容量，concurrency 列为实际负载档位。

## 投机推理

```bash
# 默认不传 speculative-config
SPECULATIVE_METHOD=none bash decode/capture.sh
# MTP：默认 1 个投机 token，可显式调整。
SPECULATIVE_METHOD=MTP NUM_SPECULATIVE_TOKENS=1 bash decode/capture.sh
# DFlash：必须指定适配目标模型的草稿权重。
SPECULATIVE_METHOD=Dflash DRAFT_MODEL_PATH=/实际草稿模型路径 bash decode/capture.sh
```

MTP 默认复用目标模型的 MTP 权重。DFlash 未指定 token 数时交由草稿模型配置推断；
需要显式指定时用 NUM_SPECULATIVE_TOKENS，并与草稿模型训练配置一致。
SPECULATIVE_CONFIG 可传 JSON 覆盖其他字段，但 method 必须与选择一致。
这些选项也适用于 mimo.sh / sweep-mimo.sh。模型是否具备 MTP 权重、DFlash 是否适配该
MiMo 模型、以及与 DecodeBenchConnector / CUDA Graph 的组合，需服务器实际验证；
脚本提供配置接口，不能根据 dry-run 宣称支持。

## EPLB / 后端

ENABLE_EPLB=1 开启 EP1 的 EPLB，EP0 始终关闭。
可配置 EPLB_WINDOW_SIZE、EPLB_STEP_INTERVAL、EPLB_NUM_REDUNDANT_EXPERTS、
EPLB_USE_ASYNC、EPLB_COMMUNICATOR、EPLB_LOG_BALANCEDNESS、EPLB_LOG_BALANCEDNESS_INTERVAL，
或用 EPLB_CONFIG JSON 覆盖。默认窗口 8、间隔 16、冗余 0、异步 true。
MOE_BACKEND、ATTENTION_BACKEND、ALL2ALL_BACKEND、EXPERT_PLACEMENT_STRATEGY
只作用于 EP1；空计算后端由 vLLM 自动选择。

## Decode 数据含义

当前负载使用自定义 DecodeBenchConnector，在真实显存布局中填充模拟 KV，跳过前
KV_LEN 个输入 token，只对最后一个输入 token 执行前向计算，之后生成输出。
默认 startup_fill=true、fill_mean=0.015、fill_std=0。
本地 `/host/vllm` **必须包含该连接器**；启动时从 `VLLM_SOURCE` 指定的源码检查。
模拟 KV 不包含真实 Prefill 或 P→D 传输，也不保证真实请求的专家路由分布。
Decode 吞吐使用实际输出 token / 正式 benchmark 墙钟耗时，不把模拟 KV 当作生成 token。
