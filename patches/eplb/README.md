# MiMo EPLB 源码适配

`runtime.patch` 是相对于 vLLM Python 包目录的最小补丁，包含四个文件的修改：

- `model_executor/models/mimo_v2.py`：注册 MoE/EPLB 元数据和专家权重接口。
- `model_executor/layers/fused_moe/routed_experts.py`：以 uint8 字节视图传输 MXFP4 E8M0 scales。
- `distributed/eplb/eplb_state.py`、`distributed/elastic_ep/elastic_execute.py`：统一配置缓存键。

补丁原始基线为 vLLM `v0.31.0`，commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`。
同一最小 diff 已在本机 `0.32.1rc1.dev223+gdb3b0f0fa.d20261029` 四个文件上以
`--fuzz=0` 合并，并用于后续混部与 4P4D 运行。其他版本需要逐项核对；
版本字符串相同也不能证明目标文件相同。补丁不附带完整 vLLM、模型、Humming 或 NIXL。

在目标源码副本上检查补丁是否适用；下面命令只做预检，不修改源码：

```bash
patch --dry-run --fuzz=0 -p1 -d /path/to/vllm/vllm < patches/eplb/runtime.patch
```

正式合并前停止使用该源码的服务，并保存目标文件。已接入该补丁的运行环境无需再次应用。
本目录提供便于审阅和移植的最小 diff。可安装的 vLLM 0.31.0 文件替换包位于
[v0.31.0_eplb/](../../v0.31.0_eplb/README.md)，其中安装脚本会校验版本和文件哈希。
本机安装快照和回滚备份不上传。

开启 EPLB 还要求 EP 开启，以及物理专家数能被该端 EP 大小整除。
MiMo 的 256 个逻辑专家在 EP6 时需要增加副本，例如
`EPLB_NUM_REDUNDANT_EXPERTS=2` 得到 258 个物理专家。
