# vLLM 0.31.0：本地源码 EPLB 补丁

适用于以下启动方式：

```bash
export PYTHONPATH="/host/vllm${PYTHONPATH:+:$PYTHONPATH}"
/usr/bin/python3 -m vllm serve /models/MiMo-V2.6-Flash-MOPD
```

安装脚本默认定位 `/host/vllm`，可通过 `VLLM_SOURCE` 覆盖。
本目录根部的安装脚本严格要求 0.31.0 及对应文件哈希，不能直接用于 0.32 开发版。
`VALIDATION.txt` 保留原始验证记录；此次发布只整理安装包，没有重新安装或启动模型。

本机使用的 `0.32.1rc1.dev223+gdb3b0f0fa.d20261029` 适配包单独保存在
[adapted-0.32.1rc1.dev223-20261109-174603/](adapted-0.32.1rc1.dev223-20261109-174603/README.md)。
它有独立的 `install.py`、`manifest.json`、`files/` 和 `SHA256SUMS`，须进入对应子目录使用。
两个安装包都只接受已知的原始文件或已应用文件；本机安装备份不随仓库发布。

本包基于官方 `v0.31.0` 标签，commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`，只移植 MiMo EPLB 注册、E8M0 scale 的字节视图以及配置缓存键一致性修复。文件内容保留 0.31.0 基线，不使用上级目录中的 0.32 完整文件。

官方源码参考：<https://github.com/vllm-project/vllm/tree/v0.31.0>

## 1. 将整个目录传到目标机器

在本机执行，替换用户名和服务器地址：

```bash
# 在本仓库根目录执行。
scp -r v0.31.0 user@server:/tmp/eplb-v031
```

确保 `/tmp/eplb-v031` 是新目录，避免重复复制后出现嵌套目录。也可在目标机器完整克隆本仓库后进入 `v0.31.0/`。

以下命令均在目标机器执行：

```bash
cd /tmp/eplb-v031
sha256sum -c SHA256SUMS
/usr/bin/python3 install.py
```

默认只检查，打印 Python 路径、vLLM 版本、实际包路径及四个文件的状态，不启动 GPU、不写入目标。

- `BASE`：内容与官方 0.31.0 对应原文件一致，可以应用此包。
- `APPLIED`：此文件已具有本包修复，无需重复覆盖。
- `DIFFERENT` / `MISSING`：目标是有定制修改或文件结构不同的版本；工具会停止。请提供检查输出和对应源文件，按目标内容合并。版本字符串同为 0.31.0 不足以证明文件一致。

四个路径都相对于打印出的 `Package` 目录：

```text
model_executor/models/mimo_v2.py
model_executor/layers/fused_moe/routed_experts.py
distributed/eplb/eplb_state.py
distributed/elastic_ep/elastic_execute.py
```

例如 `Package=/usr/local/lib/python3.12/dist-packages/vllm` 时，第一项实际为 `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py`。不要预先假定 Python 小版本或 site-packages/dist-packages 位置。

## 2. 停止服务，备份并应用

停止使用该 Python/vLLM 环境的现有服务。在有目标文件写权限的账号下执行：

```bash
cd /tmp/eplb-v031
BACKUP_DIR="/mnt/pfs/eplb-backups/vllm031-$(date +%Y%m%d-%H%M%S)-$$"
/usr/bin/python3 install.py --apply --backup-dir "$BACKUP_DIR"
/usr/bin/python3 install.py
```

备份位置可改为其他持久、可写目录。目标目录已存在时不会覆盖原备份。安装前校验全部目标文件；全部备份成功后才逐文件替换，保留文件模式和所有者。正常异常路径会尝试还原已写入的文件。完成后四项应为 `APPLIED`。不需要修改启动器、卸载 vLLM 或重编译 CUDA 扩展。

`runtime.patch` 提供包目录相对路径的最小 diff，供定制版本审阅和移植。安装工具只接受已知原始/已接入内容，没有强制覆盖选项。

## 3. 沿用原启动命令，启用 EPLB

保留原有 TP/DP、GPU 可见性、显存、端口及其他参数，在原命令中添加以下参数；已存在的选项不要重复追加：

```text
--enable-expert-parallel
--enable-eplb
--eplb-config '{"window_size":8,"step_interval":128,"num_redundant_experts":0,"use_async":true,"log_balancedness":true,"log_balancedness_interval":16}'
```

这是用于观察 EPLB 的示例配置，不是性能最优配置。有效 EP 组需要多张 GPU；不要因添加 EPLB 删除原有并行设置。异步迁移的推进依赖 forward 次数，热身应持续到所有 MoE 层完成一轮迁移。

启动后检查日志中是否出现 `EPLB is enabled`、实际 communicator、非零负载统计以及真实重平衡。`step_interval=128` 且从 3/4 进度开始时，首次触发约需 32 个 engine step，短请求测试可能还未触发。不要把“没有立即看到重平衡日志”直接解释为接入失败。

此补丁以目标环境本来就能正确加载 MiMo/MXFP4 为前提，不附带 Humming、NIXL、模型权重或其他模型兼容性修复。

## 4. 回滚

停止服务后，用同一个 Python 执行：

```bash
/usr/bin/python3 /tmp/eplb-v031/install.py --restore "$BACKUP_DIR"
```

回滚会校验备份和当前文件，并拒绝覆盖安装后的其他修改。再次启动服务时去掉 `--enable-eplb` 及对应配置。

## 验证范围

本包已在官方 0.31.0 文件快照上验证补丁应用/反向应用、语法、安装备份、回滚、重复执行及不匹配拒绝流程。**尚未在 vLLM 0.31.0 环境进行 GPU 启动或推理验证，也尚未读取目标服务器的实际文件。** 上级安装包的 0.32 GPU 测试结果不能当作此版本的测试结果；安装后需在目标机器验证。
