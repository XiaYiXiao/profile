# EPLB 适配包：vLLM 0.32.1rc1.dev223

此目录适用于本机运行版本 `0.32.1rc1.dev223+gdb3b0f0fa.d20261029` 的已知源码文件，
将上一级 0.31.0 最小 EPLB diff 移植到该版本的四个文件中。
`manifest.json` 记录版本、补丁哈希及每个目标文件的安装前/后 SHA256；
版本或文件内容不匹配时拒绝安装，不强制覆盖定制代码。

从此目录执行以下检查，`VLLM_SOURCE` 指向包含 `vllm/` 的源码根目录：

```bash
sha256sum -c SHA256SUMS
VLLM_SOURCE=/host/vllm /usr/bin/python3 install.py
```

默认只读检查。确定目标环境并停止使用它的服务后，再安装：

```bash
VLLM_SOURCE=/host/vllm /usr/bin/python3 install.py --apply \
  --backup-dir /data/eplb-backups/my-install
```

备份目录必须尚不存在。已安装的四个文件会显示 `APPLIED`，无需重复安装。
安装器保留原文件并记录回滚信息；备份留在安装机器，不加入 Git。
停止服务后可用同一环境回滚，安装后的额外修改不会被静默覆盖：

```bash
VLLM_SOURCE=/host/vllm /usr/bin/python3 install.py \
  --restore /data/eplb-backups/my-install
```

该适配包已用于本机混部多轮与 4P4D 运行。6P2D 的完整 GPU 推理验收仍未完成，
请同时参考 [部署文档](../../performance_eva/README.md) 中的专家数整除和 KV 显存限制。
这些运行记录不代表上一级原始 0.31.0 包已完成 GPU 验证。
