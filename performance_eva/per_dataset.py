
"""
通过 evalscope 跑 perf 测试并生成 csv & xlsx。
支持注入四类 Header：
1. session-id：五种模式 per_req / global / fixed_n / per_conv / per_conv_fixed_n
   （多轮对话下 per_conv* 让同一会话的多个 turn 共享 sid）
2. X-SMG-Routing-Key：值与 session-id 完全一致，随 session 策略同步生成
3. request-id：每次请求全新随机 uuid4
4. x-request-id：每次请求全新随机 uuid4
约束：同一条请求三个 uuid 类 header 全部不重复
压测完成自动导出 session-id:X-SMG-Routing-Key:request-id:x-request-id 结构化映射 csv 明细

== 用法示例 ==

1. 单轮 basic：1k → 1k，三种并发 sweep
   python per_dataset.py \\
       --ip 127.0.0.1 --port 10011 --use-chat \\
       --model /mnt/shared/models/DeepSeek-R1-0528 \\
       --tokenizer-path /mnt/shared/models/DeepSeek-R1-0528 \\
       --input-length 1024 --output-length 1024 \\
       --parallel "1 4 8" --number "100 400 800"

2. 单轮 + 长稳 1 小时（自动放大 number 到 10^9）
   python per_dataset.py \\
       --parallel 16 --duration 3600 \\
       --warmup-num 0.05 \\
       --input-length 1024 --output-length 1024 \\
       --ip ... --use-chat --model ... --tokenizer-path ...

3. 单轮 + 长度区间（input/output 都采样）
   python per_dataset.py \\
       --input-length-min 256 --input-length-max 2048 \\
       --output-length-min 256 --output-length-max 1024 \\
       --parallel 4 --number 200 ...

4. 单轮 + 预生成 random dataset（避免每次都生成）：
   # 一次性生成 5000 条 1k token prompt
   python per_dataset.py \\
       --prepare-dataset --prepare-num 5000 \\
       --input-length 1024 \\
       --tokenizer-path /data/models/your-model \\
       --prepared-dataset-path prompts.txt
   # 之后所有 perf 跑都复用（文件存在 → 直接构造请求，跳过分词和长度检查）
   python per_dataset.py \\
       --parallel 8 --number 5000 \\
       --prepared-dataset-path prompts.txt ...

5. 单轮 + cache 命中（带 prefix，prefix_length 算入总输入长度）
   python per_dataset.py \\
       --prefix-length 512 \\
       --input-length 512 \\
       --parallel 4 --number 200

6. 多轮会话（3 个用户各跑自己的多轮对话，per_conv 让每个对话共享 sid）
   python per_dataset.py \\
       --parallel 3 \\
       --multi-turn --min-turns 3 --max-turns 5 \\
       --multi-turn-dataset random_multi_turn \\
       --tokenizer-path /data/models/your-model \\
       --number 3 --duration 1800 \\
       --inject-session-id --session-mode per_conv

7. 多轮 + 长稳 + warmup + sid 注入（4 用户在线）
   python per_dataset.py \\
       --parallel 4 \\
       --multi-turn --min-turns 3 --max-turns 5 \\
       --multi-turn-dataset random_multi_turn \\
       --tokenizer-path /data/models/your-model \\
       --input-length 1024 --output-length 1024 \\
       --duration 3600 --warmup-num 0.05 \\
       --inject-session-id --session-mode per_conv \\
       --inject-req-ids

8. 多轮 + 预生成 custom_multi_turn：
   # 先生成 1000 条 5-12 turn 的多轮对话
   python per_dataset.py \\
       --prepare-dataset --prepare-num 1000 \\
       --multi-turn --min-turns 5 --max-turns 12 \\
       --input-length 1000 \\
       --tokenizer-path /data/models/your-model \\
       --prepared-dataset-path prompts_multiturn.jsonl
   # 用同一份文件反复跑（结果可复现 → cache 命中率对比准确）
   python per_dataset.py \\
       --multi-turn --multi-turn-dataset custom_multi_turn \\
       --multi-turn-dataset-path prompts_multiturn.jsonl \\
       --parallel 8 --duration 1800 ...

== 输出 ==
- outputs/<微秒时间戳_UUID>/loop_N/sweep_N_pP_nM/.../benchmark_summary.json （逐 run 指标）
- 同一 sweep 目录下 benchmark_data.db / workload_timeline.json 等 EvalScope 原始产物
- 本次输出目录中的 summary.csv / summary.xlsx（汇总所有 loop/sweep）
- request_map.csv / request_ids.log（开启 Header 或多轮插件时）
- --output-dir 可指定尚不存在的独占目录；相对输出文件名以此目录为根

"""

import os
import sys
import csv
import json
import time
import uuid
import hashlib
import tempfile
from contextlib import contextmanager
import click
import itertools
import pandas as pd
from glob import glob
from datetime import datetime
from pathlib import Path
import re as _re
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.plugin.datasets.line_by_line import LineByLineDatasetPlugin
from evalscope.perf.plugin.registry import register_api, register_dataset


@register_dataset('prepared_requests')
class _PreparedRequestsDataset(LineByLineDatasetPlugin):
    """Read saved prompts verbatim, without tokenization or length filtering."""

    def prepare_messages(self, prompt):
        # No tokenizer also makes EvalScope's normal length check use characters,
        # which can discard long prompts. Bypass that check, not just tokenization.
        if self.query_parameters.apply_chat_template:
            return [{'role': 'user', 'content': prompt}]
        return prompt


class _ServerUsageOnly:
    """Use authoritative server counts; never estimate tokens from text/chunks."""

    def parse_responses(self, responses, request=None, **kwargs):
        for response in reversed(responses):
            usage = response.get('usage') if isinstance(response, dict) else None
            if not isinstance(usage, dict):
                continue
            counts = (usage.get('prompt_tokens'), usage.get('completion_tokens'))
            if all(type(count) is int and count >= 0 for count in counts):
                return counts
        raise ValueError(
            '已保存数据集使用服务端 token 统计，但响应缺少有效的 '
            'usage.prompt_tokens / usage.completion_tokens；'
            '请确保服务及代理返回完整 usage（流式请求已启用 include_usage）。'
        )


@register_api('openai_server_usage')
class _ServerUsageOpenaiPlugin(_ServerUsageOnly, OpenaiPlugin):
    pass


@register_api('openai_session')
class _OpenaiSessionPlugin(OpenaiPlugin):
    """OpenAI 兼容接口，支持多组ID请求头注入
    Header列表：
        session-id, X-SMG-Routing-Key(复用sid), request-id, x-request-id
    session-id 模式：per_req / global / fixed_n
    X-SMG-Routing-Key 跟随session-id取值，仅开启session注入时生效
    request-id & x-request-id：每条请求全新uuid，三者互不相同
    内存缓存全量请求四元ID映射，结束后导出req_map.csv
    """
    # Header 常量定义
    SESSION_HEADER = 'session-id'
    SMG_ROUTING_HEADER = 'X-SMG-Routing-Key'
    REQ_ID_HEADER = 'request-id'
    X_REQ_ID_HEADER = 'x-request-id'
    SID_LOG_PATH = '/tmp/evalscope_sids.log'

    # session-id 全局缓存
    _global_sid = None
    _sid_pool = []
    _pool_cycle = None
    # per_conv* 模式：conversation 指纹 -> sid 的稳定映射
    _conv_sid_map: dict = {}
    session_mode = "per_req"
    session_count = 1

    # 功能开关
    enable_session_inject = False
    enable_reqid_inject = False
    # 多轮模式开关：开启后 process_request 会从 body 抽取 conversation 指纹
    multi_turn_mode = False

    # 全局存储所有请求四元ID映射（每条请求一条记录）
    req_mapping = []

    @classmethod
    def reset_all_cache(cls):
        """完整压测前重置所有会话缓存与请求映射列表，隔离多轮loop"""
        cls._global_sid = None
        cls._sid_pool = []
        cls._pool_cycle = None
        cls._conv_sid_map = {}
        cls.req_mapping = []

    def _gen_unique_uuids(self, count: int, exclude: str = None) -> list[str]:
        """生成指定数量互不重复的uuid4，支持排除指定已有ID"""
        ids = set()
        if exclude is not None:
            ids.add(exclude)
        while len(ids) < count + (1 if exclude else 0):
            ids.add(str(uuid.uuid4()))
        res = list(ids)
        if exclude is not None:
            res.remove(exclude)
        return res[:count]

    @staticmethod
    def _extract_conv_id(body) -> str:
        """从请求体抽取会话指纹（用于 per_conv / per_conv_fixed_n 模式）。

        取 body.messages 中第一条 role=user 的 content 截断到 256 字符。
        多模态 content list 时拼接所有 type=text 的部分。
        非 dict / 无 messages / 没有 user 消息时返回 None（调用方需兜底）。
        """
        if not isinstance(body, dict):
            return None
        messages = body.get('messages')
        if not isinstance(messages, list) or not messages:
            return None
        for m in messages:
            if not isinstance(m, dict) or m.get('role') != 'user':
                continue
            content = m.get('content', '')
            if isinstance(content, list):
                texts = [
                    p.get('text', '') for p in content
                    if isinstance(p, dict) and p.get('type') == 'text'
                ]
                content = ''.join(texts)
            if not isinstance(content, str):
                content = str(content)
            return content[:256]
        return None

    async def process_request(self, client_session, url, headers, body):
        per_req_headers = {**headers}
        sid = None
        smg_routing_key = None
        req_id = None
        x_req_id = None
        now_ns = time.time_ns()

        # 0. 多轮模式：从 body 抽取会话指纹，用于 per_conv* 模式跨 turn 稳定 sid
        conv_id = self._extract_conv_id(body) if self.multi_turn_mode else None

        # 1. 生成 session-id
        if self.enable_session_inject:
            if self.session_mode == "per_req":
                sid = str(uuid.uuid4())
            elif self.session_mode == "global":
                if self._global_sid is None:
                    self._global_sid = str(uuid.uuid4())
                    try:
                        with open(self.SID_LOG_PATH, 'a') as f:
                            f.write(f'{now_ns} sid={self._global_sid} [GLOBAL_SEED]\n')
                    except Exception:
                        pass
                sid = self._global_sid
            elif self.session_mode == "fixed_n":
                if not self._sid_pool:
                    self._sid_pool = [str(uuid.uuid4()) for _ in range(self.session_count)]
                    self._pool_cycle = itertools.cycle(self._sid_pool)
                    try:
                        with open(self.SID_LOG_PATH, 'a') as f:
                            for s in self._sid_pool:
                                f.write(f'{now_ns} pool_seed={s} [POOL_SEED]\n')
                    except Exception:
                        pass
                sid = next(self._pool_cycle)
            elif self.session_mode == "per_conv":
                # 一条多轮对话 = 一个稳定 sid；同一对话的所有 turn 共用
                # 兜底：conv_id 抽取失败时退化为 per_req
                if conv_id is None:
                    sid = str(uuid.uuid4())
                elif conv_id not in self._conv_sid_map:
                    self._conv_sid_map[conv_id] = str(uuid.uuid4())
                    try:
                        with open(self.SID_LOG_PATH, 'a') as f:
                            f.write(f'{now_ns} conv_seed={self._conv_sid_map[conv_id]} [CONV_SEED]\n')
                    except Exception:
                        pass
                    sid = self._conv_sid_map[conv_id]
                else:
                    sid = self._conv_sid_map[conv_id]
            elif self.session_mode == "per_conv_fixed_n":
                # N 个 sid 的池，按 conv_id 稳定分配（同一对话永远拿到同一个 sid）
                if not self._sid_pool:
                    self._sid_pool = [str(uuid.uuid4()) for _ in range(self.session_count)]
                    self._pool_cycle = itertools.cycle(self._sid_pool)
                    try:
                        with open(self.SID_LOG_PATH, 'a') as f:
                            for s in self._sid_pool:
                                f.write(f'{now_ns} pool_seed={s} [POOL_SEED]\n')
                    except Exception:
                        pass
                if conv_id is None:
                    sid = next(self._pool_cycle)
                elif conv_id not in self._conv_sid_map:
                    self._conv_sid_map[conv_id] = next(self._pool_cycle)
                    sid = self._conv_sid_map[conv_id]
                else:
                    sid = self._conv_sid_map[conv_id]
            # X-SMG-Routing-Key 直接复用session-id的值
            smg_routing_key = sid

        # 2. 生成 request-id / x-request-id（强制和sid不重复）
        if self.enable_reqid_inject:
            req_id, x_req_id = self._gen_unique_uuids(count=2, exclude=sid)

        # 3. 写入headers
        if sid:
            per_req_headers[self.SESSION_HEADER] = sid
            per_req_headers[self.SMG_ROUTING_HEADER] = smg_routing_key
        if req_id:
            per_req_headers[self.REQ_ID_HEADER] = req_id
        if x_req_id:
            per_req_headers[self.X_REQ_ID_HEADER] = x_req_id

        # 4. 内存记录本次请求四元ID映射（用于最终导出req_map.csv）
        self.req_mapping.append([
            now_ns,
            sid or "",
            smg_routing_key or "",
            req_id or "",
            x_req_id or ""
        ])

        # 5. 原始日志落地 /tmp 留存溯源
        log_parts = [f"{now_ns}"]
        if sid:
            log_parts.append(f"sid={sid}")
            log_parts.append(f"smg_routing={smg_routing_key}")
        if req_id:
            log_parts.append(f"reqid={req_id}")
        if x_req_id:
            log_parts.append(f"xreqid={x_req_id}")
        try:
            with open(self.SID_LOG_PATH, 'a') as f:
                f.write(" ".join(log_parts) + "\n")
        except Exception:
            pass

        return await super().process_request(client_session, url, per_req_headers, body)

    @classmethod
    def dump_req_mapping(cls, output_path: str):
        """导出内存中全量请求四元ID映射到csv文件"""
        if not cls.req_mapping:
            print("无请求ID映射数据，跳过生成req_map文件")
            return
        headers = [
            "timestamp_ns",
            "session-id",
            "X-SMG-Routing-Key",
            "request-id",
            "x-request-id"
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(cls.req_mapping)
        print(f"请求四元ID映射明细已保存至: {output_path}")


@register_api('openai_session_server_usage')
class _ServerUsageSessionPlugin(_ServerUsageOnly, _OpenaiSessionPlugin):
    pass


def _sample_random_prompts(
    tokenizer, allowed, num_prompts: int, in_min: int, in_max: int, rng
):
    """采样 num_prompts 条 prompt 文本，返回 List[str]。

    每条长度若 in_min==in_max 则固定，否则在 [in_min, in_max] 均匀采样。
    """
    import numpy as np
    if in_min == in_max:
        lengths = [in_min] * num_prompts
    else:
        lengths = rng.integers(in_min, in_max + 1, size=num_prompts).tolist()
    out = []
    for n_tokens in lengths:
        offsets = rng.integers(0, len(allowed), size=int(n_tokens))
        token_ids = allowed[offsets].tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        out.append(text.replace('\n', ' ').replace('\r', ' '))
    return out


def _build_allowed_tokens(tokenizer):
    """构造 allowed_tokens：过滤 special + byte-fallback tokens。

    复用 evalscope RandomDatasetPlugin._get_byte_fallback_token_ids() 的过滤逻辑。
    """
    import numpy as np
    full_vocab = len(tokenizer)
    prohibited = set(tokenizer.all_special_ids)
    byte_re = _re.compile(r'^<0x[0xA-Fa-f0-9]{2}>$')
    try:
        vocab_dict = tokenizer.get_vocab()
        for tok_str, tok_id in vocab_dict.items():
            if byte_re.match(tok_str):
                prohibited.add(tok_id)
    except Exception:
        for i in range(full_vocab):
            try:
                if byte_re.match(tokenizer.convert_ids_to_tokens(i) or ''):
                    prohibited.add(i)
            except Exception:
                pass
    return np.array(sorted(set(range(full_vocab)) - prohibited))


@contextmanager
def _new_dataset_file(path):
    """Publish a complete dataset without overwriting an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=f'.{path.name}.', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            yield handle
            handle.flush()
            os.link(temporary, path)  # Atomic, and fails if another writer won.
        finally:
            temporary.unlink(missing_ok=True)


def _fit_single_line_tokens(tokenizer, token_ids, target, allowed, rng):
    """Match the token count after the same whitespace cleanup as line_by_line."""
    for _ in range(20):
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        text = text.replace('\n', ' ').replace('\r', ' ').strip()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == target:
            return text, token_ids
        if len(token_ids) > target:
            token_ids = token_ids[:target]
        else:
            token_ids += rng.choice(allowed, size=target - len(token_ids)).tolist()
    raise click.ClickException(f'无法生成精确 {target} token 的单行文本，请更换 seed 或 tokenizer')


def _shared_prefix_prompts(tokenizer, allowed, count, prefix_length, in_min, in_max, rng):
    """Generate unique prompts with an exact, stable token prefix before chat templating."""
    _, prefix_ids = _fit_single_line_tokens(
        tokenizer, rng.choice(allowed, size=prefix_length).tolist(),
        prefix_length, allowed, rng,
    )
    seen = set()
    for _ in range(count):
        suffix_length = int(rng.integers(in_min, in_max + 1))
        for _attempt in range(100):
            candidate = prefix_ids + rng.choice(allowed, size=suffix_length).tolist()
            text, ids = _fit_single_line_tokens(
                tokenizer, candidate, prefix_length + suffix_length, allowed, rng,
            )
            digest = hashlib.sha256(text.encode('utf-8')).digest()
            # A suffix can merge with the prefix's final token. Retry instead of
            # silently changing the common prefix or reusing a complete prompt.
            if ids[:prefix_length] == prefix_ids and digest not in seen:
                seen.add(digest)
                yield text
                break
        else:
            raise click.ClickException('无法生成新的共享前缀请求，请增加后缀长度或更换 seed')


def build_random_dataset(
    output_path: str,
    tokenizer_path: str,
    num_prompts: int,
    in_min: int,
    in_max: int,
    multi_turn: bool = False,
    turns_min: int = 1,
    turns_max: int = 1,
    seed: int = 42,
    prefix_length: int = 0,
):
    """预生成 random 数据集到本地文件。

    multi_turn=False（默认）：
      输出 line_by_line 格式 — 每行一个 prompt 文本。

    multi_turn=True：
      输出 custom_multi_turn 格式 — 每行一个 OpenAI messages JSON 数组；
      每条对话的 turn 数从 [turns_min, turns_max] 均匀采样；
      每个 user turn 的内容按 [in_min, in_max] 区间采样 token 序列并 decode。

    单轮 prefix_length > 0 时，共享前缀只生成一次；逐条校验重新分词后的
    prefix_length + suffix_length 和前缀 token 一致性，完整 prompt 不重复。
    长度不含聊天模板。其他模式保留原有随机采样长度（重新分词后可能变化）。
    """
    import numpy as np
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from evalscope.perf.plugin.datasets.utils import load_tokenizer

    if not output_path:
        raise click.BadParameter('--prepare-dataset 必须指定非空 --prepared-dataset-path')
    if Path(output_path).exists():
        raise click.BadParameter(f'拒绝覆盖已有数据集: {output_path}；直接复用或更换路径')
    if num_prompts <= 0 or in_min <= 0 or in_max < in_min or prefix_length < 0:
        raise click.BadParameter('请求数和输入长度必须为正，min <= max，prefix-length >= 0')
    if multi_turn and prefix_length:
        raise click.BadParameter('多轮数据集不支持 --prefix-length，请使用单轮模式')
    if multi_turn and not (1 <= turns_min <= turns_max):
        raise click.BadParameter('多轮要求 1 <= min-turns <= max-turns')

    print(f"[prepare] 加载 tokenizer: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        raise click.BadParameter(f"无法加载 tokenizer: {tokenizer_path}")

    allowed = _build_allowed_tokens(tokenizer)
    full_vocab = len(tokenizer)
    excluded = full_vocab - len(allowed)
    print(f"[prepare] vocab={full_vocab}, allowed={len(allowed)} "
          f"(excluded {excluded} special/byte-fallback)")

    rng = np.random.default_rng(seed)

    if multi_turn:
        turns_per_conv = rng.integers(turns_min, turns_max + 1, size=num_prompts).tolist()
        total_turns = sum(turns_per_conv)
        print(f"[prepare] 多轮模式：{num_prompts} 条对话，每条 turn ∈ [{turns_min}, {turns_max}] "
              f"（总 ≈ {total_turns} turns）→ {output_path}")
        written = 0
        with _new_dataset_file(output_path) as f:
            for conv_idx, n_turns in enumerate(turns_per_conv):
                # 每条对话的所有 turn 用一次连续采样（一次性 np.random 比逐 turn 循环快）
                prompts = _sample_random_prompts(
                    tokenizer, allowed, n_turns, in_min, in_max, rng
                )
                # 关键：在 user 消息之间插入 assistant 占位（空 content）
                # evalscope custom_multi_turn 的 _split_into_turns() 用
                # assistant 消息作 turn 边界；不加占位 = 整个对话塌缩成 1 个 turn。
                # assistant content 是空串，evalscope multi_turn runner 会
                # 丢弃 dataset assistant 内容，跑时用真实模型回复填充。
                messages = []
                for i, p in enumerate(prompts):
                    if i > 0:
                        messages.append({'role': 'assistant', 'content': ''})
                    messages.append({'role': 'user', 'content': p})
                f.write(json.dumps(messages, ensure_ascii=False) + '\n')
                written += 1
                if (written % 100 == 0) and written < num_prompts:
                    print(f"[prepare] {written}/{num_prompts} conversations 已生成")
    else:
        if in_min == in_max:
            mode_str = f"固定 {in_min} tokens"
        else:
            mode_str = f"区间 [{in_min}, {in_max}] tokens"
        print(f"[prepare] 单轮模式：{num_prompts} 条 prompt ({mode_str}) → {output_path}")
        if prefix_length:
            print(f'[prepare] 共享前缀 {prefix_length} tokens + 不同后缀 {mode_str}；'
                  '逐条校验长度及去重（不含聊天模板）')
            prompts = _shared_prefix_prompts(
                tokenizer, allowed, num_prompts, prefix_length, in_min, in_max, rng,
            )
        else:
            prompts = (_sample_random_prompts(tokenizer, allowed, 1, in_min, in_max, rng)[0]
                       for _ in range(num_prompts))
        with _new_dataset_file(output_path) as f:
            for i, prompt in enumerate(prompts):
                f.write(prompt + '\n')
                if ((i + 1) % 200 == 0) and (i + 1) < num_prompts:
                    print(f"[prepare] {i + 1}/{num_prompts} 已生成")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    summary = (
        f"[prepare] 完成：{num_prompts} 条 conversations, 总 {total_turns} turns"
        if multi_turn else
        f"[prepare] 完成：{num_prompts} 条 prompts"
    )
    print(f"{summary}, 文件大小 {size_mb:.1f} MB")


def extract_json_to_csv(input_dir, output_csv, input_length, output_length, loop, concurrency: int = None):
    exclude_fields = ["Number of concurrency", "Input Length", "Output Length", "Loop", "Concurrency"]
    all_json_fields = []
    temp_rows = []
    file_exists = os.path.isfile(output_csv)

    for json_file in glob(os.path.join(input_dir, "**", "benchmark_summary.json"), recursive=True):
        with open(json_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"警告: 无法解析JSON文件: {json_file}")
                continue
        for field in data.keys():
            if field not in exclude_fields and field not in all_json_fields:
                all_json_fields.append(field)
        row_base = [input_length, output_length, loop, int(concurrency)]
        temp_rows.append(dict(zip(["Input Length", "Output Length", "Loop", "Concurrency"], row_base),
                              **{k: v for k, v in data.items() if k not in exclude_fields}))

    headers = ["Input Length", "Output Length", "Loop", "Concurrency"] + all_json_fields
    if not temp_rows:
        raise RuntimeError(f"没有评估结果: {input_dir}")
    frame = pd.DataFrame(temp_rows, columns=headers)
    if file_exists:
        frame = pd.concat([pd.read_csv(output_csv), frame], ignore_index=True)
    frame.to_csv(output_csv, index=False)



def parse_csv_to_xlsx(in_csv, out_xlsx):
    df = pd.read_csv(in_csv)

    # Support both older EvalScope labels (seconds) and 1.14 labels (milliseconds).
    if "Average time per output token (s)" in df:
        tpot = pd.to_numeric(df["Average time per output token (s)"], errors="coerce")
    elif "Avg TPOT (ms)" in df:
        tpot = pd.to_numeric(df["Avg TPOT (ms)"], errors="coerce") / 1000
    else:
        tpot = pd.Series(float("nan"), index=df.index)
    df["单轮次 Avg TPS"] = 1 / tpot.where(tpot > 0)
    throughput = next((key for key in ("Output token throughput (tok/s)", "Output Throughput (tok/s)")
                       if key in df), None)
    df["单轮次平均单路吞吐（tok/s）"] = (
        pd.to_numeric(df[throughput], errors="coerce") / df["Concurrency"]
        if throughput else float("nan")
    )

    avg_tps_all_round = (
        df.groupby(["Input Length", "Output Length", "Concurrency"])["单轮次 Avg TPS"]
        .mean()
        .rename("所有轮次平均 TPS")
        .reset_index()
    )
    df = df.merge(avg_tps_all_round, on=["Input Length", "Output Length", "Concurrency"], how="left")

    avg_tp_all_round = (
        df.groupby(["Input Length", "Output Length", "Concurrency"])["单轮次平均单路吞吐（tok/s）"]
        .mean()
        .rename("所有轮次平均单路吞吐（tok/s）")
        .reset_index()
    )
    df = df.merge(avg_tp_all_round, on=["Input Length", "Output Length", "Concurrency"], how="left")

    base_cols = ["Input Length", "Output Length", "Loop", "Concurrency"]
    tps_calc_cols = [
        "单轮次 Avg TPS",
        "单轮次平均单路吞吐（tok/s）",
        "所有轮次平均 TPS",
        "所有轮次平均单路吞吐（tok/s）"
    ]
    json_cols = [
        col for col in df.columns
        if col not in base_cols and col not in tps_calc_cols
    ]
    custom_columns = base_cols + json_cols + tps_calc_cols
    df = df[custom_columns]

    wb = Workbook()
    ws = wb.active
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append([None if pd.isna(value) else value for value in r])
    df.to_csv(in_csv, index=False)
    wb.save(out_xlsx)
    print("性能汇总xlsx结果已保存: ", out_xlsx)


@click.command(
    context_settings=dict(help_option_names=["-h", "--help"])
)
@click.option("--ip", type=str, default="127.0.0.1", show_default=True, help="AMaaS 管理 IP")
@click.option("--port", type=str, default=10011, show_default=True, help="AMaaS API 端口")
@click.option(
    "--scheme",
    type=click.Choice(["http", "https"]),
    default="http",
    show_default=True,
    help="URL scheme.",
)
@click.option(
    "--include-port/--no-include-port",
    default=True,
    show_default=True,
    help="是否在 URL 里带 :port。关闭时走默认端口 (http 80 / https 443)。",
)
@click.option("--rate", type=float, default=-1, show_default=True,
              help="目标调度速率（请求/秒，随机间隔）；-1 不限速，实际发送仍受 parallel 限制")
@click.option("--parallel", type=str, default="1 4", show_default=True, help="并发数, 请用引号引起来, 如 '1 4'")
@click.option("--number", type=str, default=None, show_default="1 4", help=(
    "请求数（空格分隔多值，与 --parallel 一一配对）。\n"
    "  不传：仅在 --duration 模式下用超大值（10^9），由 --duration 软退出；\n"
    "        其他情况默认 '1 4'（向后兼容）。"
))
@click.option("--model", type=str, default="/mnt/shared/models/DeepSeek-R1-0528", show_default=True, help="模型名称")
@click.option("--tokenizer-path", type=str, default="/mnt/shared/models/DeepSeek-R1-0528", show_default=True)
@click.option(
    "--api-key",
    type=str,
    default="EMPTY",
    show_default=True,
    help="API Key, 从 AMaaS 上创建.",
)
@click.option("--input-length", type=int, default=128, show_default=True, help="Prompt 单值长度（与 --input-length-min/max 互斥）")
@click.option("--input-length-min", type=int, default=None, show_default=True, help="Prompt 长度区间下界；同时给 --input-length-max 启用区间采样")
@click.option("--input-length-max", type=int, default=None, show_default=True, help="Prompt 长度区间上界；同时给 --input-length-min 启用区间采样")
@click.option("--output-length", type=int, default=512, show_default=True, help="Output 单值长度（与 --output-length-min/max 互斥）")
@click.option("--output-length-min", type=int, default=None, show_default=True, help="Output 长度区间下界；同时给 --output-length-max 启用区间采样（走 evalscope 原生 max_tokens IntOrRange）")
@click.option("--output-length-max", type=int, default=None, show_default=True, help="Output 长度区间上界；同时给 --output-length-min 启用区间采样")
@click.option("--prefix-length", type=int, default=0, show_default=True, help="prefix cache 命中，通常用于测试 cache 命中率，与 input-length 相加才是总的输入长度.")
@click.option("--read-timeout", type=int, default=3600, show_default=True)
@click.option(
    "--duration",
    type=float,
    default=None,
    show_default=True,
    help=(
        "单次压测的墙钟预算（秒）。软退出语义：到点后不再启动新请求，"
        "但已发出去的会跑完才退出（多轮模式下为 trace-soft-exit，已领到的对话会跑完全部 turn）。\n"
        "与 --number 同时给时，先到先停（推荐用法）；都不给则按 --number 走"
    )
)
@click.option(
    "--warmup-num",
    type=float,
    default=0,
    show_default=True,
    help=(
        "Warmup 请求/对话数（不计入 metrics）：\n"
        "  0        : 禁用（默认）；\n"
        "  >= 1     : 绝对数量；\n"
        "  0 < x < 1: 与 --number 的比例（如 0.1 = 10%）。\n"
        "多轮模式下单位为「对话」而不是请求；warmup 对话的所有 turn 都会被发但不计入 metrics。"
    )
)
@click.option("--loop", type=int, default=1, show_default=True)
@click.option("--name", type=str, default="appauto-bench", show_default=True, help="任务名称")
@click.option("--test-connection", is_flag=True, show_default=True)
@click.option("--debug", is_flag=True, show_default=True)
@click.option(
    "--allow-failed-requests",
    is_flag=True,
    help="存在成功请求时，允许部分请求失败并继续；失败计数保留，零成功仍报错。",
)
@click.option("--use-chat", is_flag=True, show_default=True)
@click.option("--output-dir", type=click.Path(file_okay=False), default=None, help="本次运行的独占输出目录，必须不存在")
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--output-csv", type=str, default=None, show_default=True, help="性能汇总csv文件名，不填自动用时间戳")
@click.option("--output-xlsx", type=str, default=None, show_default=True, help="性能汇总xlsx文件")
@click.option("--output-req-map", type=str, default=None, show_default=True, help="四元ID映射明细csv，默认时间戳_req_map.csv")
@click.option(
    "--inject-session-id/--no-inject-session-id",
    default=False,
    show_default=True,
    help="开启注入 session-id header，同步自动携带 X-SMG-Routing-Key"
)
@click.option(
    "--inject-req-ids/--no-inject-req-ids",
    default=False,
    show_default=True,
    help="开启同时注入 request-id 与 x-request-id，每条请求全新不重复uuid"
)
@click.option(
    "--session-mode",
    type=click.Choice(["per_req", "global", "fixed_n", "per_conv", "per_conv_fixed_n"]),
    default="per_req",
    show_default=True,
    help=(
        "session-id模式：\n"
        "  per_req=每请求独立；\n"
        "  global=全局共用1个；\n"
        "  fixed_n=自定义N个sid池轮询；\n"
        "  per_conv=一条多轮对话共用1个sid（多轮模式推荐）；\n"
        "  per_conv_fixed_n=N个sid池按对话稳定分配（多轮+容量受控时推荐）"
    )
)
@click.option(
    "--session-count",
    type=int,
    default=1,
    show_default=True,
    help=(
        "fixed_n / per_conv_fixed_n 模式下 sid 池容量；"
        "per_conv_fixed_n 建议 >= --parallel，确保每个并发用户可分配独立 sid"
    )
)
@click.option(
    "--multi-turn/--no-multi-turn",
    default=False,
    show_default=True,
    help="开启多轮对话模式（与 --multi-turn-dataset 配合使用）；开启后 --prefix-length 不再生效"
)
@click.option(
    "--min-turns",
    type=int,
    default=1,
    show_default=True,
    help="多轮模式下，每条对话的最小 user turn 数（仅 random_multi_turn / swe_smith 生效）"
)
@click.option(
    "--max-turns",
    type=int,
    default=1,
    show_default=True,
    help=(
        "多轮模式下，每条对话的最大 user turn 数；\n"
        "  random_multi_turn：必填且 >= 1；\n"
        "  custom_multi_turn：可选，截断长对话"
    )
)
@click.option(
    "--multi-turn-dataset",
    type=click.Choice(["random_multi_turn", "custom_multi_turn"]),
    default="random_multi_turn",
    show_default=True,
    help=(
        "多轮模式使用的 dataset：\n"
        "  random_multi_turn：运行时动态合成（需 --tokenizer-path）；\n"
        "  custom_multi_turn    ：从 JSONL 文件读取（需 --multi-turn-dataset-path）。"
        "\n搭配 --prepare-dataset 预生成多轮对话时，显式用 custom_multi_turn。"
    )
)
@click.option(
    "--multi-turn-dataset-path",
    type=str,
    default=None,
    show_default=True,
    help=(
        "custom_multi_turn 模式下读取的 JSONL 路径；每行一个 OpenAI messages 数组。"
        "\n可由 --prepare-dataset --multi-turn 预先生成到 --prepared-dataset-path 后传入。"
    )
)
@click.option(
    "--write-csv/--no-write-csv",
    default=True,
    show_default=True,
    help="是否在跑完后落性能csv + 转 xlsx。关闭则只跑压测、不做性能汇总。",
)
@click.option(
    "--prepare-dataset/--no-prepare-dataset",
    default=False,
    show_default=True,
    help=(
        "仅预生成 random 数据集到本地文本文件并退出（不跑压测）。\n"
        "  开启时：写到 --prepared-dataset-path，拒绝覆盖；单轮支持共享 --prefix-length；\n"
        "  关闭时：若 --prepared-dataset-path 已存在则直接读取，跳过分词/长度检查，token 数使用服务端 usage；\n"
        "          否则走原有的 --dataset random 重新生成（保持原行为）。"
    )
)
@click.option(
    "--prepared-dataset-path",
    type=str,
    default="prompts_random.txt",
    show_default=True,
    help="数据集本地路径，每行一个完整 prompt；复用时跳过客户端分词及长度检查，不再追加 prefix 或调整 input-length。"
)
@click.option(
    "--prepare-num",
    type=int,
    default=1000,
    show_default=True,
    help="--prepare-dataset 模式下要生成的 prompt 条数。"
)
def runner(
    ip,
    port,
    scheme,
    include_port,
    rate,
    parallel,
    number,
    model,
    api_key,
    input_length,
    input_length_min,
    input_length_max,
    output_length,
    output_length_min,
    output_length_max,
    prefix_length,
    read_timeout,
    duration,
    warmup_num,
    tokenizer_path,
    loop,
    name,
    debug,
    allow_failed_requests,
    use_chat,
    test_connection,
    output_dir,
    seed,
    output_csv,
    output_xlsx,
    output_req_map,
    inject_session_id,
    inject_req_ids,
    session_mode,
    session_count,
    multi_turn,
    min_turns,
    max_turns,
    multi_turn_dataset,
    multi_turn_dataset_path,
    write_csv,
    prepare_dataset,
    prepared_dataset_path,
    prepare_num,
):
    # Preparation mode exits before reserving an evaluation directory.
    if multi_turn:
        use_chat = True
    if loop < 1 or session_count < 1:
        raise click.BadParameter("--loop / --session-count 必须 >= 1")

    number_list = [int(n) for n in number.split()] if number else None
    parallel_list = [int(p) for p in parallel.split()]

    # ---- prepare-dataset 模式：只生成数据集并退出，不跑压测 ----
    if prepare_dataset:
        if not tokenizer_path:
            raise click.BadParameter(
                "--prepare-dataset 必须指定 --tokenizer-path（用于合成 token 至目标长度）"
            )
        if prepare_num <= 0:
            raise click.BadParameter("--prepare-num 必须 > 0")
        if multi_turn and (max_turns < 1):
            raise click.BadParameter(
                "multi-turn 模式下 --prepare-dataset 需配合 --min-turns/--max-turns"
            )
        # 用 [in_min, in_max] 作为生成区间；range 模式下等效于均匀采样
        in_min_g = input_length_min if (input_length_min is not None) else input_length
        in_max_g = input_length_max if (input_length_max is not None) else input_length
        build_random_dataset(
            output_path=prepared_dataset_path,
            tokenizer_path=tokenizer_path,
            num_prompts=prepare_num,
            in_min=int(in_min_g),
            in_max=int(in_max_g),
            multi_turn=multi_turn,
            turns_min=int(min_turns) if multi_turn else 1,
            turns_max=int(max_turns) if multi_turn else 1,
            seed=seed,
            prefix_length=prefix_length,
        )
        sys.exit(0)

    if input_length <= 0 or output_length <= 0 or prefix_length < 0 or read_timeout <= 0:
        raise click.BadParameter("长度和读取超时必须为正数，prefix-length 必须 >= 0")
    if any(v is not None and v <= 0 for v in
           (input_length_min, input_length_max, output_length_min, output_length_max)):
        raise click.BadParameter("长度区间端点必须为正数")
    if multi_turn and (min_turns < 1 or min_turns > max_turns):
        raise click.BadParameter("多轮要求 1 <= min-turns <= max-turns")

    # ---- number / duration 协同决策 ----
    # 不传 --number 时：长稳模式（duration 给定）→ 用超大值；其他情况回退 '1 4'
    UNLIMITED = 10**9
    if not number_list:
        if duration is not None:
            number_list = [UNLIMITED] * max(1, len(parallel_list))
            print(f"[长稳] 未传 --number，自动放大到 {UNLIMITED}（由 --duration 软退出）")
        else:
            number_list = [1, 4]
            print(f"[提示] --number 默认 '1 4'（未传 --duration）；传 --duration 即可长稳跑")

    if len(number_list) == 1:
        number_list *= len(parallel_list)
    if len(number_list) != len(parallel_list):
        raise click.BadParameter("--number 必须是单值或与 --parallel 等长")
    if not parallel_list or any(p < 1 for p in parallel_list) or any(n < 1 for n in number_list):
        raise click.BadParameter("并发和请求数必须为正整数")

    # ---- duration 校验 ----
    if duration is not None and duration <= 0:
        raise click.BadParameter("--duration 必须 > 0")
    if duration is not None:
        print(f"[软退出] duration={duration}s（与 --number 先到先停）")

    # ---- warmup 校验 ----
    if warmup_num < 0:
        raise click.BadParameter("--warmup-num 必须 >= 0")
    if warmup_num > 0:
        if warmup_num < 1:
            print(f"[warmup] ratio={warmup_num} → 实际 warmup 数 = max(1, int({warmup_num} * number))")
        else:
            print(f"[warmup] {int(warmup_num)} {'对话' if multi_turn else '请求'}预热（不计入 metrics）")

    # ---- 长度区间解析与互斥校验 ----
    # input 长度区间：min/max 必须同时给，否则报缺一半
    in_min_given = input_length_min is not None
    in_max_given = input_length_max is not None
    if in_min_given != in_max_given:
        raise click.BadParameter(
            "--input-length-min / --input-length-max 必须同时给，否则回退到 --input-length 单值"
        )
    out_min_given = output_length_min is not None
    out_max_given = output_length_max is not None
    if out_min_given != out_max_given:
        raise click.BadParameter(
            "--output-length-min / --output-length-max 必须同时给，否则回退到 --output-length 单值"
        )
    # 与单值模式互斥：用户改了 --input-length 默认值就给全，否则算单值模式
    if (in_min_given or in_max_given) and input_length != 128:
        raise click.BadParameter("--input-length 与 --input-length-min/max 互斥，请只用一个")
    if (out_min_given or out_max_given) and output_length != 512:
        raise click.BadParameter("--output-length 与 --output-length-min/max 互斥，请只用一个")
    in_len_range = (input_length_min, input_length_max) if in_min_given else None
    out_len_range = (output_length_min, output_length_max) if out_min_given else None
    if in_len_range:
        if in_len_range[0] > in_len_range[1]:
            raise click.BadParameter(f"--input-length-min({in_len_range[0]}) > --input-length-max({in_len_range[1]})")
        print(f"[区间] input length ∈ [{in_len_range[0]}, {in_len_range[1]}]（每次请求均匀采样）")
    if out_len_range:
        if out_len_range[0] > out_len_range[1]:
            raise click.BadParameter(f"--output-length-min({out_len_range[0]}) > --output-length-max({out_len_range[1]})")
        print(f"[区间] output length ∈ [{out_len_range[0]}, {out_len_range[1]}]（每次请求均匀采样）")

    url_host = f"{scheme}://{ip}" + (f":{int(port)}" if include_port else "")
    url = f"{url_host}/v1/completions" if not use_chat else f"{url_host}/v1/chat/completions"

    # ---- 多轮模式前置校验 ----
    if multi_turn:
        if max_turns < 1:
            raise click.BadParameter("--max-turns 必须 >= 1（多轮模式下对话至少 1 个 turn）")
        if multi_turn_dataset == "random_multi_turn" and not tokenizer_path:
            raise click.BadParameter(
                "random_multi_turn 模式下必须指定 --tokenizer-path（用于合成 prompt 至目标长度）"
            )
        if multi_turn_dataset == "custom_multi_turn" and not multi_turn_dataset_path:
            raise click.BadParameter(
                "custom_multi_turn 模式下必须指定 --multi-turn-dataset-path"
            )
        if prefix_length:
            print(
                "[提示] 多轮模式下 --prefix-length 不再生效（仅 random 单轮 dataset 支持），已忽略。"
            )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex
    run_root = Path(output_dir or Path("outputs") / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=False)

    def output_file(value, default, suffix):
        path = Path(value or default)
        if path.suffix != suffix:
            path = Path(str(path) + suffix)
        path = path if path.is_absolute() else run_root / path
        if path.exists():
            raise click.BadParameter(f"拒绝覆盖已有文件: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    output_csv = output_file(output_csv, "summary", ".csv")
    output_xlsx = output_file(output_xlsx, "summary", ".xlsx")
    output_req_map_path = output_file(output_req_map, "request_map", ".csv")
    _OpenaiSessionPlugin.SID_LOG_PATH = str(run_root / "request_ids.log")
    print(f"本次输出目录: {run_root}")

    failed_sweeps = []
    # 配置插件全局开关与参数
    use_custom_plugin = inject_session_id or inject_req_ids or multi_turn
    if use_custom_plugin:
        _OpenaiSessionPlugin.reset_all_cache()
        _OpenaiSessionPlugin.enable_session_inject = inject_session_id
        _OpenaiSessionPlugin.enable_reqid_inject = inject_req_ids
        _OpenaiSessionPlugin.session_mode = session_mode
        _OpenaiSessionPlugin.session_count = session_count
        # 多轮模式开启后，插件会从 body 抽取 conversation 指纹以支持 per_conv* sid 策略
        _OpenaiSessionPlugin.multi_turn_mode = multi_turn

        print("==== Header 注入配置 ====")
        if inject_session_id:
            print(f"session-id + X-SMG-Routing-Key 开启 | 模式: {session_mode} | 会话池数量: {session_count}")
        if inject_req_ids:
            print("request-id + x-request-id 开启，每条请求生成互不重复uuid4")
        if multi_turn:
            print(f"多轮模式开启 | dataset={multi_turn_dataset} | turns=[{min_turns}, {max_turns}]")
        print(f"请求四元ID映射文件输出路径: {output_req_map_path}")
        print("========================\n")

    # 循环执行多轮压测
    for i in range(0, int(loop)):
        print(f" loop {i} ".center(100, "*"))
        for sweep_index, (p, n) in enumerate(zip(parallel_list, number_list)):
            print(f" Running parallel: {p}, number: {n} ".center(80, "-"))

            # 公共参数（两种模式共享）
            common_kwargs = dict(
                parallel=[p],
                number=[n],
                model=model,
                url=url,
                api_key=api_key,
                api="openai_session" if use_custom_plugin else "openai",
                read_timeout=int(read_timeout),
                tokenizer_path=tokenizer_path or None,
                extra_args={"ignore_eos": True},
                seed=seed + i,
                outputs_dir=str(run_root / f"loop_{i:03d}" / f"sweep_{sweep_index:03d}_p{p}_n{n}"),
                name=f"{name}-{i}-p{p}",
                debug=debug,
                no_test_connection=not test_connection,
                stream=True,
                rate=float(rate) or -1,
            )
            # duration：与 number 共同生效，先到先停
            if duration is not None:
                common_kwargs["duration"] = float(duration)
            # warmup：multi-turn 时按「对话」预热，单轮按「请求」预热
            if warmup_num > 0:
                common_kwargs["warmup_num"] = float(warmup_num)
            # output 长度：单值 → min_tokens=max_tokens=定值强制固定；
            # 区间 → max_tokens=[min, max] 走 evalscope IntOrRange 采样（min_tokens 不设）
            if out_len_range:
                common_kwargs["max_tokens"] = list(out_len_range)
            else:
                common_kwargs["min_tokens"] = int(output_length)
                common_kwargs["max_tokens"] = int(output_length)

            # input 长度：根据模式分流到 Arguments 的 min/max_prompt_length
            if in_len_range:
                in_min, in_max = in_len_range
            else:
                in_min = in_max = int(input_length)

            if multi_turn:
                # 多轮模式：dataset 由用户显式指定（random_multi_turn / custom_multi_turn / ...）
                # 用 prepare 预生成的多轮文件时，显式指定：
                #   --multi-turn-dataset custom_multi_turn --multi-turn-dataset-path <file>
                multi_kwargs = dict(
                    dataset=multi_turn_dataset,
                    multi_turn=True,
                    min_turns=int(min_turns),
                    max_turns=int(max_turns),
                )
                if multi_turn_dataset == "custom_multi_turn":
                    multi_kwargs["dataset_path"] = multi_turn_dataset_path
                if multi_turn_dataset == "random_multi_turn":
                    # random_multi_turn 用 min/max_prompt_length 控制每 turn 长度
                    multi_kwargs["min_prompt_length"] = in_min
                    multi_kwargs["max_prompt_length"] = in_max
                task_cfg = Arguments(**common_kwargs, **multi_kwargs)
            else:
                # Saved single-turn prompts were prepared separately. Reuse them
                # without loading a client tokenizer or filtering by token/char length.
                if prepared_dataset_path and os.path.isfile(prepared_dataset_path):
                    if i == 0 and p == parallel_list[0]:
                        print(f"[预生成复用] prepared_requests ← {prepared_dataset_path}")
                        print('[预生成复用] 直接构造请求，跳过客户端分词和长度检查；token 数使用服务端 usage。')
                    prepared_kwargs = dict(
                        common_kwargs,
                        tokenizer_path=None,
                        api='openai_session_server_usage' if use_custom_plugin else 'openai_server_usage',
                    )
                    task_cfg = Arguments(
                        **prepared_kwargs,
                        dataset="prepared_requests",
                        dataset_path=prepared_dataset_path,
                    )
                else:
                    task_cfg = Arguments(
                        **common_kwargs,
                        dataset="random",
                        prefix_length=prefix_length,
                        min_prompt_length=in_min,
                        max_prompt_length=in_max,
                    )

            run_perf_benchmark(task_cfg)
            json_dir = task_cfg.outputs_dir
            summaries = list(Path(json_dir).rglob("benchmark_summary.json"))
            if not summaries:
                raise RuntimeError(f"未生成 benchmark_summary.json: {json_dir}")
            for summary_path in summaries:
                summary = json.loads(summary_path.read_text())
                # EvalScope versions use slightly different spellings.
                success = summary.get("Success Requests", summary.get("Succeed Requests", 0))
                failed = summary.get("Failed Requests", 0)
                if float(success) <= 0:
                    failed_sweeps.append(str(summary_path))
                elif float(failed) > 0:
                    if allow_failed_requests:
                        click.echo(
                            f"[警告] 本轮成功 {success} 条、失败 {failed} 条；"
                            f"已启用 --allow-failed-requests，保留失败记录并继续。结果: {summary_path}",
                            err=True,
                        )
                    else:
                        failed_sweeps.append(str(summary_path))
            if write_csv:
                extract_json_to_csv(json_dir, output_csv,
                                    str(in_len_range) if in_len_range else input_length,
                                    str(out_len_range) if out_len_range else output_length, i, p)

    # 全部压测完成后统一导出文件
    if write_csv:
        print(f"\n性能汇总csv结果已保存至: {output_csv}")
        parse_csv_to_xlsx(output_csv, output_xlsx)

    # 导出四元ID映射明细
    if use_custom_plugin:
        _OpenaiSessionPlugin.dump_req_mapping(output_req_map_path)
    if failed_sweeps:
        raise click.ClickException(f"部分评估无成功请求或包含失败请求，结果已保留: {failed_sweeps}")


if __name__ == "__main__":
    runner(obj={})
