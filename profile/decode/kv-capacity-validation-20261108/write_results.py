"""Render a compact Chinese evidence note from validated machine-readable data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
result = json.loads((DATA / "evidence-summary.json").read_text())
integrity = json.loads((DATA / "source-integrity.json").read_text())
assert not integrity["changed_files"]
assert len(result["cases"]) == 4
case_map = {c["summary"]["case"]: c for c in result["cases"]}
long_case = case_map["long_n4096_kv10240"]
short_case = case_map["short_n4096_kv1024"]
repeat = case_map["repeat_n4096_kv10240"]
assert long_case["summary"]["allocation_failures"] > 0
assert repeat["summary"]["allocation_failures"] > 0
assert short_case["summary"]["allocation_failures"] == 0
assert short_case["summary"]["metrics_peak_global_running"] == 4096
example = next(r for r in result["failure_examples"]
               if r["case"] == "long_n4096_kv10240" and r["dp_rank"] == 0)
cfg = result["engine_configurations"]["0"]
lines = [
    "结论：在本次 TP1 / DP4、EP关闭、无投机配置下，已直接验证 KV block 不足限制了 decode 并发。",
    "",
    "所有实验使用同一服务实例，max_num_seqs=4096/DP，max_num_batched_tokens=8120/DP，GPU memory utilization=0.90。",
    "每请求输出64 token、ignore_eos=true，使用直接token ID输入。与原capture的1000输出token、随机文本输入不同，因此不用于吞吐比较。",
    "",
    "已完成实验：",
]
for name in ("sanity_n512_kv10240", "long_n4096_kv10240", "short_n4096_kv1024", "repeat_n4096_kv10240"):
    s = case_map[name]["summary"]
    lines.append(f"{name}: 请求={s['requests']}，完成={s['completed']}，错误={s['errors']}；"
                 f"各DP实际单步请求数峰值={s['per_dp_peak_scheduled_requests']}；"
                 f"metrics全局running峰值={s['metrics_peak_global_running']:.0f}；KV分配失败次数={s['allocation_failures']}。")
lines += [
    "",
    "最直接的原始证据（同一个调度step）：",
    f"DP0 step={example['step']}，scheduled_requests={example['scheduled_requests']}，scheduled_tokens={example['scheduled_tokens']}，"
    f"token预算={example['token_budget']}，running={example['running']}，waiting={example['waiting']}。",
    f"等待请求的模拟KV={example['external_tokens']} token，本步新计算={example['new_tokens']} token。",
    f"原分配器计算需要{example['required_blocks']}个blocks，但实际只剩{example['available_blocks']}个，返回None；"
    "watermark与reserved_blocks均为0。这不是客户端代理推算的分配失败。",
    f"原始分配记录：{example['failure_source']}:{example['failure_line']}",
    f"原始调度记录：{example['step_source']}:{example['step_line']}",
    "",
    f"DP0总block数={cfg['total_blocks']}，启动可用={cfg['initial_free_blocks']}。"
    f"按首个长KV请求所需{example['required_blocks']} blocks估算，可容纳"
    f"{cfg['initial_free_blocks']//example['required_blocks']}条请求，与实际单步峰值一致。",
    "缩短KV后，同样4096请求全部进入running，无KV分配失败；恢复长KV后并发上限重新下降。",
    "因此，长KV条件下限制并发的直接原因是KV blocks不足，并非4096并发配置未生效，也不是8120的单步token预算先耗尽。",
    "",
    "解释与边界：",
    "KV分配失败指某次调度无法分配空间，不是HTTP请求失败；所有请求最终均完成。",
    "每DP单步峰值与全局running是两种口径；不能把异步DP的峰值相加当作同步GPU batch。",
    "部分长KV步骤发生抢占与重算，可能消耗完整token预算；所引用的直接证据来自每请求仅调度1 token的步骤。",
    "观测补丁仅存在于独立服务的进程内，原函数参数与返回值不变；日志有时间开销，不用于精确性能比较。",
    f"验证后复核{integrity['checked_files']}个源码文件的SHA-256：改动文件数={len(integrity['changed_files'])}。",
    "仅验证了baseline EP关闭配置，不把这一结果冒充其他TP/EP/投机配置的逐步实测。",
    "",
    "详细结果：data/evidence-summary.json、case-summary.csv、per-dp-summary.csv、allocation-failure-examples.csv。",
    "脚本使用方法见README.txt。",
]
(ROOT / "RESULTS.txt").write_text("\n".join(lines)+"\n")
print(ROOT / "RESULTS.txt")
