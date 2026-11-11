"""Bounded-memory P/D metric sampling; counters are differenced per label series.

No prompts are stored. Missing metrics, resets and scrape failures remain visible;
missing cache observations must never be interpreted as a zero-percent hit rate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request

from prometheus_client.parser import text_string_to_metric_families


HISTOGRAMS = (
    'time_to_first_token_seconds', 'inter_token_latency_seconds',
    'request_time_per_output_token_seconds', 'e2e_request_latency_seconds',
    'request_queue_time_seconds', 'request_prefill_time_seconds',
    'request_decode_time_seconds', 'request_inference_time_seconds',
    'request_prefill_kv_computed_tokens',
)
COUNTERS = (
    'prefix_cache_queries', 'prefix_cache_hits', 'external_prefix_cache_queries',
    'external_prefix_cache_hits', 'prompt_tokens', 'prompt_tokens_by_source',
    'prompt_tokens_cached', 'generation_tokens', 'num_preemptions', 'request_success',
)
GAUGES = ('num_requests_running', 'num_requests_waiting', 'num_requests_waiting_by_reason',
          'num_requests_kv_fetch_by_stage', 'kv_cache_usage_perc', 'gpu_cache_usage_perc')
FAMILIES = {'vllm:' + n for n in (*HISTOGRAMS, *COUNTERS, *GAUGES)}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Metrics interval/timeout must be finite and positive')
    return result


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def parse_metrics(text):
    rows = []
    for family in text_string_to_metric_families(text):
        if family.name not in FAMILIES and not family.name.startswith(
                ('vllm:nixl_', 'vllm:mooncake_', 'eva_pd_')):
            continue
        for sample in family.samples:
            if sample.name.endswith('_created') or not math.isfinite(sample.value):
                continue
            rows.append(dict(name=sample.name, labels=sample.labels, value=sample.value,
                             kind=family.type))
    if not rows:
        raise ValueError('No supported vLLM/proxy metrics found')
    return rows


def series_key(row):
    return row['name'], tuple(sorted(row['labels'].items()))


class Window:
    """Keep raw series separate until after subtraction (including DP ranks)."""
    def __init__(self):
        self.previous = {}
        self.totals = {}
        self.samples = 0
        self.errors = 0
        self.resets = 0
        self.gaps = 0
        self.first_time = None
        self.last_time = None
        self.last_error = None

    def observe(self, rows, timestamp):
        self.samples += 1
        if self.first_time is None:
            self.first_time = timestamp
        self.last_time = timestamp
        current = {}
        for row in rows:
            key = series_key(row)
            if key in current:
                raise ValueError(f'Duplicate metric series: {key}')
            current[key] = row
        for key, row in current.items():
            stat = self.totals.setdefault(key, dict(row, delta=0., intervals=0,
                                                   resets=0, gaps=0, observations=0,
                                                   sample_sum=0., maximum=row['value']))
            stat['value'] = row['value']
            stat['observations'] += 1
            stat['sample_sum'] += row['value']
            stat['maximum'] = max(stat['maximum'], row['value'])
            if row['kind'] not in ('counter', 'histogram'):
                continue
            before = self.previous.get(key)
            if before is None:
                if self.samples > 1:
                    # Dynamic labels cannot be assumed to have started at zero.
                    stat['gaps'] += 1
                    self.gaps += 1
                continue
            difference = row['value'] - before['value']
            if difference < 0:
                stat['resets'] += 1
                self.resets += 1
                continue
            stat['delta'] += difference
            stat['intervals'] += 1
        missing = self.previous.keys() - current.keys()
        for key in missing:
            self.totals[key]['gaps'] += 1
            self.gaps += 1
        self.previous = current

    def delta(self, name, labels=None):
        rows = [r for r in self.totals.values() if r['name'] == name and
                all(r['labels'].get(k) == v for k, v in (labels or {}).items())]
        if not rows or any(not r['intervals'] or r['resets'] or r['gaps'] for r in rows):
            return None
        return sum(r['delta'] for r in rows)

    def gauge(self, name):
        rows = [r for r in self.totals.values() if r['name'] == name]
        return [dict(labels=r['labels'], latest=r['value'], sample_mean=r['sample_sum'] / r['observations'],
                     max=r['maximum']) for r in rows]

    def cache(self, prefix='prefix_cache', labels=None):
        hits = self.delta('vllm:' + prefix + '_hits_total', labels)
        queries = self.delta('vllm:' + prefix + '_queries_total', labels)
        valid = hits is not None and queries is not None and 0 <= hits <= queries
        return dict(hit_tokens=hits, queried_tokens=queries,
                    hit_rate=hits / queries if valid and queries > 0 else None,
                    status=('measured' if queries > 0 else 'no_queries') if valid else 'unavailable')

    def histogram(self, name):
        total, count = self.delta(name + '_sum'), self.delta(name + '_count')
        return dict(count=count, sum=total,
                    mean=total / count if total is not None and count and count > 0 else None)

    def summary(self):
        seconds = (self.last_time - self.first_time) if self.samples > 1 else 0
        rate = lambda n, labels=None: (None if not seconds or self.delta(n, labels) is None
                                      else self.delta(n, labels) / seconds)
        cache_labels = {tuple(sorted(r['labels'].items())) for r in self.totals.values()
                        if r['name'] == 'vllm:prefix_cache_queries_total'}
        connector = sorted({r['name'][:-4] for r in self.totals.values()
                            if r['name'].startswith(('vllm:nixl_', 'vllm:mooncake_'))
                            and r['name'].endswith('_sum')})
        hist = {n: self.histogram('vllm:' + n) for n in HISTOGRAMS}
        tpot = hist['request_time_per_output_token_seconds']['mean']
        return dict(
            quality=dict(successful_scrapes=self.samples, failed_scrapes=self.errors,
                         reset_series_events=self.resets, series_gap_events=self.gaps,
                         last_error=self.last_error, first_sample_unix=self.first_time,
                         last_sample_unix=self.last_time, observed_seconds=seconds),
            prefix_cache=self.cache(), external_prefix_cache=self.cache('external_prefix_cache'),
            prefix_cache_by_labels=[dict(labels=dict(labels), **self.cache(labels=dict(labels)))
                                    for labels in sorted(cache_labels)],
            prompt_tokens_per_second=rate('vllm:prompt_tokens_total'),
            computed_prefill_tokens_per_second=rate('vllm:prompt_tokens_by_source_total', {'source': 'local_compute'}),
            output_tokens_per_second=rate('vllm:generation_tokens_total'),
            inverse_mean_tpot_tokens_per_second=1 / tpot if tpot and tpot > 0 else None,
            preemptions=self.delta('vllm:num_preemptions_total'),
            request_success=self.delta('vllm:request_success_total'),
            latencies=hist,
            gauges={n: self.gauge('vllm:' + n) for n in GAUGES},
            kv_transfer={n: self.histogram(n) for n in connector},
            kv_transfer_metrics_status='available' if connector else 'not_exposed',
            connector_errors={r['name']: self.delta(r['name']) for r in self.totals.values()
                              if r['name'].startswith(('vllm:nixl_', 'vllm:mooncake_'))
                              and r['kind'] == 'counter' and r['name'].endswith('_total')},
            proxy_stages={r['name'][:-4]: self.histogram(r['name'][:-4]) for r in self.totals.values()
                          if r['name'].startswith('eva_pd_') and r['name'].endswith('_sum')},
            proxy_requests_by_outcome={r['labels']['outcome']: self.delta(r['name'], r['labels'])
                                       for r in self.totals.values() if r['name'] == 'eva_pd_requests_total'},
            proxy_inflight=self.gauge('eva_pd_inflight'),
        )


def bottleneck_hints(roles):
    """Evidence, not a claim of exclusive causation; all thresholds are explicit."""
    hints = []
    for role in ('prefill', 'decode'):
        data = roles.get(role)
        if not data:
            continue
        q = data['latencies']['request_queue_time_seconds']['mean']
        if q is not None and q > 0.1:
            hints.append(dict(component=role, evidence=f'mean queue time {q:.4f}s > 0.1s',
                              interpretation='存在排队；需结合并发、GPU 利用率和时间序列判断容量压力。'))
        if data['preemptions'] and data['preemptions'] > 0:
            hints.append(dict(component=role, evidence=f"preemptions={data['preemptions']}",
                              interpretation='存在抢占，检查 KV 容量、并发和长上下文。'))
        for name, value in data['connector_errors'].items():
            if value and value > 0:
                hints.append(dict(component='kv_transfer', evidence=f'{role}: {name}={value}',
                                  interpretation='KV 传输异常，先排查通信与超时。'))
    return dict(status='signals_found' if hints else 'insufficient_evidence', signals=hints,
                limitation='不是瓶颈定论。TTFT 含 P 排队/执行、KV 传输、D 排队及网络；阶段均值不能直接相减得到纯传输时间。')


def gpu_snapshot(roles):
    if not roles:
        return None
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                check=True, timeout=3)
        rows = []
        for values in csv.reader(io.StringIO(result.stdout)):
            index, uuid, util, used, total, power = (v.strip() for v in values)
            assigned = next((role for role, ids in roles.items() if int(index) in ids), None)
            if assigned:
                number = lambda v: float(v) if v.replace('.', '', 1).isdigit() else None
                rows.append(dict(role=assigned, index=int(index), uuid=uuid,
                                 utilization_percent=number(util), memory_used_mib=number(used),
                                 memory_total_mib=number(total), power_watts=number(power)))
        return dict(devices=rows)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return dict(error=str(exc))


class MetricsCollector:
    def __init__(self, root, targets, *, interval=5., timeout=3., gpu_roles=None,
                 scope='deployment', authorization=None):
        self.root = Path(root)
        self.targets = targets
        self.interval = positive_seconds(interval)
        self.timeout = positive_seconds(timeout)
        self.gpu_roles = gpu_roles
        self.scope = scope
        self.authorization = authorization
        self.windows = {role: Window() for role in targets}
        self.stop_event = threading.Event()
        self.thread = None
        self.fatal_error = None
        self.started_at = utc_now()
        self.pool = ThreadPoolExecutor(max_workers=max(1, len(targets) + 1))
        self.final_report = None

    def fetch(self, role, url):
        record = dict(role=role, url=url, timestamp=utc_now(), unix_time=time.time())
        try:
            headers = {'Accept': 'text/plain'}
            if self.authorization:
                headers['Authorization'] = self.authorization
            # Do not send local telemetry through an inherited HTTP_PROXY.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(urllib.request.Request(url, headers=headers), timeout=self.timeout) as response:
                rows = parse_metrics(response.read().decode('utf-8'))
            record.update(samples=rows, unix_time=time.time())
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        return record

    def sample(self):
        futures = [self.pool.submit(self.fetch, role, url) for role, url in self.targets.items()]
        gpu = self.pool.submit(gpu_snapshot, self.gpu_roles) if self.gpu_roles else None
        records = [f.result() for f in futures]
        with (self.root / 'samples.jsonl').open('a') as output:
            for record in records:
                window = self.windows[record['role']]
                if 'error' not in record:
                    try:
                        window.observe(record['samples'], record['unix_time'])
                    except Exception as exc:
                        record['error'] = f'{type(exc).__name__}: {exc}'
                if 'error' in record:
                    window.errors += 1
                    window.last_error = record['error']
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
            if gpu:
                output.write(json.dumps(dict(role='gpu', timestamp=utc_now(), unix_time=time.time(),
                                             **gpu.result())) + '\n')
        self.publish('collecting')

    def publish(self, status):
        roles = {role: window.summary() for role, window in self.windows.items()}
        if status == 'completed' and any(w.errors or w.samples < 2 or w.resets or w.gaps
                                         for w in self.windows.values()):
            status = 'partial'
        self.final_report = dict(
            schema_version=1, scope=self.scope, status=status, started_at=self.started_at,
            updated_at=utc_now(), interval_seconds=self.interval, targets=self.targets,
            fatal_error=self.fatal_error, roles=roles, pd_analysis=bottleneck_hints(roles),
            notes=[
                '命中率是采集窗口内各 DP 的 hit token 增量之和 / query token 增量之和；不是请求命中率。',
                '不使用 EvalScope 多轮估算。D external 命中与 P/D 本地 prefix 命中分别记录。',
                'null 表示无查询、未暴露、缺少基线或计数器重置/series 变更；参见 quality。',
                '服务指标覆盖此窗口内所有客户端；run_load 窗口含准备、warmup、全部 sweep 及尾部刷数。',
                '速率以观测壁钟时长为分母；聚合 token/s 与 inverse_mean_tpot 不同。',
                'samples.jsonl 保留时间、标签、直方图 buckets 和部署侧 GPU 样本；可用于分段分析。',
                '只有 connector 实际暴露的 Prometheus 指标才有纯 KV 传输统计；部分 Mooncake 版本尚未实现此接口。',
            ])
        atomic_json(self.root / 'summary.json', self.final_report)
        lines = ['role,prefix_hit_tokens,prefix_query_tokens,prefix_hit_rate,external_hit_rate,computed_prefill_tokens_s,output_tokens_s']
        for role, data in roles.items():
            values = [role, data['prefix_cache']['hit_tokens'], data['prefix_cache']['queried_tokens'],
                      data['prefix_cache']['hit_rate'], data['external_prefix_cache']['hit_rate'],
                      data['computed_prefill_tokens_per_second'], data['output_tokens_per_second']]
            lines.append(','.join('' if v is None else str(v) for v in values))
        temp = self.root / 'summary.csv.tmp'
        temp.write_text('\n'.join(lines) + '\n')
        temp.replace(self.root / 'summary.csv')

    def start(self):
        try:
            self.root.mkdir(parents=True, exist_ok=False)
        except BaseException:
            self.pool.shutdown(wait=True)
            raise
        try:
            self.sample()  # Synchronous baseline BEFORE releasing the workload.
        except Exception as exc:
            self.fatal_error = str(exc)
        self.thread = threading.Thread(target=self.run, name='server-metrics', daemon=True)
        self.thread.start()
        return self

    def run(self):
        try:
            while not self.stop_event.wait(self.interval):
                self.sample()
        except Exception as exc:
            self.fatal_error = str(exc)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        try:
            self.sample()
            self.publish('failed' if self.fatal_error else 'completed')
        finally:
            self.pool.shutdown(wait=True)
        return self.final_report


def load_targets(mode, url, overrides=None):
    targets = ({'aggre': url + '/metrics'} if mode == 'aggre' else
               {'prefill': url + '/metrics/prefill', 'decode': url + '/metrics/decode',
                'proxy': url + '/metrics'})
    targets.update({k: v for k, v in (overrides or {}).items() if v})
    return targets


def main():
    parser = argparse.ArgumentParser(description='Attach metrics sampling to an existing deployment; no restart or load.')
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New, non-existing directory')
    parser.add_argument('--interval', type=positive_seconds, default=5.)
    args = parser.parse_args()
    cfg = json.loads(args.deployment.read_text())
    targets = {s['role']: s['url'].rstrip('/') + '/metrics' for s in cfg['servers']}
    if cfg['deployment_mode'] != 'aggre':
        targets['proxy'] = cfg['url'].rstrip('/') + '/metrics'
    collector = MetricsCollector(args.output_dir, targets, interval=args.interval,
                                 gpu_roles={s['role']: s['gpus'] for s in cfg['servers']}, scope='attached')
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())
    collector.start()
    print(f'Metrics: {args.output_dir}', flush=True)
    try:
        while not stopped.wait(1):
            if collector.fatal_error:
                raise RuntimeError(collector.fatal_error)
    finally:
        collector.stop()


if __name__ == '__main__':
    main()
