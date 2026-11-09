"""Read completed aggregate/PD captures; write comparisons without rerunning GPUs."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys

BASE = Path(__file__).resolve().parents[1]
SAMPLE = re.compile(r'^([^\s{]+)(?:\{([^}]*)\})?\s+([^\s]+)')
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')
TIMINGS = ('request_queue_time_seconds', 'request_prefill_time_seconds',
           'request_decode_time_seconds', 'e2e_request_latency_seconds',
           'time_to_first_token_seconds')


def read_metrics(path):
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith('#'):
            continue
        match = SAMPLE.match(line)
        if match:
            values[match[1], tuple(sorted(LABEL.findall(match[2] or '')))] = float(match[3])
    return values


def difference(before, after, name, **labels):
    return sum(value - before.get(key, 0) for key, value in after.items()
               if key[0] == 'vllm:' + name
               and all(dict(key[1]).get(k) == v for k, v in labels.items()))


def histogram_mean(before, after, name):
    count = difference(before, after, name + '_count')
    return difference(before, after, name + '_sum') / count if count else None


def latest_run(mode):
    runs = [p.parent for p in (BASE / mode).glob('*/status.json')
            if json.loads(p.read_text())['status'] == 'complete']
    if not runs:
        raise ValueError(f'No completed {mode} capture')
    return max(runs, key=lambda p: (p / 'status.json').stat().st_mtime)


def write_csv(path, rows):
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def p_log_samples(run, result):
    # Benchmark date is saved at completion; timestamp precision is one second.
    end = dt.datetime.strptime(result['date'], '%Y%m%d-%H%M%S')
    start = end - dt.timedelta(seconds=result['duration'])
    pattern = re.compile(r'INFO (\d\d-\d\d \d\d:\d\d:\d\d).*Running: (\d+) reqs, '
                         r'Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%')
    samples = []
    for number, line in enumerate((run / 'prefill/server.log').read_text().split('\n'), 1):
        if match := pattern.search(line):
            timestamp = dt.datetime.strptime(f'{end.year}-{match[1]}', '%Y-%m-%d %H:%M:%S')
            if start <= timestamp <= end:
                samples.append(dict(line=number, timestamp=timestamp.isoformat(),
                                    running=int(match[2]), waiting=int(match[3]),
                                    kv_usage_percent=float(match[4])))
    return samples


def analyze(runs, output):
    output.mkdir(parents=True, exist_ok=False)
    client_rows, stage_rows, raw = [], [], {}
    provenance, run_configs = {}, {}

    def record(path):
        provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    for mode, run in runs.items():
        assert json.loads((run / 'status.json').read_text())['status'] == 'complete', run
        cfg = json.loads((run / 'config.json').read_text())
        run_configs[mode] = cfg
        for filename in ('config.json', 'status.json', 'runtime.json'):
            record(run / filename)
        for server in cfg['servers']:
            record(run / server['role'] / 'server.log')
            record(run / server['role'] / 'server.command.txt')
        for concurrency in cfg['concurrency']:
            folder = run / 'benchmarks' / f'concurrency{concurrency}'
            for phase, rounds in [('warmup', cfg['warmup_rounds']), ('eval', cfg['repeats'])]:
                for number in range(1, rounds + 1):
                    stem = f'{phase}-round{number:03d}'
                    path = folder / (stem + '.json')
                    data = json.loads(path.read_text())
                    record(path)
                    n = cfg['num_prompts']
                    assert data['completed'] == n and data['failed'] == 0 and not any(data['errors'])
                    assert data['input_lens'] == [cfg['input_len']] * n
                    assert data['output_lens'] == [cfg['output_len']] * n
                    assert data['total_input_tokens'] == n * cfg['input_len']
                    assert data['total_output_tokens'] == n * cfg['output_len']
                    assert data['duration'] > 0
                    events = [(s, 1) for s in data['start_times']]
                    events += [(s + duration, -1) for s, duration in zip(data['start_times'], data['latencies'])]
                    current = peak = 0
                    for _, change in sorted(events):
                        current += change
                        peak = max(peak, current)
                    assert current == 0 and peak <= concurrency
                    first_tokens = sorted(s + t for s, t in zip(data['start_times'], data['ttfts']))
                    gaps = [b - a for a, b in zip(first_tokens, first_tokens[1:])]
                    row = dict(mode=mode, phase=phase, concurrency=concurrency, round=number,
                               completed=n, failed=data['failed'], duration_s=data['duration'],
                               request_s=n / data['duration'],
                               input_tokens_s=data['total_input_tokens'] / data['duration'],
                               output_tokens_s=data['total_output_tokens'] / data['duration'],
                               mean_ttft_s=data['mean_ttft_ms'] / 1000,
                               p99_ttft_s=data['p99_ttft_ms'] / 1000,
                               mean_tpot_ms=data['mean_tpot_ms'], p99_itl_ms=data['p99_itl_ms'],
                               mean_e2e_s=data['mean_e2el_ms'] / 1000,
                               p99_e2e_s=data['p99_e2el_ms'] / 1000,
                               measured_peak_inflight=peak,
                               mean_inflight=sum(data['latencies']) / data['duration'],
                               mean_after_first_token_inflight=sum(l - t for l, t in zip(data['latencies'], data['ttfts'])) / data['duration'],
                               mean_first_token_gap_s=statistics.mean(gaps),
                               median_first_token_gap_s=statistics.median(gaps))
                    client_rows.append(row)
                    raw[mode, phase, concurrency, number] = data
                    for server in cfg['servers']:
                        role = server['role']
                        before_path = folder / f'{stem}.{role}.before.metrics'
                        after_path = folder / f'{stem}.{role}.after.metrics'
                        before, after = read_metrics(before_path), read_metrics(after_path)
                        record(before_path)
                        record(after_path)
                        stage = dict(mode=mode, phase=phase, concurrency=concurrency,
                                     round=number, role=role)
                        for name in TIMINGS:
                            count = difference(before, after, name + '_count')
                            assert count == n, (path, role, name, count)
                            stage[name] = histogram_mean(before, after, name)
                        for name in ('num_preemptions_total', 'nixl_num_failed_transfers_total',
                                     'nixl_num_kv_expired_reqs_total', 'external_prefix_cache_hits_total'):
                            stage[name] = difference(before, after, name)
                        for name in ('nixl_xfer_time_seconds', 'nixl_post_time_seconds', 'nixl_bytes_transferred'):
                            stage[name + '_per_operation'] = histogram_mean(before, after, name)
                        stage['nixl_operations'] = difference(before, after, 'nixl_xfer_time_seconds_count')
                        stage['nixl_bytes_total'] = difference(before, after, 'nixl_bytes_transferred_sum')
                        for source in ('local_compute', 'external_kv_transfer'):
                            stage['prompt_tokens_' + source] = difference(before, after, 'prompt_tokens_by_source_total', source=source)
                        for engine in ('0', '1'):
                            stage['success_engine_' + engine] = difference(before, after, 'request_success_total', engine=engine)
                        stage_rows.append(stage)

    for key in ('num_prompts', 'input_len', 'output_len', 'seed', 'max_model_len', 'max_num_seqs', 'ep', 'nsys', 'eplb'):
        assert run_configs['aggre'][key] == run_configs['pd'][key], key
    write_csv(output / 'client_metrics.csv', client_rows)
    write_csv(output / 'stage_metrics.csv', stage_rows)
    measured = [r for r in client_rows if r['phase'] == 'eval']
    stage_eval = [r for r in stage_rows if r['phase'] == 'eval']
    comparisons = []
    common_concurrency = sorted(set(run_configs['aggre']['concurrency']) & set(run_configs['pd']['concurrency']))
    for concurrency in common_concurrency:
        a = next(r for r in measured if r['mode'] == 'aggre' and r['concurrency'] == concurrency)
        d = next(r for r in measured if r['mode'] == 'pd' and r['concurrency'] == concurrency)
        p = next(r for r in stage_eval if r['mode'] == 'pd' and r['role'] == 'prefill' and r['concurrency'] == concurrency)
        comparisons.append(dict(concurrency=concurrency,
                                pd_throughput_change_percent=100 * (d['output_tokens_s'] / a['output_tokens_s'] - 1),
                                pd_ttft_ratio=d['mean_ttft_s'] / a['mean_ttft_s'],
                                pd_tpot_change_percent=100 * (d['mean_tpot_ms'] / a['mean_tpot_ms'] - 1),
                                p_queue_share_of_pd_ttft=p['request_queue_time_seconds'] / d['mean_ttft_s']))
    write_csv(output / 'comparisons.csv', comparisons)
    samples = {str(c): p_log_samples(runs['pd'], raw['pd', 'eval', c, 1])
               for c in run_configs['pd']['concurrency']}
    evidence = dict(runs={k: str(v) for k, v in runs.items()}, comparisons=comparisons,
                    source_sha256=provenance, formal_completed=sum(r['completed'] for r in measured),
                    all_completed_including_warmup=sum(r['completed'] for r in client_rows),
                    common_concurrency=common_concurrency,
                    unmatched_concurrency={mode: sorted(set(cfg['concurrency']) - set(common_concurrency))
                                           for mode, cfg in run_configs.items()},
                    token_budgets={mode: cfg['max_num_batched_tokens'] for mode, cfg in run_configs.items()},
                    pd_prefill_log_samples=samples,
                    caveats=[
                        'One formal round per concurrency; warmup is not an independent randomized replicate.',
                        'P/D topology and GPUs available for prefill differ from the aggregate deployment.',
                        'NIXL duration means are per transfer operation across ranks, not request wall-clock latency.',
                        'Native max_concurrent_requests uses one-second bins; exact interval overlap is used here.',
                        'Request stage metrics are wall-clock intervals, not GPU kernel execution time.',
                        'No Nsight trace or continuous GPU utilization sample is available.'
                    ])
    if len(set(evidence['token_budgets'].values())) > 1:
        evidence['caveats'].append('Token budgets also differ between these runs.')
    (output / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    plot(output, measured, stage_eval, raw, run_configs)
    return evidence


def plot(output, clients, stages, raw, configs):
    os.environ['MPLCONFIGDIR'] = str(output / '.matplotlib')
    # Optional plotting wheels stay with analysis artifacts, outside serving dependencies.
    plotting_deps = BASE / 'analysis/.plot-deps'
    if plotting_deps.is_dir():
        sys.path.insert(0, str(plotting_deps))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'aggre': '#2563eb', 'pd': '#d97706'}
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for mode in ('aggre', 'pd'):
        rows = sorted((r for r in clients if r['mode'] == mode and r['round'] == 1), key=lambda r: r['concurrency'])
        x = [r['concurrency'] for r in rows]
        for ax, field, title, unit in (
                (axs[0, 0], 'output_tokens_s', 'End-to-end output throughput', 'tokens / second'),
                (axs[0, 1], 'mean_ttft_s', 'Mean client time to first token', 'seconds'),
                (axs[1, 0], 'mean_tpot_ms', 'Mean time per output token', 'milliseconds')):
            ax.plot(x, [r[field] for r in rows], 'o-', color=colors[mode], label=mode)
            ax.set(title=title, ylabel=unit, xlabel='Client concurrency', xticks=x)
            ax.grid(alpha=0.2)
            ax.legend()
    x = sorted(configs['pd']['concurrency'])
    p = [next(r for r in stages if r['role'] == 'prefill' and r['concurrency'] == c and r['round'] == 1) for c in x]
    d = [next(r for r in clients if r['mode'] == 'pd' and r['concurrency'] == c and r['round'] == 1) for c in x]
    queue = [r['request_queue_time_seconds'] for r in p]
    prefill = [r['request_prefill_time_seconds'] for r in p]
    rest = [r['mean_ttft_s'] - q - f for r, q, f in zip(d, queue, prefill)]
    ax = axs[1, 1]
    ax.bar(range(len(x)), queue, color='#b45309', label='P queue')
    ax.bar(range(len(x)), prefill, bottom=queue, color='#fbbf24', label='P prefill interval')
    ax.bar(range(len(x)), rest, bottom=[q + f for q, f in zip(queue, prefill)], color='#94a3b8', label='Other TTFT overhead')
    ax.set(title='PD mean TTFT decomposition', ylabel='seconds', xlabel='Client concurrency', xticks=range(len(x)), xticklabels=x)
    ax.legend()
    # Read budgets from saved runs so historical and corrected captures retain their actual settings.
    cfg = configs['pd']
    topology = []
    for mode, config in configs.items():
        servers = ', '.join(f"{s['role']} DP{s['dp']} x TP{s['tp']}" for s in config['servers'])
        topology.append(f"{mode}: {servers}, budget {config['max_num_batched_tokens']}")
    fig.suptitle(f"{cfg['num_prompts']} requests per point | input {cfg['input_len']:,} / output {cfg['output_len']} | "
                 f"EP {'on' if cfg['ep'] else 'off'}\n" + ' | '.join(topology), fontsize=11)
    fig.savefig(output / 'overview.png', dpi=180)
    plt.close(fig)

    common = set(configs['aggre']['concurrency']) & set(configs['pd']['concurrency'])
    timeline_concurrency = {max(configs['pd']['concurrency'])}
    if common:
        timeline_concurrency.add(max(common))
    for concurrency in sorted(timeline_concurrency):
        modes = [mode for mode in ('aggre', 'pd') if concurrency in configs[mode]['concurrency']]
        fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 8),
                                 sharex=True, sharey=True, squeeze=False, layout='constrained')
        axs = axes[0]
        for ax, mode in zip(axs, modes):
            data = raw[mode, 'eval', concurrency, 1]
            t0 = min(data['start_times'])
            ordered = sorted(zip(data['start_times'], data['ttfts'], data['latencies']))
            for i, (start, ttft, latency) in enumerate(ordered):
                ax.barh(i, ttft, left=start - t0, height=0.75, color='#fbbf24')
                ax.barh(i, latency - ttft, left=start - t0 + ttft, height=0.75, color='#2563eb')
            ax.set(title=mode, xlabel='Seconds from first request', xlim=(0, None))
            ax.grid(axis='x', alpha=0.2)
        axs[0].set(ylabel='Request, ordered by send time', ylim=(cfg['num_prompts'], -1))
        fig.suptitle(f'Concurrency {concurrency}: client-observed request timeline\n'
                     'Yellow includes queue, prefill and first-token overhead; blue starts at first token', fontsize=11)
        axs[-1].legend(handles=[Patch(color='#fbbf24', label='Before first token'),
                               Patch(color='#2563eb', label='After first token')], loc='upper right')
        fig.savefig(output / f'request_timeline_c{concurrency}.png', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggre-run', type=Path)
    parser.add_argument('--pd-run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    runs = {'aggre': args.aggre_run or latest_run('aggre'), 'pd': args.pd_run or latest_run('pd')}
    output = args.output or BASE / 'analysis' / ('comparison_' + dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
    evidence = analyze(runs, output)
    print(json.dumps({'output': str(output), 'comparisons': evidence['comparisons'],
                      'formal_completed': evidence['formal_completed']}, indent=2))
