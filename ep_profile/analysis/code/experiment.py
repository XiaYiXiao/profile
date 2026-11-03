"""Shared round validation and evaluation-only summaries (no GPU dependencies)."""

import csv
import json
import math
from pathlib import Path
import statistics
import sys


def group_names(config):
    prefix = 'eplb' if config.get('COMPARISON', 'ep') == 'eplb' else 'ep'
    return [f'{prefix}{i}' for i in range(2)]


def result_stem(phase, number):
    if phase not in ('warmup', 'eval') or number < 1:
        raise ValueError(f'Invalid round: {phase} {number}')
    return f'{"warmup" if phase == "warmup" else "repeat"}{number}'


def load_bench(folder):
    """Aggregate evaluation durations, not trace spans or warmup requests."""
    config = json.loads((folder.parent / 'config.json').read_text())
    repeats = int(config.get('REPEATS', config.get('repeats', 1)))
    runs = [json.loads((folder / f'repeat{i}.json').read_text())
            for i in range(1, repeats + 1)]
    bench = dict(runs[0])
    for key in ('num_prompts', 'completed', 'failed', 'duration',
                'total_input_tokens', 'total_output_tokens'):
        bench[key] = sum(r[key] for r in runs)
    bench['mean_ttft_ms'] = sum(r['mean_ttft_ms'] * r['completed'] for r in runs) / bench['completed']
    ttfts = sorted(t * 1000 for r in runs for t in r.get('ttfts', []))
    if len(ttfts) == bench['completed']:
        pos = .99 * (len(ttfts) - 1)
        low = math.floor(pos)
        bench['p99_ttft_ms'] = ttfts[low] + (ttfts[math.ceil(pos)] - ttfts[low]) * (pos - low)
    elif repeats > 1:
        bench['p99_ttft_ms'] = None  # Percentiles cannot be averaged.
    tps = [r['total_input_tokens'] / r['duration'] for r in runs]
    bench.update(evaluation_rounds=repeats, input_tokens_per_s=bench['total_input_tokens'] / bench['duration'],
                 per_round_input_tokens_per_s=tps, mean_input_tokens_per_s=statistics.mean(tps),
                 stdev_input_tokens_per_s=statistics.stdev(tps) if repeats > 1 else 0.0)
    return bench


def validate_round(root, group, phase, number, returncode=0):
    config = json.loads((root / 'config.json').read_text())
    path = root / group / f'{result_stem(phase, number)}.json'
    data, errors = {}, []
    try:
        data = json.loads(path.read_text())
        n, il, ol = (int(config[k]) for k in ('NUM_PROMPTS', 'INPUT_LEN', 'OUTPUT_LEN'))
        checks = {
            'exit_code': returncode == 0,
            'counts': data.get('num_prompts') == n and data.get('completed') == n and data.get('failed') == 0,
            'tokens': data.get('total_input_tokens') == n * il and data.get('total_output_tokens') == n * ol,
            'lengths': data.get('input_lens') == [il] * n and data.get('output_lens') == [ol] * n,
            'errors': data.get('errors') == [''] * n,
            'concurrency': data.get('max_concurrency') == int(config['CONCURRENCY']),
            'duration': isinstance(data.get('duration'), (int, float)) and math.isfinite(data['duration']) and data['duration'] > 0,
        }
        errors = [k for k, passed in checks.items() if not passed]
    except (OSError, ValueError, TypeError) as error:
        errors.append(str(error))
    status = 'invalid' if errors else 'ok'
    tps = '' if errors else data['total_input_tokens'] / data['duration']
    summary = root / ('warmup-summary.csv' if phase == 'warmup' else 'summary.csv')
    exists = summary.exists()
    with summary.open('a', newline='') as stream:
        writer = csv.writer(stream)
        if not exists:
            writer.writerow(['group', 'phase', 'round', 'exit_code', 'status', 'completed', 'failed',
                             'duration_s', 'input_tokens_per_s', 'mean_ttft_ms', 'p99_ttft_ms',
                             'result_file', 'validation_error'])
        writer.writerow([group, phase, number, returncode, status, data.get('completed', ''),
                         data.get('failed', ''), data.get('duration', ''), tps,
                         data.get('mean_ttft_ms', ''), data.get('p99_ttft_ms', ''),
                         str(path.relative_to(root)), '; '.join(errors)])
    if errors:
        raise ValueError(f'{group} {phase} {number} invalid: {errors}')
    print(f'{group} {phase} {number}: input_tokens/s={tps:.3f}', flush=True)


def summarize(root):
    config = json.loads((root / 'config.json').read_text())
    keys = ('evaluation_rounds', 'num_prompts', 'completed', 'failed', 'duration', 'total_input_tokens',
            'input_tokens_per_s', 'mean_input_tokens_per_s', 'stdev_input_tokens_per_s',
            'per_round_input_tokens_per_s', 'mean_ttft_ms', 'p99_ttft_ms')
    result = {group: {k: v for k, v in load_bench(root / group).items() if k in keys}
              for group in group_names(config)}
    (root / 'evaluation-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    action, directory = sys.argv[1:3]
    if action == 'validate':
        group, phase, number, rc = sys.argv[3:]
        validate_round(Path(directory), group, phase, int(number), int(rc))
    elif action == 'summarize':
        print(json.dumps(summarize(Path(directory)), indent=2))
    else:
        raise SystemExit(f'Unknown action: {action}')
