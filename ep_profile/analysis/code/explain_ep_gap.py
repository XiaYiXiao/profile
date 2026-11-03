"""Read-only analysis of CUPTI ReduceScatter arrival skew across matching ranks.

Usage: python explain_ep_gap.py CAPTURE_DIR [--output-dir OUTPUT_DIR]
Outputs JSON, per-collective CSV, and an SVG example under analysis/ep_gap-RUN.
With --output-dir, all artifacts are written to that directory instead.
No NCCL plugin channel timestamps are used for duration measurements.
"""

import argparse
import bisect
import collections
import csv
import html
import json
from pathlib import Path
import sqlite3
import statistics

from analyze import category, merge, seconds
from experiment import group_names, load_bench


def distribution(values):
    ordered = sorted(values)
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                p90=ordered[int(0.9 * (len(ordered) - 1))], max=max(values))


def analyze(root, ep):
    config = json.loads((root / 'config.json').read_text())
    group_name = group_names(config)[ep]
    folder = root / group_name
    ep_enabled = config.get('group_settings', {}).get(group_name, {}).get(
        'ep_enabled', config.get('COMPARISON') == 'eplb' or ep == 1)
    with sqlite3.connect(f"file:{folder / 'trace.sqlite'}?mode=ro", uri=True) as con:
        rows = con.execute("""
            SELECT k.start, k.end, k.deviceId, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start
        """).fetchall()
    assert all(b >= a for a, b, _, _ in rows)
    lo, hi = min(r[0] for r in rows), max(r[1] for r in rows)
    by_device = {d: [r for r in rows if r[2] == d] for d in range(4)}
    rs = {d: [r for r in rr if "ncclDevKernel_ReduceScatter" in r[3]]
          for d, rr in by_device.items()}
    gemms = {d: [r for r in rr if "void humming<" in r[3]]
             for d, rr in by_device.items()}
    if not all(rs.values()) or not all(gemms.values()):
        raise ValueError(
            f'{folder}: expert_skew requires Humming GEMM and NCCL ReduceScatter '
            'kernels on all four trace devices; this backend/trace is unsupported.'
        )
    gemm_ends = {d: [r[1] for r in rr] for d, rr in gemms.items()}
    assert len({len(v) for v in gemms.values()}) == 1
    assert len(gemms[0]) % 2 == 0
    gemm_pairs = [
        [sum(gemms[d][j][1] - gemms[d][j][0] for j in [i, i+1])/1e9
         for d in range(4)]
        for i in range(0, len(gemms[0]), 2)
    ]
    gemm_pair_stats = dict(
        pairs_per_rank=len(gemm_pairs),
        sum_rank_mean_s=sum(statistics.mean(v) for v in gemm_pairs),
        sum_rank_max_s=sum(max(v) for v in gemm_pairs),
        max_over_mean=distribution([max(v)/statistics.mean(v) for v in gemm_pairs]),
        slowest_rank_counts=dict(collections.Counter(
            max(range(4), key=lambda d: v[d]) for v in gemm_pairs)),
        interpretation='Sum of per-MoE-invocation max(GEMM1+GEMM2) is a compute '
                       'critical-path proxy, not an exact wall-time decomposition.',
    )
    groups = [[0, 1, 2, 3]] if ep_enabled else [[0, 2], [1, 3]]
    records, group_stats, examples = [], [], []
    for group in groups:
        assert len({len(rs[d]) for d in group}) == 1
        prior_end = -1
        starts, ends, before, after = [], [], [], []
        used_gemms = set()
        for ordinal, collective in enumerate(zip(*(rs[d] for d in group))):
            first = min(r[0] for r in collective)
            last = max(r[0] for r in collective)
            end = max(r[1] for r in collective)
            # Equal counts/order and non-overlapping collective envelopes guard
            # against an ordinal shift. EP0 uses two independent DP groups.
            assert first >= prior_end
            assert min(r[1] for r in collective) >= last
            prior_end = end
            starts.append((last - first) / 1e6)
            ends.append((end - min(r[1] for r in collective)) / 1e6)
            before.append(sum(last - r[0] for r in collective) / 1e9)
            after.append(sum(r[1] - last for r in collective) / 1e9)
            example = dict(ordinal=ordinal, group=group, label=group_name.upper(),
                           first_rs_relative_s=(first - lo) / 1e9,
                           start_skew_ms=starts[-1], ranks=[])
            for r in collective:
                a, b, d, _ = r
                gi = bisect.bisect_right(gemm_ends[d], a)
                preceding = gemms[d][gi-2:gi]
                assert len(preceding) == 2 and gi % 2 == 0
                assert (d, gi) not in used_gemms
                used_gemms.add((d, gi))
                rec = dict(ep=int(ep_enabled), configuration=group_name, group="-".join(map(str, group)),
                           ordinal=ordinal, device=d, start_ns=a, end_ns=b,
                           last_rank_start_ns=last, duration_ms=(b-a)/1e6,
                           before_last_arrival_ms=(last-a)/1e6,
                           after_last_arrival_ms=(b-last)/1e6,
                           gemm1_ms=(preceding[0][1]-preceding[0][0])/1e6,
                           gemm2_ms=(preceding[1][1]-preceding[1][0])/1e6)
                records.append(rec)
                example['ranks'].append(dict(**rec, gemms=preceding))
            examples.append(example)
        group_stats.append(dict(group=group, collectives=len(starts),
                                start_skew_ms=distribution(starts),
                                end_spread_ms=distribution(ends),
                                before_last_arrival_gpu_s=sum(before),
                                after_last_arrival_gpu_s=sum(after)))
    categories = collections.Counter()
    kernels = collections.Counter()
    for a, b, d, name in rows:
        categories[category(name)] += (b-a)/1e9
        kernels[name] += (b-a)/1e9
    rank_stats = []
    for d, rr in by_device.items():
        busy = seconds(merge([(r[0], r[1]) for r in rr]))
        rank_stats.append(dict(device=d, busy_s=busy, gap_s=(hi-lo)/1e9-busy))
    bench = load_bench(folder)
    summary = dict(ep=int(ep_enabled), configuration=group_name, evaluation_rounds=bench["evaluation_rounds"], span_s=(hi-lo)/1e9, duration_s=bench['duration'],
                   input_tps=bench['total_input_tokens']/bench['duration'],
                   input_tokens=bench['total_input_tokens'],
                   completed=bench['completed'], failed=bench['failed'],
                   mean_ttft_ms=bench['mean_ttft_ms'], category_gpu_s=dict(categories),
                   kernel_gpu_s=dict(kernels), ranks=rank_stats, groups=group_stats,
                   gemm_pair_stats=gemm_pair_stats)
    summary['rs_gpu_s'] = sum(v for k, v in kernels.items()
                              if 'ncclDevKernel_ReduceScatter' in k)
    summary['rs_before_last_arrival_gpu_s'] = sum(
        g['before_last_arrival_gpu_s'] for g in group_stats)
    summary['rs_after_last_arrival_gpu_s'] = sum(
        g['after_last_arrival_gpu_s'] for g in group_stats)
    assert abs(summary['rs_gpu_s'] - summary['rs_before_last_arrival_gpu_s']
               - summary['rs_after_last_arrival_gpu_s']) < 1e-6
    return summary, records, examples


def plot_example(example, output):
    ranks = example['ranks']
    origin = min(r['gemms'][0][0] for r in ranks)
    finish = max(r['end_ns'] for r in ranks)
    scale = 900 / ((finish-origin)/1e6)
    colors = dict(gemm='#2874b5', wait='#ec9635', tail='#74969b')
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="390" '
           'viewBox="0 0 1120 390"><rect width="1120" height="390" fill="white"/>',
           '<g font-family="sans-serif" fill="#203347">',
           f'<text x="25" y="30" font-size="21">{html.escape(example.get("label", "EP1"))}: expert GEMM skew and ReduceScatter wait</text>',
           f'<text x="25" y="55" font-size="14">RS ordinal {example["ordinal"]} '
           f'(zero-based), {example["first_rs_relative_s"]:.6f} s after first GPU kernel</text>']
    for tick in range(0, int((finish-origin)/1e6)+1, 10):
        x = 170+tick*scale
        svg.append(f'<path d="M{x} 80 V300" stroke="#e1e7ed"/>'
                   f'<text x="{x}" y="320" text-anchor="middle" font-size="12">{tick}</text>')
    for i, r in enumerate(ranks):
        y = 95+i*50
        svg.append(f'<text x="25" y="{y+20}" font-size="15">trace GPU {r["device"]}</text>')
        segments = [(g[0], g[1], 'gemm') for g in r['gemms']]
        segments += [(r['start_ns'], r['last_rank_start_ns'], 'wait'),
                     (r['last_rank_start_ns'], r['end_ns'], 'tail')]
        for a, b, kind in segments:
            x, width = 170+(a-origin)/1e6*scale, (b-a)/1e6*scale
            label = html.escape(f'{kind}: {(b-a)/1e6:.3f} ms')
            svg.append(f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" '
                       f'height="28" fill="{colors[kind]}"><title>{label}</title></rect>')
    svg.append('<text x="620" y="344" font-size="13">Time from first GEMM start (ms)</text>')
    for i, (kind, label) in enumerate([
        ('gemm', 'Expert GEMM'), ('wait', 'RS before last rank enters'),
        ('tail', 'RS after last rank enters')]):
        x = 25+i*350
        svg.append(f'<rect x="{x}" y="365" width="14" height="14" fill="{colors[kind]}"/>'
                   f'<text x="{x+22}" y="377" font-size="14">{label}</text>')
    svg.append('</g></svg>')
    output.write_text(''.join(svg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_dir', type=Path)
    parser.add_argument('--output-dir', type=Path,
                        help='Output directory (default: analysis/ep_gap-RUN)')
    args = parser.parse_args()
    root = args.capture_dir.resolve()
    output = (args.output_dir.resolve() if args.output_dir else
              Path(__file__).resolve().parents[1] / f'ep_gap-{root.name}')
    summaries, records, example = [], [], None
    for ep in [0, 1]:
        summary, rr, examples = analyze(root, ep)
        summaries.append(summary)
        records.extend(rr)
        if ep == 1:
            # Keep the historical example for long runs; use the last available
            # collective for short runs, independently of which rank is slowest.
            example = examples[min(500, len(examples) - 1)]
            summary['max_skew_example'] = max(examples, key=lambda e: e['start_skew_ms'])
            summary['illustrated_example'] = example
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'reduce_scatter.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / 'evidence.json').write_text(json.dumps(dict(
        capture_dir=str(root),
        method='Pair ReduceScatter kernels by execution order within verified communicator '
               'membership. Assert equal rank counts and disjoint successive collective '
               'envelopes. Split CUPTI duration at latest rank start. Pre-arrival time '
               'includes possible communication progress and is not a pure network-wait '
               'counter; post-arrival time is not total wire transmission time. '
               'Preceding two Humming kernels are paired separately on each rank. '
               'Expert token histograms and hardware counters were not captured.',
        data=summaries), ensure_ascii=False, indent=2))
    plot_example(example, output / 'expert_skew.svg')
    for summary in summaries:
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ['kernel_gpu_s', 'max_skew_example', 'illustrated_example']},
                         ensure_ascii=False))
    print(output)


if __name__ == '__main__':
    main()
