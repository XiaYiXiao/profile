"""CPU-only regression checks: python -m unittest discover -s analysis/code -p test_experiment_rounds.py."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import collect
import experiment
import explain_ep_gap

BASE = Path(__file__).resolve().parents[2]


def result(duration=10):
    return dict(num_prompts=2, completed=2, failed=0, duration=duration,
                total_input_tokens=20, total_output_tokens=2, input_lens=[10, 10],
                output_lens=[1, 1], errors=['', ''], max_concurrency=2,
                mean_ttft_ms=1500, p99_ttft_ms=1990, ttfts=[1, 2])


def config(mode='ep', repeats=3):
    return dict(COMPARISON=mode, REPEATS=repeats, WARMUP_ROUNDS=2,
                NUM_PROMPTS=2, INPUT_LEN=10, OUTPUT_LEN=1, CONCURRENCY=2)


class RoundTests(unittest.TestCase):
    def preview(self, script, **overrides):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'must-not-be-created'
            env = dict(os.environ, PYTHON_BIN=sys.executable, DRY_RUN='1',
                       RESULT_DIR=str(destination), WARMUP_ROUNDS='2', REPEATS='3')
            env.update(overrides)
            output = subprocess.check_output(['bash', str(BASE / script)], env=env, text=True)
            self.assertFalse(destination.exists())
            commands = [shlex.split(line) for line in output.splitlines()
                        if line.startswith(('env ', 'nsys '))]
            servers = [c for c in commands if '-m' in c and c[c.index('-m') + 1:c.index('-m') + 3] == ['vllm', 'serve']]
            benches = [c for c in commands if 'bench' in c]
            return output, servers, benches

    def test_all_entrypoints_full_warmup_and_three_evaluations(self):
        for script in ('reproduce.sh', 'capture.sh', 'eplb_cmp.sh'):
            with self.subTest(script=script):
                _, servers, benches = self.preview(script)
                self.assertEqual(len(servers), 2)
                self.assertEqual(len(benches), 10)
                expected = [f'{phase}{i}.json' for phase, count in [('warmup', 2), ('repeat', 3)]
                            for i in range(1, count + 1) for _ in range(2)]
                self.assertEqual([c[c.index('--result-filename') + 1] for c in benches], expected)
                for c in benches:
                    self.assertEqual(c[c.index('--num-warmups') + 1], '0')
                    self.assertNotIn('--profile', c)
                normalized = []
                for c in benches:
                    c = c.copy()
                    c[1] = 'CUDA_VISIBLE_DEVICES=GROUP'
                    for flag in ('--base-url', '--result-dir', '--result-filename'):
                        c[c.index(flag) + 1] = 'GROUP_OR_ROUND'
                    normalized.append(c)
                self.assertTrue(all(c == normalized[0] for c in normalized))
                self.assertEqual('--enable-expert-parallel' in servers[0], script == 'eplb_cmp.sh')
                self.assertIn('--enable-expert-parallel', servers[1])
                self.assertNotIn('--enable-eplb', servers[0])
                self.assertIn('--enable-eplb', servers[1])

    def test_comparison_only_changes_eplb_and_group_resources(self):
        _, servers, _ = self.preview('eplb_cmp.sh', ENABLE_EPLB='0',
                                     MOE_BACKEND='humming', ATTENTION_BACKEND='TRITON_ATTN_DIFFKV')
        normalized = []
        for c in servers:
            c = c[c.index('env'):]
            c[1] = 'GPU_GROUP'
            for flag in ('--port', '--data-parallel-rpc-port', '--master-port'):
                c[c.index(flag) + 1] = 'GROUP_PORT'
            if '--enable-eplb' in c:
                c.remove('--enable-eplb')
                pos = c.index('--eplb-config')
                del c[pos:pos + 2]
            normalized.append(c)
        self.assertEqual(*normalized)

    def test_round_overrides_and_disable_eplb(self):
        for script in ('reproduce.sh', 'capture.sh', 'eplb_cmp.sh'):
            _, _, benches = self.preview(script, WARMUP_ROUNDS='0', REPEATS='5')
            self.assertEqual(len(benches), 10)
            self.assertTrue(all('warmup' not in c[c.index('--result-filename') + 1] for c in benches))
            _, _, benches = self.preview(script, WARMUP_ROUNDS='', warmup_round='4', REPEATS='1')
            self.assertEqual(len(benches), 10)
        _, servers, _ = self.preview('capture.sh', ENABLE_EPLB='0')
        self.assertTrue(all('--enable-eplb' not in c for c in servers))

    def test_invalid_round_counts_rejected(self):
        for script in ('reproduce.sh', 'capture.sh', 'eplb_cmp.sh'):
            for values in ({'WARMUP_ROUNDS': '-1'}, {'REPEATS': '0'}):
                env = dict(os.environ, PYTHON_BIN=sys.executable, DRY_RUN='1', **values)
                p = subprocess.run(['bash', str(BASE / script)], env=env, capture_output=True)
                self.assertNotEqual(p.returncode, 0)

    def run_collector(self, root, failure=False, mode='ep'):
        manifest = config(mode)
        (root / 'config.json').write_text(json.dumps(manifest))
        plans = [dict(group=g, folder=root / g, serve=['server', g],
                      bench=['bench', '--result-filename', 'repeat1.json', '--result-dir', str(root / g)],
                      url='http://' + g) for g in experiment.group_names(manifest)]
        events = []

        class Process:
            pid = 123456789  # All signal operations are mocked.
            returncode = 0

            def poll(self):
                return None

        def launch(command, log):
            if command[0] == 'server':
                events.append('server:' + command[1])
            else:
                name = command[command.index('--result-filename') + 1]
                events.append(name)
                data = result(1000 if name.startswith('warmup') else 10)
                if failure and name == 'warmup2.json':
                    data['failed'] = 1
                (log.parent / name).write_text(json.dumps(data))
            return Process()

        def urlopen(req, **kwargs):
            if isinstance(req, str):
                response = io.BytesIO(b'ok')
                response.status = 200
                return contextlib.nullcontext(response)
            self.assertEqual(req.get_method(), 'POST')
            events.append(req.full_url.rsplit('/', 1)[-1])
            return contextlib.nullcontext(io.BytesIO(b''))

        with patch.dict(os.environ, WARMUP_ROUNDS='2', REPEATS='3', STARTUP_TIMEOUT='1', BENCH_TIMEOUT='1'), \
                patch.object(collect, 'ROOT', root), patch.object(collect, 'launch', side_effect=launch), \
                patch.object(collect, 'server_children', return_value=[]), \
                patch.object(collect, 'wait_jobs'), patch.object(collect, 'snapshot'), \
                patch.object(collect.urllib.request, 'urlopen', side_effect=urlopen), \
                patch.object(collect.os, 'kill'), patch.object(collect, 'terminate_all') as cleanup, \
                patch.object(collect, 'export_check') as export, contextlib.redirect_stdout(io.StringIO()):
            if failure:
                with self.assertRaisesRegex(RuntimeError, 'invalid'):
                    collect.collect_parallel(plans, manifest)
                self.assertEqual(export.call_count, 0)
            else:
                collect.collect_parallel(plans, manifest)
                self.assertEqual(export.call_count, 2)
            cleanup.assert_called_once()
        return events, manifest

    def test_collector_lifecycle_and_evaluation_only_aggregation(self):
        for mode in ('ep', 'eplb'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                events, manifest = self.run_collector(root, mode=mode)
                self.assertEqual(events[:2], [f'server:{g}' for g in experiment.group_names(manifest)])
                self.assertEqual(events[2:6], ['warmup1.json'] * 2 + ['warmup2.json'] * 2)
                self.assertEqual(events[6:8], ['start_profile'] * 2)
                self.assertEqual(events[8:14], [f'repeat{i}.json' for i in range(1, 4) for _ in range(2)])
                self.assertEqual(events[14:], ['stop_profile'] * 2)
                self.assertEqual(len(manifest['rounds']), 5)
                with (root / 'summary.csv').open() as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 6)
                with (root / 'warmup-summary.csv').open() as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 4)
                summary = json.loads((root / 'evaluation-summary.json').read_text())
                for values in summary.values():
                    self.assertEqual(values['duration'], 30)
                    self.assertEqual(values['completed'], 6)
                    self.assertEqual(values['input_tokens_per_s'], 2)
                    self.assertEqual(values['p99_ttft_ms'], 2000)

    def test_failed_warmup_prevents_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events, _ = self.run_collector(root, failure=True)
            self.assertNotIn('start_profile', events)
            self.assertFalse((root / 'summary.csv').exists())
            self.assertFalse((root / 'evaluation-summary.json').exists())

    def test_eplb0_analysis_uses_four_rank_ep_collectives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'config.json').write_text(json.dumps(config('eplb', 1)))
            folder = root / 'eplb0'
            folder.mkdir()
            (folder / 'repeat1.json').write_text(json.dumps(result()))
            with sqlite3.connect(folder / 'trace.sqlite') as db:
                db.execute('CREATE TABLE StringIds (id INTEGER, value TEXT)')
                db.executemany('INSERT INTO StringIds VALUES (?,?)', [(1, 'void humming<test>'), (2, 'ncclDevKernel_ReduceScatter')])
                db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER, demangledName INTEGER)')
                for rank in range(4):
                    db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)',
                                   [(0, 10 + rank, rank, 1), (20, 30 + rank, rank, 1), (40 + rank, 60, rank, 2)])
            summary, _, _ = explain_ep_gap.analyze(root, 0)
            self.assertEqual(summary['ep'], 1)
            self.assertEqual(summary['configuration'], 'eplb0')
            self.assertEqual([g['group'] for g in summary['groups']], [[0, 1, 2, 3]])


if __name__ == '__main__':
    unittest.main()
