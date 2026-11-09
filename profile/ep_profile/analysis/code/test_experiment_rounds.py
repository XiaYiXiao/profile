"""CPU-only regression checks: python -m unittest discover -s analysis/code -p test_experiment_rounds.py."""

import contextlib
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
import analyze
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
                       NSYS_BIN='nsys', VLLM_SOURCE='/host/vllm',
                       RESULT_DIR=str(destination), WARMUP_ROUNDS='2', REPEATS='3')
            env.pop('ENABLE_EPLB', None)
            env['EP_SOLO'] = '0'
            env.update(overrides)
            output = subprocess.check_output(['bash', str(BASE / script)], env=env, text=True)
            self.assertFalse(destination.exists())
            commands = [shlex.split(line) for line in output.splitlines()
                        if line.startswith(('env ', 'nsys '))]
            servers = [c for c in commands if 'vllm' in c
                       and c[c.index('vllm') + 1] == 'serve']
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
                self.assertEqual('--enable-eplb' in servers[1], script == 'eplb_cmp.sh')

    def test_capture_explicitly_enables_eplb(self):
        _, servers, _ = self.preview('capture.sh', ENABLE_EPLB='1')
        self.assertNotIn('--enable-eplb', servers[0])
        self.assertIn('--enable-eplb', servers[1])

    def test_solo_preview_only_uses_ep1_resources(self):
        output, servers, benches = self.preview(
            'capture.sh', EP_SOLO='1', GPU_EP0='unused', PORT_EP0='unused')
        self.assertEqual(len(servers), 1)
        self.assertEqual(len(benches), 5)
        self.assertIn('--enable-expert-parallel', servers[0])
        self.assertNotIn('--enable-eplb', servers[0])
        self.assertIn('CUDA_VISIBLE_DEVICES=4,5,6,7', servers[0])
        self.assertEqual(servers[0][servers[0].index('--port') + 1], '8001')
        self.assertIn('smoke_ep1', output)
        self.assertNotIn('smoke_ep0', output)
        self.assertTrue(all(c[c.index('--result-dir') + 1].endswith('/ep1') for c in benches))

    def test_solo_smoke_only_launches_one_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, EP_SOLO='1', COMPARISON='ep'), \
                    patch.object(collect, 'ROOT', root), \
                    patch.object(collect, 'smoke_command', return_value=['smoke']) as command, \
                    patch.object(collect, 'launch'), patch.object(collect, 'wait_jobs'), \
                    patch.object(collect, 'terminate_all'), patch.object(collect, 'export_check') as export, \
                    contextlib.redirect_stdout(io.StringIO()):
                collect.run_smokes()
            command.assert_called_once_with(1)
            export.assert_called_once_with(root / 'smoke_ep1')
            self.assertFalse((root / 'smoke_ep0').exists())

    def test_solo_allows_busy_unused_gpus_and_rejects_busy_selected_gpu(self):
        inventory = '\n'.join(f'{i}, GPU-{i}' for i in range(8))
        with patch.dict(os.environ, GPU_EP1='4,5,6,7'):
            with patch.object(collect.subprocess, 'check_output', side_effect=[inventory, 'GPU-0, 123']):
                collect.check_gpu_availability([1])
            with patch.object(collect.subprocess, 'check_output', side_effect=[inventory, 'GPU-4, 123']):
                with self.assertRaisesRegex(RuntimeError, 'Selected GPU processes'):
                    collect.check_gpu_availability([1])

    def test_invalid_solo_settings_rejected(self):
        for overrides in [dict(EP_SOLO='2')]:
            with self.assertRaises(ValueError):
                experiment.group_indices(overrides)
            env = dict(os.environ, PYTHON_BIN=sys.executable, DRY_RUN='1', **overrides)
            p = subprocess.run(['bash', str(BASE / 'capture.sh')], env=env, capture_output=True)
            self.assertNotEqual(p.returncode, 0)

    def test_solo_eplb_preview_only_runs_enabled_group(self):
        output, servers, benches = self.preview(
            'capture.sh', EP_SOLO='1', COMPARISON='eplb', ENABLE_EPLB='1',
            GPU_EP0='unused', PORT_EP0='unused')
        self.assertEqual(len(servers), 1)
        self.assertEqual(len(benches), 5)
        self.assertIn('--enable-expert-parallel', servers[0])
        self.assertIn('--enable-eplb', servers[0])
        self.assertIn('CUDA_VISIBLE_DEVICES=4,5,6,7', servers[0])
        self.assertTrue(all(c[c.index('--result-dir') + 1].endswith('/eplb1') for c in benches))
        self.assertNotIn('smoke_ep0', output)
        self.assertEqual(experiment.group_names({'EP_SOLO': '1', 'COMPARISON': 'eplb'}), ['eplb1'])

    def test_installed_cli_server_detection(self):
        self.assertTrue(collect.is_vllm_server(
            [b'/usr/bin/python3', b'/usr/local/bin/vllm', b'serve', b'/model']))
        self.assertFalse(collect.is_vllm_server(
            [b'/usr/bin/python3', b'/usr/local/bin/vllm', b'bench', b'serve']))

    def test_source_runtime_metadata_without_git(self):
        from types import SimpleNamespace
        module = SimpleNamespace(__version__='0.32.1', __file__='/host/vllm/vllm/__init__.py')
        with patch.dict(os.environ, VLLM_SOURCE='/host/vllm'), \
                patch.dict(sys.modules, vllm=module), \
                patch.object(collect.subprocess, 'check_output', side_effect=OSError):
            info = collect.vllm_runtime_info()
        self.assertIsNone(info['vllm_git_head'])
        self.assertEqual(info['vllm_version'], '0.32.1')
        self.assertEqual(info['vllm_module_path'], module.__file__)

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

    def test_overlapping_devices_and_ports_rejected(self):
        for values, message in [({'GPU_EP1': '0,5,6,7'}, 'GPU 0 重复'),
                                ({'PORT_EP1': '8000'}, '端口 8000 重复')]:
            env = dict(os.environ, PYTHON_BIN=sys.executable, DRY_RUN='1', **values)
            p = subprocess.run(['bash', str(BASE / 'reproduce.sh')], env=env,
                               capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn(message, p.stderr)

    def run_collector(self, root, failure=False, mode='ep', solo=False):
        manifest = config(mode)
        manifest['EP_SOLO'] = '1' if solo else '0'
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
                self.assertEqual(export.call_count, len(plans))
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
                self.assertFalse((root / 'summary.csv').exists())
                self.assertFalse((root / 'warmup-summary.csv').exists())
                self.assertFalse((root / 'evaluation-summary.json').exists())
                for group in experiment.group_names(manifest):
                    self.assertEqual(len(list((root / group).glob('repeat*.json'))), 3)
                    self.assertEqual(len(list((root / group).glob('warmup*.json'))), 2)
                # Analysis remains available as a separate, explicit operation.
                summary = experiment.summarize(root)
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

    def test_solo_collector_completes_all_rounds_and_only_summarizes_ep1(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events, manifest = self.run_collector(root, solo=True)
            self.assertEqual(events, ['server:ep1', 'warmup1.json', 'warmup2.json',
                                     'start_profile', 'repeat1.json', 'repeat2.json',
                                     'repeat3.json', 'stop_profile'])
            self.assertFalse((root / 'ep0').exists())
            self.assertFalse((root / 'evaluation-summary.json').exists())
            self.assertFalse((root / 'summary.csv').exists())
            summary = experiment.summarize(root)
            self.assertEqual(list(summary), ['ep1'])
            self.assertEqual(summary['ep1']['completed'], 6)
            self.assertTrue(all(len(r['clients']) == 1 for r in manifest['rounds']))

    def test_solo_report_has_no_comparison_ratio(self):
        settings = dict(config(), EP_SOLO='1', SERVED_MODEL_NAME='MiMo', nsys_version='test',
                        GPU_EP1='4,5,6,7', MAX_NUM_BATCHED_TOKENS=32768, MAX_NUM_SEQS=256)
        bench = dict(result(), input_tokens_per_s=2, per_round_input_tokens_per_s=[2, 2, 2])
        data = dict(ep='ep1', bench=bench, category_s={'MoE': 1}, ranks=[], nccl=[], kernels=[],
                    warnings=[], jit_warnings=[], nccl_plugin_workers=4, timeline=[], span_s=1)
        report = analyze.render(Path('/capture'), settings, [data])
        self.assertIn('EP=1 单组实验', report)
        self.assertIn('EP1 使用 GPU 4,5,6,7', report)
        self.assertNotIn('相对 EP0', report)
        self.assertNotIn('本轮吞吐变化为', report)
        settings.update(EP_SOLO='0', GPU_EP0='0,1,2,3')
        comparison = analyze.render(Path('/capture'), settings, [dict(data, ep='ep0'), data])
        self.assertIn('EP1 相对 EP0 的输入吞吐变化', comparison)

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

            # The same four-rank trace must be analyzable with only an ep1 folder.
            settings = dict(config('ep', 1), EP_SOLO='1')
            (root / 'config.json').write_text(json.dumps(settings))
            folder.rename(root / 'ep1')
            with patch.object(sys, 'argv', ['explain_ep_gap.py', str(root), '--output-dir', str(root)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                explain_ep_gap.main()
            evidence = json.loads((root / 'evidence.json').read_text())
            self.assertEqual([s['configuration'] for s in evidence['data']], ['ep1'])
            self.assertTrue((root / 'reduce_scatter.csv').is_file())
            self.assertIn('EP1:', (root / 'expert_skew.svg').read_text())


if __name__ == '__main__':
    unittest.main()
