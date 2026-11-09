"""CPU checks for decode commands, resume identity and capture boundaries."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from speculative_config import resolve
import decode_state
spec = importlib.util.spec_from_file_location('decode_capture', BASE / 'collect.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class DecodeTests(unittest.TestCase):
    def preview(self, script='capture.sh', **overrides):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'not-created'
            env = dict(os.environ, PYTHON_BIN=sys.executable, DRY_RUN='1', RESULT_DIR=str(root),
                       SEQ_COUNTS='16 32', SPECULATIVE_METHOD='none', SPECULATIVE_CONFIG='',
                       ENABLE_EPLB='0', NUM_SPECULATIVE_TOKENS='', DRAFT_MODEL_PATH='', WARMUP_ROUNDS='2',
                       LOG_ONLY='0', SMOKE_ONLY='0', KV_LEN='8192', MAX_OUTPUT='500')
            env.update(overrides)
            text = subprocess.check_output(['bash', str(BASE / script)], env=env, text=True)
            self.assertFalse(root.exists())
            commands = [shlex.split(line) for line in text.splitlines()
                        if line.startswith(('env ', '/usr/', sys.executable))]
            return text, commands

    def test_default_groups_budget_and_workload(self):
        text, commands = self.preview(MAX_NUM_BATCHED_TOKENS='2048')
        servers = [c for c in commands if '--tensor-parallel-size' in c]
        self.assertEqual(len(servers), 2)
        for index, c in enumerate(servers):
            self.assertIn(f'CUDA_VISIBLE_DEVICES={"0,1,2,3" if index % 2 == 0 else "4,5,6,7"}', c)
            self.assertEqual(c[c.index('--max-num-batched-tokens') + 1], '8120')
            self.assertEqual(c[c.index('--max-num-seqs') + 1], '32')
            self.assertEqual(c[c.index('--tensor-parallel-size') + 1], '1')
            self.assertEqual(c[c.index('--data-parallel-size') + 1], '4')
            self.assertEqual('--enable-expert-parallel' in c, index % 2 == 1)
            self.assertNotIn('--enable-eplb', c)
            self.assertNotIn('--speculative-config', c)
        benches = [c for c in commands if 'bench' in c]
        self.assertEqual(len(benches), 12)
        for c in benches:
            self.assertEqual(c[c.index('--random-input-len') + 1], '8193')
            self.assertEqual(json.loads(c[c.index('--extra-body') + 1]),
                             {'truncate_prompt_tokens': 8193})
            self.assertEqual(c[c.index('--random-output-len') + 1], '500')
            self.assertEqual(c[c.index('--num-warmups') + 1], '0')
        formal = [c for c in benches if 'warmup' not in c[c.index('--result-filename') + 1]]
        self.assertEqual([c[c.index('--num-prompts') + 1] for c in formal], ['16','16','32','32'])
        self.assertEqual([c[c.index('--max-concurrency') + 1] for c in formal], ['16','16','32','32'])

    def test_log_only_preview_keeps_both_groups_without_nsight(self):
        text, commands = self.preview(LOG_ONLY='1', NSYS_BIN='/missing/nsys')
        servers = [c for c in commands if '--tensor-parallel-size' in c]
        self.assertEqual(len(servers), 2)
        for ep, command in enumerate(servers):
            self.assertEqual(command[0], 'env')
            self.assertIn(f'CUDA_VISIBLE_DEVICES={"0,1,2,3" if ep == 0 else "4,5,6,7"}', command)
            self.assertEqual('--enable-expert-parallel' in command, ep == 1)
            self.assertNotIn('--profiler-config', command)
            self.assertEqual(command[command.index('--max-num-seqs') + 1], '32')
            self.assertEqual(command[command.index('--max-num-batched-tokens') + 1], '8120')
        self.assertEqual(len([c for c in commands if 'bench' in c]), 12)
        for forbidden in ('/missing/nsys', '/start_profile', '/stop_profile', '--smoke'):
            self.assertNotIn(forbidden, text)

    def test_log_only_main_does_not_require_nsight(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'results'
            model = Path(temp) / 'model'
            model.mkdir()
            for filename in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
                (model / filename).write_text('{}')
            (model / 'model.safetensors.index.json').write_text('{"weight_map": {}}')
            env = dict(DRY_RUN='0', SMOKE_ONLY='0', LOG_ONLY='1', SEQ_COUNTS='16',
                       WARMUP_ROUNDS='2', METRICS_INTERVAL='1', STARTUP_TIMEOUT='1', BENCH_TIMEOUT='1',
                       GPU_EP0='0,1,2,3', GPU_EP1='4,5,6,7', PORT_EP0='8000', PORT_EP1='8001',
                       DP_RPC_PORT_EP0='29550', DP_RPC_PORT_EP1='29650',
                       MASTER_PORT_EP0='29501', MASTER_PORT_EP1='29601',
                       VLLM_SOURCE='/host/vllm', MODEL_PATH=str(model), SERVED_MODEL_NAME='MiMo',
                       KV_LEN='8192', MAX_OUTPUT='500', MAX_NUM_BATCHED_TOKENS='8120', SEED='42',
                       SPECULATIVE_METHOD='none', SPECULATIVE_CONFIG='', NSYS_BIN='/missing/nsys')
            selected = [dict(ep=ep, folder=root/f'ep{ep}', serve=['server']) for ep in (0, 1)]
            def check_output(command, **kwargs):
                self.assertEqual(command, ['nvidia-smi', 'topo', '-m'])
                return 'topology\n'
            with patch.dict(os.environ, env), patch.object(capture, 'ROOT', root), \
                    patch.object(capture, 'plans', return_value=selected), \
                    patch.object(capture, 'check_gpu_availability'), \
                    patch.object(capture.socket, 'socket'), \
                    patch.object(capture.subprocess, 'run'), \
                    patch.object(capture.subprocess, 'check_output', side_effect=check_output), \
                    patch.object(capture, 'vllm_runtime_info', return_value={}), \
                    patch.object(capture, 'run_smokes') as smokes, \
                    patch.object(capture, 'collect') as collect, \
                    contextlib.redirect_stdout(io.StringIO()):
                capture.main()
            smokes.assert_not_called()
            collect.assert_called_once_with(selected, [16])
            config = json.loads((root / 'config.json').read_text())
            self.assertEqual(config['capture_mode'], 'logs_only')
            self.assertNotIn('nsys_version', config)

    def test_invalid_log_only_settings_fail_before_launch(self):
        for overrides in (dict(LOG_ONLY='2'), dict(LOG_ONLY='true'),
                          dict(LOG_ONLY='1', SMOKE_ONLY='1')):
            env = dict(os.environ, PYTHON_BIN=sys.executable, DRY_RUN='1', SMOKE_ONLY='0')
            env.update(overrides)
            result = subprocess.run(['bash', str(BASE / 'capture.sh')], env=env,
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('LOG_ONLY', result.stderr)

    def test_sweep_single_server_budget_and_load_order(self):
        _, commands = self.preview('sweep-mimo.sh', SEQ_COUNTS='32 16', MAX_NUM_BATCHED_TOKENS='2048')
        servers = [c for c in commands if '--tensor-parallel-size' in c]
        self.assertEqual(len(servers), 1)
        self.assertEqual(servers[0][servers[0].index('--max-num-batched-tokens') + 1], '8120')
        self.assertEqual(servers[0][servers[0].index('--max-num-seqs') + 1], '32')
        benches = [c for c in commands if 'bench' in c]
        formal = [c for c in benches if 'warmup' not in c[c.index('--result-filename') + 1]]
        for flag in ('--num-prompts', '--max-concurrency'):
            self.assertEqual([c[c.index(flag) + 1] for c in formal], ['32', '16'])
            self.assertEqual([c[c.index(flag) + 1] for c in benches], ['32'] * 3 + ['16'] * 3)
        self.assertTrue(all(c[c.index('--num-warmups') + 1] == '0' for c in benches))
        self.assertTrue(all(json.loads(c[c.index('--extra-body') + 1]) ==
                            {'truncate_prompt_tokens': 8193} for c in benches))

    def test_capture_preserves_load_order(self):
        _, commands = self.preview(SEQ_COUNTS='32 16')
        formal = [c for c in commands if 'bench' in c and 'warmup' not in c[c.index('--result-filename') + 1]]
        self.assertEqual([c[c.index('--num-prompts') + 1] for c in formal], ['32','32','16','16'])

    def test_warmup_round_count_and_workload(self):
        for script, groups in [('capture.sh', 2), ('sweep-mimo.sh', 1)]:
            for rounds in (0, 3):
                with self.subTest(script=script, rounds=rounds):
                    _, commands = self.preview(script, SEQ_COUNTS='16', WARMUP_ROUNDS=str(rounds),
                                               NUM_WARMUPS='99', KV_LEN='1024')
                    benches = [c for c in commands if 'bench' in c]
                    warmups = [c for c in benches if 'warmup' in c[c.index('--result-filename') + 1]]
                    self.assertEqual(len(warmups), rounds * groups)
                    self.assertEqual(len(benches), (rounds + 1) * groups)
                    for command in benches:
                        self.assertEqual(command[command.index('--random-input-len') + 1], '1025')
                        self.assertEqual(json.loads(command[command.index('--extra-body') + 1]),
                                         {'truncate_prompt_tokens': 1025})
                        for flag in ('--num-prompts', '--max-concurrency'):
                            self.assertEqual(command[command.index(flag) + 1], '16')
                        self.assertEqual(command[command.index('--num-warmups') + 1], '0')
                    filenames = {c[c.index('--result-filename') + 1] for c in warmups}
                    self.assertEqual(len(filenames), rounds)

    def test_invalid_warmup_rounds_fail_before_launch(self):
        for script in ('capture.sh', 'sweep-mimo.sh'):
            for rounds in ('-1', '1.5', 'abc'):
                env = dict(os.environ, PYTHON_BIN=sys.executable, DRY_RUN='1', WARMUP_ROUNDS=rounds)
                result = subprocess.run(['bash', str(BASE / script)], env=env, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)

    def test_mtp_dflash_and_eplb_scope(self):
        for method, extra in [('MTP', {}), ('Dflash', {'DRAFT_MODEL_PATH': '/draft'})]:
            _, commands = self.preview(SPECULATIVE_METHOD=method, ENABLE_EPLB='1', **extra)
            servers = [c for c in commands if '--tensor-parallel-size' in c]
            for index, c in enumerate(servers):
                config = json.loads(c[c.index('--speculative-config') + 1])
                self.assertEqual(config['method'], method.lower())
                self.assertEqual('--enable-eplb' in c, index % 2 == 1)
                if method == 'Dflash':
                    self.assertEqual(config['model'], '/draft')

    def test_invalid_speculation(self):
        for env in [dict(SPECULATIVE_METHOD='dflash', DRAFT_MODEL_PATH='', SPECULATIVE_CONFIG=''),
                    dict(SPECULATIVE_METHOD='mtp', NUM_SPECULATIVE_TOKENS='0')]:
            with patch.dict(os.environ, env):
                with self.assertRaises(ValueError):
                    resolve()

    def test_invalid_budget_and_overlapping_groups_fail_before_launch(self):
        for overrides in [dict(SEQ_COUNTS='8121'), dict(GPU_EP1='0,5,6,7')]:
            env = dict(os.environ, PYTHON_BIN=sys.executable, DRY_RUN='1',
                       SEQ_COUNTS='16', MAX_NUM_BATCHED_TOKENS='2048',
                       GPU_EP0='0,1,2,3', GPU_EP1='4,5,6,7')
            env.update(overrides)
            result = subprocess.run(['bash', str(BASE / 'capture.sh')], env=env,
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_fixed_and_historical_result_names(self):
        self.assertEqual(decode_state.name(64, {'max_num_batched_tokens':'2048'}), 'batch2048-concurrency64')
        self.assertEqual(decode_state.name(64, {'batch_token_extra':'48'}), 'batch112-concurrency64')
        with patch.dict(os.environ, SPECULATIVE_METHOD='none', MAX_NUM_BATCHED_TOKENS='2048'):
            first = decode_state.desired()
        with patch.dict(os.environ, SPECULATIVE_METHOD='mtp', MAX_NUM_BATCHED_TOKENS='2048'):
            second = decode_state.desired()
        self.assertNotEqual(first, second)
        self.assertEqual(first['max_num_batched_tokens'], '8120')

    def test_result_validation_rejects_actual_input_length_drift(self):
        # Reproduce seq4096: every request succeeds, but one prompt is 8194
        # tokens. Fixing request construction must not weaken this check.
        seq = 4096
        config = dict(max_num_batched_tokens='8120', input_len='8193', max_output='500')
        data = dict(num_prompts=seq, completed=seq, failed=0, max_concurrency=seq,
                    total_input_tokens=33558528, total_output_tokens=2048000,
                    input_lens=[8193] * seq, output_lens=[500] * seq, duration=10)
        for metric in ('tpot', 'itl'):
            for prefix in ('mean', 'p50', 'p90', 'p95', 'p99'):
                data[f'{prefix}_{metric}_ms'] = 1
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for filename in ('warmup-round001.json', decode_state.name(seq, config) + '.json'):
                path = root / filename
                path.write_text(json.dumps(data))
                decode_state.valid_result(root, seq, config, filename)
                bad = dict(data, input_lens=data['input_lens'][:-1] + [8194],
                           total_input_tokens=33558529)
                path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, 'total_input_tokens: expected 33558528, got 33558529'):
                    decode_state.valid_result(root, seq, config, filename)
                # Changing just the total must not hide the oversized request.
                bad['total_input_tokens'] = data['total_input_tokens']
                path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, 'invalid input_lens'):
                    decode_state.valid_result(root, seq, config, filename)

    def test_resume_tracks_capacity_load_order_and_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, MAX_NUM_SEQS='32'):
            root = Path(temp)
            with contextlib.redirect_stdout(io.StringIO()):
                decode_state.initialize(root, [16, 32], False, False)
                decode_state.initialize(root, [16, 32], False, True)
            self.assertEqual(set(decode_state.rows(root)), {16, 32})
            self.assertEqual({row['max_num_seqs'] for row in decode_state.rows(root).values()}, {'32'})
            with self.assertRaisesRegex(SystemExit, 'load_order'):
                decode_state.initialize(root, [32, 16], False, True)
            with patch.dict(os.environ, MAX_NUM_SEQS='64'):
                with self.assertRaisesRegex(SystemExit, 'server_max_num_seqs'):
                    decode_state.initialize(root, [16, 32], False, True)
            with patch.dict(os.environ, WARMUP_ROUNDS='3'):
                with self.assertRaisesRegex(SystemExit, 'warmup_rounds'):
                    decode_state.initialize(root, [16, 32], False, True)
            path = root / 'config.txt'
            path.write_text('\n'.join(line for line in path.read_text().splitlines()
                                      if not line.startswith('server_lifecycle=')))
            with self.assertRaisesRegex(SystemExit, 'server_lifecycle'):
                decode_state.initialize(root, [16, 32], False, True)

    def run_fake_sweep(self, fail_seq='', fail_phase='bench', warmup_rounds='2'):
        """Exercise the actual shell loop and cleanup using a lightweight server."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('sweep-mimo.sh', 'settings.sh', 'decode_state.py', 'summarize-mimo.py'):
                shutil.copy(BASE / name, root / name)
            model = root / 'model'
            model.mkdir()
            for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
                (model / name).write_text('{}')
            (model / 'model.safetensors.index.json').write_text('{"weight_map": {}}')
            (root / 'vllm').mkdir()
            (root / 'vllm' / '__init__.py').write_text('')
            runtime = root / 'vllm' / '__main__.py'
            runtime.write_text('''import json, os, signal, sys, time
from pathlib import Path
def event(kind, **data):
    with open(os.environ['FAKE_EVENTS'], 'a') as out:
        out.write(json.dumps(dict(kind=kind, **data)) + '\\n')
def arg(flag): return sys.argv[sys.argv.index(flag) + 1]
if sys.argv[1] == 'server':
    event('server', capacity=os.environ['MAX_NUM_SEQS'], budget=os.environ['MAX_NUM_BATCHED_TOKENS'])
    def stop(*_):
        event('stop')
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    while True: time.sleep(0.01)
n = int(arg('--num-prompts'))
phase = 'warmup' if 'warmup' in arg('--result-filename') else 'bench'
event(phase, count=n, concurrency=int(arg('--max-concurrency')), filename=arg('--result-filename'))
assert arg('--num-warmups') == '0'
if str(n) == os.environ.get('FAKE_FAIL_SEQ') and phase == os.environ.get('FAKE_FAIL_PHASE'): sys.exit(3)
data = dict(num_prompts=n, completed=n, failed=0, max_concurrency=n, duration=10,
            total_input_tokens=n*8193, total_output_tokens=n*500,
            input_lens=[8193]*n, output_lens=[500]*n)
for metric in ('tpot', 'itl'):
    for prefix in ('mean', 'p50', 'p90', 'p95', 'p99'): data[f'{prefix}_{metric}_ms'] = 1
(Path(arg('--result-dir')) / arg('--result-filename')).write_text(json.dumps(data))
''')
            (root / 'mimo.sh').write_text('''#!/bin/bash
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
exec "$PYTHON_BIN" -m vllm server
''')
            (root / 'sample-metrics.py').write_text('import time; time.sleep(300)\n')
            binaries = root / 'bin'
            binaries.mkdir()
            stubs = {
                'setsid': f'#!{sys.executable}\nimport os, sys\nos.setsid()\nos.execvp(sys.argv[1], sys.argv[1:])\n',
                'timeout': '#!/bin/bash\nshift 3\nexec "$@"\n',
                'flock': '#!/bin/bash\nexit 0\n',
                'curl': '#!/bin/bash\n[[ -s "$FAKE_EVENTS" ]]\n',
            }
            for name, content in stubs.items():
                path = binaries / name
                path.write_text(content)
                path.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=sys.executable, VLLM_SOURCE=str(root),
                       PATH=str(binaries) + os.pathsep + os.environ['PATH'], MODEL_PATH=str(model),
                       RESULT_DIR=str(root / 'results'), SEQ_COUNTS='32 16 64', DRY_RUN='0', FORCE_RERUN='0',
                       FAKE_EVENTS=str(root / 'events.jsonl'), FAKE_FAIL_SEQ=fail_seq, FAKE_FAIL_PHASE=fail_phase,
                       WARMUP_ROUNDS=warmup_rounds,
                       STARTUP_TIMEOUT='10', BENCH_TIMEOUT='10', STOP_TIMEOUT='1')
            result = subprocess.run(['bash', str(root / 'sweep-mimo.sh')], env=env,
                                    text=True, capture_output=True, timeout=45)
            self.assertTrue((root / 'events.jsonl').exists(), result.stdout + result.stderr)
            events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
            return result, events, decode_state.rows(root / 'results')

    def test_sweep_reuses_model_until_all_loads_finish(self):
        result, events, rows = self.run_fake_sweep()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([e['kind'] for e in events], ['server'] + ['warmup', 'warmup', 'bench'] * 3 + ['stop'])
        self.assertEqual(events[0]['capacity'], '64')
        self.assertEqual(events[0]['budget'], '8120')
        self.assertEqual([e['count'] for e in events if e['kind'] == 'bench'], [32, 16, 64])
        self.assertTrue(all(e['count'] == e['concurrency'] for e in events if e['kind'] in ('bench', 'warmup')))
        self.assertEqual([e['count'] for e in events if e['kind'] == 'warmup'], [32, 32, 16, 16, 64, 64])
        self.assertEqual({r['max_num_seqs'] for r in rows.values()}, {'64'})
        self.assertTrue(all(r['status'] == 'finished' and r['exit_code'] == '0' for r in rows.values()))
        self.assertIn('seqs=16: output_tokens/s=800.000', result.stdout)

    def test_sweep_failed_load_stops_without_model_restart(self):
        result, events, rows = self.run_fake_sweep(fail_seq='16')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([e['kind'] for e in events], ['server'] + ['warmup', 'warmup', 'bench'] * 2 + ['stop'])
        self.assertEqual(rows[64]['status'], 'pending')

    def test_sweep_zero_warmup_rounds(self):
        result, events, _ = self.run_fake_sweep(warmup_rounds='0')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([e['kind'] for e in events], ['server', 'bench', 'bench', 'bench', 'stop'])

    def test_sweep_failed_warmup_stops_before_formal_load(self):
        result, events, rows = self.run_fake_sweep(fail_seq='32', fail_phase='warmup')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([e['kind'] for e in events], ['server', 'warmup', 'stop'])
        self.assertEqual(rows[32]['status'], 'warmup_failed')
        self.assertEqual(rows[16]['status'], 'pending')

    def test_capture_excludes_warmup_and_exports_after_shutdown(self):
        self.check_capture_lifecycle(log_only=False)

    def test_log_only_runs_all_loads_without_trace_or_metrics(self):
        self.check_capture_lifecycle(log_only=True)

    def check_capture_lifecycle(self, log_only):
        events = []
        launched = []
        class Process:
            pid = 123456789
            def poll(self): return None
        env = dict(WARMUP_ROUNDS='2', KV_LEN='8192', MAX_OUTPUT='500', STARTUP_TIMEOUT='1',
                   BENCH_TIMEOUT='1', METRICS_INTERVAL='1', MAX_NUM_BATCHED_TOKENS='8120',
                   VLLM_SOURCE='/host/vllm', MODEL_PATH='/model', SERVED_MODEL_NAME='MiMo',
                   PORT_EP0='8000', PORT_EP1='8001', SEED='42', LOG_ONLY='1' if log_only else '0')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = [dict(ep=i, folder=root/f'ep{i}', serve=['server'], url=f'http://ep{i}') for i in (0,1)]
            def launch(command, log):
                launched.append(command)
                log.write_text('running\n')
                if command == ['server']:
                    events.append('server')
                if 'bench' in command:
                    n = int(command[command.index('--num-prompts') + 1])
                    self.assertEqual(n, int(command[command.index('--max-concurrency') + 1]))
                    filename = command[command.index('--result-filename') + 1]
                    events.append(filename)
                    data = dict(num_prompts=n, completed=n, failed=0, max_concurrency=int(command[command.index('--max-concurrency') + 1]),
                        total_input_tokens=n*8193, total_output_tokens=n*500, input_lens=[8193]*n,
                        output_lens=[500]*n, duration=10)
                    for metric in ('tpot','itl'):
                        for prefix in ('mean','p50','p90','p95','p99'):
                            data[f'{prefix}_{metric}_ms'] = 1
                    (log.parent / filename).write_text(json.dumps(data))
                return Process()
            def healthy(*args, **kwargs):
                response=io.BytesIO(b'')
                response.status=200
                return contextlib.nullcontext(response)
            with patch.dict(os.environ, env), patch.object(capture, 'ROOT', root), \
                    patch.object(capture, 'launch', side_effect=launch), \
                    patch.object(capture, 'wait_jobs'), patch.object(capture, 'snapshot') as snapshot, \
                    patch.object(capture, 'boundary', side_effect=lambda _, action:events.append(action)), \
                    patch.object(capture, 'terminate_all'), patch.object(capture, 'server_children', return_value=[]), \
                    patch.object(capture.os, 'kill'), patch.object(capture.urllib.request, 'urlopen', side_effect=healthy), \
                    patch.object(capture, 'export_check', side_effect=lambda _:events.append('export')), \
                    contextlib.redirect_stdout(io.StringIO()):
                capture.collect(selected, [16, 32])
            timeline = json.loads((root / 'load-events.json').read_text())
            loads = [e for e in timeline if 'phase' in e]
            self.assertEqual([(e['phase'], e['concurrency'], e['round']) for e in loads],
                             [('warmup', 16, 1), ('warmup', 16, 2), ('warmup', 32, 1),
                              ('warmup', 32, 2), ('eval', 16, 1), ('eval', 32, 1)])
            self.assertTrue(all(e['status'] == 'ok' and len(e['clients']) == 2 for e in loads))
            self.assertTrue((root / 'ep0' / 'server.log').parent.is_dir())
            self.assertTrue((root / 'ep1' / 'seq32' / 'batch8120-concurrency32.json').is_file())
            if log_only:
                snapshot.assert_not_called()
                self.assertTrue(all(command == ['server'] or 'bench' in command for command in launched))
                self.assertTrue(all('event' not in entry for entry in timeline))
                for ep in (0, 1):
                    self.assertTrue((root / f'ep{ep}' / 'server.log').is_file())
                    for seq in (16, 32):
                        for filename in ('warmup-round001.log', 'warmup-round002.log', 'eval.log'):
                            self.assertTrue((root / f'ep{ep}' / f'seq{seq}' / filename).is_file())
        self.assertEqual(events, ['server', 'server'] +
            (['warmup-round001.json'] * 2 + ['warmup-round002.json'] * 2) * 2 + ([] if log_only else ['start']) + [
            'batch8120-concurrency16.json','batch8120-concurrency16.json',
            'batch8120-concurrency32.json','batch8120-concurrency32.json'] +
            ([] if log_only else ['stop','export','export']))

    def test_startup_failure_cleans_owned_app_groups_without_export(self):
        with tempfile.TemporaryDirectory() as temp:
            selected = [dict(ep=i, folder=Path(temp)/f'ep{i}', serve=['server'], url=f'http://ep{i}')
                        for i in (0, 1)]
            servers = [SimpleNamespace(pid=100+i, poll=lambda: None) for i in (0, 1)]
            with patch.dict(os.environ, STARTUP_TIMEOUT='1'), \
                    patch.object(capture, 'launch', side_effect=servers), \
                    patch.object(capture, 'server_children', side_effect=lambda proc: [proc.pid+10]), \
                    patch.object(capture.os, 'getpgid', side_effect=lambda pid: pid), \
                    patch.object(capture.time, 'monotonic', side_effect=[0, 2]), \
                    patch.object(capture.urllib.request, 'urlopen', side_effect=OSError('not ready')), \
                    patch.object(capture, 'terminate_all') as cleanup, \
                    patch.object(capture, 'export_check') as export:
                with self.assertRaises(TimeoutError):
                    capture.collect(selected, [16, 32])
            cleanup.assert_called_once_with(servers, {110, 111})
            export.assert_not_called()


if __name__ == '__main__':
    unittest.main()
