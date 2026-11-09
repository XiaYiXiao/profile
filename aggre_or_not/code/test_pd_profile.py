"""CPU checks for the P/D wire protocol and benchmark integrity."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from collect_pd import validate_result
from pd_proxy import create_app


class CaptureTopologyTests(unittest.TestCase):
    """Exercise the real shell entry points without importing GPU libraries."""

    def run_entry(self, entry='capture_pd_6p2d.sh', heads=64, **overrides):
        base = Path(__file__).resolve().parents[1]
        validation = base / 'validation'
        validation.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='cpu_6p2d_', dir=validation) as folder:
            root = Path(folder)
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text(json.dumps({
                'model_type': 'mimo_v2', 'num_attention_heads': heads,
                'swa_num_attention_heads': heads,
            }))
            env = {k: v for k, v in os.environ.items()
                   if k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH')}
            env.update(DRY_RUN='1', ENABLE_NSYS='0', PYTHON_BIN=sys.executable,
                       MODEL_PATH=str(model), OUTPUT_ROOT=str(root),
                       RESULT_DIR=str(root / 'result'), PYTHONDONTWRITEBYTECODE='1')
            env.update(overrides)
            proc = subprocess.run(['bash', str(base / entry)], env=env,
                                  capture_output=True, text=True, timeout=30)
            result = root / 'result'
            cfg = json.loads((result / 'config.json').read_text()) if result.exists() else None
            status = json.loads((result / 'status.json').read_text()) if result.exists() else None
            sources = sorted(p.name for p in (result / 'sources').glob('*'))
            return proc, cfg, status, sources, (result / 'runtime.json').exists()

    def test_new_entry_keeps_requested_six_plus_two_topology_and_records_entry(self):
        proc, cfg, status, sources, runtime_exists = self.run_entry()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(status['status'], 'dry_run')
        self.assertEqual([(s['gpus'], s['dp'], s['tp']) for s in cfg['servers']],
                         [(list(range(6)), 3, 2), ([6, 7], 2, 1)])
        self.assertTrue(cfg['ep'])
        self.assertEqual(cfg['max_num_batched_tokens'], 16384)
        self.assertEqual(cfg['model_parallelism_check']['status'], 'checked')
        self.assertEqual(cfg['model_parallelism_check']['errors'], [])
        self.assertIn('capture_pd_6p2d.sh', sources)
        self.assertIn('capture_pd.sh', sources)
        self.assertFalse(runtime_exists)
        for server in cfg['servers']:
            cmd = server['command']
            self.assertEqual(cmd[cmd.index('--tensor-parallel-size') + 1], str(server['tp']))
            self.assertEqual(cmd[cmd.index('--data-parallel-size') + 1], str(server['dp']))
            self.assertEqual(cmd[cmd.index('--data-parallel-size-local') + 1], str(server['dp']))
            self.assertIn('--enable-expert-parallel', cmd)
            self.assertIn('CUDA_VISIBLE_DEVICES=' + ','.join(map(str, server['gpus'])), cmd)

    def test_original_pd_entry_retains_four_plus_four_defaults(self):
        proc, cfg, status, _, _ = self.run_entry(entry='capture_pd.sh')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual([(s['gpus'], s['dp'], s['tp']) for s in cfg['servers']],
                         [([0, 1, 2, 3], 2, 2), ([4, 5, 6, 7], 4, 1)])
        self.assertFalse(cfg['ep'])
        self.assertEqual(cfg['model_parallelism_check']['errors'], [])
        self.assertEqual(status['status'], 'dry_run')

    def test_aggregate_entry_still_requires_eight_gpus(self):
        proc, cfg, status, _, _ = self.run_entry(entry='capture_aggre.sh')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual([(s['gpus'], s['dp'], s['tp']) for s in cfg['servers']],
                         [(list(range(8)), 2, 4)])
        self.assertEqual(status['status'], 'dry_run')

    def test_rejects_overlapping_devices_wrong_products_and_wrong_total(self):
        for changes in [
            {'D_GPUS': '5,6'},
            {'P_TP': '3'},
            {'P_GPUS': '0,1,2,3,4', 'P_TP': '1', 'P_DP': '5', 'D_GPUS': '5,6'},
            {'P_GPUS': '0,1,2,3,4,4'},
        ]:
            with self.subTest(changes=changes):
                proc, cfg, _, _, _ = self.run_entry(**changes)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIsNone(cfg)
                self.assertTrue('GPU' in proc.stderr or 'TP * DP' in proc.stderr)

    def test_incompatible_preflight_stops_before_runtime_import_or_gpu_query(self):
        proc, cfg, status, _, runtime_exists = self.run_entry(
            DRY_RUN='0', PREFLIGHT_ONLY='1', P_TP='3', P_DP='2')
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(status['status'], 'failed')
        self.assertIn('num_attention_heads=64', status['error'])
        self.assertIn('TP=3', status['error'])
        self.assertFalse(runtime_exists)
        self.assertEqual(cfg['servers'][0]['tp'], 3)
        self.assertNotIn('nvidia-smi', proc.stderr)

    def test_ep_can_still_be_disabled_for_both_roles(self):
        proc, cfg, status, _, _ = self.run_entry(ENABLE_EP='0')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(cfg['ep'])
        self.assertEqual(cfg['model_parallelism_check']['errors'], [])
        self.assertEqual(status['status'], 'dry_run')
        for server in cfg['servers']:
            self.assertNotIn('--enable-expert-parallel', server['command'])
            self.assertNotIn('--all2all-backend', server['command'])


class ProxyTests(unittest.TestCase):
    def request(self, handler, body):
        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(handler)
        with patch('pd_proxy.httpx.AsyncClient', side_effect=lambda **kwargs:
                   real_client(transport=transport, **kwargs)):
            with TestClient(create_app('http://prefill', 'http://decode', 10)) as client:
                return client.post('/v1/completions', json=body)

    def test_stream_retains_decode_budget_and_usage(self):
        seen = []
        kv = dict(do_remote_prefill=True, remote_engine_id='p-dp0',
                  remote_block_ids=[[1, 2]], remote_host='127.0.0.1', remote_port=5557)
        stream = ('data: {"choices":[{"text":"ok"}]}\n\n'
                  'data: {"usage":{"prompt_tokens":50000,"completion_tokens":500}}\n\n'
                  'data: [DONE]\n\n')

        def handler(request):
            body = json.loads(request.content)
            seen.append((request, body))
            if request.url.host == 'prefill':
                self.assertEqual(body['max_tokens'], 1)
                self.assertFalse(body['stream'])
                self.assertNotIn('stream_options', body)
                self.assertNotIn('min_tokens', body)
                self.assertTrue(body['kv_transfer_params']['do_remote_decode'])
                return httpx.Response(200, json={'kv_transfer_params': kv})
            self.assertEqual(body['max_tokens'], 500)
            self.assertEqual(body['min_tokens'], 500)
            self.assertEqual(body['prompt'], 'same prompt')
            self.assertEqual(body['kv_transfer_params'], kv)
            self.assertTrue(body['stream_options']['include_usage'])
            return httpx.Response(200, text=stream)

        response = self.request(handler, dict(prompt='same prompt', max_tokens=500,
                                              min_tokens=500, stream=True,
                                              stream_options={'include_usage': True}))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, stream)
        self.assertTrue(response.headers['content-type'].startswith('text/event-stream'))
        self.assertEqual(seen[0][0].headers['X-Request-Id'], seen[1][0].headers['X-Request-Id'])

    def test_missing_kv_never_calls_decode(self):
        calls = []

        def handler(request):
            calls.append(request.url.host)
            return httpx.Response(200, json={'choices': []})

        response = self.request(handler, dict(prompt='prompt', max_tokens=500, stream=True))
        self.assertEqual(response.status_code, 502)
        self.assertEqual(calls, ['prefill'])

    def test_decode_http_error_is_not_a_success_stream(self):
        def handler(request):
            if request.url.host == 'prefill':
                return httpx.Response(200, json={'kv_transfer_params': dict(
                    do_remote_prefill=True, remote_engine_id='p', remote_block_ids=[[1]],
                    remote_host='127.0.0.1', remote_port=5557)})
            return httpx.Response(503, json={'error': 'decode unavailable'})

        response = self.request(handler, dict(prompt='prompt', max_tokens=500, stream=True))
        self.assertEqual(response.status_code, 503)


class ResultTests(unittest.TestCase):
    def test_reject_partial_or_wrong_token_counts(self):
        cfg = dict(num_prompts=2, input_len=50000, output_len=500)
        valid = dict(completed=2, failed=0, input_lens=[50000, 50000],
                     output_lens=[500, 500], total_input_tokens=100000,
                     total_output_tokens=1000, errors=['', ''], duration=1.0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            path.write_text(json.dumps(valid))
            validate_result(path, cfg)
            for bad in ({'completed': 1}, {'output_lens': [500, 499]},
                        {'input_lens': [49999, 50000]}, {'errors': ['', 'failure']},
                        {'duration': 0}):
                with self.subTest(bad=bad):
                    path.write_text(json.dumps(dict(valid, **bad)))
                    with self.assertRaises(RuntimeError):
                        validate_result(path, cfg)


if __name__ == '__main__':
    unittest.main()
