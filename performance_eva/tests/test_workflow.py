"""CPU protocol and actual EvalScope smoke tests; no model/GPU is started."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import httpx
import pandas as pd
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
import uvicorn

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('eva_proxy', BASE / 'code/pd_proxy.py')
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
KV = dict(do_remote_prefill=True, remote_engine_id='p', remote_block_ids=[1], remote_host='127.0.0.1', remote_port=6100)


@pytest.mark.parametrize('mode,topology,port', [
    ('aggre', [(2, 4)], 19000), ('pd', [(2, 2), (4, 1)], 19100),
    ('pd_6p2d', [(3, 2), (2, 1)], 19200),
])
def test_deployment_dry_run(mode, topology, port, tmp_path):
    env = dict(os.environ, DRY_RUN='1', PYTHON_BIN=sys.executable, OUTPUT_ROOT=str(tmp_path))
    proc = subprocess.run(['bash', str(BASE / f'deploy_{mode}.sh')], env=env,
                          text=True, capture_output=True, check=True)
    cfg = json.loads(proc.stdout)
    assert [(s['dp'], s['tp']) for s in cfg['servers']] == topology
    assert cfg['port'] == port and cfg['host'] == '0.0.0.0'
    assert len(cfg['ports']) == len(set(cfg['ports']))
    assert cfg['ep'] and cfg['eplb']
    assert not list(tmp_path.iterdir())


def test_port_overlap_rejected():
    proc = subprocess.run(['bash', str(BASE / 'deploy_pd.sh')],
                          env=dict(os.environ, DRY_RUN='1', P_NIXL_PORT='6200'), capture_output=True)
    assert proc.returncode != 0
    assert b'disjoint' in proc.stderr


@pytest.mark.parametrize('endpoint', ['/v1/completions', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
def test_proxy(endpoint, stream):
    seen = []
    def upstream(req):
        if req.url.path == '/health':
            return httpx.Response(200)
        if req.url.path == '/v1/models':
            return httpx.Response(200, json={'data': [{'id': 'test-model'}]})
        body = json.loads(req.content)
        seen.append((req, body))
        if req.url.host == 'prefill':
            assert body['max_tokens'] == 1 and not body['stream']
            assert 'stream_options' not in body and 'min_tokens' not in body
            return httpx.Response(200, json={'kv_transfer_params': KV})
        assert body['max_tokens'] == 4 and body['kv_transfer_params'] == KV
        return httpx.Response(200, content=b'data: [DONE]\n\n' if stream else b'{"choices":[]}',
                              headers={'content-type': 'text/event-stream' if stream else 'application/json'})
    app = proxy.create_app('http://prefill', 'http://decode', 10, httpx.MockTransport(upstream))
    with TestClient(app) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/v1/models').json()['data'][0]['id'] == 'test-model'
        body = dict(model='test-model', max_tokens=4, min_tokens=4, stream=stream, stream_options={})
        body.update(messages=[{'role': 'user', 'content': 'hello'}] if 'chat' in endpoint else [], prompt='hello')
        result = client.post(endpoint, json=body, headers={'session-id': 'sid', 'request-id': 'rid', 'x-request-id': 'xrid', 'X-SMG-Routing-Key': 'sid'})
        assert result.status_code == 200
        assert result.text == ('data: [DONE]\n\n' if stream else '{"choices":[]}')
    assert len(seen) == 2
    for req, _ in seen:
        assert req.url.path == endpoint
        assert req.headers['session-id'] == 'sid' and req.headers['x-request-id'] == 'xrid'
        assert req.headers['request-id'] == 'rid' and req.headers['x-smg-routing-key'] == 'sid'


@pytest.mark.parametrize('failure', ['missing_kv', 'prefill_error', 'decode_error'])
def test_proxy_failure(failure):
    called = []
    def upstream(req):
        called.append(req.url.host)
        if req.url.host == 'prefill':
            return httpx.Response(500 if failure == 'prefill_error' else 200,
                                  json={} if failure == 'missing_kv' else {'kv_transfer_params': KV})
        return httpx.Response(429, json={'error': 'busy'})
    with TestClient(proxy.create_app('http://prefill', 'http://decode', 10, httpx.MockTransport(upstream))) as client:
        result = client.post('/v1/chat/completions', json={'messages': [], 'max_tokens': 4})
    assert result.status_code == (429 if failure == 'decode_error' else 502)
    assert len(called) == (2 if failure == 'decode_error' else 1)


@pytest.fixture(scope='module')
def mock_server():
    app = FastAPI()
    @app.get('/health')
    def health():
        return {'status': 'ok'}
    @app.get('/v1/models')
    def models():
        return {'data': [{'id': 'test-model'}]}
    @app.post('/v1/completions')
    @app.post('/v1/chat/completions')
    async def completions(req: Request):
        body = await req.json()
        async def chunks():
            for token in ('hello', ' world', '!'):
                choice = {'index': 0, 'delta': {'content': token}, 'finish_reason': None} if 'messages' in body else {'index': 0, 'text': token, 'finish_reason': None}
                yield 'data: ' + json.dumps({'id': 'mock', 'object': 'chat.completion.chunk', 'model': 'test-model', 'choices': [choice]}) + '\n\n'
                await asyncio.sleep(0.01)
            yield 'data: ' + json.dumps({'choices': [], 'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}}) + '\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(chunks(), media_type='text/event-stream')
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(.05)
    assert server.started
    yield port
    server.should_exit = True
    thread.join(timeout=10)
    sock.close()


@pytest.mark.parametrize('kind', ['completion', 'chat', 'multi'])
def test_real_evalscope(tmp_path, mock_server, kind):
    dataset = tmp_path / ('prompts.jsonl' if kind == 'multi' else 'prompts.txt')
    if kind == 'multi':
        dataset.write_text(json.dumps([{'role': 'user', 'content': 'one'}, {'role': 'assistant', 'content': ''}, {'role': 'user', 'content': 'two'}]) + '\n')
        extra = ['--multi-turn', '--multi-turn-dataset', 'custom_multi_turn', '--multi-turn-dataset-path', str(dataset), '--min-turns', '2', '--max-turns', '2']
    else:
        dataset.write_text('hello\nworld\n')
        extra = ['--prepared-dataset-path', str(dataset)] + (['--use-chat'] if kind == 'chat' else [])
    command = ['bash', str(BASE / 'run_load.sh'), '--results-root', str(tmp_path / 'results'),
               '--port', str(mock_server), '--model', 'test-model', '--tokenizer-path', '',
               '--parallel', '1 2', '--number', '2', '--input-length', '5', '--output-length', '3',
               '--loop', '2', '--inject-session-id', '--inject-req-ids', *extra]
    env = dict(os.environ, LOAD_PYTHON_BIN=sys.executable)
    proc = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stdout[-5000:] + proc.stderr[-5000:]
    manifest_path, = (tmp_path / 'results').rglob('manifest.json')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'completed'
    root = manifest_path.parent / 'perf'
    assert (root / 'summary.xlsx').is_file()
    frame = pd.read_csv(root / 'summary.csv')
    assert len(frame) == 4
    assert len(list(root.rglob('benchmark_summary.json'))) == 4
    assert len(list(root.rglob('benchmark_data.db'))) == 4
    assert (root / 'request_map.csv').is_file()
    assert (root / 'request_ids.log').is_file()


def test_concurrent_identical_runs(tmp_path, mock_server):
    data = tmp_path / 'prompts.txt'
    data.write_text('hello\n')
    cmd = ['bash', str(BASE / 'run_load.sh'), '--results-root', str(tmp_path / 'results'),
           '--port', str(mock_server), '--model', 'test-model', '--tokenizer-path', '',
           '--prepared-dataset-path', str(data), '--parallel', '1', '--number', '1',
           '--input-length', '5', '--output-length', '3']
    env = dict(os.environ, LOAD_PYTHON_BIN=sys.executable)
    procs = [subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    for proc in procs:
        out, _ = proc.communicate(timeout=90)
        assert proc.returncode == 0, out[-4000:]
    manifests = list((tmp_path / 'results').rglob('manifest.json'))
    assert len(manifests) == 2
    assert manifests[0].parent.parent == manifests[1].parent.parent
    assert len({json.loads(p.read_text())['run_id'] for p in manifests}) == 2
    for path in manifests:
        assert len(pd.read_csv(path.parent / 'perf/summary.csv')) == 1


def test_direct_output_and_invalid_sweep(tmp_path, monkeypatch):
    sys.path.insert(0, str(BASE))
    import per_dataset
    from click.testing import CliRunner
    runner = CliRunner()
    def fake_perf(args):
        root = Path(args.outputs_dir)
        root.mkdir(parents=True)
        (root / 'benchmark_summary.json').write_text(json.dumps({
            'Concurrency': args.parallel[0], 'Success Requests': 2, 'Failed Requests': 0,
            'Avg TPOT (ms)': 10, 'Output Throughput (tok/s)': 100,
        }))
    monkeypatch.setattr(per_dataset, 'run_perf_benchmark', fake_perf)
    root = tmp_path / 'direct'
    args = ['--output-dir', str(root), '--parallel', '1 2', '--number', '2',
            '--output-csv', 'custom.csv', '--output-xlsx', 'different.xlsx']
    result = runner.invoke(per_dataset.runner, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert (root / 'different.xlsx').is_file()
    assert not (root / 'different.csv').exists()
    assert len(pd.read_csv(root / 'custom.csv')) == 2
    assert runner.invoke(per_dataset.runner, args).exit_code != 0
    assert len(pd.read_csv(root / 'custom.csv')) == 2
    bad = runner.invoke(per_dataset.runner, ['--parallel', '1 2 3', '--number', '2 4'])
    assert bad.exit_code != 0 and '等长' in bad.output


def test_failed_metrics_preserve_summary(tmp_path, monkeypatch):
    sys.path.insert(0, str(BASE))
    import per_dataset
    from click.testing import CliRunner
    def failed_perf(args):
        root = Path(args.outputs_dir)
        root.mkdir(parents=True)
        (root / 'benchmark_summary.json').write_text(json.dumps({'Success Requests': 0, 'Failed Requests': 2}))
    monkeypatch.setattr(per_dataset, 'run_perf_benchmark', failed_perf)
    root = tmp_path / 'failed'
    result = CliRunner().invoke(per_dataset.runner, ['--output-dir', str(root), '--parallel', '1', '--number', '2'])
    assert result.exit_code != 0
    assert (root / 'summary.csv').is_file() and (root / 'summary.xlsx').is_file()
