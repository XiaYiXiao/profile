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
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
import uvicorn

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('eva_proxy', BASE / 'code/pd_proxy.py')
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
KV = dict(do_remote_prefill=True, remote_engine_id='p', remote_block_ids=[1], remote_host='127.0.0.1', remote_port=6100)


@pytest.mark.parametrize('mode,topology,port', [
    ('aggre', [(2, 4)], 19000), ('pd', [(2, 2), (2, 2)], 19100),
    ('pd_6p2d', [(3, 2), (1, 2)], 19200),
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
                          env=dict(os.environ, DRY_RUN='1', KV_CONNECTOR='NixlConnector', P_NIXL_PORT='6200'), capture_output=True)
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


def mooncake_registry():
    return {str(dp): {'engine_id': f'p-dp{dp}',
                     'worker_addr': {str(tp): {'0': f'127.0.0.1:{7000 + dp * 2 + tp}'}
                                     for tp in range(2)}}
            for dp in range(3)}


def mooncake_app(upstream):
    return proxy.create_app('http://prefill', 'http://decode', 10,
                            httpx.MockTransport(upstream), kv_connector='MooncakeConnector',
                            prefill_bootstrap_url='http://bootstrap',
                            prefill_dp_size=3, prefill_tp_size=2)


@pytest.mark.parametrize('endpoint', ['/v1/completions', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
def test_mooncake_proxy_routes_matching_dp_and_transfer_ids(endpoint, stream):
    seen = []
    def upstream(req):
        if req.url.host == 'bootstrap':
            assert req.url.path == '/query'
            return httpx.Response(200, json=mooncake_registry())
        if req.url.path == '/health':
            return httpx.Response(200)
        body = json.loads(req.content)
        seen.append((req, body))
        if req.url.host == 'prefill':
            assert body['max_tokens'] == body['max_completion_tokens'] == 1
            assert not body['stream'] and 'min_tokens' not in body and 'stream_options' not in body
            return httpx.Response(200, json={'choices': []})
        assert body['max_tokens'] == body['max_completion_tokens'] == 4
        assert 'x-data-parallel-rank' not in req.headers
        return httpx.Response(200, content=b'data: [DONE]\n\n' if stream else b'{"choices":[]}')
    with TestClient(mooncake_app(upstream)) as client:
        assert client.get('/health').status_code == 200
        for _ in range(4):
            result = client.post(endpoint, json={'model': 'test', 'prompt': 'hello',
                                 'messages': [], 'max_tokens': 4, 'max_completion_tokens': 4,
                                 'min_tokens': 4, 'stream': stream, 'stream_options': {}},
                                 headers={'x-request-id': 'reused-id', 'session-id': 'session'})
            assert result.status_code == 200
            assert result.text == ('data: [DONE]\n\n' if stream else '{"choices":[]}')
    transfer_ids = []
    for idx in range(4):
        (p_req, p_body), (d_req, d_body) = seen[2 * idx:2 * idx + 2]
        assert p_req.headers['x-data-parallel-rank'] == str(idx % 3)
        p_kv, d_kv = p_body['kv_transfer_params'], d_body['kv_transfer_params']
        assert p_kv['do_remote_decode'] and not p_kv['do_remote_prefill']
        assert d_kv == dict(do_remote_decode=False, do_remote_prefill=True,
                            remote_bootstrap_addr='http://bootstrap',
                            remote_engine_id=f'p-dp{idx % 3}', transfer_id=p_kv['transfer_id'])
        transfer_ids.append(p_kv['transfer_id'])
        for req in (p_req, d_req):
            assert req.url.path == endpoint
            assert req.headers['x-request-id'] == 'reused-id'
            assert req.headers['session-id'] == 'session'
    assert len(set(transfer_ids)) == 4


@pytest.mark.parametrize('failure', ['empty', 'missing_tp', 'missing_engine', 'invalid_json'])
def test_mooncake_proxy_rejects_incomplete_bootstrap(failure):
    registry = mooncake_registry()
    if failure == 'empty':
        registry = {}
    elif failure == 'missing_tp':
        del registry['1']['worker_addr']['1']
    elif failure == 'missing_engine':
        del registry['2']['engine_id']
    def upstream(req):
        assert req.url.host == 'bootstrap'
        return httpx.Response(200, text='invalid') if failure == 'invalid_json' else httpx.Response(200, json=registry)
    with pytest.raises(RuntimeError, match='Invalid Mooncake bootstrap registry'):
        with TestClient(mooncake_app(upstream)):
            pass


@pytest.mark.parametrize('failure', ['prefill', 'decode'])
def test_mooncake_proxy_preserves_upstream_failures(failure):
    called = []
    def upstream(req):
        if req.url.host == 'bootstrap':
            return httpx.Response(200, json=mooncake_registry())
        called.append(req.url.host)
        if req.url.host == 'prefill':
            return httpx.Response(500 if failure == 'prefill' else 200, json={'choices': []})
        return httpx.Response(429, json={'error': 'busy'})
    with TestClient(mooncake_app(upstream)) as client:
        response = client.post('/v1/completions', json={'prompt': 'test', 'max_tokens': 4})
    assert response.status_code == (502 if failure == 'prefill' else 429)
    assert called == (['prefill'] if failure == 'prefill' else ['prefill', 'decode'])


@pytest.fixture(scope='module')
def mock_server():
    app = FastAPI()
    @app.get('/health')
    def health():
        return {'status': 'ok'}
    @app.get('/v1/models')
    def models():
        return {'data': [{'id': 'test-model'}, {'id': 'missing-usage'}, {'id': 'partial-failures'}]}
    @app.post('/v1/completions')
    @app.post('/v1/chat/completions')
    async def completions(req: Request):
        body = await req.json()
        prompt = body.get('prompt') or body.get('messages', [{}])[0].get('content')
        if body['model'] == 'partial-failures' and prompt == 'fail':
            return JSONResponse({'error': 'simulated request failure'}, status_code=503)
        async def chunks():
            for token in ('hello', ' world', '!'):
                choice = {'index': 0, 'delta': {'content': token}, 'finish_reason': None} if 'messages' in body else {'index': 0, 'text': token, 'finish_reason': None}
                yield 'data: ' + json.dumps({'id': 'mock', 'object': 'chat.completion.chunk', 'model': 'test-model', 'choices': [choice]}) + '\n\n'
                await asyncio.sleep(0.01)
            if body['model'] != 'missing-usage':
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
    command = ['bash', str(BASE / 'run_load.sh'), '--no-reset-prefix-cache', '--results-root', str(tmp_path / 'results'),
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
    cmd = ['bash', str(BASE / 'run_load.sh'), '--no-reset-prefix-cache', '--results-root', str(tmp_path / 'results'),
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


def test_saved_requests_fail_when_server_omits_usage(tmp_path, mock_server):
    dataset = tmp_path / 'prompts.txt'
    dataset.write_text('saved prompt\n')
    command = [sys.executable, str(BASE / 'per_dataset.py'),
               '--port', str(mock_server), '--model', 'missing-usage', '--use-chat',
               '--tokenizer-path', '', '--prepared-dataset-path', str(dataset),
               '--parallel', '1', '--number', '1', '--output-length', '3',
               '--output-dir', str(tmp_path / 'results'), '--no-write-csv', '--allow-failed-requests']
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert result.returncode != 0
    assert 'usage.prompt_tokens / usage.completion_tokens' in result.stdout + result.stderr


@pytest.mark.parametrize('allow_failed, prompts, expected_success, expected_status', [
    (False, ['fail', 'hello'], 1, 'failed'),
    (True, ['fail', 'hello'], 1, 'completed_with_errors'),
    (True, ['fail', 'fail'], 0, 'failed'),
])
def test_request_failure_policy_with_real_evalscope(
        tmp_path, mock_server, allow_failed, prompts, expected_success, expected_status):
    dataset = tmp_path / 'prompts.txt'
    dataset.write_text('\n'.join(prompts) + '\n')
    command = ['bash', str(BASE / 'run_load.sh'), '--no-reset-prefix-cache', '--no-server-metrics',
               '--results-root', str(tmp_path / 'results'), '--port', str(mock_server),
               '--model', 'partial-failures', '--tokenizer-path', '', '--use-chat',
               '--prepared-dataset-path', str(dataset), '--parallel', '1', '--number', '2',
               '--output-length', '3']
    if allow_failed:
        command.append('--allow-failed-requests')
    result = subprocess.run(command, env=dict(os.environ, LOAD_PYTHON_BIN=sys.executable),
                            capture_output=True, text=True, timeout=60)
    assert (result.returncode == 0) == (expected_status == 'completed_with_errors'), result.stdout + result.stderr
    manifest_path, = (tmp_path / 'results').rglob('manifest.json')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == expected_status
    frame = pd.read_csv(manifest_path.parent / 'perf/summary.csv')
    assert frame['Success Requests'].tolist() == [expected_success]
    assert frame['Failed Requests'].tolist() == [2 - expected_success]
    assert (manifest_path.parent / 'perf/summary.xlsx').is_file()
    if expected_status == 'completed_with_errors':
        assert manifest['request_counts'] == {'success': 1, 'failed': 1}
        assert '保留失败记录并继续' in result.stdout
        assert '本轮成功 1 条、失败 1 条' in (manifest_path.parent / 'load.log').read_text()


def test_reset_failure_prevents_load_and_is_archived(tmp_path, mock_server):
    command = ['bash', str(BASE / 'run_load.sh'), '--results-root', str(tmp_path / 'results'),
               '--port', str(mock_server), '--model', 'test-model', '--tokenizer-path', '',
               '--parallel', '1', '--number', '1', '--no-server-metrics', '--allow-failed-requests']
    result = subprocess.run(command, env=dict(os.environ, LOAD_PYTHON_BIN=sys.executable),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    manifest_path, = (tmp_path / 'results').rglob('manifest.json')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'failed'
    assert manifest['prefix_cache_reset']['status'] == 'failed'
    assert 'load was not started' in manifest['error']
    assert not (manifest_path.parent / 'perf').exists()
    assert not (manifest_path.parent / 'server_metrics').exists()


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


def test_allowed_failed_requests_keep_counts_and_continue(tmp_path, monkeypatch):
    sys.path.insert(0, str(BASE))
    import per_dataset
    from click.testing import CliRunner

    seen = []
    def partial_perf(args):
        seen.append(args.parallel[0])
        root = Path(args.outputs_dir)
        root.mkdir(parents=True)
        (root / 'benchmark_summary.json').write_text(json.dumps({
            'Success Requests': 1 if len(seen) == 1 else 2,
            'Failed Requests': 1 if len(seen) == 1 else 0,
        }))

    monkeypatch.setattr(per_dataset, 'run_perf_benchmark', partial_perf)
    root = tmp_path / 'partial'
    result = CliRunner().invoke(per_dataset.runner, [
        '--output-dir', str(root), '--parallel', '1 2', '--number', '2', '--allow-failed-requests',
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert seen == [1, 2]
    assert '保留失败记录并继续' in result.output
    assert pd.read_csv(root / 'summary.csv')['Failed Requests'].tolist() == [1, 0]
    assert (root / 'summary.xlsx').is_file()


def test_load_uses_deployed_tokenizer_with_explicit_override(tmp_path):
    deployment = tmp_path / 'deployment.json'
    deployment.write_text(json.dumps(dict(status='ready', deployment_mode='aggre',
                                         port=19000, model='custom', model_path='/models/custom')))
    cmd = [sys.executable, str(BASE / 'code/load.py'), '--dry-run',
           '--deployment', str(deployment)]
    for extra, expected in [([], '/models/custom'),
                            (['--tokenizer-path', '/other/tokenizer'], '/other/tokenizer')]:
        result = subprocess.run(cmd + extra, text=True, capture_output=True, check=True)
        manifest = json.loads(result.stdout)
        command = manifest['command']
        assert command[command.index('--tokenizer-path') + 1] == expected
        # The default load must fit even the smallest deployment (6P2D).
        assert (int(command[command.index('--input-length') + 1]) +
                int(command[command.index('--output-length') + 1])) <= 16384
