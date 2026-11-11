"""Real Prometheus parsing and HTTP/EvalScope integration without a GPU/model."""
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
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest
import pytest
import uvicorn

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'code'))
from server_metrics import MetricsCollector, Window, load_targets, parse_metrics

spec = importlib.util.spec_from_file_location('metrics_proxy', BASE / 'code/pd_proxy.py')
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
KV = dict(do_remote_prefill=True, remote_engine_id='p', remote_block_ids=[1], remote_host='127.0.0.1', remote_port=6100)


def exposition(queries, hits, external=0):
    lines = []
    for name, values in [('prefix_cache_queries', queries), ('prefix_cache_hits', hits),
                         ('external_prefix_cache_queries', queries),
                         ('external_prefix_cache_hits', [external] * len(queries))]:
        lines.append(f'# TYPE vllm:{name}_total counter')
        lines.extend(f'vllm:{name}_total{{model_name="test,model",engine="{i}"}} {v}'
                     for i, v in enumerate(values))
    lines += ['# TYPE vllm:prompt_tokens_by_source_total counter',
              f'vllm:prompt_tokens_by_source_total{{source="local_compute"}} {sum(queries) - sum(hits)}']
    return '\n'.join(lines) + '\n'


def test_weighted_cache_delta_and_computed_rate():
    window = Window()
    window.observe(parse_metrics(exposition([1000, 1000], [900, 900], 500)), 10)
    window.observe(parse_metrics(exposition([1100, 1900], [1000, 900], 900)), 20)
    data = window.summary()
    assert data['prefix_cache']['hit_rate'] == pytest.approx(.1)  # Not mean(100%, 0%).
    assert data['prefix_cache']['queried_tokens'] == 1000
    assert data['external_prefix_cache']['hit_rate'] == pytest.approx(.8)
    assert data['computed_prefill_tokens_per_second'] == pytest.approx(90)
    assert [row['hit_rate'] for row in data['prefix_cache_by_labels']] == [1, 0]


def test_no_queries_missing_and_resets_are_not_zero_hits():
    window = Window()
    window.observe(parse_metrics(exposition([100], [80])), 1)
    assert window.summary()['prefix_cache']['hit_rate'] is None
    window.observe(parse_metrics(exposition([100], [80])), 2)
    assert window.summary()['prefix_cache']['status'] == 'no_queries'
    window.observe(parse_metrics(exposition([10], [8])), 3)
    assert window.summary()['prefix_cache']['hit_rate'] is None
    assert window.resets > 0
    window.observe(parse_metrics(exposition([20, 100], [16, 80])), 4)
    assert window.summary()['prefix_cache']['hit_rate'] is None
    assert window.gaps > 0
    with pytest.raises(ValueError):
        parse_metrics('<html>not metrics</html>')
    with pytest.raises(ValueError, match='No supported'):
        parse_metrics('# TYPE unrelated gauge\nunrelated 1\n')


def test_histogram_delta_and_transfer_metrics():
    text = '''# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_sum{engine="0"} 10
vllm:request_queue_time_seconds_count{engine="0"} 2
vllm:request_queue_time_seconds_bucket{engine="0",le="+Inf"} 2
# TYPE vllm:nixl_xfer_time_seconds histogram
vllm:nixl_xfer_time_seconds_sum 1
vllm:nixl_xfer_time_seconds_count 2
'''
    rows = parse_metrics(text)
    window = Window()
    window.observe(rows, 0)
    window.observe([dict(row, value=row['value'] * 2) for row in rows], 10)
    assert window.summary()['latencies']['request_queue_time_seconds']['mean'] == 5
    assert window.summary()['kv_transfer']['vllm:nixl_xfer_time_seconds']['mean'] == .5


@pytest.mark.parametrize('mode', ['aggre', 'pd', 'pd_6p2d'])
def test_deploy_enables_collection_without_launch(mode, tmp_path):
    env = dict(os.environ, DRY_RUN='1', PYTHON_BIN=sys.executable, OUTPUT_ROOT=str(tmp_path),
               ENABLE_METRICS='1', METRICS_INTERVAL='2.5', METRICS_GPU='0')
    result = subprocess.run(['bash', str(BASE / f'deploy_{mode}.sh')], env=env,
                            capture_output=True, text=True, check=True)
    cfg = json.loads(result.stdout)
    assert cfg['metrics']['enabled'] and cfg['metrics']['interval_seconds'] == 2.5
    assert not cfg['metrics']['gpu_enabled']
    for server in cfg['servers']:
        assert '--enable-prompt-tokens-details' in server['command']
    if mode != 'aggre':
        assert '--trace-path' in cfg['proxy_command']
    assert not list(tmp_path.iterdir())
    env['ENABLE_METRICS'] = '0'
    result = subprocess.run(['bash', str(BASE / f'deploy_{mode}.sh')], env=env,
                            capture_output=True, text=True, check=True)
    cfg = json.loads(result.stdout)
    assert not cfg['metrics']['enabled']
    assert '--trace-path' not in cfg.get('proxy_command', [])


class DelayedStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        await asyncio.sleep(.03)
        yield b'data: {"choices":[{"delta":{"content":"'
        await asyncio.sleep(.01)
        yield b'hello"}}]}\n\ndata: [DONE]\n\n'


def test_proxy_stage_trace_and_metrics_forwarding(tmp_path):
    trace = tmp_path / 'requests.jsonl'
    def upstream(req):
        if req.url.path == '/metrics':
            return httpx.Response(200, text=exposition([100], [80]))
        if req.url.host == 'prefill':
            return httpx.Response(200, json=dict(kv_transfer_params=KV, usage={
                'prompt_tokens': 100, 'prompt_tokens_details': {'cached_tokens': 80}}))
        return httpx.Response(200, stream=DelayedStream())
    app = proxy.create_app('http://prefill', 'http://decode', 10,
                           httpx.MockTransport(upstream), trace_path=trace)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json={'messages': [], 'stream': True},
                               headers={'x-request-id': 'test-id'})
        assert 'hello' in response.text and '[DONE]' in response.text
        rows = parse_metrics(client.get('/metrics').text)
        assert any(r['name'] == 'eva_pd_requests_total' and r['labels']['outcome'] == 'success'
                   and r['value'] == 1 for r in rows)
        assert client.get('/metrics/prefill').status_code == 200
        assert client.get('/metrics/decode').status_code == 200
        assert client.get('/metrics/arbitrary-url').status_code == 404
    data = json.loads(trace.read_text())
    assert data['request_id'] == 'test-id' and data['outcome'] == 'success'
    assert data['decode_first_content_seconds'] >= .04
    assert data['ttft_seconds'] >= data['decode_first_content_seconds']
    assert data['prefill_cached_tokens'] == 80
    assert 'hello' not in trace.read_text() and 'messages' not in data


@pytest.mark.parametrize('failure', ['prefill', 'decode', 'metadata', 'sse'])
def test_proxy_records_failures(tmp_path, failure):
    trace = tmp_path / 'requests.jsonl'
    def upstream(req):
        if req.url.host == 'prefill':
            return httpx.Response(500 if failure == 'prefill' else 200,
                                  json={} if failure == 'metadata' else {'kv_transfer_params': KV})
        return httpx.Response(429 if failure == 'decode' else 200,
                              content=b'data: {"error":{"message":"failed"}}\n\ndata: [DONE]\n\n')
    with TestClient(proxy.create_app('http://prefill', 'http://decode', 10,
                                    httpx.MockTransport(upstream), trace_path=trace)) as client:
        client.post('/v1/completions', json={'prompt': 'not logged', 'stream': True})
    data = json.loads(trace.read_text())
    assert data['outcome'] == ('stream_error' if failure == 'sse' else failure + '_error')
    assert data['ttft_seconds'] is None


@pytest.fixture
def telemetry_server():
    registry = CollectorRegistry()
    queries = Counter('vllm:prefix_cache_queries', 'tokens', registry=registry)
    hits = Counter('vllm:prefix_cache_hits', 'tokens', registry=registry)
    queries.inc(1000)
    hits.inc(900)  # Prior traffic must be excluded.
    app = FastAPI()
    @app.get('/health')
    def health():
        return {'status': 'ok'}
    @app.get('/v1/models')
    def models():
        return {'data': [{'id': 'test-model'}]}
    @app.get('/metrics/prefill')
    @app.get('/metrics/decode')
    @app.get('/metrics')
    def metrics():
        return Response(generate_latest(registry), media_type='text/plain')
    Gauge('vllm:num_requests_running', 'running', registry=registry)
    Gauge('vllm:num_requests_waiting', 'waiting', registry=registry)
    Gauge('eva_pd_inflight', 'proxy active', registry=registry)
    @app.post('/reset_prefix_cache')
    def reset_cache():
        return {'success': True}
    @app.post('/v1/completions')
    async def completions(request: Request):
        queries.inc(100)
        hits.inc(80)
        async def events():
            for word in ('hi', '!'):
                yield 'data: ' + json.dumps({'choices': [{'index': 0, 'text': word}]}) + '\n\n'
                await asyncio.sleep(.01)
            yield 'data: ' + json.dumps({'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 2, 'total_tokens': 102}}) + '\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(events(), media_type='text/event-stream')
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(.01)
    assert server.started
    yield f'http://127.0.0.1:{port}', queries, hits
    server.should_exit = True
    thread.join(5)
    sock.close()


def test_periodic_sampler_handles_failed_endpoint_and_final_flush(tmp_path, telemetry_server):
    url, queries, hits = telemetry_server
    collector = MetricsCollector(tmp_path / 'metrics', {'prefill': url + '/metrics', 'decode': url + '/missing'},
                                 interval=.03, timeout=.2).start()
    queries.inc(200)
    hits.inc(100)
    time.sleep(.1)
    report = collector.stop()
    assert not collector.thread.is_alive()
    assert report['status'] == 'partial'
    assert report['roles']['prefill']['prefix_cache']['hit_rate'] == .5
    assert report['roles']['decode']['prefix_cache']['hit_rate'] is None
    rows = [json.loads(line) for line in (tmp_path / 'metrics/samples.jsonl').read_text().splitlines()]
    assert any(r.get('error') for r in rows) and len(rows) >= 4
    assert (tmp_path / 'metrics/summary.csv').is_file()


def test_load_archives_server_cache_with_real_evalscope(tmp_path, telemetry_server):
    url, _, _ = telemetry_server
    dataset = tmp_path / 'input.txt'
    dataset.write_text('hello\nworld\n')
    command = ['bash', str(BASE / 'run_load.sh'), '--mode', 'pd', '--port', url.rsplit(':', 1)[1],
               '--results-root', str(tmp_path / 'results'), '--model', 'test-model', '--tokenizer-path', '',
               '--prepared-dataset-path', str(dataset), '--parallel', '1', '--number', '2',
               '--input-length', '100', '--output-length', '2', '--metrics-interval', '.1',
               '--metrics-settle-seconds', '0']
    result = subprocess.run(command, env=dict(os.environ, LOAD_PYTHON_BIN=sys.executable),
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    manifest_path, = (tmp_path / 'results').rglob('manifest.json')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'completed'
    assert manifest['prefix_cache_reset']['status'] == 'completed'
    assert manifest['server_metrics']['prefix_cache']['prefill']['hit_rate'] == .8
    report = json.loads((manifest_path.parent / 'server_metrics/summary.json').read_text())
    assert report['roles']['prefill']['prefix_cache']['queried_tokens'] == 200
    assert report['roles']['decode']['prefix_cache']['hit_rate'] == .8
    assert (manifest_path.parent / 'perf/summary.csv').is_file()
    assert 'prefill prefix cache hit rate: 80.00%' in result.stdout


def test_target_resolution_never_infers_remote_loopback():
    targets = load_targets('pd', 'https://remote:19100')
    assert targets['prefill'] == 'https://remote:19100/metrics/prefill'
    assert load_targets('aggre', 'http://remote:19000') == {'aggre': 'http://remote:19000/metrics'}
    assert load_targets('pd', 'http://remote', {'prefill': 'http://local:19110/metrics'})['prefill'] == 'http://local:19110/metrics'


def test_deployment_collector_lifecycle(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('metrics_deploy', BASE / 'code/deploy.py')
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    events = []
    cfg = dict(root=str(tmp_path / 'deployment'), host='127.0.0.1', port=0, mode='pd',
               deployment_mode='pd', startup_timeout=1, proxy_command=['proxy'],
               servers=[dict(role='prefill', url='http://p', command=['p'], gpus=[0]),
                        dict(role='decode', url='http://d', command=['d'], gpus=[1])],
               metrics=dict(enabled=True, directory=str(tmp_path / 'deployment/metrics'),
                            interval_seconds=5, timeout_seconds=3, gpu_enabled=True))
    monkeypatch.setattr(deploy, 'configuration', lambda: cfg)
    monkeypatch.setattr(deploy, 'enabled', lambda _: False)
    monkeypatch.setattr(deploy, 'prepare_environment', lambda _: None)
    monkeypatch.setattr(deploy, 'preflight', lambda _: {})
    class Job:
        pid = 123
        def poll(self):
            return None
    monkeypatch.setattr(deploy, 'launch', lambda command, _: (events.append(('launch', command)), Job())[1])
    monkeypatch.setattr(deploy, 'server_children', lambda _: [])
    monkeypatch.setattr(deploy, 'terminate_all', lambda *_: events.append(('terminate',)))
    class Health:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(deploy.urllib.request, 'urlopen',
                        lambda url, **_: (events.append(('health', url)), Health())[1])
    class Collector:
        fatal_error = None
        def __init__(self, root, targets, **kwargs):
            assert targets == {'prefill': 'http://p/metrics', 'decode': 'http://d/metrics',
                               'proxy': 'http://127.0.0.1:0/metrics'}
        def start(self):
            events.append(('baseline',))
            assert cfg['status'] == 'starting'
            return self
        def stop(self):
            events.append(('stop_collector',))
    monkeypatch.setattr(deploy, 'MetricsCollector', Collector)
    def interrupt(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(deploy.time, 'sleep', interrupt)
    monkeypatch.setattr(deploy.signal, 'signal', lambda *_: None)
    deploy.main()
    assert [e[0] for e in events] == [
        'launch', 'launch', 'health', 'health', 'launch', 'health',
        'baseline', 'stop_collector', 'terminate']
    assert json.loads((tmp_path / 'deployment/deployment.json').read_text())['status'] == 'stopped'
