import io
import json
from pathlib import Path
import sys
import urllib.error

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import cache_reset
from pd_proxy import create_app


def gauges(busy=0):
    return '\n'.join(f'vllm:num_requests_{name}{{engine="{rank}"}} {busy if rank == 1 else 0}'
                     for name in ('running', 'waiting') for rank in range(2)).encode()


@pytest.mark.parametrize('failure', [None, 'busy_rank1', 'missing_gauges', 'false', '404'])
def test_reset_checks_all_workers_and_response(monkeypatch, failure):
    calls = []
    def request(req, **kwargs):
        calls.append(req)
        if req.get_method() == 'POST':
            if failure == '404':
                raise urllib.error.HTTPError(req.full_url, 404, 'Not Found', {}, None)
            return io.BytesIO(json.dumps({'success': failure != 'false'}).encode())
        return io.BytesIO(b'vllm:prefix_cache_hits_total 2' if failure == 'missing_gauges'
                          else gauges(1 if failure == 'busy_rank1' else 0))
    monkeypatch.setattr(cache_reset.urllib.request, 'urlopen', request)
    report = {}
    if failure:
        with pytest.raises(RuntimeError, match='load was not started'):
            cache_reset.reset_prefix_cache('http://server', {'aggre': 'http://server/metrics'},
                                           report, timeout=.001, idle_seconds=0)
        assert report['status'] == 'failed'
    else:
        cache_reset.reset_prefix_cache('http://server', {'aggre': 'http://server/metrics'},
                                       report, api_key='test', idle_seconds=0)
        assert report['status'] == 'completed'
        assert calls[-1].get_header('Authorization') == 'Bearer test'
        assert 'reset_running_requests=false&reset_external=false' in calls[-1].full_url
    if failure in ('busy_rank1', 'missing_gauges'):
        assert not any(req.get_method() == 'POST' for req in calls)


@pytest.mark.parametrize('decode_success', [True, False])
def test_pd_resets_both_backends(decode_success):
    calls = []
    def upstream(req):
        calls.append(req)
        assert req.method == 'POST' and req.url.path == '/reset_prefix_cache'
        assert req.url.params['reset_running_requests'] == 'false'
        assert req.url.params['reset_external'] == 'false'
        return httpx.Response(200, json={'success': req.url.host == 'prefill' or decode_success})
    with TestClient(create_app('http://prefill', 'http://decode', 10,
                               httpx.MockTransport(upstream))) as client:
        result = client.post('/reset_prefix_cache')
    assert result.status_code == 200
    assert result.json()['success'] is decode_success
    assert [r.url.host for r in calls] == ['prefill', 'decode']


@pytest.mark.parametrize('queries', [0, 100])
def test_unset_initial_gauges_require_zero_counters(monkeypatch, queries):
    exposition = ('# TYPE vllm:num_requests_running gauge\n'
                  '# TYPE vllm:num_requests_waiting gauge\n'
                  '# TYPE vllm:prefix_cache_queries_total counter\n'
                  f'vllm:prefix_cache_queries_total{{engine="0"}} {queries}\n')
    def request(req, **kwargs):
        return io.BytesIO(b'{"success":true}' if req.get_method() == 'POST' else exposition.encode())
    monkeypatch.setattr(cache_reset.urllib.request, 'urlopen', request)
    report = {}
    if queries:
        with pytest.raises(RuntimeError, match='missing idle gauges'):
            cache_reset.reset_prefix_cache('http://server', {'aggre': 'http://server/metrics'}, report, idle_seconds=0)
    else:
        cache_reset.reset_prefix_cache('http://server', {'aggre': 'http://server/metrics'}, report, idle_seconds=0)
        assert report['initial_gauges_unset'] == {'aggre': True}
        assert report['status'] == 'completed'
