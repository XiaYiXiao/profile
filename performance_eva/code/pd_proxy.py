"""Streaming NIXL/Mooncake proxy for local benchmarks.

Protocols follow vLLM's NIXL and Mooncake example proxies.
The prefill token is discarded; D receives the original prompt and output budget,
so the user receives exactly OUTPUT_LEN tokens, not OUTPUT_LEN + 1.
"""
import argparse
from contextlib import asynccontextmanager
import itertools
import json
import logging
from pathlib import Path
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger('pd_proxy')


class ContentTracker:
    """Recognize content SSEs, including split frames; role/usage events are not tokens."""
    def __init__(self):
        self.buffer = b''
        self.content = False
        self.done = False
        self.error = False

    def feed(self, chunk):
        self.buffer += chunk
        self.buffer = self.buffer.replace(b'\r\n', b'\n')
        while b'\n\n' in self.buffer:
            frame, self.buffer = self.buffer.split(b'\n\n', 1)
            data = b'\n'.join(line[5:].lstrip() for line in frame.splitlines() if line.startswith(b'data:'))
            if data.strip() == b'[DONE]':
                self.done = True
                continue
            if not data:
                continue
            try:
                event = json.loads(data)
                if not isinstance(event, dict):
                    continue
                self.error |= bool(event.get('error'))
                for choice in event.get('choices') or []:
                    delta = choice.get('delta') or {}
                    if choice.get('text') or any(delta.get(key) for key in (
                            'content', 'reasoning_content', 'reasoning', 'tool_calls')):
                        self.content = True
            except (ValueError, TypeError, AttributeError):
                self.error = True
        if len(self.buffer) > 4 * 1024 * 1024:
            # Bound monitoring memory; forwarding is unaffected.
            self.buffer = b''
            self.error = True


def prefill_body(original):
    body = dict(original)
    body.update(stream=False, max_tokens=1, kv_transfer_params={
        'do_remote_decode': True, 'do_remote_prefill': False,
        'remote_engine_id': None, 'remote_block_ids': None,
        'remote_host': None, 'remote_port': None})
    for key in ('stream_options', 'min_tokens', 'min_completion_tokens'):
        body.pop(key, None)
    if 'max_completion_tokens' in body:
        body['max_completion_tokens'] = 1
    return body


def transfer_params(response):
    params = response.get('kv_transfer_params')
    if (not isinstance(params, dict) or not params.get('do_remote_prefill') or
            not params.get('remote_engine_id') or not params.get('remote_block_ids') or
            not params.get('remote_host') or not params.get('remote_port')):
        raise HTTPException(502, 'Prefill returned no usable NIXL KV metadata; refusing decode recomputation')
    return params


async def mooncake_routes(client, bootstrap_url, dp_size, tp_size):
    response = await client.get(bootstrap_url.rstrip('/') + '/query')
    response.raise_for_status()
    try:
        entries = response.json()
        if not isinstance(entries, dict) or set(entries) != {str(i) for i in range(dp_size)}:
            raise ValueError('incomplete DP registry')
        routes = []
        for rank in range(dp_size):
            entry = entries[str(rank)]
            engine_id, workers = entry['engine_id'], entry['worker_addr']
            if not isinstance(engine_id, str) or not engine_id:
                raise ValueError('missing engine_id')
            if not isinstance(workers, dict) or set(workers) != {str(i) for i in range(tp_size)}:
                raise ValueError('incomplete TP registry')
            for stages in workers.values():
                if (not isinstance(stages, dict) or set(stages) != {'0'} or
                        not isinstance(stages['0'], str) or not stages['0']):
                    raise ValueError('missing PP0 worker address')
            routes.append((rank, engine_id))
        if len({engine_id for _, engine_id in routes}) != dp_size:
            raise ValueError('duplicate DP engine_id')
        return routes
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f'Invalid Mooncake bootstrap registry: {exc}') from exc


def create_app(prefill_url, decode_url, timeout, transport=None, *,
               kv_connector='NixlConnector', prefill_bootstrap_url=None,
               prefill_dp_size=1, prefill_tp_size=1, trace_path=None):
    if kv_connector not in ('NixlConnector', 'MooncakeConnector'):
        raise ValueError('Unsupported KV connector')
    use_mooncake = kv_connector == 'MooncakeConnector'
    if use_mooncake and (not prefill_bootstrap_url or prefill_dp_size < 1 or prefill_tp_size < 1):
        raise ValueError('Mooncake requires a bootstrap URL and positive P DP/TP sizes')
    registry = CollectorRegistry()
    buckets = (.001, .01, .05, .1, .25, .5, 1, 2, 5, 10, 30, 60, 120, 300, 600)
    stages = {name: Histogram('eva_pd_' + name, help_text, buckets=buckets, registry=registry)
              for name, help_text in {
                  'prefill_http_seconds': 'P HTTP round trip, including P queue and compute.',
                  'decode_headers_seconds': 'D dispatch to HTTP headers, not KV transfer duration.',
                  'decode_first_content_seconds': 'D dispatch to first content SSE, including KV/queue/network.',
                  'ttft_seconds': 'Proxy admission to first content SSE; no client network.',
                  'request_seconds': 'Proxy request lifetime, including interrupted/failed requests.',
              }.items()}
    outcomes = Counter('eva_pd_requests', 'Requests by proxy outcome.', ['outcome'], registry=registry)
    for outcome in ('success', 'prefill_error', 'metadata_error', 'decode_error', 'stream_error', 'cancelled'):
        outcomes.labels(outcome)
    inflight = Gauge('eva_pd_inflight', 'Requests active at the proxy.', registry=registry)

    @asynccontextmanager
    async def lifespan(app):
        app.state.trace = None
        if trace_path:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            app.state.trace = open(trace_path, 'a', buffering=1)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False, transport=transport,
                                     limits=httpx.Limits(max_connections=None)) as client:
            app.state.client = client
            if use_mooncake:
                routes = await mooncake_routes(client, prefill_bootstrap_url,
                                               prefill_dp_size, prefill_tp_size)
                app.state.prefill_routes = itertools.cycle(routes)
            try:
                yield
            finally:
                if app.state.trace:
                    app.state.trace.close()

    app = FastAPI(lifespan=lifespan)

    @app.get('/metrics')
    async def metrics():
        return Response(generate_latest(registry), headers={'Content-Type': CONTENT_TYPE_LATEST})

    @app.get('/metrics/{role}')
    async def backend_metrics(role: str, request: Request):
        if role not in ('prefill', 'decode'):
            raise HTTPException(404, 'Unknown backend')
        url = prefill_url if role == 'prefill' else decode_url
        headers = {'Authorization': request.headers['authorization']} if 'authorization' in request.headers else {}
        try:
            result = await request.app.state.client.get(url + '/metrics', timeout=2, headers=headers)
            return Response(result.content, status_code=result.status_code,
                            headers={'Content-Type': result.headers.get('content-type', 'text/plain')})
        except httpx.HTTPError as exc:
            raise HTTPException(502, 'Backend metrics unavailable') from exc

    @app.get('/health')
    async def health(request: Request):
        try:
            for url in (prefill_url, decode_url):
                response = await request.app.state.client.get(url + '/health', timeout=2)
                response.raise_for_status()
            if use_mooncake:
                await mooncake_routes(request.app.state.client, prefill_bootstrap_url,
                                      prefill_dp_size, prefill_tp_size)
        except (httpx.HTTPError, RuntimeError) as exc:
            raise HTTPException(503, 'P/D service is not healthy') from exc
        return {'status': 'ok'}

    @app.post('/reset_prefix_cache')
    async def reset_prefix_cache(request: Request):
        headers = {'Authorization': request.headers['authorization']} if 'authorization' in request.headers else {}
        if inflight._value.get() != 0:
            raise HTTPException(409, 'Cannot reset prefix cache while proxy requests are active')
        results = {}
        for role, url in (('prefill', prefill_url), ('decode', decode_url)):
            try:
                response = await request.app.state.client.post(
                    url + '/reset_prefix_cache',
                    params={'reset_running_requests': 'false', 'reset_external': 'false'},
                    headers=headers, timeout=30)
                response.raise_for_status()
                results[role] = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise HTTPException(502, f'{role} prefix cache reset failed: {exc}') from exc
        return {'success': all(isinstance(r, dict) and r.get('success') is True
                               for r in results.values()), 'backends': results}

    @app.get('/v1/models')
    async def models(request: Request):
        try:
            response = await request.app.state.client.get(decode_url + '/v1/models')
            return Response(response.content, status_code=response.status_code,
                            media_type='application/json')
        except httpx.HTTPError as exc:
            raise HTTPException(502, 'Decode service unavailable') from exc

    @app.post('/v1/chat/completions')
    @app.post('/v1/completions')
    async def completions(request: Request):
        original = await request.json()
        client = request.app.state.client
        if not isinstance(original, dict):
            raise HTTPException(400, 'Expected a JSON object')
        endpoint = request.url.path
        request_id = request.headers.get('x-request-id') or str(uuid.uuid4())
        headers = {key: request.headers[key] for key in
                   ('session-id', 'x-smg-routing-key', 'request-id', 'authorization')
                   if key in request.headers}
        headers['x-request-id'] = request_id
        started = time.perf_counter()
        trace = dict(trace_id=uuid.uuid4().hex, request_id=request_id, endpoint=endpoint,
                     started_unix=time.time(), stream=bool(original.get('stream')),
                     prefill_http_seconds=None, decode_headers_seconds=None,
                     decode_first_content_seconds=None, ttft_seconds=None)
        inflight.inc()
        finished = False

        def finish(outcome):
            nonlocal finished
            if finished:
                return
            finished = True
            inflight.dec()
            trace.update(outcome=outcome, finished_unix=time.time(),
                         request_seconds=time.perf_counter() - started)
            outcomes.labels(outcome).inc()
            for name, metric in stages.items():
                if trace.get(name) is not None:
                    metric.observe(trace[name])
            if request.app.state.trace:
                try:
                    request.app.state.trace.write(json.dumps(trace, ensure_ascii=False) + '\n')
                except OSError:
                    logger.exception('Could not write proxy request metrics')

        phase = 'prefill'
        upstream = None
        try:
            p_body, p_headers = prefill_body(original), dict(headers)
            if use_mooncake:
                dp_rank, engine_id = next(request.app.state.prefill_routes)
                # Transfer IDs must stay unique even if the client reuses its request ID.
                transfer_id = 'xfer-' + uuid.uuid4().hex
                p_body['kv_transfer_params'] = dict(do_remote_decode=True,
                                                    do_remote_prefill=False,
                                                    transfer_id=transfer_id)
                p_headers['x-data-parallel-rank'] = str(dp_rank)
                trace['prefill_dp_rank'] = dp_rank
            p_start = time.perf_counter()
            pref = await client.post(prefill_url + endpoint,
                                     json=p_body, headers=p_headers)
            trace['prefill_http_seconds'] = time.perf_counter() - p_start
            trace['prefill_status'] = pref.status_code
            pref.raise_for_status()
            phase = 'metadata'
            p_result = pref.json()
            p_usage = p_result.get('usage') or {}
            trace['prefill_prompt_tokens'] = p_usage.get('prompt_tokens')
            trace['prefill_cached_tokens'] = (p_usage.get('prompt_tokens_details') or {}).get('cached_tokens')
            if use_mooncake:
                params = dict(do_remote_decode=False, do_remote_prefill=True,
                              remote_bootstrap_addr=prefill_bootstrap_url,
                              remote_engine_id=engine_id, transfer_id=transfer_id)
            else:
                params = transfer_params(p_result)
            body = dict(original, kv_transfer_params=params)
            phase = 'decode'
            d_start = time.perf_counter()
            upstream = await client.send(client.build_request(
                'POST', decode_url + endpoint, json=body, headers=headers), stream=True)
            trace['decode_headers_seconds'] = time.perf_counter() - d_start
            trace['decode_status'] = upstream.status_code
            if upstream.status_code != 200:
                error = await upstream.aread()
                await upstream.aclose()
                finish('decode_error')
                return Response(error, status_code=upstream.status_code, media_type='application/json')
        except httpx.HTTPError as exc:
            finish(phase + '_error')
            if upstream:
                await upstream.aclose()
            logger.exception('P/D request %s failed', request_id)
            raise HTTPException(502, 'P/D upstream request failed') from exc
        except HTTPException:
            finish('metadata_error')
            raise
        except (ValueError, TypeError, AttributeError) as exc:
            finish('metadata_error')
            raise HTTPException(502, 'Malformed prefill metadata') from exc
        except BaseException:
            finish('cancelled')
            if upstream:
                await upstream.aclose()
            raise

        async def stream():
            outcome = 'cancelled'
            tracker = ContentTracker()
            try:
                async for chunk in upstream.aiter_bytes():
                    if original.get('stream'):
                        tracker.feed(chunk)
                        if tracker.content and trace['ttft_seconds'] is None:
                            now = time.perf_counter()
                            trace['ttft_seconds'] = now - started
                            trace['decode_first_content_seconds'] = now - d_start
                    yield chunk
                outcome = ('stream_error' if original.get('stream') and (tracker.error or not tracker.done)
                           else 'success')
            except Exception:
                outcome = 'stream_error'
                raise
            finally:
                try:
                    await upstream.aclose()
                finally:
                    finish(outcome)

        return StreamingResponse(stream(), media_type=(
            'text/event-stream' if original.get('stream') else 'application/json'))

    return app


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--prefill-url', required=True)
    parser.add_argument('--decode-url', required=True)
    parser.add_argument('--timeout', type=float, default=7200)
    parser.add_argument('--kv-connector', choices=['NixlConnector', 'MooncakeConnector'],
                        default='NixlConnector')
    parser.add_argument('--prefill-bootstrap-url')
    parser.add_argument('--prefill-dp-size', type=int, default=1)
    parser.add_argument('--prefill-tp-size', type=int, default=1)
    parser.add_argument('--trace-path', help='Append per-request stage timings (no prompts or generated text)')
    args = parser.parse_args()
    uvicorn.run(create_app(args.prefill_url, args.decode_url, args.timeout,
                          kv_connector=args.kv_connector,
                          prefill_bootstrap_url=args.prefill_bootstrap_url,
                          prefill_dp_size=args.prefill_dp_size,
                          prefill_tp_size=args.prefill_tp_size, trace_path=args.trace_path),
                host=args.host, port=args.port)
