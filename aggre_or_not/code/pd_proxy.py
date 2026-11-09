"""Streaming NIXL pull proxy for local benchmarks, failing closed on missing KV.

Protocol follows vLLM tests/v1/kv_connector/nixl_integration/toy_proxy_server.py.
The prefill token is discarded; D receives the original prompt and output budget,
so the user receives exactly OUTPUT_LEN tokens, not OUTPUT_LEN + 1.
"""
import argparse
from contextlib import asynccontextmanager
import logging
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger('pd_proxy')


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


def create_app(prefill_url, decode_url, timeout):
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=timeout, trust_env=False,
                                     limits=httpx.Limits(max_connections=None)) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get('/health')
    async def health(request: Request):
        try:
            for url in (prefill_url, decode_url):
                response = await request.app.state.client.get(url + '/health', timeout=2)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(503, 'P/D service is not healthy') from exc
        return {'status': 'ok'}

    @app.post('/v1/completions')
    async def completions(request: Request):
        original = await request.json()
        client = request.app.state.client
        request_id = str(uuid.uuid4())
        headers = {'X-Request-Id': request_id}
        try:
            pref = await client.post(prefill_url + '/v1/completions',
                                     json=prefill_body(original), headers=headers)
            pref.raise_for_status()
            body = dict(original, kv_transfer_params=transfer_params(pref.json()))
            upstream = await client.send(client.build_request(
                'POST', decode_url + '/v1/completions', json=body, headers=headers), stream=True)
            if upstream.status_code != 200:
                error = await upstream.aread()
                await upstream.aclose()
                return Response(error, status_code=upstream.status_code, media_type='application/json')
        except httpx.HTTPError as exc:
            logger.exception('P/D request %s failed', request_id)
            raise HTTPException(502, 'P/D upstream request failed') from exc

        async def stream():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(stream(), media_type=(
            'text/event-stream' if original.get('stream') else 'application/json'))

    return app


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--prefill-url', required=True)
    parser.add_argument('--decode-url', required=True)
    parser.add_argument('--timeout', type=float, default=7200)
    args = parser.parse_args()
    uvicorn.run(create_app(args.prefill_url, args.decode_url, args.timeout),
                host='127.0.0.1', port=args.port)
