"""Serve persistently using the capture scripts' vLLM command builder (no workloads)."""
import json
import importlib.util
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import sys
import time
import urllib.request
from datetime import datetime
from uuid import uuid4

BASE = Path(__file__).resolve().parents[1]
# Share command construction and process ownership with the reference experiments.
# Importing collect_pd does not invoke its collection entry point.
sys.path.insert(0, str(BASE.parent / 'aggre_or_not/code'))
from collect_pd import (enabled, integer, model_parallelism_check, preflight,
                        prepare_environment, resolve_eplb, server_command)
from runtime import launch, terminate_all, server_children


def resolve_speculative():
    """Use exactly the same JSON merge/validation as decode/capture.sh."""
    source = BASE.parent / 'profile/decode/speculative_config.py'
    spec = importlib.util.spec_from_file_location('decode_speculative_config', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve()


def configuration():
    e = os.environ
    mode = e['DEPLOY_MODE']
    if mode not in ('aggre', 'pd', 'pd_6p2d'):
        raise ValueError('Invalid DEPLOY_MODE')
    cfg = dict(mode='aggre' if mode == 'aggre' else 'pd', deployment_mode=mode,
               ep=enabled('ENABLE_EP'), eplb=enabled('ENABLE_EPLB'), nsys=False,
               prefix_caching=enabled('ENABLE_PREFIX_CACHING'),
               startup_timeout=integer('STARTUP_TIMEOUT'))
    cfg['speculative_config'] = resolve_speculative()
    for key in ('MAX_MODEL_LEN', 'MAX_NUM_BATCHED_TOKENS', 'MAX_NUM_SEQS', 'KV_BLOCK_SIZE', 'REQUEST_TIMEOUT'):
        cfg[key.lower()] = integer(key)
    if cfg['max_num_batched_tokens'] < cfg['max_num_seqs']:
        raise ValueError('MAX_NUM_BATCHED_TOKENS must be >= MAX_NUM_SEQS')
    memory = float(e['GPU_MEMORY_UTILIZATION'])
    if not math.isfinite(memory) or not 0 < memory < 1:
        raise ValueError('GPU_MEMORY_UTILIZATION must be in (0, 1)')
    cfg['servers'], ports, devices = [], [], []
    roles = [('aggre', '')] if mode == 'aggre' else [('prefill', 'P_'), ('decode', 'D_')]
    for role, prefix in roles:
        gpus = [int(g) for g in e[prefix + 'GPUS'].split(',')]
        tp, dp = integer(prefix + 'TP'), integer(prefix + 'DP')
        if tp * dp != len(gpus) or any(g < 0 for g in gpus):
            raise ValueError(f'{role}: TP * DP must equal GPU count')
        # Aggregate uses the unprefixed switches; PD exposes each end separately.
        ep = enabled(prefix + 'ENABLE_EP')
        eplb = enabled(prefix + 'ENABLE_EPLB')
        if eplb and not ep:
            raise ValueError(f'{role}: {prefix}ENABLE_EPLB requires {prefix}ENABLE_EP=1')
        server = dict(role=role, gpus=gpus, tp=tp, dp=dp, ep=ep, eplb=eplb,
                      prefix_caching=cfg['prefix_caching'],
                      speculative_config=cfg['speculative_config'])
        for key in ('PORT', 'DP_RPC_PORT', 'MASTER_PORT'):
            server[key.lower()] = integer(prefix + key)
            ports.append(server[key.lower()])
        if mode != 'aggre':
            server['nixl_port'] = integer(prefix + 'NIXL_PORT')
            ports.extend(range(server['nixl_port'], server['nixl_port'] + dp))
        server['url'] = f"http://127.0.0.1:{server['port']}"
        cfg['servers'].append(server)
        devices.extend(gpus)
    if len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError('Select eight distinct GPUs; P and D cannot overlap')
    # Top-level switches summarize whether any end enables the feature;
    # servers[] records the exact, possibly asymmetric, configuration.
    cfg['ep'] = any(s['ep'] for s in cfg['servers'])
    cfg['eplb'] = any(s['eplb'] for s in cfg['servers'])
    cfg['eplb_config'] = resolve_eplb() if cfg['eplb'] else None
    port = cfg['servers'][0]['port'] if mode == 'aggre' else integer('PROXY_PORT')
    if mode != 'aggre':
        ports.append(port)
        cfg['kv_extra_config'] = json.loads(e.get('KV_EXTRA_CONFIG') or '{}')
        if not isinstance(cfg['kv_extra_config'], dict):
            raise ValueError('KV_EXTRA_CONFIG must be a JSON object')
        if e['KV_CACHE_LAYOUT'] != 'LBHNC':
            raise ValueError('Heterogeneous TP requires KV_CACHE_LAYOUT=LBHNC')
    if len(ports) != len(set(ports)) or any(p < 1 or p > 65535 for p in ports):
        raise ValueError('HTTP, RPC, master and NIXL port ranges must be valid and disjoint')
    cfg.update(ports=ports, port=port, host=e['HOST'], url=f'http://127.0.0.1:{port}',
               model=e['SERVED_MODEL_NAME'], model_path=e['MODEL_PATH'])
    cfg['environment'] = {key: value for key, value in e.items() if
                          key.startswith(('P_', 'D_', 'KV_', 'UCX_', 'EPLB_', 'SPECULATIVE_')) or key in (
                              'MODEL_PATH', 'SERVED_MODEL_NAME', 'PYTHON_BIN', 'VLLM_SOURCE',
                              'DEPS_DIR', 'ENABLE_EP', 'ENABLE_EPLB', 'ENABLE_PREFIX_CACHING', 'ALL2ALL_BACKEND',
                              'EXPERT_PLACEMENT_STRATEGY', 'GPU_MEMORY_UTILIZATION',
                              'MOE_BACKEND', 'ATTENTION_BACKEND', 'EXPERIMENT_NAME',
                              'NUM_SPECULATIVE_TOKENS', 'DRAFT_MODEL_PATH', 'TP', 'DP', 'GPUS')}
    cfg['deployment_id'] = datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '_' + uuid4().hex
    cfg['root'] = str(Path(e['OUTPUT_ROOT']).resolve() / 'deployments' / mode / cfg['deployment_id'])
    for server in cfg['servers']:
        server['eplb_config'] = cfg['eplb_config'] if server['eplb'] else None
        server['command'] = server_command(dict(cfg, ep=server['ep'], eplb=server['eplb']), server)
        if server['prefix_caching']:
            index = server['command'].index('--no-enable-prefix-caching')
            server['command'][index] = '--enable-prefix-caching'
        if server['speculative_config'] is not None:
            server['command'] += ['--speculative-config', json.dumps(server['speculative_config'])]
        if mode == 'aggre':
            i = server['command'].index('--host')
            server['command'][i + 1] = cfg['host']
    if mode != 'aggre':
        cfg['proxy_command'] = [e['PYTHON_BIN'], str(BASE / 'code/pd_proxy.py'),
                                '--host', cfg['host'], '--port', str(port),
                                '--prefill-url', cfg['servers'][0]['url'],
                                '--decode-url', cfg['servers'][1]['url'],
                                '--timeout', str(cfg['request_timeout'])]
    return cfg


def main():
    cfg = configuration()
    if enabled('DRY_RUN'):
        cfg['model_parallelism_check'] = model_parallelism_check(cfg)
        print(json.dumps(cfg, indent=2))
        return
    root = Path(cfg['root'])
    root.mkdir(parents=True, exist_ok=False)
    cfg['status'] = 'starting'
    jobs, groups = [], set()

    def save():
        temp = root / 'deployment.json.tmp'
        temp.write_text(json.dumps(cfg, indent=2))
        temp.replace(root / 'deployment.json')

    def stopped(signum, frame):
        raise KeyboardInterrupt

    def check():
        for proc in jobs:
            for child in server_children(proc):
                try:
                    groups.add(os.getpgid(child))
                except ProcessLookupError:
                    pass
            if proc.poll() is not None:
                raise RuntimeError(f'Service exited: PID={proc.pid}; inspect {root}')

    def ready(url):
        deadline = time.monotonic() + cfg['startup_timeout']
        while time.monotonic() < deadline:
            check()
            try:
                with urllib.request.urlopen(url + '/health', timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            time.sleep(2)
        raise TimeoutError(f'Service startup timed out: {url}')

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    save()
    try:
        prepare_environment(cfg)
        # Bind the public interface too, rather than checking only loopback.
        with socket.socket() as sock:
            sock.bind((cfg['host'], cfg['port']))
        cfg['preflight'] = preflight(cfg)
        if enabled('PREFLIGHT_ONLY'):
            cfg['status'] = 'preflight_passed'
            return
        for server in cfg['servers']:
            folder = root / server['role']
            folder.mkdir()
            (folder / 'command.txt').write_text(shlex.join(server['command']) + '\n')
            jobs.append(launch(server['command'], folder / 'server.log'))
        for server in cfg['servers']:
            # Non-wildcard aggregate binding may not include loopback.
            url = server['url']
            if cfg['mode'] == 'aggre' and cfg['host'] not in ('0.0.0.0', '127.0.0.1'):
                url = f"http://{cfg['host']}:{cfg['port']}"
            ready(url)
        if cfg['mode'] == 'pd':
            (root / 'proxy.command.txt').write_text(shlex.join(cfg['proxy_command']) + '\n')
            jobs.append(launch(cfg['proxy_command'], root / 'proxy.log'))
        cfg['url'] = f"http://{'127.0.0.1' if cfg['host'] == '0.0.0.0' else cfg['host']}:{cfg['port']}"
        ready(cfg['url'])
        cfg.update(status='ready', pids=[p.pid for p in jobs])
        save()
        print(f"READY {cfg['url']} (bind {cfg['host']}); deployment: {root / 'deployment.json'}", flush=True)
        while True:
            check()
            time.sleep(1)
    except KeyboardInterrupt:
        cfg['status'] = 'stopped'
    except BaseException as exc:
        cfg.update(status='failed', error=str(exc))
        raise
    finally:
        terminate_all(jobs, groups)
        save()
        print(f"Deployment {cfg['status']}: {root}", flush=True)


if __name__ == '__main__':
    main()
