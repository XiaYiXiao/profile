"""Aggregate vs. real NIXL P/D serving; invoked by capture_{aggre,pd}.sh."""
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request

from eplb_config import resolve as resolve_eplb

BASE = Path(__file__).resolve().parents[1]


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.log.write(text)
        return self.console.write(text)

    def flush(self):
        self.log.flush()
        self.console.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def integer(name, minimum=1):
    value = int(os.environ[name])
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


def enabled(name):
    if os.environ[name] not in ('0', '1'):
        raise ValueError(f'{name} must be 0 or 1')
    return os.environ[name] == '1'


def configuration():
    e = os.environ
    mode = e['PROFILE_MODE']
    if mode not in ('aggre', 'pd'):
        raise ValueError('PROFILE_MODE must be aggre or pd')
    cfg = dict(mode=mode, root=str(Path(e['RESULT_DIR']).resolve()),
               concurrency=[int(x) for x in e['CONCURRENCIES'].replace(',', ' ').split()],
               ep=enabled('ENABLE_EP'), nsys=enabled('ENABLE_NSYS'), eplb=enabled('ENABLE_EPLB'),
               dry_run=enabled('DRY_RUN'), preflight_only=enabled('PREFLIGHT_ONLY'))
    if cfg['eplb'] and not cfg['ep']:
        raise ValueError('ENABLE_EPLB=1 requires ENABLE_EP=1；关闭 EP 时请同时关闭 EPLB')
    for name in ('NUM_PROMPTS', 'INPUT_LEN', 'OUTPUT_LEN', 'MAX_MODEL_LEN',
                 'MAX_NUM_BATCHED_TOKENS', 'MAX_NUM_SEQS', 'REPEATS',
                 'STARTUP_TIMEOUT', 'BENCH_TIMEOUT'):
        cfg[name.lower()] = integer(name)
    cfg['warmup_rounds'] = integer('WARMUP_ROUNDS', 0)
    cfg['seed'] = integer('SEED', 0)
    cfg['kv_block_size'] = integer('KV_BLOCK_SIZE')
    if not cfg['concurrency'] or any(x < 1 or x > cfg['num_prompts'] for x in cfg['concurrency']):
        raise ValueError('CONCURRENCIES must be between 1 and NUM_PROMPTS')
    if len(set(cfg['concurrency'])) != len(cfg['concurrency']):
        raise ValueError('Duplicate concurrency')
    if cfg['input_len'] + cfg['output_len'] > cfg['max_model_len']:
        raise ValueError('INPUT_LEN + OUTPUT_LEN exceeds MAX_MODEL_LEN')
    if cfg['max_num_batched_tokens'] < cfg['max_num_seqs']:
        raise ValueError('MAX_NUM_BATCHED_TOKENS must be >= MAX_NUM_SEQS')
    memory = float(e['GPU_MEMORY_UTILIZATION'])
    if not math.isfinite(memory) or not 0 < memory < 1:
        raise ValueError('GPU_MEMORY_UTILIZATION must be in (0, 1)')
    cfg['eplb_config'] = resolve_eplb() if cfg['eplb'] else None
    cfg['servers'] = []
    ports, all_gpus = [], []
    for role, prefix in ([('aggre', '')] if mode == 'aggre' else
                         [('prefill', 'P_'), ('decode', 'D_')]):
        gpus = [int(x) for x in e[prefix + 'GPUS'].split(',')]
        tp, dp = integer(prefix + 'TP'), integer(prefix + 'DP')
        if tp * dp != len(gpus) or any(x < 0 for x in gpus):
            raise ValueError(f'{role}: TP * DP = {tp * dp} must match '
                             f'the selected GPU count ({len(gpus)})')
        server = dict(role=role, gpus=gpus, tp=tp, dp=dp)
        for key in ('PORT', 'DP_RPC_PORT', 'MASTER_PORT'):
            server[key.lower()] = integer(prefix + key)
            ports.append(server[key.lower()])
        if mode == 'pd':
            server['nixl_port'] = integer(prefix + 'NIXL_PORT')
            ports.extend(range(server['nixl_port'], server['nixl_port'] + dp))
        server['url'] = f"http://127.0.0.1:{server['port']}"
        cfg['servers'].append(server)
        all_gpus.extend(gpus)
    if len(all_gpus) != 8 or len(set(all_gpus)) != 8:
        raise ValueError('GPU lists must contain eight distinct GPUs; P and D cannot overlap')
    if mode == 'pd':
        ports.append(integer('PROXY_PORT'))
        extra = json.loads(e.get('KV_EXTRA_CONFIG') or '{}')
        if not isinstance(extra, dict):
            raise ValueError('KV_EXTRA_CONFIG must be a JSON object')
        cfg['kv_extra_config'] = extra
        if e['KV_CACHE_LAYOUT'] != 'LBHNC':
            raise ValueError('This heterogeneous TP experiment requires KV_CACHE_LAYOUT=LBHNC')
    if len(ports) != len(set(ports)) or any(not 0 < p < 65536 for p in ports):
        raise ValueError('HTTP, RPC, master and NIXL port ranges must be valid and disjoint')
    cfg['ports'] = ports
    cfg['url'] = (f"http://127.0.0.1:{e['PROXY_PORT']}" if mode == 'pd'
                  else cfg['servers'][0]['url'])
    return cfg


def prepare_environment(cfg):
    e = os.environ
    source = e['VLLM_SOURCE']
    paths = [source]
    if e.get('DEPS_DIR') and Path(e['DEPS_DIR']).is_dir():
        paths.append(e['DEPS_DIR'])
    paths += [p for p in e.get('PYTHONPATH', '').split(os.pathsep) if p]
    e['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(paths))
    sys.path[:0] = paths[:2]
    for key, value in dict(PYTHONUNBUFFERED='1', VLLM_USE_RUST_FRONTEND='0',
                           VLLM_USE_RUST_BENCH='0', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', VLLM_ALLOW_LONG_MAX_MODEL_LEN='1').items():
        e[key] = value
    e.setdefault('VLLM_ENGINE_READY_TIMEOUT_S', e['STARTUP_TIMEOUT'])
    e.setdefault('CUDA_HOME', '/usr/local/cuda')
    e['PATH'] = e['CUDA_HOME'] + '/bin:' + e['PATH']
    e['LD_LIBRARY_PATH'] = e['CUDA_HOME'] + '/lib64:' + e.get('LD_LIBRARY_PATH', '')
    e['NO_PROXY'] = '127.0.0.1,localhost,' + e.get('NO_PROXY', '')
    e['no_proxy'] = e['NO_PROXY']
    # Shared compile caches, but temporary files and profiler output belong to this run.
    e.setdefault('VLLM_CACHE_ROOT', str(Path(e['OUTPUT_ROOT']).resolve() / '.cache/vllm'))
    e.setdefault('TRITON_CACHE_DIR', str(Path(e['OUTPUT_ROOT']).resolve() / '.cache/triton'))
    e.setdefault('XDG_CACHE_HOME', str(Path(e['OUTPUT_ROOT']).resolve() / '.cache'))
    e.setdefault('FLASHINFER_WORKSPACE_BASE', str(Path(e['OUTPUT_ROOT']).resolve()))
    # vLLM uses Unix sockets: keep the path short enough for sockaddr_un.
    e['TMPDIR'] = str(Path(e['OUTPUT_ROOT']).resolve() / '.tmp' / str(os.getpid()))
    Path(e['TMPDIR']).mkdir(parents=True, exist_ok=True)
    e['VLLM_KV_CACHE_LAYOUT'] = e.get('KV_CACHE_LAYOUT', 'LBHNC')


def server_command(cfg, server):
    e = os.environ
    cmd = [e['PYTHON_BIN'], '-m', 'vllm', 'serve', e['MODEL_PATH'],
           '--served-model-name', e['SERVED_MODEL_NAME'], '--host', '127.0.0.1',
           '--port', str(server['port']), '--tensor-parallel-size', str(server['tp']),
           '--data-parallel-size', str(server['dp']), '--data-parallel-size-local', str(server['dp']),
           '--pipeline-parallel-size', '1', '--distributed-executor-backend', 'mp',
           '--data-parallel-backend', 'mp', '--data-parallel-address', '127.0.0.1',
           '--data-parallel-rpc-port', str(server['dp_rpc_port']),
           '--master-port', str(server['master_port']),
           '--trust-remote-code', '--dtype', 'bfloat16', '--kv-cache-dtype', 'auto',
           '--block-size', e.get('KV_BLOCK_SIZE', '16'),
           '--gpu-memory-utilization', e['GPU_MEMORY_UTILIZATION'],
           '--max-model-len', e['MAX_MODEL_LEN'], '--max-num-seqs', e['MAX_NUM_SEQS'],
           '--max-num-batched-tokens', e['MAX_NUM_BATCHED_TOKENS'],
           '--enable-chunked-prefill', '--no-enable-prefix-caching', '--generation-config', 'vllm']
    if cfg['ep']:
        cmd += ['--enable-expert-parallel', '--all2all-backend', e['ALL2ALL_BACKEND'],
                '--expert-placement-strategy', e['EXPERT_PLACEMENT_STRATEGY']]
    for name in ('MOE_BACKEND', 'ATTENTION_BACKEND'):
        if e.get(name):
            cmd.extend(['--' + name.lower().replace('_', '-'), e[name]])
    if cfg['eplb']:
        cmd += ['--enable-eplb', '--eplb-config', json.dumps(cfg['eplb_config'])]
    local_env = dict(CUDA_VISIBLE_DEVICES=','.join(map(str, server['gpus'])))
    if cfg['mode'] == 'pd':
        kv = dict(kv_connector='NixlConnector', kv_role='kv_both',
                  kv_load_failure_policy='fail', kv_connector_extra_config=cfg['kv_extra_config'])
        cmd += ['--kv-transfer-config', json.dumps(kv)]
        local_env.update(VLLM_NIXL_SIDE_CHANNEL_HOST='127.0.0.1',
                         VLLM_NIXL_SIDE_CHANNEL_PORT=str(server['nixl_port']))
    cmd = ['env'] + [f'{k}={v}' for k, v in local_env.items()] + cmd
    if cfg['nsys']:
        from runtime import trace_command
        cmd += ['--profiler-config', '{"profiler":"cuda"}']
        cmd = trace_command(Path(cfg['root']) / server['role'] / 'trace') + cmd
    return cmd


def bench_command(cfg, concurrency, folder, filename):
    e = os.environ
    return [e['PYTHON_BIN'], '-m', 'vllm', 'bench', 'serve', '--backend', 'vllm',
            '--base-url', cfg['url'], '--endpoint', '/v1/completions',
            '--model', e['SERVED_MODEL_NAME'], '--tokenizer', e['MODEL_PATH'],
            '--dataset-name', 'random', '--random-input-len', e['INPUT_LEN'],
            '--random-output-len', e['OUTPUT_LEN'], '--random-range-ratio', '0',
            '--random-prefix-len', '0', '--num-prompts', e['NUM_PROMPTS'],
            '--request-rate', 'inf', '--max-concurrency', str(concurrency),
            '--seed', e['SEED'], '--ignore-eos', '--num-warmups', '0',
            '--ready-check-timeout-sec', '0',
            '--extra-body', json.dumps({'truncate_prompt_tokens': cfg['input_len']}),
            '--percentile-metrics', 'ttft,tpot,itl,e2el', '--metric-percentiles', '50,90,95,99',
            '--save-result', '--save-detailed', '--result-dir', str(folder),
            '--result-filename', filename]


def validate_result(path, cfg):
    data = json.loads(path.read_text())
    n = cfg['num_prompts']
    if data.get('completed') != n or data.get('failed', 0) or any(data.get('errors', [])):
        raise RuntimeError(f'Failed/incomplete benchmark: {path}')
    for key, target in [('input_lens', cfg['input_len']), ('output_lens', cfg['output_len'])]:
        if data.get(key) != [target] * n:
            raise RuntimeError(f'{path}: {key} must be exactly {n} x {target}')
    if (data.get('total_input_tokens') != n * cfg['input_len'] or
            data.get('total_output_tokens') != n * cfg['output_len'] or
            not math.isfinite(data['duration']) or data['duration'] <= 0):
        raise RuntimeError(f'Invalid token totals or duration: {path}')
    return data


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def model_parallelism_check(cfg):
    """Read local model JSON only; this does not validate full backend support."""
    path = Path(os.environ['MODEL_PATH']) / 'config.json'
    result = dict(config_path=str(path), status='checked', errors=[],
                  note='Attention-head divisibility only; not full vLLM/NIXL validation.')
    if not path.is_file():
        result.update(status='unavailable', errors=[f'MODEL_PATH/config.json is missing: {path}'])
        return result
    try:
        model = json.loads(path.read_text())
        model = model.get('text_config', model)
        if not isinstance(model, dict):
            raise ValueError('Model text configuration must be a JSON object')
    except (ValueError, AttributeError) as exc:
        result.update(status='invalid', errors=[f'Invalid model config: {exc}'])
        return result
    heads = {key: model[key] for key in ('num_attention_heads', 'swa_num_attention_heads')
             if isinstance(model.get(key), int) and model[key] > 0}
    result['attention_heads'] = heads
    for server in cfg['servers']:
        for key, count in heads.items():
            if count % server['tp']:
                result['errors'].append(
                    f"{server['role']}: {key}={count} 不能被 TP={server['tp']} 整除；"
                    f"DP={server['dp']} 不改变 TP 的整除要求。"
                    "请使用兼容的模型或显式调整 TP/DP；脚本不会自动修改布局。")
    if result['errors']:
        result['status'] = 'incompatible'
    elif not heads:
        result['status'] = 'not_checked'
    return result


def preflight(cfg):
    check = model_parallelism_check(cfg)
    if check['errors']:
        raise RuntimeError('\n'.join(check['errors']))
    import vllm
    result = dict(vllm_version=vllm.__version__, vllm_module=vllm.__file__)
    if cfg['mode'] == 'pd':
        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
        if NixlWrapper is None:
            raise RuntimeError('NIXL is unavailable; install the version in VLLM_SOURCE/requirements/kv_connectors.txt into DEPS_DIR')
        backends = cfg['kv_extra_config'].get('backends', ['UCX'])
        agent = NixlWrapper(f'pd-preflight-{os.getpid()}', nixl_agent_config(backends=backends))
        result['nixl_version'] = importlib.metadata.version('nixl')
        result['nixl_backends'] = backends
        del agent
    if cfg['nsys']:
        result['nsys_version'] = subprocess.check_output([os.environ['NSYS_BIN'], '--version'], text=True).strip()
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    devices = {int(line.split(',')[0]): line.split(',')[1].strip() for line in inventory.splitlines()}
    gpus = {g for s in cfg['servers'] for g in s['gpus']}
    if not gpus <= devices.keys():
        raise RuntimeError('Requested GPUs are unavailable')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    busy = [line for line in processes.splitlines() if line.split(',')[0].strip() in {devices[g] for g in gpus}]
    if busy:
        raise RuntimeError(f'GPUs are busy; existing processes are left untouched: {busy}')
    for port in cfg['ports']:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    result['gpu_inventory'] = inventory
    return result


def profile_boundary(cfg, action):
    for server in cfg['servers']:
        req = urllib.request.Request(server['url'] + f'/{action}_profile', data=b'{}',
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            response.read()


def collect(cfg, workloads):
    from runtime import launch, server_children, snapshot, terminate_all, wait_jobs
    root = Path(cfg['root'])
    servers, jobs, groups = [], [], set()
    tracing = False

    def check_servers():
        for proc in servers:
            for child in server_children(proc):
                try:
                    groups.add(os.getpgid(child))
                except ProcessLookupError:
                    pass
            if proc.poll() is not None:
                raise RuntimeError(f'Service exited: PID={proc.pid}; inspect server.log/proxy.log')

    def wait_ready(url):
        deadline = time.monotonic() + cfg['startup_timeout']
        while True:
            check_servers()
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f'Waiting for {url}')
            time.sleep(2)

    summaries = []
    try:
        for server in cfg['servers']:
            servers.append(launch(server['command'], root / server['role'] / 'server.log'))
        for server in cfg['servers']:
            wait_ready(server['url'] + '/health')
        if cfg['mode'] == 'pd':
            servers.append(launch(cfg['proxy_command'], root / 'proxy.log'))
            wait_ready(cfg['url'] + '/health')
        for work in workloads:
            if cfg['nsys'] and work['phase'] == 'eval' and not tracing:
                profile_boundary(cfg, 'start')
                tracing = True
            check_servers()
            folder = Path(work['folder'])
            print(f"{cfg['mode']} {work['phase']} concurrency={work['concurrency']} round={work['round']}", flush=True)
            for server in cfg['servers']:
                snapshot(server['url'], folder / f"{work['stem']}.{server['role']}.before.metrics")
            job = launch(work['command'], folder / (work['stem'] + '.log'))
            jobs.append(job)
            wait_jobs([job], cfg['bench_timeout'], servers)
            data = validate_result(folder / (work['stem'] + '.json'), cfg)
            for server in cfg['servers']:
                snapshot(server['url'], folder / f"{work['stem']}.{server['role']}.after.metrics")
            if work['phase'] == 'eval':
                row = dict(mode=cfg['mode'], enable_ep=int(cfg['ep']),
                           concurrency=work['concurrency'], round=work['round'],
                           completed=data['completed'], duration_s=data['duration'],
                           input_tokens_s=data['total_input_tokens'] / data['duration'],
                           output_tokens_s=data['total_output_tokens'] / data['duration'])
                for key in ('mean_ttft_ms', 'p50_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms',
                            'p50_tpot_ms', 'p99_tpot_ms', 'p99_itl_ms', 'p99_e2el_ms'):
                    row[key] = data.get(key)
                summaries.append(row)
                with (root / 'summary.csv').open('w') as out:
                    writer = csv.DictWriter(out, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(summaries)
            print(f"validated: input={data['total_input_tokens']}, output={data['total_output_tokens']}, "
                  f"output_tokens/s={data['total_output_tokens'] / data['duration']:.3f}", flush=True)
        if tracing:
            profile_boundary(cfg, 'stop')
            tracing = False
            # capture-range-end=stop stops tracing, not the application. Ask the
            # owned vLLM applications to exit so Nsight can finalize the reports.
            for proc in servers[:len(cfg['servers'])]:
                targets = server_children(proc) or ([proc.pid] if proc.poll() is None else [])
                for pid in targets:
                    try:
                        os.kill(pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
            wait_jobs(servers[:len(cfg['servers'])], 300, check_returncodes=False)
    finally:
        if tracing:
            try:
                profile_boundary(cfg, 'stop')
            except Exception as exc:
                print(f'Failed to stop profiler: {exc}', file=sys.stderr)
        terminate_all(jobs + servers, groups)
    if cfg['nsys']:
        for server in cfg['servers']:
            folder = root / server['role']
            reports = list(folder.glob('trace*.nsys-rep'))
            if len(reports) != 1:
                raise RuntimeError(f'Expected one Nsight report in {folder}')
            with (folder / 'export.log').open('w') as out:
                subprocess.run([os.environ['NSYS_BIN'], 'export', '--type=sqlite',
                                '--output=' + str(folder / 'trace.sqlite'), str(reports[0])],
                               stdout=out, stderr=subprocess.STDOUT, check=True, timeout=300)
            with sqlite3.connect(folder / 'trace.sqlite') as db:
                workers = db.execute('SELECT COUNT(DISTINCT globalPid) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
            if workers < len(server['gpus']):
                raise RuntimeError(f'Incomplete worker traces in {folder}: {workers}')


def main():
    cfg = configuration()
    root = Path(cfg['root'])
    # Never overwrite a previous run, including dry runs and failed runs.
    root.mkdir(parents=True, exist_ok=False)
    log = (root / 'collector.log').open('w', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    prepare_environment(cfg)
    # Import shared process/Nsight helpers only after runtime environment setup.
    for server in cfg['servers']:
        (root / server['role']).mkdir()
        server['command'] = server_command(cfg, server)
        (root / server['role'] / 'server.command.txt').write_text(shlex.join(server['command']) + '\n')
    if cfg['mode'] == 'pd':
        cfg['proxy_command'] = [os.environ['PYTHON_BIN'], str(BASE / 'code/pd_proxy.py'),
                                '--port', os.environ['PROXY_PORT'],
                                '--prefill-url', cfg['servers'][0]['url'],
                                '--decode-url', cfg['servers'][1]['url'],
                                '--timeout', os.environ['BENCH_TIMEOUT']]
        (root / 'proxy.command.txt').write_text(shlex.join(cfg['proxy_command']) + '\n')
    workloads = []
    for phase, rounds in [('warmup', cfg['warmup_rounds']), ('eval', cfg['repeats'])]:
        for concurrency in cfg['concurrency']:
            folder = root / 'benchmarks' / f'concurrency{concurrency}'
            folder.mkdir(parents=True, exist_ok=True)
            for number in range(1, rounds + 1):
                stem = f'{phase}-round{number:03d}'
                command = bench_command(cfg, concurrency, folder, stem + '.json')
                workloads.append(dict(phase=phase, concurrency=concurrency, round=number,
                                      stem=stem, folder=str(folder), command=command))
                (folder / (stem + '.command.txt')).write_text(shlex.join(command) + '\n')
    cfg['workloads'] = workloads
    cfg['model_parallelism_check'] = model_parallelism_check(cfg)
    if os.environ.get('CAPTURE_ENTRYPOINT'):
        cfg['capture_entrypoint'] = str(Path(os.environ['CAPTURE_ENTRYPOINT']).resolve())
    cfg['runtime_environment'] = {k: os.environ[k] for k in (
        'PYTHONPATH', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'TMPDIR', 'VLLM_KV_CACHE_LAYOUT',
        'XDG_CACHE_HOME', 'FLASHINFER_WORKSPACE_BASE',
        'VLLM_ENGINE_READY_TIMEOUT_S', 'VLLM_USE_RUST_FRONTEND', 'VLLM_USE_RUST_BENCH',
        'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'UCX_TLS', 'UCX_NET_DEVICES') if k in os.environ}
    write_json(root / 'config.json', cfg)
    sources = root / 'sources'
    sources.mkdir()
    source_paths = [BASE / f"capture_{cfg['mode']}.sh", Path(__file__),
                    Path(__file__).with_name('pd_proxy.py'), Path(__file__).with_name('eplb_config.py'),
                    Path(__file__).with_name('runtime.py')]
    if cfg.get('capture_entrypoint'):
        source_paths.append(Path(cfg['capture_entrypoint']))
    for path in dict.fromkeys(source_paths):
        shutil.copy2(path, sources / path.name)
    print(f"Result directory: {root}", flush=True)
    status = dict(status='running', started_unix=time.time())
    write_json(root / 'status.json', status)
    try:
        if cfg['dry_run']:
            for error in cfg['model_parallelism_check']['errors']:
                print(f'WARNING: 仅生成命令，当前配置不能正式启动：{error}', flush=True)
            for server in cfg['servers']:
                print(shlex.join(server['command']))
            if cfg['mode'] == 'pd':
                print(shlex.join(cfg['proxy_command']))
            for work in workloads:
                print(shlex.join(work['command']))
            status['status'] = 'dry_run'
        else:
            write_json(root / 'runtime.json', preflight(cfg))
            if cfg['preflight_only']:
                status['status'] = 'preflight_passed'
            else:
                collect(cfg, workloads)
                status['status'] = 'complete'
    except BaseException as exc:
        status.update(status='failed', error=str(exc))
        raise
    finally:
        status['finished_unix'] = time.time()
        write_json(root / 'status.json', status)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
