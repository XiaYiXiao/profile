"""Run concurrent EP0/EP1 decode loads, optionally with Nsight traces (stdlib)."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / '../ep_profile/analysis/code'))
# Reuse the prefill process ownership, Nsight export and integrity checks.
from collect import (check_gpu_availability, export_check, launch, run_smokes,
                     server_children, smoke_command, snapshot, terminate_all,
                     trace_command, vllm_runtime_info, wait_jobs)
from decode_state import valid_result
from speculative_config import resolve

ROOT = Path(os.environ.get('RESULT_DIR', '/tmp/mimo-decode-profile')).resolve()
PYTHON = os.environ.get('PYTHON_BIN', sys.executable)


def profiling_enabled():
    return os.environ.get('LOG_ONLY', '0') == '0'


def bench(folder, ep, seq, warmup_round=0):
    filename = (f'warmup-round{warmup_round:03d}.json' if warmup_round
                else f"batch{os.environ['MAX_NUM_BATCHED_TOKENS']}-concurrency{seq}.json")
    input_len = int(os.environ['KV_LEN']) + 1
    return [PYTHON, '-m', 'vllm', 'bench', 'serve', '--backend', 'vllm',
            '--base-url', f"http://127.0.0.1:{os.environ[f'PORT_EP{ep}']}", '--endpoint', '/v1/completions',
            '--model', os.environ['SERVED_MODEL_NAME'], '--tokenizer', os.environ['MODEL_PATH'],
            '--dataset-name', 'random', '--random-input-len', str(input_len),
            # Random text can re-tokenize above the requested length. Cap the
            # actual server input; keep the result length checks exact.
            '--extra-body', json.dumps({'truncate_prompt_tokens': input_len}),
            '--random-output-len', os.environ['MAX_OUTPUT'], '--random-range-ratio', '0',
            '--random-prefix-len', '0', '--num-prompts', str(seq), '--request-rate', 'inf',
            '--max-concurrency', str(seq), '--seed', os.environ['SEED'],
            '--num-warmups', '0', '--ready-check-timeout-sec', '0', '--ignore-eos',
            '--percentile-metrics', 'ttft,tpot,itl,e2el', '--metric-percentiles', '50,90,95,99',
            '--save-result', '--save-detailed', '--result-dir', str(folder), '--result-filename', filename]


def plans(max_sequences):
    result = []
    for ep in (0, 1):
        folder = ROOT / f'ep{ep}'
        env = dict(os.environ, EP=str(ep), CUDA_VISIBLE_DEVICES=os.environ[f'GPU_EP{ep}'],
                   PORT=os.environ[f'PORT_EP{ep}'], DP_RPC_PORT=os.environ[f'DP_RPC_PORT_EP{ep}'],
                   MASTER_PORT=os.environ[f'MASTER_PORT_EP{ep}'], MAX_NUM_SEQS=str(max_sequences), DRY_RUN='1')
        preview = subprocess.check_output(['bash', str(BASE / 'mimo.sh')], env=env, text=True)
        command = shlex.split(preview.splitlines()[-1])
        command = ['env', f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}"] + command
        if profiling_enabled():
            command += ['--profiler-config', '{"profiler":"cuda"}']
            command = trace_command(folder / 'trace') + command
        result.append(dict(ep=ep, folder=folder, serve=command,
                           url=f"http://127.0.0.1:{env['PORT']}"))
    return result


def boundary(selected, action):
    def request(plan):
        req = urllib.request.Request(plan['url'] + f'/{action}_profile', data=b'{}',
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            response.read()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(request, selected))


def validate_bench(folder, seq, warmup_round=0):
    filename = f'warmup-round{warmup_round:03d}.json' if warmup_round else None
    data = valid_result(folder, seq, dict(max_output=os.environ['MAX_OUTPUT'],
                        input_len=str(int(os.environ['KV_LEN']) + 1),
                        max_num_batched_tokens=os.environ['MAX_NUM_BATCHED_TOKENS']), filename)
    label = f'warmup round={warmup_round}' if warmup_round else 'eval'
    print(f"{folder.parent.name} seq={seq} {label}: "
          f"output_tokens/s={data['total_output_tokens'] / data['duration']:.3f}", flush=True)


def workloads(seqs, warmup):
    for seq in seqs:
        for number in range(1, int(os.environ['WARMUP_ROUNDS']) + 1) if warmup else (0,):
            yield seq, number


def collect(selected, seqs):
    profiling = profiling_enabled()
    servers, clients, samplers = [], [], []
    app_groups = set()
    try:
        for plan in selected:
            plan['folder'].mkdir(parents=True)
            (plan['folder'] / 'server.command.txt').write_text(shlex.join(plan['serve']) + '\n')
            servers.append(launch(plan['serve'], plan['folder'] / 'server.log'))
        deadline = time.monotonic() + int(os.environ['STARTUP_TIMEOUT'])
        while True:
            ready = []
            for plan, proc in zip(selected, servers):
                # Nsight may exit before the app; remember owned app groups
                # while they are still descendants of the launcher.
                for child in server_children(proc):
                    try:
                        app_groups.add(os.getpgid(child))
                    except ProcessLookupError:
                        pass
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited: {plan['folder'] / 'server.log'}")
                try:
                    with urllib.request.urlopen(plan['url'] + '/health', timeout=2) as response:
                        ready.append(response.status == 200)
                except OSError:
                    ready.append(False)
            if all(ready):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Waiting for decode servers')
            time.sleep(2)
        # Both modes run all warmups before all formal loads.
        events = []
        for warmup in (True, False):
            if not warmup and profiling:
                boundary(selected, 'start')
                events.append(dict(event='profile_started', time_unix_ns=time.time_ns()))
            for seq, number in workloads(seqs, warmup):
                current = []
                stem = f'warmup-round{number:03d}' if warmup else 'eval'
                entry = dict(phase='warmup' if warmup else 'eval', round=number or 1,
                             concurrency=seq, clients=[], status='running')
                events.append(entry)
                for plan in selected:
                    folder = plan['folder'] / f'seq{seq}'
                    folder.mkdir(exist_ok=True)
                    if profiling:
                        snapshot(plan['url'], folder / f'{stem}.metrics-before.txt')
                    command = bench(folder, plan['ep'], seq, number)
                    (folder / f'{stem}.command.txt').write_text(shlex.join(command) + '\n')
                    if not warmup and profiling:
                        samplers.append(launch([PYTHON, str(BASE / 'sample-metrics.py'),
                            plan['url'] + '/metrics', str(folder / 'eval.metrics-series.jsonl'),
                            os.environ['METRICS_INTERVAL'], str(folder / 'eval.log')], folder / 'sampler.log'))
                    entry['clients'].append(dict(ep=plan['ep'], launched_unix_ns=time.time_ns()))
                    proc = launch(command, folder / f'{stem}.log')
                    current.append(proc)
                    clients.append(proc)
                (ROOT / 'load-events.json').write_text(json.dumps(events, indent=2))
                wait_jobs(current, int(os.environ['BENCH_TIMEOUT']), servers=servers)
                entry['finished_unix_ns'] = time.time_ns()
                if samplers:
                    terminate_all(samplers)
                    samplers = []
                for plan in selected:
                    folder = plan['folder'] / f'seq{seq}'
                    validate_bench(folder, seq, number)
                    if profiling:
                        snapshot(plan['url'], folder / f'{stem}.metrics-after.txt')
                entry['status'] = 'ok'
                (ROOT / 'load-events.json').write_text(json.dumps(events, indent=2))
            if not warmup and profiling:
                boundary(selected, 'stop')
                events.append(dict(event='profile_stopped', time_unix_ns=time.time_ns()))
                (ROOT / 'load-events.json').write_text(json.dumps(events, indent=2))
        # Stop the directly launched service, or Nsight's service child so
        # Nsight can finish writing its report before owned-process cleanup.
        for proc in servers:
            children = server_children(proc) if profiling else []
            for pid in children or [proc.pid]:
                try:
                    os.kill(pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        wait_jobs(servers, 180, check_returncodes=False)
    finally:
        terminate_all(samplers + clients + servers, app_groups)
    if profiling:
        for plan in selected:
            export_check(plan['folder'])


def main():
    for key in ('DRY_RUN', 'SMOKE_ONLY', 'LOG_ONLY'):
        if os.environ.get(key, '0') not in ('0', '1'):
            raise ValueError(f'{key} must be 0 or 1')
    profiling = profiling_enabled()
    if not profiling and os.environ['SMOKE_ONLY'] == '1':
        raise ValueError('SMOKE_ONLY checks Nsight; use LOG_ONLY=0 or SMOKE_ONLY=0')
    seqs = [int(value) for value in os.environ['SEQ_COUNTS'].split()]
    if not seqs or len(set(seqs)) != len(seqs) or min(seqs) <= 0:
        raise ValueError('SEQ_COUNTS must contain unique positive integers')
    rounds = os.environ['WARMUP_ROUNDS']
    if int(rounds) < 0 or str(int(rounds)) != rounds:
        raise ValueError('WARMUP_ROUNDS must be a nonnegative integer')
    interval = float(os.environ['METRICS_INTERVAL'])
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('METRICS_INTERVAL must be positive and finite')
    for key in ('STARTUP_TIMEOUT', 'BENCH_TIMEOUT'):
        if int(os.environ[key]) <= 0:
            raise ValueError(f'{key} must be positive')
    gpus = [int(gpu) for ep in (0, 1) for gpu in os.environ[f'GPU_EP{ep}'].split(',')]
    if len(set(gpus)) != 8 or len(gpus) != 8:
        raise ValueError('Two disjoint four-GPU groups required')
    ports = [int(os.environ[f'{prefix}{ep}']) for ep in (0, 1)
             for prefix in ('PORT_EP', 'DP_RPC_PORT_EP', 'MASTER_PORT_EP')]
    if len(set(ports)) != 6 or any(not 0 < port <= 65535 for port in ports):
        raise ValueError('Six distinct valid ports required')
    selected = plans(max(seqs))
    if os.environ['DRY_RUN'] == '1':
        if profiling:
            for ep in (0, 1):
                print(shlex.join(smoke_command(ep)))
        if os.environ['SMOKE_ONLY'] == '1':
            return
        print(f'Decode: one model start per group; max_num_seqs={max(seqs)}; output={ROOT}')
        if not profiling:
            print('LOG_ONLY=1: run warmups and formal loads; save logs and benchmark JSON')
        for plan in selected:
            print(shlex.join(plan['serve']))
        for warmup in (True, False):
            if not warmup and profiling:
                print('POST /start_profile on both services; continuous capture of all formal loads')
            for seq, number in workloads(seqs, warmup):
                print(f"{'warmup' if warmup else 'eval'} round={number or 1} concurrency={seq}")
                for plan in selected:
                    print(shlex.join(bench(plan['folder'] / f'seq{seq}', plan['ep'], seq, number)))
        print('POST /stop_profile on both services; shutdown and export after all loads'
              if profiling else 'Shutdown both services after all loads')
        return
    if not (Path(os.environ.get('VLLM_SOURCE') or '/host/vllm') / 'vllm' / '__init__.py').is_file():
        raise FileNotFoundError('VLLM_SOURCE must point to a local vLLM checkout')
    subprocess.run([PYTHON, '-c',
        'import importlib.util; assert importlib.util.find_spec('
        '"vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector"), '
        '"Local vLLM is missing DecodeBenchConnector"'], check=True)
    check_gpu_availability([0, 1])
    for port in ports:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    model = Path(os.environ['MODEL_PATH'])
    for filename in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'):
        if not (model / filename).is_file():
            raise FileNotFoundError(model / filename)
    for shard in set(json.loads((model / 'model.safetensors.index.json').read_text())['weight_map'].values()):
        if not (model / shard).is_file():
            raise FileNotFoundError(model / shard)
    ROOT.mkdir(parents=True, exist_ok=False)
    config = {key: value for key, value in os.environ.items() if key in (
        'MODEL_PATH', 'VLLM_SOURCE', 'PYTHON_BIN', 'TP', 'DP', 'GPU_EP0', 'GPU_EP1', 'SEQ_COUNTS', 'LOG_ONLY',
        'MAX_NUM_BATCHED_TOKENS', 'KV_LEN', 'MAX_OUTPUT', 'WARMUP_ROUNDS', 'SEED', 'ENABLE_EPLB',
        'KV_STARTUP_FILL', 'KV_FILL_MEAN', 'KV_FILL_STD', 'CUDAGRAPH_MODE', 'MAX_MODEL_LEN',
        'NSYS_TRACE', 'NCCL_TRACE', 'CUDA_GRAPH_TRACE', 'GPU_MEMORY_UTILIZATION')}
    config.update(speculative_config=resolve(), **vllm_runtime_info())
    if profiling:
        config['nsys_version'] = subprocess.check_output([os.environ['NSYS_BIN'], '--version'], text=True).strip()
    config.update(server_max_num_seqs=max(seqs), server_lifecycle='single_start',
                  capture_mode='continuous_formal_loads' if profiling else 'logs_only', load_order=seqs)
    config['servers'] = [dict(ep=p['ep'], command=p['serve']) for p in selected]
    config['commands'] = [{'seq': seq, 'groups': [dict(ep=p['ep'],
        warmup=[bench(p['folder'] / f'seq{seq}', p['ep'], seq, number)
                for number in range(1, int(rounds) + 1)],
        bench=bench(p['folder'] / f'seq{seq}', p['ep'], seq)) for p in selected]}
        for seq in seqs]
    (ROOT / 'config.json').write_text(json.dumps(config, indent=2))
    (ROOT / 'topology.txt').write_text(subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True))
    if profiling:
        # Shared prefill smoke checks cover the two selected physical groups.
        os.environ['COMPARISON'], os.environ['EP_SOLO'] = 'ep', '0'
        run_smokes()
    if os.environ['SMOKE_ONLY'] == '1':
        return
    collect(selected, seqs)
    print('Decode data collection complete (no analysis):' if profiling
          else 'Decode log-only run complete:', ROOT, flush=True)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
