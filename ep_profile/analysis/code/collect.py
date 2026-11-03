"""MiMo: full benchmark warmup/evaluation rounds, comparing EP or EPLB."""
import datetime
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request

from experiment import group_names, result_stem, summarize, validate_round

BASE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('RESULT_DIR', '/tmp/mimo-ep-profile')).resolve()
PYTHON = os.environ.get('PYTHON_BIN', sys.executable)
NSYS = os.environ.get('NSYS_BIN', 'nsys')


def smoke_worker(rank, rendezvous):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=4)
    x = torch.ones(1 << 20, device='cuda')
    gathered = torch.empty(4 * x.numel(), device='cuda')
    reduced = torch.empty_like(x)
    dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.profiler.start()
    for step in range(3):
        with torch.cuda.nvtx.range(f'smoke_rank{rank}_step{step}'):
            x.fill_(rank + 1)
            dist.all_gather_into_tensor(gathered, x)
            dist.reduce_scatter_tensor(reduced, gathered)
            dist.all_reduce(x)
            assert x[0].item() == 10
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    dist.destroy_process_group()


def run(cmd, log, env=None, timeout=300):
    with open(log, 'w') as out:
        result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT,
                                env=env, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Command exit {result.returncode}: {shlex.join(cmd)}; see {log}')


def trace_command(output):
    return [NSYS, 'profile', '--sample=none', '--cpuctxsw=none',
            '--trace=' + os.environ['NSYS_TRACE'],
            '--nccl-trace=' + os.environ['NCCL_TRACE'],
            '--cuda-graph-trace=' + os.environ['CUDA_GRAPH_TRACE'],
            '--trace-fork-before-exec=true', '--capture-range=cudaProfilerApi',
            '--capture-range-end=stop', '--output=' + str(output)]


def export_check(folder):
    reports = list(folder.glob('trace*.nsys-rep'))
    if len(reports) != 1:
        raise RuntimeError(f'Expected one Nsight report in {folder}, got {reports}')
    db = folder / 'trace.sqlite'
    run([NSYS, 'export', '--type=sqlite', '--include-json=true', '--dynamic-tables=all',
         '--output=' + str(db), str(reports[0])],
        folder / 'export.log', timeout=300)
    return validate_database(folder)


def validate_database(folder):
    db = folder / 'trace.sqlite'
    reports = list(folder.glob('trace*.nsys-rep'))
    with sqlite3.connect(db) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        kernels = con.execute('SELECT deviceId,globalPid,COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY 1,2').fetchall()
        nccl = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                for t in sorted(tables) if 'NCCL' in t and not t.startswith('ENUM')}
        nvtx_nccl = con.execute("SELECT COUNT(*) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE n.end IS NOT NULL AND COALESCE(n.text,s.value,'') IN ('AllGather','ReduceScatter','AllReduce','Broadcast','Send','Recv')").fetchone()[0] if 'NVTX_EVENTS' in tables else 0
        plugin_pids = con.execute("SELECT DISTINCT globalPid FROM DIAGNOSTIC_EVENT WHERE text LIKE 'NCCL plugin init%'").fetchall()
        invalid_intervals = con.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end < start').fetchone()[0]
        payloads = con.execute('SELECT COUNT(*) FROM NVTX_EVENTS WHERE jsonText IS NOT NULL').fetchone()[0]
        channel_warnings = con.execute("SELECT COUNT(*) FROM DIAGNOSTIC_EVENT WHERE text LIKE 'KernelCh range begin%'").fetchone()[0]
    check = {'kernels_by_device_process': kernels, 'nccl_tables': nccl,
             'nccl_nvtx_events': nvtx_nccl, 'nccl_plugin_processes': len(plugin_pids),
             'extended_payload_events':payloads, 'invalid_cupti_intervals':invalid_intervals,
             'nccl_channel_timestamp_warnings':channel_warnings,'report': reports[0].name}
    (folder / 'validation.json').write_text(json.dumps(check, indent=2))
    if len({k[0] for k in kernels}) != 4 or len({k[1] for k in kernels}) != 4:
        raise RuntimeError(f'Four GPU worker traces required: {check}')
    if not any(nccl.values()) and not nvtx_nccl:
        raise RuntimeError(f'No NCCL events: {check}')
    if len(plugin_pids) < 4:
        raise RuntimeError(f'NCCL profiler plugin missing on workers: {check}')
    if invalid_intervals or not payloads:
        raise RuntimeError(f'CUPTI timestamps or NCCL payload validation failed: {check}')
    print('Trace validated:', folder, check, flush=True)
    return check


def cleanup(proc):
    if proc is None:
        return
    # nsys starts the application in its own process group.
    groups = {proc.pid}
    for child in descendants(proc.pid):
        try:
            args = Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
            if b'vllm' in args and b'serve' in args:
                groups.add(os.getpgid(int(child)))
        except (ProcessLookupError, FileNotFoundError):
            pass
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        proc.poll()
        alive = []
        for group in groups:
            try:
                os.killpg(group, 0)
                alive.append(group)
            except ProcessLookupError:
                pass
        if not alive:
            return
        time.sleep(1)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()


def descendants(pid):
    """nsys-launcher inserts another process between nsys and the API server."""
    try:
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except FileNotFoundError:
        return []
    return children + [p for child in children for p in descendants(child)]


def snapshot(url, path):
    with urllib.request.urlopen(url + '/metrics', timeout=15) as r:
        path.write_bytes(r.read())


def launch(command, log, env=None):
    with log.open('w') as out:
        return subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)


def wait_jobs(jobs, timeout, servers=(), check_returncodes=True):
    """Poll all jobs together, so either side failing stops the experiment."""
    deadline = time.monotonic() + timeout
    pending = set(range(len(jobs)))
    while pending:
        for server in servers:
            if server.poll() is not None:
                raise RuntimeError(f'Profile server exited early: pid={server.pid}')
        for index in list(pending):
            rc = jobs[index].poll()
            if rc is not None:
                pending.remove(index)
                if check_returncodes and rc:
                    raise RuntimeError(f'Job {index} failed with exit {rc}; inspect its log')
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Jobs {sorted(pending)} exceeded {timeout}s')
            time.sleep(1)


def terminate_all(jobs, app_groups=()):
    # Notify both sides before waiting for either to exit. Only owned groups.
    for group in set(app_groups) | {p.pid for p in jobs}:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for proc in jobs:
        cleanup(proc)
    # A target can become reparented after its nsys launcher exits.
    for group in app_groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def server_children(proc):
    found = []
    for child in descendants(proc.pid):
        try:
            args = Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
            if b'vllm' in args and b'serve' in args:
                found.append(int(child))
        except FileNotFoundError:
            pass
    return found


def make_plans():
    # Obtain ONE command template; actual round counts are executed below.
    # No warmup is hidden in the client: all phases use the identical benchmark.
    env = os.environ.copy()
    env.update(DRY_RUN='1', REPEATS='1', WARMUP_ROUNDS='0', BENCH_PARALLEL='1', RESULT_DIR=str(ROOT))
    preview = subprocess.check_output(['bash', str(BASE / 'reproduce.sh')], env=env, text=True)
    commands = [shlex.split(line) for line in preview.splitlines() if line.startswith('env ')]
    if len(commands) != 4:
        raise ValueError(f'Expected two server and two benchmark commands: {preview}')
    plans = []
    for ep, group in enumerate(group_names(os.environ)):
        folder = ROOT / group
        serve = commands[ep] + ['--profiler-config', '{"profiler":"cuda"}']
        plans.append({'index':ep, 'group':group, 'folder':folder,
                      'serve':trace_command(folder/'trace') + serve,
                      'bench':commands[ep+2],
                      'url':'http://127.0.0.1:' + serve[serve.index('--port')+1]})
    return plans


def smoke_command(ep):
    folder = ROOT / f'smoke_ep{ep}'
    return trace_command(folder/'trace') + [
        'env', 'CUDA_VISIBLE_DEVICES=' + os.environ[f'GPU_EP{ep}'],
        PYTHON, str(Path(__file__).resolve()), '--smoke', str(folder/'rendezvous')]


def run_smokes():
    jobs = []
    try:
        for ep in [0, 1]:
            folder = ROOT / f'smoke_ep{ep}'
            folder.mkdir()
            jobs.append(launch(smoke_command(ep), folder/'collect.log'))
        print('Both four-GPU toolchain checks launched concurrently.', flush=True)
        wait_jobs(jobs, 240)
    finally:
        terminate_all(jobs)
    for ep in [0, 1]:
        export_check(ROOT/f'smoke_ep{ep}')


def round_command(plan, phase, number):
    command = plan['bench'].copy()
    command[command.index('--result-filename') + 1] = result_stem(phase, number) + '.json'
    return command


def save_manifest(manifest):
    (ROOT / 'config.json').write_text(json.dumps(manifest, indent=2) + '\n')


def profile_boundary(plans, action, manifest):
    # One capture window across ALL evaluation rounds, compatible with nsys stop.
    # Start/stop both sides concurrently; requests begin only after both respond.
    def request(plan):
        before = time.time_ns()
        req = urllib.request.Request(plan['url'] + '/' + action + '_profile',
                                     data=b'{}', headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            response.read()
        return {'group': plan['group'], 'requested_unix_ns': before,
                'acknowledged_unix_ns': time.time_ns()}
    with ThreadPoolExecutor(max_workers=len(plans)) as pool:
        events = list(pool.map(request, plans))
    manifest.setdefault('profiler_events', []).append({'action': action, 'groups': events})
    save_manifest(manifest)

def collect_parallel(plans, manifest):
    servers, benches, app_groups = [], [], set()
    try:
        # Start both services before waiting for readiness.
        for plan in plans:
            plan['folder'].mkdir()
            proc = launch(plan['serve'], plan['folder']/'server.log')
            servers.append(proc)
            print(f'{plan["group"]} loading under Nsight: pid={proc.pid}', flush=True)
        deadline = time.monotonic() + int(os.environ['STARTUP_TIMEOUT'])
        ready = [False]*len(plans)
        while not all(ready):
            for i, (plan, proc) in enumerate(zip(plans, servers)):
                for child in server_children(proc):
                    try:
                        app_groups.add(os.getpgid(child))
                    except ProcessLookupError:
                        pass
                if proc.poll() is not None:
                    raise RuntimeError(f'{plan["group"]} server exited: {plan["folder"]}/server.log')
                try:
                    with urllib.request.urlopen(plan['url']+'/health', timeout=2) as r:
                        ready[i] = r.status == 200
                except OSError:
                    ready[i] = False
            if time.monotonic() > deadline:
                raise TimeoutError('Timed out waiting for both services to become ready')
            if not all(ready):
                time.sleep(2)
        print('Both services ready; full benchmark warmup rounds, then evaluation.', flush=True)
        manifest['rounds'] = []
        for phase, count in [('warmup', int(os.environ['WARMUP_ROUNDS'])),
                             ('eval', int(os.environ['REPEATS']))]:
            if phase == 'eval':
                for plan in plans:
                    snapshot(plan['url'], plan['folder'] / 'metrics-before.txt')
                profile_boundary(plans, 'start', manifest)
            for number in range(1, count + 1):
                stem = result_stem(phase, number)
                entry = {'phase': phase, 'round': number, 'clients': []}
                manifest['rounds'].append(entry)
                current = []
                for plan in plans:
                    snapshot(plan['url'], plan['folder'] / f'{stem}.metrics-before.txt')
                for plan in plans:
                    command = round_command(plan, phase, number)
                    (plan['folder'] / f'{stem}.command.txt').write_text(shlex.join(command) + '\n')
                    entry['clients'].append({'group': plan['group'], 'launched_unix_ns': time.time_ns(),
                                              'result': f"{plan['group']}/{stem}.json"})
                    proc = launch(command, plan['folder'] / f'{stem}.log')
                    current.append(proc)
                    benches.append(proc)
                save_manifest(manifest)
                print(f'{phase} round {number}/{count}: both clients launched', flush=True)
                wait_jobs(current, int(os.environ['BENCH_TIMEOUT']), servers=servers,
                          check_returncodes=False)
                entry['both_clients_finished_unix_ns'] = time.time_ns()
                # Validate both sides, including failed rounds, before proceeding.
                failures = []
                for plan, proc in zip(plans, current):
                    try:
                        validate_round(ROOT, plan['group'], phase, number, proc.returncode)
                    except ValueError as error:
                        failures.append(str(error))
                    snapshot(plan['url'], plan['folder'] / f'{stem}.metrics-after.txt')
                entry['status'] = 'invalid' if failures else 'ok'
                save_manifest(manifest)
                if failures:
                    raise RuntimeError('; '.join(failures))
            if phase == 'eval':
                profile_boundary(plans, 'stop', manifest)
                for plan in plans:
                    snapshot(plan['url'], plan['folder'] / 'metrics-after.txt')
        summarize(ROOT)
        # Neither service is shut down and no SQLite export starts until all rounds finish.
        for proc in servers:
            children = server_children(proc)
            if children:
                for child in children:
                    try:
                        os.kill(child, signal.SIGINT)
                    except ProcessLookupError:
                        pass
            elif proc.poll() is None:
                os.kill(proc.pid, signal.SIGINT)
        wait_jobs(servers, 180, check_returncodes=False)
    finally:
        terminate_all(benches+servers, app_groups)
    for plan in plans:
        export_check(plan['folder'])


def main():
    for name, minimum in [('WARMUP_ROUNDS', 0), ('REPEATS', 1)]:
        value = os.environ[name]
        if not value.isascii() or not value.isdecimal() or int(value) < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    plans = make_plans()
    if os.environ.get('DRY_RUN') == '1':
        print('MiMo:', '/'.join(group_names(os.environ)), 'on separate GPU groups. No processes launched. Output:', ROOT)
        print('Startup timeouts:', {key:os.environ[key] for key in
                                   ['STARTUP_TIMEOUT','VLLM_ENGINE_READY_TIMEOUT_S']})
        print('Stage 1: launch both toolchain checks, then wait for both:')
        for ep in [0, 1]:
            print(shlex.join(smoke_command(ep)))
        if os.environ.get('SMOKE_ONLY') == '1':
            return
        print('Stage 2: launch both services, then wait until BOTH are healthy:')
        for plan in plans:
            print(shlex.join(plan['serve']))
        for phase, count in [('warmup', int(os.environ['WARMUP_ROUNDS'])),
                             ('eval', int(os.environ['REPEATS']))]:
            if phase == 'eval':
                print('POST /start_profile on BOTH services; one continuous evaluation capture')
            for number in range(1, count + 1):
                print(f'Stage 3: {phase} round {number}/{count}; launch BOTH, validate BOTH before next round:')
                for plan in plans:
                    print(shlex.join(round_command(plan, phase, number)))
            if phase == 'eval':
                print('POST /stop_profile on BOTH services after ALL evaluation rounds')
        print('Stage 4: generate expert skew visualization and evidence in the result directory:')
        print(shlex.join([PYTHON, str(Path(__file__).with_name('explain_ep_gap.py')),
                          str(ROOT), '--output-dir', str(ROOT)]))
        return
    for name in ['SMOKE_ONLY','DRY_RUN']:
        if os.environ.get(name,'0') not in ('0','1'):
            raise ValueError(f'{name} must be 0 or 1')
    busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy:
        raise RuntimeError(f'GPU processes still active; leaving them untouched: {busy}')
    for plan in plans:
        for flag in ['--port','--data-parallel-rpc-port','--master-port']:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',int(plan['serve'][plan['serve'].index(flag)+1])))
    model_root = Path(os.environ['MODEL_PATH'])
    model_config = json.loads((model_root/'config.json').read_text())
    for name in ['tokenizer.json','tokenizer_config.json','model.safetensors.index.json']:
        if not (model_root/name).is_file():
            raise FileNotFoundError(model_root/name)
    weights = json.loads((model_root/'model.safetensors.index.json').read_text())['weight_map']
    for name in set(weights.values()):
        if not (model_root/name).is_file():
            raise FileNotFoundError(model_root/name)
    ROOT.mkdir(parents=True, exist_ok=False)
    keys = ['TP','DP','GPU_EP0','GPU_EP1','NUM_PROMPTS','NUM_WARMUPS','WARMUP_ROUNDS','REPEATS','COMPARISON','CONCURRENCY',
            'INPUT_LEN','OUTPUT_LEN','MAX_NUM_BATCHED_TOKENS','MAX_NUM_SEQS','MAX_MODEL_LEN',
            'GPU_MEMORY_UTILIZATION','ALL2ALL_BACKEND','SEED','MODEL_PATH','SERVED_MODEL_NAME',
            'VLLM_SOURCE','NSYS_TRACE','NCCL_TRACE','CUDA_GRAPH_TRACE',
            'STARTUP_TIMEOUT','VLLM_ENGINE_READY_TIMEOUT_S']
    manifest = {key:os.environ[key] for key in keys}
    # Requested options only; vLLM may fall back to another placement/backend.
    manifest.update({key:os.environ.get(key, default) for key, default in
                     [('MOE_BACKEND', ''), ('ATTENTION_BACKEND', ''),
                      ('EXPERT_PLACEMENT_STRATEGY', 'linear')]})
    manifest.update(backend_parameter_scope='both_groups' if os.environ['COMPARISON']=='eplb' else 'ep1_only')
    # The shared command builder resolves per-field settings and JSON overrides.
    # Read the effective config from group 1's command, not the parent env.
    ep1_command = plans[1]['serve']
    eplb_enabled = '--enable-eplb' in ep1_command
    manifest.update(ENABLE_EPLB=eplb_enabled,
                    EPLB_CONFIG=json.loads(ep1_command[ep1_command.index('--eplb-config')+1])
                    if eplb_enabled else None,
                    eplb_enabled_by_group={p['group']:'--enable-eplb' in p['serve'] for p in plans},
                    group_settings={p['group']:{'ep_enabled':'--enable-expert-parallel' in p['serve'],
                                               'eplb_enabled':'--enable-eplb' in p['serve']} for p in plans})
    manifest.update(created_at=datetime.datetime.now().astimezone().isoformat(),
                    nsys_version=subprocess.check_output([NSYS,'--version'],text=True).strip(),
                    vllm_git_head=subprocess.check_output(['git','-C',os.environ['VLLM_SOURCE'],'rev-parse','HEAD'],text=True).strip(),
                    repeats=int(os.environ['REPEATS']), warmup_rounds=int(os.environ['WARMUP_ROUNDS']),
                    sequential=False, bench_parallel=True,
                    profile_start='After ALL warmup rounds. One continuous capture across ALL evaluation rounds, including client startup and inter-round gaps. Throughput excludes these gaps.',
                    model_config=model_config,
                    commands=[{k:v for k,v in plan.items() if k!='folder'} for plan in plans])
    (ROOT/'config.json').write_text(json.dumps(manifest,indent=2))
    run(['nvidia-smi','topo','-m'],ROOT/'topology.txt')
    run_smokes()
    if os.environ.get('SMOKE_ONLY') == '1':
        return
    collect_parallel(plans,manifest)
    subprocess.run([PYTHON, str(Path(__file__).with_name('explain_ep_gap.py')),
                    str(ROOT), '--output-dir', str(ROOT)], check=True)
    print('Expert skew visualization:', ROOT/'expert_skew.svg', flush=True)
    subprocess.run([PYTHON,str(Path(__file__).with_name('analyze.py')),str(ROOT)],check=True)
    print('Concurrent collection and analysis complete:',ROOT,flush=True)


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}; cleaning up owned processes')


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--smoke':
        import torch.multiprocessing as mp
        mp.spawn(smoke_worker,args=('file://'+sys.argv[2],),nprocs=4,join=True)
    else:
        signal.signal(signal.SIGTERM,interrupted)
        main()
