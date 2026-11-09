"""Independent load launcher. Native per_dataset options are forwarded unchanged."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import urllib.request
from uuid import uuid4

BASE = Path(__file__).resolve().parents[1]
PORTS = {'aggre': 19000, 'pd': 19100, 'pd_6p2d': 19200}


def slug(value):
    return re.sub(r'[^\w.-]+', '_', str(value)).strip('._')[:90] or 'unnamed'


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=PORTS, default=None)
    parser.add_argument('--deployment', type=Path, help='本机或复制的 deployment.json')
    parser.add_argument('--results-root', type=Path, default=BASE / 'results')
    parser.add_argument('--label', default='experiment')
    parser.add_argument('--dry-run', action='store_true', help='仅打印有效参数和命令，不发请求或创建目录')
    parser.add_argument('--perf-help', action='store_true', help='显示 per_dataset.py 的全部负载参数')
    args, native = parser.parse_known_args()
    if native[:1] == ['--']:
        native = native[1:]
    if args.perf_help:
        return subprocess.call([sys.executable, str(BASE / 'per_dataset.py'), '--help'])
    deployment = json.loads(args.deployment.read_text()) if args.deployment else None
    if deployment and deployment.get('status') != 'ready':
        parser.error('deployment.json 尚未 ready；请等待启动成功')
    mode = args.mode or (deployment['deployment_mode'] if deployment else 'aggre')
    if deployment and args.mode and args.mode != deployment['deployment_mode']:
        parser.error('--mode 与 deployment.json 不一致')
    defaults = dict(ip='127.0.0.1', port=str(deployment['port'] if deployment else PORTS[mode]),
                    model=deployment['model'] if deployment else 'MiMo-V2.6-Flash-MOPD',
                    tokenizer_path='/models/MiMo-V2.6-Flash-MOPD',
                    parallel='2 4 8 16' if mode == 'aggre' else '2 4 8 16 32',
                    input_length=50000, output_length=500, number='50', api_key='EMPTY')
    # Range options must not collide with wrapper-provided fixed-length defaults.
    for axis, original in [('input', 128), ('output', 512)]:
        if any(a.split('=')[0] in (f'--{axis}-length-min', f'--{axis}-length-max') for a in native):
            defaults[axis + '_length'] = original
    if any(a.split('=')[0] == '--duration' for a in native):
        defaults['number'] = None
    sys.path.insert(0, str(BASE))
    try:
        from per_dataset import runner
    except ImportError as exc:
        raise RuntimeError('负载依赖缺失；用 LOAD_PYTHON_BIN 指定已安装 requirements-load.txt 的 Python') from exc
    with runner.make_context('per_dataset.py', native, default_map=defaults) as context:
        params = context.params
    if params['prepare_dataset']:
        # Dataset preparation remains a separate, explicit action.
        return subprocess.call([sys.executable, str(BASE / 'per_dataset.py'), *native])
    for key in ('output_dir', 'output_csv', 'output_xlsx', 'output_req_map'):
        if params[key]:
            parser.error(f'负载入口统一管理 {key}，请使用 --results-root；自定义文件路径可直接调用 per_dataset.py')
    if params['multi_turn']:
        params['use_chat'] = True
    # Verify the sweep before making any network requests.
    parallel = [int(p) for p in params['parallel'].split()]
    number = [int(n) for n in params['number'].split()] if params['number'] else [10**9]
    if not parallel or any(p <= 0 for p in parallel) or any(n <= 0 for n in number):
        parser.error('并发和请求数必须为正整数')
    if len(number) not in (1, len(parallel)):
        parser.error('--number 必须是单值或与 --parallel 等长')
    digest_params = {k: v for k, v in params.items() if k not in ('api_key', 'ip', 'port')}
    fingerprint = hashlib.sha256(json.dumps(digest_params, sort_keys=True).encode()).hexdigest()[:12]
    workload = f"{'multi' if params['multi_turn'] else 'single'}_in{params['input_length']}_out{params['output_length']}_{slug(args.label)}_{fingerprint}"
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '_' + uuid4().hex
    root = args.results_root.resolve() / mode / slug(params['model']) / workload / run_id
    params['output_dir'] = str(root / 'perf')
    # Serialize Click's parsed values, retaining explicit false switches and native types.
    command = [sys.executable, str(BASE / 'per_dataset.py')]
    for option in runner.params:
        value = params[option.name]
        if value is None:
            continue
        if option.is_flag:
            if value:
                command.append(option.opts[0])
            elif option.secondary_opts:
                command.append(option.secondary_opts[0])
        else:
            command.extend([option.opts[0], str(value)])
    safe_params = dict(params, api_key='<redacted>')
    safe_command = command.copy()
    if '--api-key' in safe_command:
        safe_command[safe_command.index('--api-key') + 1] = '<redacted>'
    manifest = dict(run_id=run_id, mode=mode, status='starting', parameters=safe_params,
                    deployment=deployment, command=safe_command, results=str(root),
                    versions={p: importlib.metadata.version(p) for p in ('evalscope', 'pandas', 'openpyxl')})
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    root.mkdir(parents=True, exist_ok=False)

    def save():
        temp = root / 'manifest.json.tmp'
        temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        temp.replace(root / 'manifest.json')

    child = None
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        port = f":{params['port']}" if params['include_port'] else ''
        url = f"{params['scheme']}://{params['ip']}{port}"
        manifest['endpoint'] = url
        # Health may be JSON or plain text; only the status code matters.
        with urllib.request.urlopen(url + '/health', timeout=10) as response:
            if response.status != 200:
                raise RuntimeError('Health check failed')
        models = get_json(url + '/v1/models')
        if params['model'] not in [m['id'] for m in models.get('data', [])]:
            raise RuntimeError(f"目标服务未提供模型 {params['model']}")
        manifest['status'] = 'running'
        save()
        print(f'Results: {root}', flush=True)
        with (root / 'load.log').open('w') as log:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True,
                                     env=dict(os.environ, PYTHONUNBUFFERED='1'))
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
        if code:
            raise RuntimeError(f'per_dataset.py exited with status {code}; see load.log')
        manifest['status'] = 'completed'
        return 0
    except KeyboardInterrupt:
        manifest.update(status='interrupted', error='Load interrupted')
        return 130
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        manifest['finished_at'] = datetime.now().isoformat()
        save()
        print(f"{manifest['status']}: {root}", flush=True)


if __name__ == '__main__':
    sys.exit(main())
