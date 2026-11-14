"""Local command construction and deployment preflight checks.

Extracted from aggre_or_not/code/collect_pd.py so performance_eva can be copied
on its own. Collection workloads and source-path injection are not needed here.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess


def integer(name, minimum=1):
    value = int(os.environ[name])
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


def enabled(name):
    if os.environ[name] not in ('0', '1'):
        raise ValueError(f'{name} must be 0 or 1')
    return os.environ[name] == '1'


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
            raise RuntimeError('NIXL is unavailable in the deployment Python environment; '
                               'install a version compatible with its vLLM package')
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
