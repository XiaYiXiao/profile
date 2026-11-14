"""Deployment option contracts: no GPU or vLLM imports required."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

BASE = Path(__file__).resolve().parents[1]


def preview(mode, entry=None, **options):
    env = {key: value for key, value in os.environ.items()
           if key in ('PATH', 'HOME', 'LANG', 'LD_LIBRARY_PATH')}
    env.update(DRY_RUN='1', PYTHON_BIN=sys.executable)
    env.update(options)
    result = subprocess.run(['bash', str(entry or BASE / f'deploy_{mode}.sh')],
                            env=env, text=True, capture_output=True, timeout=30)
    cfg = json.loads(result.stdout) if result.returncode == 0 else None
    return result, cfg


@pytest.mark.parametrize('mode,ep,eplb', [('aggre', True, True), ('pd', True, True), ('pd_6p2d', True, True)])
def test_default_switches(mode, ep, eplb):
    result, cfg = preview(mode)
    assert result.returncode == 0, result.stderr
    assert cfg['max_num_batched_tokens'] == 65536 and cfg['max_num_seqs'] == (32 if mode == 'aggre' else 512)
    assert cfg['prefix_caching'] is True
    assert cfg['reset_prefix_cache_enabled'] is True
    for server in cfg['servers']:
        cmd = server['command']
        assert cmd[cmd.index('--max-num-batched-tokens') + 1] == '65536'
        assert cmd[cmd.index('--max-num-seqs') + 1] == ('32' if mode == 'aggre' else '512')
        assert '--enable-prefix-caching' in cmd
        assert '--no-enable-prefix-caching' not in cmd
        assert server['ep'] is ep and server['eplb'] is eplb
        assert ('--enable-expert-parallel' in server['command']) is ep
        assert ('--enable-eplb' in server['command']) is eplb
        if mode == 'aggre':
            assert server['speculative_config']['method'] == 'dflash'
            assert server['speculative_config']['synthetic_acceptance_length'] == 4.2
        else:
            assert server['speculative_config'] is None


@pytest.mark.parametrize('mode', ['aggre', 'pd', 'pd_6p2d'])
@pytest.mark.parametrize('method', ['mtp', 'dflash'])
def test_speculative_forwarding(mode, method):
    extra = dict(DRAFT_MODEL_PATH='/models/draft with spaces' if method == 'dflash' else '')
    result, cfg = preview(mode, SPECULATIVE_METHOD=method, NUM_SPECULATIVE_TOKENS='7',
                          SPECULATIVE_CONFIG=json.dumps({'rejection_sample_method': 'synthetic',
                                                         'synthetic_acceptance_length': 4.2}), **extra)
    assert result.returncode == 0, result.stderr
    expected = dict(method=method, num_speculative_tokens=7, rejection_sample_method='synthetic',
                    synthetic_acceptance_length=4.2)
    if method == 'dflash':
        expected['model'] = extra['DRAFT_MODEL_PATH']
    assert cfg['speculative_config'] == expected
    for server in cfg['servers']:
        cmd = server['command']
        assert cmd.count('--speculative-config') == 1
        assert json.loads(cmd[cmd.index('--speculative-config') + 1]) == expected
        assert server['speculative_config'] == expected


def test_speculative_json_override():
    result, cfg = preview('aggre', SPECULATIVE_METHOD='mtp', NUM_SPECULATIVE_TOKENS='7',
                          SPECULATIVE_CONFIG='{"num_speculative_tokens":3,"model":"/draft"}')
    assert result.returncode == 0, result.stderr
    assert cfg['speculative_config'] == {'method': 'mtp', 'num_speculative_tokens': 3, 'model': '/draft'}


@pytest.mark.parametrize('mode', ['pd', 'pd_6p2d'])
@pytest.mark.parametrize('ep,eplb', [('0', '0'), ('1', '0'), ('1', '1')])
def test_independent_pd_switches(mode, ep, eplb):
    result, cfg = preview(mode, ENABLE_EP='0', ENABLE_EPLB='0',
                          P_ENABLE_EP='0', P_ENABLE_EPLB='0', D_ENABLE_EP=ep, D_ENABLE_EPLB=eplb)
    assert result.returncode == 0, result.stderr
    p, d = cfg['servers']
    assert cfg['ep'] is (ep == '1') and cfg['eplb'] is (eplb == '1')
    assert not p['ep'] and not p['eplb']
    assert '--enable-expert-parallel' not in p['command'] and '--enable-eplb' not in p['command']
    assert d['ep'] is (ep == '1') and d['eplb'] is (eplb == '1')
    assert ('--enable-expert-parallel' in d['command']) is (ep == '1')
    assert ('--enable-eplb' in d['command']) is (eplb == '1')


@pytest.mark.parametrize('mode', ['aggre', 'pd', 'pd_6p2d'])
def test_global_switch_compatibility(mode):
    result, cfg = preview(mode, ENABLE_EP='1', ENABLE_EPLB='1', EPLB_WINDOW_SIZE='16')
    assert result.returncode == 0, result.stderr
    for server in cfg['servers']:
        assert server['ep'] and server['eplb']
        assert server['eplb_config']['window_size'] == 16


@pytest.mark.parametrize('options', [
    {'SPECULATIVE_METHOD': 'invalid'},
    {'SPECULATIVE_METHOD': 'none', 'SPECULATIVE_CONFIG': '{}'},
    {'SPECULATIVE_METHOD': 'dflash', 'DRAFT_MODEL_PATH': ''},
    {'SPECULATIVE_METHOD': 'mtp', 'NUM_SPECULATIVE_TOKENS': '0'},
    {'SPECULATIVE_METHOD': 'mtp', 'SPECULATIVE_CONFIG': '[]'},
    {'SPECULATIVE_METHOD': 'mtp', 'SPECULATIVE_CONFIG': 'not json'},
    {'SPECULATIVE_METHOD': 'mtp', 'SPECULATIVE_CONFIG': '{"method":"dflash"}'},
    {'SPECULATIVE_METHOD': 'mtp', 'SPECULATIVE_CONFIG': '{"num_speculative_tokens":true}'},
    {'P_ENABLE_EP': '0', 'P_ENABLE_EPLB': '1'},
    {'D_ENABLE_EP': '0', 'D_ENABLE_EPLB': '1'},
])
def test_invalid_options_rejected(options):
    result, cfg = preview('pd_6p2d', **options)
    assert result.returncode != 0 and cfg is None


def test_tp_dp_override():
    result, cfg = preview('pd', P_TP='1', P_DP='4', D_TP='2', D_DP='2')
    assert result.returncode == 0, result.stderr
    assert [(s['tp'], s['dp']) for s in cfg['servers']] == [(1, 4), (2, 2)]
    for server in cfg['servers']:
        cmd = server['command']
        assert cmd[cmd.index('--tensor-parallel-size') + 1] == str(server['tp'])
        assert cmd[cmd.index('--data-parallel-size') + 1] == str(server['dp'])


def test_6p2d_does_not_source_pd(tmp_path):
    entry = tmp_path / 'deploy_pd_6p2d.sh'
    shutil.copyfile(BASE / entry.name, entry)
    (tmp_path / 'deploy_pd.sh').write_text('exit 97\n')
    (tmp_path / 'code').symlink_to(BASE / 'code', target_is_directory=True)
    result, cfg = preview('pd_6p2d', entry=entry, ENABLE_EP='1', ENABLE_EPLB='1', SPECULATIVE_METHOD='mtp')
    assert result.returncode == 0, result.stderr
    assert cfg['deployment_mode'] == 'pd_6p2d'
    assert cfg['port'] == 19200
    assert [(s['tp'], s['dp']) for s in cfg['servers']] == [(2, 3), (2, 1)]


@pytest.mark.parametrize('mode', ['aggre', 'pd', 'pd_6p2d'])
@pytest.mark.parametrize('value', ['0', '1'])
def test_prefix_caching_switch(mode, value):
    result, cfg = preview(mode, ENABLE_PREFIX_CACHING=value)
    assert result.returncode == 0, result.stderr
    assert cfg['prefix_caching'] is (value == '1')
    assert cfg['environment']['ENABLE_PREFIX_CACHING'] == value
    for server in cfg['servers']:
        assert server['prefix_caching'] is (value == '1')
        assert ('--enable-prefix-caching' in server['command']) is (value == '1')
        assert ('--no-enable-prefix-caching' in server['command']) is (value == '0')


def test_invalid_prefix_caching_switch():
    result, cfg = preview('aggre', ENABLE_PREFIX_CACHING='yes')
    assert result.returncode != 0 and cfg is None
    assert 'ENABLE_PREFIX_CACHING must be 0 or 1' in result.stderr


@pytest.mark.parametrize('mode,bootstrap_port', [('pd', 6500), ('pd_6p2d', 6600)])
@pytest.mark.parametrize('connector', ['NixlConnector', 'MooncakeConnector'])
def test_pd_connector_commands(mode, bootstrap_port, connector):
    result, cfg = preview(mode, KV_CONNECTOR=connector,
                          SPECULATIVE_METHOD='dflash', SPECULATIVE_CONFIG='',
                          KV_EXTRA_CONFIG='{"mooncake_protocol":"rdma"}' if connector == 'MooncakeConnector' else '{}')
    assert result.returncode == 0, result.stderr
    assert cfg['kv_connector'] == connector
    assert cfg['speculative_config']['method'] == 'dflash'
    assert 'rejection_sample_method' not in cfg['speculative_config']
    assert cfg['proxy_command'][cfg['proxy_command'].index('--kv-connector') + 1] == connector
    for server in cfg['servers']:
        cmd = server['command']
        transfer = json.loads(cmd[cmd.index('--kv-transfer-config') + 1])
        assert transfer['kv_connector'] == connector
        assert transfer['kv_load_failure_policy'] == 'fail'
        if connector == 'MooncakeConnector':
            assert transfer['kv_role'] == ('kv_producer' if server['role'] == 'prefill' else 'kv_consumer')
            assert transfer['kv_connector_extra_config']['mooncake_protocol'] == 'rdma'
            assert not any('NIXL' in arg for arg in cmd)
            assert 'VLLM_HOST_IP=127.0.0.1' in cmd
            assert (f'VLLM_MOONCAKE_BOOTSTRAP_PORT={bootstrap_port}' in cmd) == (server['role'] == 'prefill')
        else:
            assert transfer['kv_role'] == 'kv_both'
            assert f"VLLM_NIXL_SIDE_CHANNEL_PORT={server['nixl_port']}" in cmd
    if connector == 'MooncakeConnector':
        assert cfg['ports'].count(bootstrap_port) == 1
        assert cfg['mooncake_bootstrap_url'] in cfg['proxy_command']
        assert cfg['proxy_command'][cfg['proxy_command'].index('--prefill-dp-size') + 1] == str(cfg['servers'][0]['dp'])
        assert cfg['proxy_command'][cfg['proxy_command'].index('--prefill-tp-size') + 1] == str(cfg['servers'][0]['tp'])


@pytest.mark.parametrize('options,error', [
    ({'P_MOONCAKE_BOOTSTRAP_PORT': '19210'}, 'disjoint'),
    ({'KV_CONNECTOR': 'UnknownConnector'}, 'KV_CONNECTOR'),
    ({'KV_EXTRA_CONFIG': '{"backends":["UCX"]}'}, 'mooncake_protocol'),
    ({'KV_EXTRA_CONFIG': '{"bootstrap_addr":"http://external:8998"}'}, 'P_MOONCAKE_BOOTSTRAP_PORT'),
])
def test_mooncake_rejects_conflicting_configuration(options, error):
    result, cfg = preview('pd_6p2d', **dict({'KV_CONNECTOR': 'MooncakeConnector'}, **options))
    assert cfg is None and result.returncode != 0
    assert error in result.stderr


def test_mooncake_options_and_speculation_can_be_overridden():
    result, cfg = preview('pd_6p2d', KV_CONNECTOR='MooncakeConnector',
                          P_MOONCAKE_BOOTSTRAP_PORT='6650', SPECULATIVE_METHOD='none',
                          KV_EXTRA_CONFIG='{"mooncake_protocol":"tcp","num_workers":4}')
    assert result.returncode == 0, result.stderr
    assert cfg['speculative_config'] is None
    assert cfg['kv_extra_config'] == {'mooncake_protocol': 'tcp', 'num_workers': 4}
    assert cfg['mooncake_bootstrap_url'] == 'http://127.0.0.1:6650'
    assert [(s['tp'], s['dp']) for s in cfg['servers']] == [(2, 3), (2, 1)]


def test_container_transport_and_backend_defaults():
    result, cfg = preview('pd')
    assert result.returncode == 0, result.stderr
    assert cfg['kv_connector'] == 'NixlConnector'
    assert cfg['kv_extra_config'] == {}
    for server in cfg['servers']:
        cmd = server['command']
        assert '--moe-backend' not in cmd
    result, cfg = preview('aggre', MOE_BACKEND='marlin')
    assert result.returncode == 0, result.stderr
    cmd = cfg['servers'][0]['command']
    assert cmd[cmd.index('--moe-backend') + 1] == 'marlin'


@pytest.mark.parametrize('mode', ['aggre', 'pd', 'pd_6p2d'])
def test_image_runtime_ignores_source_and_dependency_overrides(mode, tmp_path):
    # A caller's Python search path must not replace image packages or stdlib.
    (tmp_path / 'json.py').write_text("raise RuntimeError('external module loaded')")
    result, cfg = preview(mode, VLLM_SOURCE=str(tmp_path), DEPS_DIR=str(tmp_path),
                          PYTHONPATH=str(tmp_path), PYTHONHOME=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert 'VLLM_SOURCE' not in cfg['environment']
    assert 'DEPS_DIR' not in cfg['environment']
    for server in cfg['servers']:
        cmd = server['command']
        assert cmd[cmd.index('-m') - 1] == '-I'
        assert cmd[cmd.index('-m') + 1] == 'vllm.entrypoints.cli.main'
    if mode != 'aggre':
        assert cfg['proxy_command'][1] == '-I'


@pytest.mark.parametrize('mode', ['aggre', 'pd', 'pd_6p2d'])
def test_disable_reset_prefix_cache(mode):
    result, cfg = preview(mode, ENABLE_RESET_PREFIX_CACHE='0')
    assert result.returncode == 0, result.stderr
    assert cfg['reset_prefix_cache_enabled'] is False
