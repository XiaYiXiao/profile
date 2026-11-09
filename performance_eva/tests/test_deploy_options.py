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
    assert cfg['max_num_batched_tokens'] == 65536 and cfg['max_num_seqs'] == 512
    assert cfg['prefix_caching'] is (mode != 'aggre')
    for server in cfg['servers']:
        cmd = server['command']
        assert cmd[cmd.index('--max-num-batched-tokens') + 1] == '65536'
        assert cmd[cmd.index('--max-num-seqs') + 1] == '512'
        assert ('--enable-prefix-caching' in cmd) is (mode != 'aggre')
        assert ('--no-enable-prefix-caching' in cmd) is (mode == 'aggre')
        assert server['ep'] is ep and server['eplb'] is eplb
        assert ('--enable-expert-parallel' in server['command']) is ep
        assert ('--enable-eplb' in server['command']) is eplb
        assert server['speculative_config'] is None
        assert '--speculative-config' not in server['command']


@pytest.mark.parametrize('mode', ['aggre', 'pd', 'pd_6p2d'])
@pytest.mark.parametrize('method', ['mtp', 'dflash'])
def test_speculative_forwarding(mode, method):
    extra = dict(DRAFT_MODEL_PATH='/models/draft with spaces') if method == 'dflash' else {}
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
    assert [(s['tp'], s['dp']) for s in cfg['servers']] == [(2, 3), (1, 2)]


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
