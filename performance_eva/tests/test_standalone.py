"""Exercise the uploaded-folder layout without access to sibling source trees."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

BASE = Path(__file__).resolve().parents[1]


@pytest.fixture
def standalone(tmp_path):
    folder = tmp_path / 'uploaded folder' / 'performance_eva'
    shutil.copytree(BASE, folder, ignore=shutil.ignore_patterns(
        '.venv', '.venv-load', '__pycache__', '.pytest_cache', '.DS_Store',
        '.cache', '.tmp', 'deployments', 'results', 'outputs',
    ))
    return folder


def run_standalone(folder, command, **options):
    env = {key: value for key, value in os.environ.items()
           if key in ('PATH', 'HOME', 'LANG', 'LD_LIBRARY_PATH')}
    env.update(PYTHON_BIN=sys.executable, LOAD_PYTHON_BIN=sys.executable,
               PYTHONDONTWRITEBYTECODE='1')
    env.update(options)
    # Run outside the copied folder, with no PYTHONPATH from the repository.
    return subprocess.run(command, cwd=folder.parent, env=env,
                          text=True, capture_output=True, timeout=30)


@pytest.mark.parametrize('mode,connector', [
    ('aggre', None), ('pd', 'MooncakeConnector'), ('pd', 'NixlConnector'),
    ('pd_6p2d', 'MooncakeConnector'), ('pd_6p2d', 'NixlConnector'),
])
@pytest.mark.parametrize('method', ['none', 'mtp', 'dflash'])
def test_copied_deployment(standalone, mode, connector, method):
    options = dict(DRY_RUN='1', SPECULATIVE_METHOD=method,
                   SPECULATIVE_CONFIG='', EPLB_WINDOW_SIZE='32')
    if connector:
        options['KV_CONNECTOR'] = connector
    result = run_standalone(standalone,
                            ['bash', str(standalone / f'deploy_{mode}.sh')], **options)
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)
    assert cfg['deployment_mode'] == mode
    assert Path(cfg['root']).relative_to(standalone).parts[0] == 'deployments'
    for server in cfg['servers']:
        assert server['eplb_config']['window_size'] == 32
        assert ('--speculative-config' in server['command']) is (method != 'none')
        if method != 'none':
            assert server['speculative_config']['method'] == method
    if connector:
        assert cfg['kv_connector'] == connector
        assert str(standalone / 'code/pd_proxy.py') in cfg['proxy_command']
    assert not (standalone / 'deployments').exists()


@pytest.mark.parametrize('entry', ['run_load.sh', 'code/pd_proxy.py', 'code/server_metrics.py'])
def test_copied_command_help(standalone, entry):
    command = ['bash'] if entry.endswith('.sh') else [sys.executable, '-I']
    result = run_standalone(standalone, command + [str(standalone / entry), '--help'])
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()
