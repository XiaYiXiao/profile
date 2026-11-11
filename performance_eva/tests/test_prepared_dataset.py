"""Saved shared-prefix workloads: real BPE boundaries, reuse, and file safety."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from click.testing import CliRunner
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import per_dataset


@pytest.fixture
def tokenizer(monkeypatch):
    # Include a merge and whitespace to exercise decode/encode length changes.
    vocab = {c: i for i, c in enumerate('abcde \n\r中文')}
    vocab.update({'ab': len(vocab), '<unk>': len(vocab) + 1})
    raw = Tokenizer(models.BPE(vocab, merges=[('a', 'b')], unk_token='<unk>'))
    raw.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='<unk>')
    tok.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    monkeypatch.setattr('evalscope.perf.plugin.datasets.utils.load_tokenizer', lambda _: tok)
    monkeypatch.setattr('evalscope.perf.plugin.datasets.base.load_tokenizer', lambda _: tok)
    return tok


@pytest.mark.parametrize('suffix_max', [13, 17])
def test_saved_prefix_round_trip_and_determinism(tmp_path, tokenizer, suffix_max):
    paths = [tmp_path / 'nested' / f'{i}.txt' for i in range(2)]
    for path in paths:
        per_dataset.build_random_dataset(str(path), 'test', 32, 13, suffix_max,
                                         prefix_length=33, seed=42)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    lines = paths[0].read_text().splitlines()
    assert len(lines) == len(set(lines)) == 32
    ids = [tokenizer.encode(line, add_special_tokens=False) for line in lines]
    assert all(46 <= len(row) <= 33 + suffix_max for row in ids)
    assert all(row[:33] == ids[0][:33] for row in ids)
    assert len({row[33] for row in ids}) > 1
    assert all(line == line.strip() and line for line in lines)


def test_cli_prepare_and_reuse_without_regeneration(tmp_path, tokenizer, monkeypatch):
    dataset = tmp_path / 'prompts.txt'
    cli = CliRunner()
    result = cli.invoke(per_dataset.runner, [
        '--prepare-dataset', '--prepare-num', '4', '--prefix-length', '33',
        '--input-length', '13', '--tokenizer-path', 'test',
        '--prepared-dataset-path', str(dataset),
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    original = dataset.read_bytes()
    assert len(original.splitlines()) == 4

    def should_not_generate(*args, **kwargs):
        pytest.fail('Reuse must not regenerate the dataset')

    def fake_perf(args):
        from evalscope.perf.plugin.datasets.line_by_line import LineByLineDatasetPlugin
        assert args.dataset == 'line_by_line'
        assert args.min_tokens == args.max_tokens == 1000
        rows = list(LineByLineDatasetPlugin(args).build_messages())
        assert [row[0]['content'] for row in rows] == dataset.read_text().splitlines()
        root = Path(args.outputs_dir)
        root.mkdir(parents=True)
        (root / 'benchmark_summary.json').write_text(json.dumps({
            'Success Requests': 4, 'Failed Requests': 0,
        }))

    monkeypatch.setattr(per_dataset, 'build_random_dataset', should_not_generate)
    monkeypatch.setattr(per_dataset, 'run_perf_benchmark', fake_perf)
    result = cli.invoke(per_dataset.runner, [
        '--use-chat', '--tokenizer-path', 'test', '--parallel', '4', '--number', '4',
        '--rate', '14', '--output-length', '1000', '--prefix-length', '33',
        '--input-length', '13', '--prepared-dataset-path', str(dataset),
        '--output-dir', str(tmp_path / 'result'),
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert dataset.read_bytes() == original


def test_existing_file_is_preserved(tmp_path, tokenizer):
    dataset = tmp_path / 'existing.txt'
    dataset.write_text('keep this\n')
    with pytest.raises(Exception, match='拒绝覆盖'):
        per_dataset.build_random_dataset(str(dataset), 'test', 2, 13, 13, prefix_length=33)
    assert dataset.read_text() == 'keep this\n'


def test_failed_generation_does_not_publish_partial_file(tmp_path, tokenizer, monkeypatch):
    def fails(*args, **kwargs):
        yield 'first prompt'
        raise RuntimeError('generation failed')
    monkeypatch.setattr(per_dataset, '_shared_prefix_prompts', fails)
    with pytest.raises(RuntimeError, match='generation failed'):
        per_dataset.build_random_dataset(str(tmp_path / 'data.txt'), 'test', 2, 13, 13,
                                         prefix_length=33)
    assert not list(tmp_path.iterdir())


def test_multi_turn_prefix_is_rejected(tmp_path, tokenizer):
    with pytest.raises(Exception, match='多轮数据集不支持'):
        per_dataset.build_random_dataset(str(tmp_path / 'data.jsonl'), 'test', 2, 13, 13,
                                         prefix_length=33, multi_turn=True)


def test_prepare_wrapper_dry_run_preserves_defaults(tmp_path):
    dataset = tmp_path / 'data.txt'
    proc = subprocess.run([
        'bash', str(BASE / 'run_load.sh'), '--dry-run', '--prepare-dataset',
        '--prepare-num', '3000', '--prefix-length', '8000', '--input-length', '1000',
        '--prepared-dataset-path', str(dataset),
    ], env=dict(os.environ, LOAD_PYTHON_BIN=sys.executable),
        capture_output=True, text=True, check=True)
    plan = json.loads(proc.stdout)
    assert plan['action'] == 'prepare_dataset'
    assert plan['parameters']['tokenizer_path'] == '/models/MiMo-V2.6-Flash-MOPD'
    assert plan['parameters']['prefix_length'] == 8000
    assert plan['parameters']['prepare_num'] == 3000
    assert not list(tmp_path.iterdir())
