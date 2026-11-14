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
        from evalscope.perf.plugin.registry import ApiRegistry, DatasetRegistry
        assert args.dataset == 'prepared_requests'
        assert args.tokenizer_path is None
        assert args.api == 'openai_server_usage'
        assert args.min_tokens == args.max_tokens == 1000
        rows = list(DatasetRegistry.get_class(args.dataset)(args).build_messages())
        assert [row[0]['content'] for row in rows] == dataset.read_text().splitlines()
        plugin = ApiRegistry.get_class(args.api)(args)
        assert plugin.tokenizer is None
        assert plugin.build_request(rows[0])['stream_options'] == {'include_usage': True}
        assert plugin.parse_responses([{'usage': {'prompt_tokens': 59, 'completion_tokens': 1000}}]) == (59, 1000)
        root = Path(args.outputs_dir)
        root.mkdir(parents=True)
        (root / 'benchmark_summary.json').write_text(json.dumps({
            'Success Requests': 4, 'Failed Requests': 0,
        }))

    monkeypatch.setattr(per_dataset, 'build_random_dataset', should_not_generate)
    for module in ('datasets.base', 'api.openai_api'):
        monkeypatch.setattr(f'evalscope.perf.plugin.{module}.load_tokenizer', should_not_generate)
    monkeypatch.setattr(per_dataset, 'run_perf_benchmark', fake_perf)
    result = cli.invoke(per_dataset.runner, [
        '--use-chat', '--tokenizer-path', 'test', '--parallel', '4', '--number', '4',
        '--rate', '14', '--output-length', '1000', '--prefix-length', '33',
        '--input-length', '13', '--prepared-dataset-path', str(dataset),
        '--output-dir', str(tmp_path / 'result'),
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert dataset.read_bytes() == original


@pytest.mark.parametrize('chat', [False, True])
@pytest.mark.parametrize('session_headers', [False, True])
def test_saved_long_prompts_skip_all_length_checks(tmp_path, monkeypatch, chat, session_headers):
    from evalscope.perf.plugin.datasets.base import DatasetPluginBase
    from evalscope.perf.plugin.registry import ApiRegistry, DatasetRegistry

    # Longer than EvalScope's default 131072 limit in *characters* as well.
    prompts = ['中文长前缀' * 40000 + ' suffix-a', 'another prompt']
    dataset = tmp_path / 'saved.txt'
    dataset.write_text('\n'.join(prompts) + '\n', encoding='utf-8')
    original = dataset.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail('Saved prompts must not load a tokenizer or check lengths')

    for module in ('datasets.base', 'api.openai_api'):
        monkeypatch.setattr(f'evalscope.perf.plugin.{module}.load_tokenizer', forbidden)
    monkeypatch.setattr(DatasetPluginBase, 'check_prompt_length', forbidden)

    def fake_perf(args):
        assert args.tokenizer_path is None
        assert args.api == ('openai_session_server_usage' if session_headers else 'openai_server_usage')
        rows = list(DatasetRegistry.get_class(args.dataset)(args).build_messages())
        expected = [[{'role': 'user', 'content': prompt}] for prompt in prompts] if chat else prompts
        assert rows == expected
        api = ApiRegistry.get_class(args.api)(args)
        body = api.build_request(rows[0])
        assert body['messages' if chat else 'prompt'] == expected[0]
        assert body['stream_options']['include_usage'] is True
        assert api.parse_responses([
            {'choices': [{'delta': {'content': 'one chunk'}}], 'usage': None},
            {'usage': {'prompt_tokens': 80017, 'completion_tokens': 1000}},
        ]) == (80017, 1000)
        Path(args.outputs_dir).mkdir(parents=True)
        (Path(args.outputs_dir) / 'benchmark_summary.json').write_text(json.dumps({
            'Success Requests': 2, 'Failed Requests': 0,
        }))

    monkeypatch.setattr(per_dataset, 'run_perf_benchmark', fake_perf)
    options = ['--prepared-dataset-path', str(dataset), '--tokenizer-path', 'must-not-be-loaded',
               '--parallel', '1', '--number', '2', '--output-dir', str(tmp_path / 'results'),
               '--no-write-csv']
    if chat:
        options.append('--use-chat')
    if session_headers:
        options.append('--inject-req-ids')
    result = CliRunner().invoke(per_dataset.runner, options)
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert dataset.read_bytes() == original


@pytest.mark.parametrize('usage', [None, {}, {'prompt_tokens': 12},
                                  {'prompt_tokens': 12, 'completion_tokens': None},
                                  {'prompt_tokens': -1, 'completion_tokens': 2}])
def test_saved_requests_require_server_usage(usage):
    parser = per_dataset._ServerUsageOnly()
    with pytest.raises(ValueError, match='usage.prompt_tokens'):
        parser.parse_responses([{'usage': usage, 'choices': [{'text': 'not a token count'}]}])


def test_saved_json_messages_and_request_body_are_preserved(tmp_path):
    from evalscope.perf.arguments import Arguments
    from evalscope.perf.plugin.registry import ApiRegistry, DatasetRegistry

    messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'saved input'}]
    body = {'messages': messages, 'max_tokens': 17, 'temperature': 0.7}
    dataset = tmp_path / 'saved.jsonl'
    dataset.write_text(json.dumps(messages) + '\n' + json.dumps(body) + '\n')
    args = Arguments(model='test', url='http://localhost:9999/v1/chat/completions',
                     api='openai_server_usage', dataset='prepared_requests',
                     dataset_path=str(dataset), tokenizer_path=None,
                     parallel=[1], number=[2], max_tokens=1000, stream=True)
    rows = list(DatasetRegistry.get_class(args.dataset)(args).build_messages())
    assert rows == [messages, body]
    api = ApiRegistry.get_class(args.api)(args)
    assert api.build_request(rows[0])['messages'] == messages
    payload = api.build_request(rows[1])
    assert payload['messages'] == messages
    assert payload['max_tokens'] == 17 and payload['temperature'] == 0.7
    assert payload['stream_options'] == {'include_usage': True}
    assert api.parse_responses([{'usage': {'prompt_tokens': 0, 'completion_tokens': 0}}]) == (0, 0)


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
