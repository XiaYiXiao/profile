# Bundled from profile/decode/speculative_config.py; no sibling directory is required.
"""Resolve speculative CLI configuration without importing vLLM."""
import json
import os


def resolve():
    method = os.environ.get('SPECULATIVE_METHOD', 'none').lower()
    if method not in ('none', 'mtp', 'dflash'):
        raise ValueError('SPECULATIVE_METHOD must be none, mtp or dflash')
    raw = os.environ.get('SPECULATIVE_CONFIG', '').strip()
    if method == 'none':
        if raw:
            raise ValueError('Choose mtp/dflash before setting SPECULATIVE_CONFIG')
        return None
    config = {'method': method}
    draft = os.environ.get('DRAFT_MODEL_PATH', '')
    if draft:
        config['model'] = draft
    tokens = os.environ.get('NUM_SPECULATIVE_TOKENS', '')
    if tokens:
        config['num_speculative_tokens'] = int(tokens)
    elif method == 'mtp':
        config['num_speculative_tokens'] = 1
    if raw:
        overrides = json.loads(raw)
        if not isinstance(overrides, dict):
            raise ValueError('SPECULATIVE_CONFIG must be a JSON object')
        config.update(overrides)
    if config.get('method') != method:
        raise ValueError('SPECULATIVE_CONFIG.method must match SPECULATIVE_METHOD')
    if method == 'dflash' and not config.get('model'):
        raise ValueError('DFlash requires DRAFT_MODEL_PATH or SPECULATIVE_CONFIG.model')
    if 'num_speculative_tokens' in config:
        value = config['num_speculative_tokens']
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('num_speculative_tokens must be a positive integer')
    return config


if __name__ == '__main__':
    print(json.dumps(resolve(), separators=(',', ':')))
