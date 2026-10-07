"""Pinned official DeepSeek text tokenizer for PF payload comparisons."""
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
import urllib.request

from .sql_evidence import digest


# Official model/template mapping: deepseek-ai/deepseek-recipe/docs/tokenizer.md.
REVISION = '8cadfede7063c896b944e7bae05daa3549ae97ea'
SHA256 = '81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99'


def tokenizer_config(provider, model):
    name = model.lower().rsplit('/', 1)[-1]
    # The experiment's verified API name is deepseek-flash (V4.1, 2026-09-10).
    # Do not infer tokenizers for other models or legacy aliases.
    if provider not in ('deepseek', 'opencode') or name not in ('deepseek-flash', 'deepseek-v4.1-flash'):
        return dict(status='unavailable', reason='unsupported_model', provider=provider, model=model)
    return dict(status='available', provider=provider, model=model,
                tokenizer_id='deepseek-ai/deepseek-recipe/v41', revision=REVISION,
                sha256=SHA256, library='tokenizers', library_version=version('tokenizers'),
                method='Tokenizer.encode(add_special_tokens=False)',
                url=f'https://raw.githubusercontent.com/deepseek-ai/deepseek-recipe/{REVISION}/static/tokenizers/v41/tokenizer.json')


class PayloadTokenCounter:
    def __init__(self, tokenizer, metadata):
        self.tokenizer, self.metadata = tokenizer, metadata

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


@lru_cache(maxsize=4)
def load_counter(provider, model):
    metadata = tokenizer_config(provider, model)
    if metadata['status'] != 'available':
        return None
    from tokenizers import Tokenizer
    path = Path.home() / '.cache' / 'ceobench' / 'tokenizers' / (metadata['sha256'] + '.json')
    if path.exists():
        raw = path.read_bytes()
    else:
        with urllib.request.urlopen(metadata['url'], timeout=60) as response:
            raw = response.read()
        if digest(raw) != metadata['sha256']:
            raise ValueError('Official tokenizer checksum mismatch')
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(raw) != metadata['sha256']:
        raise ValueError('Cached tokenizer checksum mismatch')
    tokenizer = Tokenizer.from_str(raw.decode())
    tokenizer.no_truncation()
    tokenizer.no_padding()
    counter = PayloadTokenCounter(tokenizer, metadata)
    sample = 'PF tokenizer check: 中文🙂\r\nvalue = 42\n'
    if counter.count(sample) <= 0 or counter.count(sample) != counter.count(sample):
        raise RuntimeError('Tokenizer self-check failed')
    return counter
