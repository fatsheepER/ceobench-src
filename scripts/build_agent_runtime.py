#!/usr/bin/env python3
"""Build the Python runtime mounted read-only at /opt/python in the agent sandbox.

The runtime is a copy of this interpreter's base CPython plus the analysis
packages the agent may import, copied at the exact versions installed in the
current environment. Nothing else from the development environment (simulator
source links, provider SDKs, build and test tools) enters the sandbox.

Usage:
    .venv/bin/python scripts/build_agent_runtime.py [--output agent-runtime]
"""

import argparse
import compileall
import importlib.metadata as metadata
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

from packaging.requirements import Requirement

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from saas_bench.run_state import tree_hash, write_json  # noqa: E402

PACKAGES = ('numpy', 'pandas', 'scikit-learn')


def closure(names):
    found, pending = {}, list(names)
    while pending:
        dist = metadata.distribution(pending.pop())
        name = dist.metadata['Name'].lower()
        if name in found:
            continue
        found[name] = dist
        for text in dist.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
    return dict(sorted(found.items()))


def build(output):
    base = Path(sys.base_prefix)
    version = f'python{sys.version_info.major}.{sys.version_info.minor}'
    staging = Path(tempfile.mkdtemp(prefix='.agent-runtime-', dir=output.parent))
    try:
        python = staging / 'python'
        site = python / 'lib' / version / 'site-packages'
        shutil.copytree(base, python, symlinks=True, ignore=lambda directory, names: (
            names if Path(directory) == base / 'lib' / version / 'site-packages' else
            [n for n in names if n.startswith('pip')] if Path(directory) == base / 'bin' else []))
        dists = closure(PACKAGES)
        for dist in dists.values():
            for file in dist.files:
                source = Path(dist.locate_file(file))
                target = site / file
                if '..' in file.parts or '__pycache__' in file.parts or not source.is_file():
                    continue  # Console scripts and caches stay out of the runtime.
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        compileall.compile_dir(site, quiet=1, workers=0)
        record = {'version': 1, 'python': platform.python_version(),
                  'packages': {name: dist.version for name, dist in dists.items()},
                  'sha256': tree_hash(python)}
        write_json(staging / 'runtime.json', record)
        if output.exists():
            shutil.rmtree(output)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'agent-runtime')
    record = build(parser.parse_args().output.resolve())
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
