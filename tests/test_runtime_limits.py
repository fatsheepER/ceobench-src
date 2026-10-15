import argparse
import json
import resource
import subprocess

import pytest

from saas_bench.process_boundary import Boundary, command_memory_limit
from saas_bench.runtime_options import add_runtime_arguments, runtime_environment


@pytest.mark.parametrize('value', ['0', 'max', 'infinity', 'unlimited'])
def test_bash_memory_limit_can_be_disabled(monkeypatch, value):
    monkeypatch.setenv('CEOBENCH_BASH_MEMORY_MAX_BYTES', value)
    assert command_memory_limit() == 0
    boundary = Boundary("python -c 'import resource,json; print(json.dumps(resource.getrlimit(resource.RLIMIT_AS)))'",
                        verified=True, memory_limit_bytes=0)
    try:
        process = subprocess.Popen(['bash', '-c', boundary.command], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = boundary.communicate(process, 5)
        assert process.returncode == 0, stderr
        assert json.loads(stdout) == list(resource.getrlimit(resource.RLIMIT_AS))
        assert boundary.record['memory_limit_bytes'] == 0
        assert boundary.record['boundary_closed'] is True
    finally:
        boundary.close()


def test_launch_runtime_options_export_only_explicit_values():
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    assert runtime_environment(parser.parse_args([])) == {}
    args = parser.parse_args(['--bash-memory-max', 'max', '--model-request-attempts', 'unlimited',
                              '--model-retry-delays', '1,2,3', '--advance-timeout', '9000'])
    assert runtime_environment(args) == {
        'CEOBENCH_BASH_MEMORY_MAX_BYTES': '0',
        'CEOBENCH_MODEL_REQUEST_ATTEMPTS': '0',
        'CEOBENCH_MODEL_RETRY_DELAYS': '1,2,3',
        'CEOBENCH_ADVANCE_TIMEOUT_SECONDS': '9000.0',
    }
