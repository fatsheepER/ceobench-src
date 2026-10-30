"""PF is a separate command with shell-safe routing corrections."""
import shlex
import subprocess
import sys

import pytest

from test_preflight_integration import packed_public


@pytest.mark.parametrize('client', ['novamind-client', 'novamind-operation'])
def test_packed_wrong_pf_route_is_nonzero_and_safely_copyable(packed_public, client):
    args = ['depend', "r1; printf 'unsafe'", '$(touch unwanted)']
    result = subprocess.run([sys.executable, str(packed_public / client), 'pf', *args], text=True, capture_output=True)
    assert result.returncode == 2
    correction = result.stderr.strip().split('Run: ', 1)[1]
    assert shlex.split(correction) == ['pf', *args]
    assert result.stdout == ''


def test_private_cli_wrong_pf_route_uses_same_correction(monkeypatch, capsys):
    from saas_bench.novamind_cli import operation_main
    args = ['depend', '$(touch unwanted)']
    monkeypatch.setattr(sys, 'argv', ['novamind-operation', 'pf', *args])
    with pytest.raises(SystemExit) as exc:
        operation_main()
    assert exc.value.code == 2
    assert shlex.split(capsys.readouterr().err.split('Run: ', 1)[1]) == ['pf', *args]
