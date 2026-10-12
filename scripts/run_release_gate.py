#!/usr/bin/env python3
"""Verify the public build and tests in disposable storage outside /tmp."""
import argparse
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from saas_bench.process_boundary import Boundary, BoundaryOpen
from saas_bench.runtime_temp import prepare_tmpdir
from saas_bench.run_state import write_json


def run(directory, tests):
    deadline = time.monotonic() + 6 * 3600
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    temporary_root = prepare_tmpdir(os.environ.get('TMPDIR', directory / 'tmp'))
    if temporary_root == Path('/tmp') or Path('/tmp') in temporary_root.parents:
        raise ValueError('Release gate temporary storage must be outside /tmp')
    source = Path(__file__).resolve().parents[1]
    record = dict(source_root=str(source), checks=[], tmpdir=str(temporary_root))
    temporary = tempfile.mkdtemp(prefix='release-', dir=temporary_root)
    cleanup_safe = True
    try:
        env = dict(os.environ, TMPDIR=temporary, CEOBENCH_TEAM_ARTIFACTS=temporary + '/team',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MALLOC_ARENA_MAX='2',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(source / 'src'))
        commands = [[sys.executable, '-c', "from pathlib import Path; from saas_bench.run_state import verify_build; verify_build(Path('public'), root=Path.cwd())"],
                    [sys.executable, '-m', 'pytest', '-ra', '--basetemp=' + temporary + '/pytest',
                     '--junitxml=' + str(directory / 'pytest.xml'), *tests]]
        for index, command in enumerate(commands):
            memory_limit = resource.getrlimit(resource.RLIMIT_AS)[1]
            memory_limit = 2 * 1024**3 if memory_limit == resource.RLIM_INFINITY else min(memory_limit, 2 * 1024**3)
            boundary = Boundary(shlex.join(command), sys.executable, verified=True,
                memory_limit_bytes=memory_limit)
            try:
                process = subprocess.Popen(['bash', '-c', boundary.command], cwd=source, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                cleanup_safe = False
                stdout, stderr = boundary.communicate(process, max(.001, deadline - time.monotonic()))
                cleanup_safe = boundary.record.get('descendants_closed') is True
                exit_code = 124 if boundary.record.get('timed_out') else process.returncode
                record['checks'].append(dict(command=command, exit_code=exit_code,
                                             process_boundary=boundary.record))
                (directory / f'check-{index}-stdout.log').write_bytes(stdout)
                (directory / f'check-{index}-stderr.log').write_bytes(stderr)
                sys.stdout.buffer.write(stdout)
                sys.stderr.buffer.write(stderr)
                if not boundary.record.get('descendants_closed'):
                    raise RuntimeError('Release gate child resources remain open')
                if exit_code:
                    break
            except BoundaryOpen as exc:
                record['checks'].append(dict(command=command, exit_code=None, process_boundary=exc.record,
                                             preserved_pid=exc.process.pid))
                (directory / f'check-{index}-stdout.log').write_bytes(exc.stdout)
                (directory / f'check-{index}-stderr.log').write_bytes(exc.stderr)
                raise
            finally:
                boundary.close()
    finally:
        try:
            if cleanup_safe:
                def writable(func, path, error):
                    if not isinstance(error, PermissionError) or func not in (os.open, os.unlink, os.rmdir):
                        raise error
                    target = Path(path) if func is os.open else Path(path).parent
                    descriptor = os.open(target, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        parent = os.fstat(descriptor)
                        if parent.st_uid != os.getuid():
                            raise error
                        os.chmod(f'/proc/self/fd/{descriptor}', parent.st_mode | 0o700)
                    finally:
                        os.close(descriptor)
                    if func is os.open:
                        shutil.rmtree(path, onexc=writable)
                    else:
                        func(path)
                shutil.rmtree(temporary, onexc=writable)
        except Exception as exc:
            record['cleanup_error'] = str(exc)
            raise
        finally:
            record['temporary_cleaned'] = not Path(temporary).exists()
            record['preserved_tmpdir'] = temporary if Path(temporary).exists() else None
            write_json(directory / 'release-gate.json', record)
    return record['checks'][-1]['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='New acceptance artifact directory outside /tmp')
    parser.add_argument('tests', nargs='*', default=['tests'])
    args = parser.parse_args()
    return run(args.directory, args.tests or ['tests'])


if __name__ == '__main__':
    raise SystemExit(main())
