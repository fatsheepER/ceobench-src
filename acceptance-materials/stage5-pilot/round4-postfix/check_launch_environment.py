"""Regression for the failed detached startup; keep the virtualenv symlink path."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
interpreter=root/'.venv/bin/python'
check=subprocess.run([str(interpreter),str(root/'acceptance-materials/stage5-pilot/supervise_postfix.py'),'--self-check'],capture_output=True,text=True)
assert check.returncode==0,check.stdout+check.stderr
assert 'supervisor self-check passed' in check.stdout
print('detached-launch environment check passed')
