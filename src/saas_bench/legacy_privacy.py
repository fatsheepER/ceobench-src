"""Keep unclassified old histories private when restoring a selected world."""
from pathlib import Path
import shutil

from .run_state import write_json


PUBLIC_HISTORY_FORMAT = 1
HISTORY_NOTICE = ('Earlier workspace files, Git and PF history, conversations, registered scripts, '
                  'and declarations are unavailable in this restored attempt. The selected business '
                  'world and simulated day were restored. Inspect current business data and rebuild '
                  'any needed analysis or scripts. Original history is retained privately.')


def quarantine_legacy_runtime(root, public, state=None):
    root, public = Path(root), Path(public)
    private = root / 'private'
    private.mkdir(parents=True, exist_ok=True)
    archive = private / 'legacy-history'
    archive.mkdir(mode=0o700)
    for path in list(private.iterdir()):
        if path != archive:
            shutil.move(str(path), archive / path.name)
    shutil.move(str(root / 'roles'), archive / 'roles')
    from .role_policy import ROLES
    for role in ROLES:
        workspace = root / 'roles' / role
        workspace.mkdir(parents=True)
        shutil.copytree(public / 'docs', workspace / 'docs')
        shutil.copy2(public / 'novamind-client', workspace / 'novamind-operation')
        (workspace / 'MEMORY.md').write_text(HISTORY_NOTICE + '\n')
    if state is None:
        return
    write_json(archive / 'team.json', state)
    state.update(_legacy_history_unavailable=True, public_history_format=PUBLIC_HISTORY_FORMAT,
                 stage='boundary', phase='boundary', week={}, messages=[], sources=[],
                 role_weeks={role: None for role in ROLES}, scripts={role: {} for role in ROLES},
                 script_results={role: [] for role in ROLES}, dashboard=None, day_result=None)
    for saved in state['roles'].values():
        saved.update(context_id=None, last_request_event=None)


def clear_legacy_scripts(conn):
    for table in ('_registered_scripts', '_team_scripts'):
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            conn.execute('DELETE FROM ' + table)
    conn.commit()


def quarantine_legacy_standalone(run, workspace, public, snapshot, session_id):
    import tempfile
    from .db_protection import load_session_db, save_session_db
    run, workspace, public, snapshot = map(Path, (run, workspace, public, snapshot))
    archive = Path(tempfile.mkdtemp(prefix='legacy-history-', dir=run))
    shutil.move(str(workspace), archive / 'agent_workspace')
    for source in list(run.glob('sql-evidence*')):
        if source.is_file():
            shutil.move(str(source), archive / source.name)
    if (run / 'private-agent').exists():
        shutil.move(str(run / 'private-agent'), archive / 'private-agent')
    if (snapshot / 'sql-evidence.sqlite').exists():
        shutil.copy2(snapshot / 'sql-evidence.sqlite', archive / 'selected-evidence.sqlite')
    workspace.mkdir()
    shutil.copytree(public / 'docs', workspace / 'docs')
    shutil.copy2(public / 'novamind-client', workspace / 'novamind-operation')
    (workspace / 'MEMORY.md').write_text(HISTORY_NOTICE + '\n')
    session = workspace / 'sessions' / session_id
    session.mkdir(parents=True)
    for name in ('world.nmdb', 'session.json'):
        shutil.copy2(snapshot / name, session / name)
    conn = load_session_db(session / 'world.nmdb')
    try:
        clear_legacy_scripts(conn)
        save_session_db(conn, session / 'world.nmdb')
    finally:
        conn.close()
    import json
    state = json.loads((snapshot / 'server_state.json').read_text())
    state.update(dashboard='', script_results=[])
    safe = run / 'private-restored-state'
    safe.mkdir(exist_ok=True)
    write_json(safe / 'server_state.json', state)
    return safe
