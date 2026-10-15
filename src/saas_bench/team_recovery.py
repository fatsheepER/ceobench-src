"""Select checkpoints for explicit continuation without changing prior attempts."""
import json
import logging
import os
from pathlib import Path

from .run_warnings import advisory


def execution_root(manifest_path, manifest):
    return Path(__file__).resolve().parents[2]


def hold_path(manifest_path):
    return Path(manifest_path).parent / 'HOLD'


def latest_checkpoint(entry, previous=None, week=None):
    from .team_checkpoint import validate_snapshot
    if week is not None and (type(week) is not int or week < 1):
        logging.warning('Recovery week is not a positive integer; week 1 starts at day 0')
    root = Path(entry['output_dir']).resolve()
    history = sorted(root.glob('attempt-*'), key=lambda path: int(path.name.removeprefix('attempt-')))
    latest = previous or {}
    lineage_attempt = history[-1] if history else root
    for attempt in history:
        record = attempt / 'attempt.json'
        if not record.exists():
            continue
        try:
            state = json.loads(record.read_text())
            if not isinstance(state, dict):
                raise ValueError('Attempt metadata is not a mapping')
            if state.get('reason') == 'missing_checkpoint' and state.get('execution_started') is False:
                continue
            latest, lineage_attempt = state, attempt
        except (OSError, ValueError, AttributeError) as exc:
            logging.warning('Recovery could not read attempt metadata %s: %s', record, exc)
    requested = {}
    if week is None:
        with advisory():
            requested = json.loads(os.environ.get('CEOBENCH_RECOVERY_CHECKPOINTS', '{}'))
    if not isinstance(requested, dict):
        logging.warning('Recovery checkpoint selection is not a mapping; using saved checkpoints')
        requested = {}
    requested = requested.get(entry['run_id'])
    if requested is not None and not isinstance(requested, str):
        logging.warning('Requested checkpoint path is not text; using saved checkpoints')
        requested = None
    candidates = [Path(requested)] if requested else [
        snapshot for attempt in history for snapshot in (attempt / 'runtime/checkpoints').glob('*')
        if snapshot.is_dir() and not snapshot.name.startswith('.')]
    if not requested and week is None and history and (latest.get('source_checkpoint') or latest.get('replay_week') == 1):
        candidates = [snapshot for snapshot in (lineage_attempt / 'runtime/checkpoints').glob('*')
            if snapshot.is_dir() and not snapshot.name.startswith('.')] + [Path(latest[key])
            for key in ('checkpoint', 'source_checkpoint') if latest.get(key)]
        candidates = list(dict.fromkeys(candidates))
    valid = []
    for candidate in candidates:
        try:
            snapshot, state = validate_snapshot(candidate)
            if not snapshot.is_relative_to(root):
                logging.warning('Recovery checkpoint belongs to another run: %s', snapshot)
            if (state['stage'] not in ('boundary', 'ceo') or
                    set(state['roles']) != {'ceo', 'growth', 'ops_finance'} or
                    any(role['identity']['run_id'] != entry['run_id'] for role in state['roles'].values())):
                logging.warning('Recovery checkpoint has invalid run identity or handoff: %s', snapshot)
            if week is not None and (state['stage'] != 'boundary' or state['day'] != 7 * (week - 1)):
                continue
            valid.append((state.get('saved_at') or '', snapshot.stat().st_mtime_ns, str(snapshot), snapshot))
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            logging.warning('Recovery skipped invalid checkpoint %s: %s', candidate, exc)
            if requested:
                return Path(requested).resolve()
    if week is not None and not valid:
        logging.warning('No saved checkpoint for week %s in %s%s', week, root,
                        '; week 1 can use sealed D0' if week == 1 else '; selected world data is missing')
    return max(valid)[-1] if valid else None


def recovery_source(manifest_path, entry, previous, directory, week=None):
    return latest_checkpoint(entry, previous, week=week)
