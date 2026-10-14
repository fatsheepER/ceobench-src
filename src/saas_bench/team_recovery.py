"""Find complete checkpoints without approvals or changes to prior attempts."""
import json
import logging
import os
from pathlib import Path


def execution_root(manifest_path, manifest):
    return Path(__file__).resolve().parents[2]


def hold_path(manifest_path):
    return Path(manifest_path).parent / 'HOLD'


def latest_checkpoint(entry, previous=None, week=None):
    from .team_checkpoint import validate_snapshot
    if week is not None and (type(week) is not int or week < 1):
        raise ValueError('Recovery week must be a positive integer; week 1 starts at day 0')
    root = Path(entry['output_dir']).resolve()
    history = sorted(root.glob('attempt-*'), key=lambda path: int(path.name.removeprefix('attempt-')))
    recovered = False
    latest = previous or {}
    for attempt in history:
        record = attempt / 'attempt.json'
        if not record.exists():
            continue
        try:
            state = json.loads(record.read_text())
            recovered |= bool(state.get('source_checkpoint'))
            if attempt == history[-1]:
                latest = state
        except (OSError, ValueError, AttributeError) as exc:
            logging.warning('Recovery could not read attempt metadata %s: %s', record, exc)
    requested = json.loads(os.environ.get('CEOBENCH_RECOVERY_CHECKPOINTS', '{}')) if week is None else {}
    if not isinstance(requested, dict) or any(not isinstance(key, str) or not isinstance(value, str)
            for key, value in requested.items()):
        raise ValueError('Recovery checkpoint selection must map run IDs to snapshot paths')
    requested = requested.get(entry['run_id'])
    candidates = [Path(requested)] if requested and not recovered else [
        snapshot for attempt in history for snapshot in (attempt / 'runtime/checkpoints').glob('*')
        if snapshot.is_dir() and not snapshot.name.startswith('.')]
    if week is None and latest.get('source_checkpoint'):
        candidates = [snapshot for snapshot in (history[-1] / 'runtime/checkpoints').glob('*')
            if snapshot.is_dir() and not snapshot.name.startswith('.')] + [Path(latest[key])
            for key in ('checkpoint', 'source_checkpoint') if latest.get(key)]
        candidates = list(dict.fromkeys(candidates))
    valid = []
    for candidate in candidates:
        try:
            snapshot, state = validate_snapshot(candidate)
            if not snapshot.is_relative_to(root):
                raise ValueError('Recovery checkpoint belongs to another run')
            if (state['stage'] not in ('boundary', 'ceo') or
                    set(state['roles']) != {'ceo', 'growth', 'ops_finance'} or
                    any(role['identity']['run_id'] != entry['run_id'] for role in state['roles'].values())):
                raise ValueError('Recovery checkpoint has invalid run identity or handoff')
            if week is not None and (state['stage'] != 'boundary' or state['day'] != 7 * (week - 1)):
                continue
            valid.append((state.get('saved_at') or '', snapshot.stat().st_mtime_ns, str(snapshot), snapshot))
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            logging.warning('Recovery skipped invalid checkpoint %s: %s', candidate, exc)
            if requested and not recovered:
                raise
    if week is not None and not valid:
        raise ValueError(f'No complete saved checkpoint for week {week}, day {7 * (week - 1)} in {root}')
    return max(valid)[-1] if valid else None


def recovery_source(manifest_path, entry, previous, directory, week=None):
    return latest_checkpoint(entry, previous, week=week)
