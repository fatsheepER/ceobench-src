"""Frozen shared-workspace contracts and offline launch checks for the D56 pair."""
import json
from contextlib import closing
from pathlib import Path
import subprocess
import tempfile

from .agent_capabilities import AgentCapabilities
from .role_policy import CALL_SPECS, RESPONSIBILITY, ROLES
from .run_state import file_hash, tree_hash, verify_build
from .run_warnings import AttemptWarnings


PROFILE = 'r5-d56'
FIFO_PROFILE = 'r5-s41-44-d56'
PROFILE_SEEDS = {PROFILE: (1048, 1049), FIFO_PROFILE: (41, 42, 43, 44)}
LAUNCHER = 'scripts/round5_main.py'


def scheduling_contract(profile, runs):
    contract = dict(mode='group_independent', queues={group:
        [r['run_id'] for r in runs if r['group'] == group] for group in ('git', 'pf')})
    if profile == FIFO_PROFILE:
        contract.update(slots={'git': 2, 'pf': 2}, max_workers=4,
            isolated_hold='continue_fifo', systemic_hold='stop_all', restart='never_replay')
    return contract


def terminal_checkpoint(entry, record):
    from .team_checkpoint import validate_snapshot, validate_runtime_data
    if (record.get('status') != 'finished' or
            record.get('resource_cleanup', {}).get('status') != 'closed'):
        raise ValueError('R5 terminal attempt is not finished and closed')
    day, result = record.get('day'), record.get('result', {})
    observation = (record.get('reason') == result.get('stop_reason') == 'observation_end' and
        day == result.get('day') == entry['stop_after_day'] == 56 and result.get('status') == 'reached_stop_day')
    bankruptcy = (record.get('reason') in ('bankrupt', 'natural_end') and
        result.get('stop_reason') == record['reason'] and result.get('status') == 'natural_bankruptcy' and
        type(day) is int and 0 <= day <= 56 and result.get('day') == day)
    if (not (observation or bankruptcy) or record.get('runtime_failure') not in
            ((None, 'natural_end') if bankruptcy else (None,))):
        raise ValueError('R5 terminal is neither observation_end D56 nor natural bankruptcy')
    snapshot = Path(record['checkpoint']).resolve()
    attempt = Path(entry['output_dir']) / f'attempt-{record["attempt"]:03}'
    if not snapshot.is_relative_to(attempt.resolve() / 'runtime/checkpoints'):
        raise ValueError('R5 terminal checkpoint is outside its attempt')
    with AttemptWarnings() as warnings:
        snapshot, state = validate_snapshot(snapshot)
        validate_runtime_data(snapshot, state)
    if warnings.rows:
        raise ValueError('R5 terminal checkpoint validation: ' + '; '.join(r['message'] for r in warnings.rows))
    pointer = json.loads((attempt / 'runtime/checkpoint.json').read_text())
    if (pointer != dict(snapshot=str(snapshot), sha256=file_hash(snapshot / 'checkpoint.json')) or
            state['day'] != day or state['seed'] != entry['seed'] or
            observation and state['stage'] != 'boundary' or state.get('failure') != record.get('runtime_failure') or
            any(role['identity']['run_id'] != entry['run_id'] for role in state['roles'].values())):
        raise ValueError('R5 terminal checkpoint day, identity, boundary or publication pointer differs')
    from .db_protection import load_session_db
    with closing(load_session_db(snapshot / 'world.nmdb')) as conn:
        row = conn.execute("SELECT state_json FROM _rng_states WHERE name='all'").fetchone()
        if not row or json.loads(row[0])['_sim_state']['current_day'] != day:
            raise ValueError('R5 terminal checkpoint world day differs')
    return dict(kind='observation_end' if observation else 'natural_bankruptcy',
        day=day, checkpoint=str(snapshot), checkpoint_sha256=pointer['sha256'], recovery_eligible=True)


def workspace_contract(group):
    return dict(workspace_layout='shared-v1',
        workspace_protocol='snapshot-v1' if group == 'pf' else 'direct-v1',
        comparison_base='independent-per-author-path' if group == 'pf' else None,
        publication='atomic-tool-candidate' if group == 'pf' else 'ordinary-shared-writes',
        git_save_points=['team_reply', 'analyst_handoff', 'week_close'],
        registered_scripts=dict(source='registered-source-snapshot', business_access='read-only',
            file_inputs='current-public-call-snapshot' if group == 'pf' else 'current-shared-files'),
        role_permissions={role: sorted(tool for tool, spec in CALL_SPECS.items()
            if role in RESPONSIBILITY.get(tool, spec.roles)) for role in ROLES},
        advance_roles=['ceo'])


def retention_contract():
    return dict(policy='retain-all-checkpoints-and-attempts', required_day=28, required_stage='boundary',
        contents=['public-files', 'git-history', 'private-authorities', 'role-memory', 'world',
                  'rng', 'drafts-and-bases', 'resolutions-and-confirmations', 'registered-source',
                  'configuration-and-source-identity', 'usage-lineage'],
        fork_execution='deferred')


def validate_contract(manifest):
    from . import team_batch as batch
    from .team_models import validate_profile_models, round_four_models
    from .runtime_options import runtime_environment_snapshot
    profile = manifest.get('profile')
    if profile not in PROFILE_SEEDS:
        raise ValueError('Expected the fifth-round main profile')
    parameters = manifest['parameters']
    seeds = PROFILE_SEEDS[profile]
    if manifest['seeds'] != list(seeds) or [(r['seed'], r['group'], r['repeat']) for r in manifest['runs']] != [
            (seed, group, 1) for seed in seeds for group in ('git', 'pf')]:
        raise ValueError('R5 requires its fixed seeds x Git/PF in ascending order')
    validate_profile_models(profile, parameters['model_configs'], manifest['pricing'])
    if parameters['model_configs'] != round_four_models(profile, manifest['pricing']):
        raise ValueError('R5 model route or request contract differs')
    if parameters.get('runtime_limits') != runtime_environment_snapshot():
        raise ValueError('R5 runtime environment differs from the frozen limits')
    for key, value in batch.PARAMETERS.items():
        if parameters.get(key) != value:
            raise ValueError('R5 runtime parameter differs: ' + key)
    if parameters.get('workspace_contracts') != {group: workspace_contract(group) for group in ('git', 'pf')}:
        raise ValueError('R5 workspace, script or responsibility contract differs')
    if parameters.get('checkpoint_retention') != retention_contract():
        raise ValueError('R5 checkpoint retention contract differs')
    capabilities = {group: AgentCapabilities.round_three(group).to_dict() for group in ('git', 'pf')}
    if parameters.get('capabilities') != capabilities:
        raise ValueError('R5 capability configuration differs')
    for entry in manifest['runs']:
        if (entry.get('workspace_contract') != workspace_contract(entry['group']) or
                entry.get('capabilities') != capabilities[entry['group']] or entry['stop_after_day'] != 56):
            raise ValueError('R5 assignment workspace, capability or observation endpoint differs')
    expected = scheduling_contract(profile, manifest['runs'])
    if manifest.get('scheduling') != expected:
        raise ValueError('R5 requires independent ascending group queues')


def check_launch(policy_path, *, driver_path):
    """Check the executable contract without credentials, model calls or business actions.

    This strict launch audit is separate from advisory operator-selected recovery.
    """
    from . import team_batch as batch
    policy = batch.read(policy_path)['identity']
    path = Path(policy['manifest'])
    with AttemptWarnings() as warnings:
        manifest = batch.validate_manifest(path)
    if warnings.rows:
        raise ValueError('R5 manifest validation: ' + '; '.join(row['message'] for row in warnings.rows))
    validate_contract(manifest)
    digest = file_hash(path)
    if digest != policy['manifest_sha256'] or digest != batch.read(path.parent / 'freeze.json')['manifest_sha256']:
        raise ValueError('R5 manifest freeze identity differs')
    for key in ('batch_id', 'source_root', 'public_dir', 'source_commit'):
        if policy[key] != manifest[key]:
            raise ValueError('R5 policy identity differs: ' + key)
    assigned = {r['run_id']: r['output_dir'] for r in manifest['runs']}
    selection = policy['runs']
    if (not selection or len({r['run_id'] for r in selection}) != len(selection) or
            any(assigned.get(r['run_id']) != r['output_dir'] for r in selection)):
        raise ValueError('R5 policy selection differs from frozen assignments')
    root, public = Path(manifest['source_root']), Path(manifest['public_dir'])
    if (root.resolve() != batch.ROOT.resolve() or
            Path(driver_path).resolve() != root.resolve() / 'scripts/team_driver.py'):
        raise ValueError('R5 executing source root or driver differs from the frozen source root')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if commit != manifest['source_commit'] or tree_hash(root / 'src') != manifest['source_sha256']:
        raise ValueError('R5 source version differs from the frozen version')
    if verify_build(public, root=root) != manifest['build']:
        raise ValueError('R5 public build differs')
    expected_scripts = {*batch.CONTROLLER_SCRIPTS, 'scripts/team_driver.py', LAUNCHER}
    if (set(manifest['controller_sha256']) != expected_scripts or
            any(file_hash(root / name) != digest for name, digest in manifest['controller_sha256'].items())):
        raise ValueError('R5 launch/controller scripts differ')
    if file_hash(root / 'scripts/team_batch.py') != manifest['host_script_sha256']:
        raise ValueError('R5 host script differs')
    from importlib.metadata import version
    if manifest['parameters']['openai_sdk_version'] != version('openai'):
        raise ValueError('R5 installed model SDK differs from the frozen version')
    models = manifest['parameters']['model_configs']
    counter = batch.load_counter(models['agent']['provider'], models['agent']['model'])
    if batch.canonical(counter.metadata if counter else None) != manifest['tokenizer']:
        raise ValueError('R5 installed tokenizer differs from the frozen version')
    configurations = {}
    for group in ('git', 'pf'):
        entry = next(r for r in manifest['runs'] if r['group'] == group)
        with tempfile.TemporaryDirectory(prefix='r5-launch-check-') as temporary:
            team = batch.make_team(manifest, dict(entry, output_dir=temporary), Path(temporary), token_counter=counter)
            try:
                configurations[group] = team.configuration
                contract = workspace_contract(group)
                if any(team.configuration[key] != contract[key] for key in ('workspace_layout', 'workspace_protocol')):
                    raise ValueError('R5 workspace configuration did not reach the runtime')
                if team.configuration['capabilities'] != entry['capabilities']:
                    raise ValueError('R5 capabilities did not reach the runtime')
                if batch.canonical(batch.prompt_templates(team)) != manifest['prompts'][group]:
                    raise ValueError('R5 current role prompts differ from the frozen prompts')
                if team.server.conn.execute('SELECT count(*) FROM business_operations').fetchone()[0]:
                    raise ValueError('R5 preparation unexpectedly contains business operations')
                if any(rr.usage.path.exists() and rr.usage.path.stat().st_size for rr in team.roles.values()):
                    raise ValueError('R5 preparation unexpectedly contains model usage')
            finally:
                batch.close_team(team)
    return dict(prompts_match=True, source_matches=True, runtime_configurations=configurations,
        model_requests=0, business_operations=0, credential_read=False, live_verified=False,
        pending=(['Parent session CloudAgent and grokwatch verification before live launch',
                  'Current provider availability, quota and budget confirmation']
                 if manifest['profile'] == FIFO_PROFILE else
                 ['Final R5-2.1 unified release gate and post-gate check',
                  'User approval of final prompt diff', 'Current provider availability, quota and budget confirmation']))


def retained_d28(entry):
    """List every historical D28 boundary without selecting or mutating a branch point."""
    from . import team_batch as batch
    from .team_checkpoint import validate_snapshot, validate_runtime_data
    rows, errors, reached = [], [], False
    for attempt in batch.attempts(entry):
        record = batch.read_attempt_record(attempt)
        if record is None:
            errors.append(dict(attempt=str(attempt), error='Attempt metadata is unavailable'))
            record = {}
        reached = reached or (record.get('day') is not None and record['day'] >= 28)
        ledger = attempt / 'runtime/checkpoints/.retention.json'
        if not ledger.exists():
            if record.get('checkpoint'):
                errors.append(dict(ledger=str(ledger), error='Checkpoint retention ledger is missing'))
            continue
        try:
            snapshots = batch.read(ledger)['snapshots']
            for saved in snapshots:
                snapshot = Path(saved['snapshot'])
                state = batch.read(snapshot / 'team.json')
                if state['day'] != 28 or state['stage'] != 'boundary':
                    continue
                row = dict(snapshot=str(snapshot), attempt=attempt.name, day=28,
                    stage='boundary', checkpoint_sha256=saved['sha256'], recovery_eligible=False)
                rows.append(row)
                try:
                    if file_hash(snapshot / 'checkpoint.json') != saved['sha256']:
                        raise ValueError('D28 retention identity differs')
                    with AttemptWarnings() as warnings:
                        snapshot, state = validate_snapshot(snapshot)
                        validate_runtime_data(snapshot, state)
                    if warnings.rows:
                        raise ValueError('; '.join(r['message'] for r in warnings.rows))
                    row.update(recovery_eligible=True, configuration=state['configuration'],
                        source_sha256=state['source_sha256'])
                    _, gaps = batch.checkpoint_usage_inventory(snapshot, {}, entry['category'])
                    row.update(usage_lineage_complete=not gaps, usage_lineage_gaps=gaps)
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                    row['error'] = str(exc)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(dict(ledger=str(ledger), error=str(exc)))
    return dict(day=28, stage='boundary', snapshots=rows, errors=errors,
        status=('retained' if all(r.get('usage_lineage_complete') for r in rows) else 'retained_with_usage_gaps')
               if rows and all(r['recovery_eligible'] for r in rows) and not errors
               else 'incomplete' if rows or errors or reached else 'not_reached')
