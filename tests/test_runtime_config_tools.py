from copy import deepcopy
import json
import sqlite3

import pytest

from saas_bench.api_server import _TOOL_DISPATCH
from saas_bench.config import BenchmarkConfig
from saas_bench.database import init_database
from saas_bench.tools import AgentTools


SETTINGS = {
    'set_targeted_ad_spend': {'targeted_spend': {'social_media': {'S1': 10}}},
    'set_targeted_dev_spend': {'targeted_spend': {'S1': 20}},
    'set_targeted_ops_spend': {
        'by_group': {'S1': 30}, 'by_plan': {'A': 40},
        'by_group_plan': {'S1': {'A': 50}}, 'by_customer': {'42': 60},
    },
    'set_ads_strength': {
        'global_strength': .1, 'by_group': {'S1': .2}, 'by_customer': {'42': .3},
    },
    'set_lead_promotion': {
        'global_promotion': 1, 'by_group': {'S1': 2},
        'by_channel': {'social_media': 3}, 'by_channel_group': {'social_media': {'S1': 4}},
    },
    'set_promotion': {
        'global_promotion': 5, 'by_group': {'S1': 6},
        'by_customer': {'42': 7}, 'by_group_plan': {'S1': {'A': 8}},
    },
}


@pytest.fixture
def tools(tmp_path):
    conn = init_database(tmp_path / 'world.db')
    tools = AgentTools(conn, 7, tmp_path / 'workspace', config=BenchmarkConfig())
    for name, args in SETTINGS.items():
        assert getattr(tools, name)(**deepcopy(args)).success
    yield tools
    conn.close()


def snapshot(tools):
    config = {key: value for key, value in vars(tools.config).items()
              if key.startswith(('targeted_', 'ads_strength_', 'lead_promotion_', 'promotion_'))}
    return deepcopy(config), [dict(row) for row in tools.conn.execute(
        'SELECT * FROM config_overrides ORDER BY id'
    )]


@pytest.mark.parametrize('name,args', [
    ('set_targeted_ops_spend', {
        'by_group': {'E1': 100}, 'by_plan': {'B': 200},
        'by_group_plan': {'E1': {'B': 300}}, 'by_customer': {'bad-id': 400},
    }),
    ('set_ads_strength', {'global_strength': .9, 'by_group': {'E1': .8}, 'by_customer': []}),
    ('set_lead_promotion', {
        'global_promotion': 10, 'by_group': {'E1': 20}, 'by_channel': {'linkedin': 30},
        'by_channel_group': {'linkedin': {'unknown-group': 40}},
    }),
    ('set_promotion', {
        'global_promotion': 10, 'by_group': {'E1': 20}, 'by_customer': {'43': 30},
        'by_group_plan': {'E1': {'invalid-plan': 40}},
    }),
])
def test_failed_multiscope_update_changes_nothing(tools, name, args):
    before = snapshot(tools)
    assert not getattr(tools, name)(**args).success
    assert snapshot(tools) == before


SCOPES = [
    ('set_targeted_ops_spend', 'by_group', 'targeted_ops_spend'),
    ('set_targeted_ops_spend', 'by_plan', 'targeted_ops_spend_by_plan'),
    ('set_targeted_ops_spend', 'by_group_plan', 'targeted_ops_spend_by_group_plan'),
    ('set_targeted_ops_spend', 'by_customer', 'targeted_ops_spend_by_customer'),
    ('set_ads_strength', 'by_group', 'ads_strength_by_group'),
    ('set_ads_strength', 'by_customer', 'ads_strength_by_customer'),
    ('set_lead_promotion', 'by_group', 'lead_promotion_by_group'),
    ('set_lead_promotion', 'by_channel', 'lead_promotion_by_channel'),
    ('set_lead_promotion', 'by_channel_group', 'lead_promotion_by_channel_group'),
    ('set_promotion', 'by_group', 'promotion_by_group'),
    ('set_promotion', 'by_customer', 'promotion_by_customer'),
    ('set_promotion', 'by_group_plan', 'promotion_by_group_plan'),
]


@pytest.mark.parametrize('name,scope,field', SCOPES)
@pytest.mark.parametrize('clear', [False, True])
def test_partial_update_keeps_other_scopes_and_can_clear(tools, name, scope, field, clear):
    before, audit = snapshot(tools)
    value = {} if clear else deepcopy(SETTINGS[name][scope])
    for key, amount in value.items():
        value[key] = {k: v * 2 for k, v in amount.items()} if isinstance(amount, dict) else amount * 2
    args = dict.fromkeys(SETTINGS[name])
    args[scope] = deepcopy(value)
    result = getattr(tools, name)(**args)
    assert result.success
    before[field] = {int(k): float(v) for k, v in value.items()} if scope == 'by_customer' else value
    config, new_audit = snapshot(tools)
    assert config == before
    assert new_audit[:-1] == audit and len(new_audit) == len(audit) + 1
    settings = json.loads(new_audit[-1]['settings_json'])
    assert settings.keys() == {'global' if key.startswith('global_') else key for key in SETTINGS[name]}
    assert settings == {k: v for k, v in result.data.items() if k in settings}
    assert settings[scope] == value


@pytest.mark.parametrize('name,scope,field', [
    ('set_ads_strength', 'global_strength', 'ads_strength_global'),
    ('set_lead_promotion', 'global_promotion', 'lead_promotion_global'),
    ('set_promotion', 'global_promotion', 'promotion_global'),
])
def test_global_partial_update_leaves_other_scopes_unchanged(tools, name, scope, field):
    before, audit = snapshot(tools)
    value = SETTINGS[name][scope] * 2
    result = getattr(tools, name)(**{scope: value})
    assert result.success and result.data['global'] == value
    before[field] = value
    config, new_audit = snapshot(tools)
    assert config == before
    assert new_audit[:-1] == audit and len(new_audit) == len(audit) + 1
    assert json.loads(new_audit[-1]['settings_json']) == result.data


def test_ops_legacy_alias_and_integer_customer_keys(tools):
    result = tools.set_targeted_ops_spend(targeted_spend={'E1': 123}, by_customer={'43': 45})
    assert result.success
    assert result.data['targeted_spend'] == result.data['by_group'] == {'E1': 123.0}
    assert tools.config.targeted_ops_spend_by_customer == {43: 45.0}
    assert tools.config.ads_strength_by_customer == {42: .3}
    assert tools.config.promotion_by_customer == {42: 7.0}
    before = snapshot(tools)
    assert not tools.set_targeted_ops_spend(targeted_spend={}, by_group={}).success
    assert snapshot(tools) == before
    assert tools.set_targeted_ops_spend(targeted_spend={}).success
    assert tools.config.targeted_ops_spend == {}


@pytest.mark.parametrize('args,changes', [
    ({'E1': 123}, {'targeted_ops_spend': {'E1': 123.0}}),
    ({'targeted_spend': {'E1': 123}}, {'targeted_ops_spend': {'E1': 123.0}}),
    ({
        'by_group': {'E1': 123}, 'by_plan': {'B': 234},
        'by_group_plan': {'E1': {'B': 345}}, 'by_customer': {'43': 456},
    }, {
        'targeted_ops_spend': {'E1': 123.0}, 'targeted_ops_spend_by_plan': {'B': 234.0},
        'targeted_ops_spend_by_group_plan': {'E1': {'B': 345.0}},
        'targeted_ops_spend_by_customer': {43: 456.0},
    }),
])
def test_ops_dispatch_supports_legacy_and_multiscope_input(tools, args, changes):
    before, audit = snapshot(tools)
    result = _TOOL_DISPATCH['set_targeted_ops_spend'](tools, args)
    assert result.success
    before.update(changes)
    config, new_audit = snapshot(tools)
    assert config == before
    assert result.data['targeted_spend'] == changes['targeted_ops_spend']
    assert new_audit[:-1] == audit and len(new_audit) == len(audit) + 1


NUMERIC_PATHS = [
    ('set_targeted_ad_spend', ('targeted_spend', 'social_media', 'S1')),
    ('set_targeted_dev_spend', ('targeted_spend', 'S1')),
    ('set_targeted_ops_spend', ('by_group', 'S1')),
    ('set_targeted_ops_spend', ('by_plan', 'A')),
    ('set_targeted_ops_spend', ('by_group_plan', 'S1', 'A')),
    ('set_targeted_ops_spend', ('by_customer', '42')),
    ('set_ads_strength', ('global_strength',)),
    ('set_ads_strength', ('by_group', 'S1')),
    ('set_ads_strength', ('by_customer', '42')),
    ('set_lead_promotion', ('global_promotion',)),
    ('set_lead_promotion', ('by_group', 'S1')),
    ('set_lead_promotion', ('by_channel', 'social_media')),
    ('set_lead_promotion', ('by_channel_group', 'social_media', 'S1')),
    ('set_promotion', ('global_promotion',)),
    ('set_promotion', ('by_group', 'S1')),
    ('set_promotion', ('by_customer', '42')),
    ('set_promotion', ('by_group_plan', 'S1', 'A')),
]


@pytest.mark.parametrize('name,path', NUMERIC_PATHS)
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), float('-inf'), True])
def test_invalid_numbers_change_neither_runtime_nor_audit(tools, name, path, invalid):
    args = deepcopy(SETTINGS[name])
    target = args
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid
    before = snapshot(tools)
    assert not getattr(tools, name)(**args).success
    assert snapshot(tools) == before


@pytest.mark.parametrize('name', SETTINGS)
def test_audit_commit_failure_changes_nothing(tools, name):
    conn = tools.conn
    before = snapshot(tools)

    class FailedCommit:
        def __getattr__(self, name):
            return getattr(conn, name)

        def commit(self):
            raise sqlite3.OperationalError('audit commit failed')

    tools.conn = FailedCommit()
    args = {key: {} if isinstance(value, dict) else value * 2
            for key, value in SETTINGS[name].items()}
    with pytest.raises(sqlite3.OperationalError, match='audit commit failed'):
        getattr(tools, name)(**args)
    assert snapshot(tools) == before
