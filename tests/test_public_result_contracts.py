"""Public data survives ToolResult serialization, HTTP, SDK calls and generated docs."""
import json
import re

import pytest

from saas_bench.config import BenchmarkConfig, CUSTOMER_GROUPS, RESEARCH_TIERS_BY_ID
from saas_bench.novamind_api import market, research
from saas_bench.tools import TOOL_DOCS
from test_public_sql import server


@pytest.fixture
def sdk(server, make_agent_tools, monkeypatch):
    server.tools = make_agent_tools(server.conn, BenchmarkConfig(), day=308)
    server.conn.execute('UPDATE ledger SET amount=10000000')
    for gid, cfg in CUSTOMER_GROUPS.items():
        initial = not gid.startswith('D_')
        server.conn.execute('INSERT INTO group_info_levels(group_id,info_level,is_discoverable) VALUES(?,?,?)',
                            (gid, int(initial), int(not initial)))
        if initial:
            server.conn.execute('INSERT OR REPLACE INTO group_insight_snapshots VALUES(?,?,?,?,?)',
                                (gid, 0, cfg.c_max_mean, cfg.q_min_mean, cfg.base_market_cap))
    server.conn.execute("INSERT INTO macroeconomic_conditions VALUES(270,54.2,'expansion',1.3,'recovering','Period average')")
    server.conn.commit()
    server.start()
    monkeypatch.setenv('NOVAMIND_API_PORT', str(server.port))
    return server.tools


def test_sdk_macro_dates(sdk, tmp_path):
    macro = market.get_market_overview()['macroeconomic']
    assert macro['measurement_day'] == 270
    assert macro['publication_delay_days'] == sdk.config.macro_pmi_publication_delay_days
    example = TOOL_DOCS['get_market_overview']['returns']['data']['macroeconomic']
    assert set(example) == set(macro)
    from saas_bench.execution_capture import finish_http
    from saas_bench.sql_evidence import SQLEvidenceStore, encoded
    from test_sql_evidence import identity
    store = SQLEvidenceStore(tmp_path / 'evidence.sqlite', identity(capture_scope='execution'))
    event = store.begin_event('public_http', dict(method='POST', path='/call',
        parsed=dict(tool='get_market_overview', args={})))
    finish_http(store, event, 200, '', encoded(dict(success=True, data={'macroeconomic': macro})), {})
    meta, _ = store.get_content(event + ':public_response')
    assert any(f['field'] == 'measurement_day' and f['value'] == 270
               for f in meta['content_time'].get('fields', []))


def test_sdk_survey_dates_and_referral_units(sdk):
    result = market.get_group_insights('S1')
    assert result['snapshot_day'] == 0
    network = result['network_influence']
    text = sdk.get_group_insights('S1').message
    for direction, symbol in [('outgoing', '→'), ('incoming', '←')]:
        for gid, rate in network[direction].items():
            line = next(line for line in text.splitlines() if symbol in line and f'({gid}):' in line)
            assert rate == pytest.approx(float(re.search(r'~([\d.]+)', line).group(1)), abs=.051)
    self_line = next(line for line in text.splitlines() if 'Self-referral:' in line)
    assert network['self_referral'] == pytest.approx(float(re.search(r'~([\d.]+)', self_line).group(1)), abs=.051)
    example = TOOL_DOCS['get_group_insights']['returns']['data']
    assert set(example) == set(result)


def test_sdk_research_list_returns_tier_summaries_without_project_dates(sdk):
    from saas_bench.novamind_api import query
    start = research.start_research_project(1)
    tier = research.list_research_projects()['tiers'][0]
    assert tier['std_days'] == RESEARCH_TIERS_BY_ID[1].std_days
    assert tier['std_quality_boost'] == RESEARCH_TIERS_BY_ID[1].std_quality_boost
    assert tier['in_progress'] == 1 and tier['completed'] == 0
    assert 'projects' not in tier
    assert 'expected_completion_day' not in json.dumps(tier)
    project = query('SELECT project_id,expected_completion_day FROM research_projects')['rows'][0]
    assert project == {k: start[k] for k in ('project_id', 'expected_completion_day')}
    sdk.conn.execute("UPDATE research_projects SET status='completed',quality_boost_applied=expected_quality_boost")
    sdk.conn.commit()
    completed = research.list_research_projects()['tiers'][0]
    assert completed['completed'] == 1 and completed['in_progress'] == 0
    assert 'projects' not in completed


def test_sdk_discovery_distinguishes_all_outcomes(sdk):
    # The minimal HTTP fixture starts with the initial groups only. Hide one to
    # exercise both discovery outcomes without initializing a whole simulation.
    sdk.conn.execute("UPDATE group_info_levels SET info_level=0,is_discoverable=1 WHERE group_id='S3'")
    sdk.conn.commit()
    sdk.config.market_research_discover_prob = 0
    missed = market.research_market()
    assert missed['status'] == 'not_found' and missed['remaining_undiscovered'] > 0
    sdk.config.market_research_discover_prob = 1
    found = market.research_market()
    assert found['status'] == 'discovered' and found['discovered_group_id']
    assert found['remaining_undiscovered'] == missed['remaining_undiscovered'] - 1
    sdk.conn.execute('UPDATE group_info_levels SET info_level=1')
    sdk.conn.commit()
    exhausted = market.research_market()
    assert exhausted['status'] == 'exhausted' and exhausted['remaining_undiscovered'] == 0
    assert missed['cost'] == found['cost'] == exhausted['cost'] == sdk.config.discovery_cost_level_1


def test_generated_docs_include_public_result_contracts(tmp_path):
    from saas_bench.docs_generator import render_api_docs
    from scripts.generate_public_docs import render_tools_reference
    render_api_docs(tmp_path / 'api')
    render_tools_reference(tmp_path / 'tools-reference.md')
    docs = {d['name']: d for d in json.loads((tmp_path / 'api/market.json').read_text())}
    assert 'snapshot_day' in docs['get_group_insights']['output_schema']
    assert 'self_referral' in docs['get_group_insights']['output_schema']['network_influence']
    reference = (tmp_path / 'tools-reference.md').read_text()
    assert 'snapshot_day' in reference and 'Output Schema' in reference
    assert "'ism_pmi'" not in reference
