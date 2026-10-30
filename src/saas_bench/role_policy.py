from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal

ROLES = ('ceo', 'growth', 'ops_finance')
READABLE_ROLES = MappingProxyType(dict.fromkeys(ROLES, ROLES))


@dataclass(frozen=True)
class CallSpec:
    category: Literal['read', 'research', 'write']
    roles: frozenset[str]


CALL_SPECS = MappingProxyType({
    **dict.fromkeys(('get_social_posts', 'get_cost_info', 'get_market_overview',
                     'get_group_insights', 'list_research_projects'), CallSpec('read', frozenset(ROLES))),
    **dict.fromkeys(('research_market', 'research_group', 'start_research_project'),
                   CallSpec('research', frozenset(('ceo',)))),
    **dict.fromkeys(('set_prices', 'send_enterprise_deal', 'reject_enterprise_deal'),
                   CallSpec('write', frozenset(('ceo',)))),
    'post_social_media': CallSpec('write', frozenset(('ceo', 'growth'))),
    **dict.fromkeys(('set_model_tiers', 'set_daily_spend', 'set_targeted_ad_spend',
                     'set_capacity_tier', 'set_usage_quotas', 'set_targeted_ops_spend',
                     'set_targeted_dev_spend', 'set_ads_strength', 'set_lead_promotion', 'set_promotion'),
                   CallSpec('write', frozenset(ROLES))),
})
CALL_POLICY = MappingProxyType({tool: spec.category for tool, spec in CALL_SPECS.items()})


@dataclass(frozen=True)
class AgentIdentity:
    run_id: str
    world_id: str
    role: str
    agent_id: str
    session_id: str

    def __post_init__(self):
        if self.role not in ROLES or not all(asdict(self).values()):
            raise ValueError('Invalid team identity')

    def fields(self):
        return asdict(self)


def require_role(role):
    if role not in ROLES:
        raise PermissionError('A host-bound role is required')


def require_call(role, tool):
    require_role(role)
    spec = CALL_SPECS.get(tool)
    if spec is None:
        raise PermissionError(f'Unknown tool: {tool}')
    if role not in spec.roles:
        permitted = 'CEO' if spec.roles == frozenset(('ceo',)) else 'CEO and Growth'
        raise PermissionError(f'Only {permitted} can call {tool}')


def require_advance(role):
    require_role(role)
    if role != 'ceo':
        raise PermissionError('Only CEO can advance time')


class RoleAudit:
    def __init__(self, path, identity):
        from threading import Lock
        self.path, self.identity, self.lock = path, identity, Lock()
        self.world_state = lambda: None

    def record(self, kind, **facts):
        import json
        import os
        import uuid
        record = dict(self.identity.fields(), event_id=uuid.uuid4().hex,
                      world_state_id=self.world_state(), kind=kind, **facts)
        with self.lock, self.path.open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        return record
