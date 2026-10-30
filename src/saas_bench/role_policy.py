from dataclasses import asdict, dataclass
from types import MappingProxyType

ROLES = ('ceo', 'growth', 'ops_finance')
READABLE_ROLES = MappingProxyType({
    'ceo': ROLES,
    'growth': ('growth', 'ceo'),
    'ops_finance': ('ops_finance', 'ceo'),
})
CALL_POLICY = MappingProxyType({
    **dict.fromkeys(('get_social_posts', 'get_cost_info', 'get_market_overview',
                     'get_group_insights', 'list_research_projects'), 'read'),
    **dict.fromkeys(('research_market', 'research_group', 'start_research_project'), 'research'),
    **dict.fromkeys(('set_prices', 'set_model_tiers', 'set_daily_spend', 'set_targeted_ad_spend',
                     'set_capacity_tier', 'set_usage_quotas', 'send_enterprise_deal',
                     'reject_enterprise_deal', 'post_social_media', 'set_targeted_ops_spend',
                     'set_targeted_dev_spend', 'set_ads_strength', 'set_lead_promotion', 'set_promotion'), 'write'),
})


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
    if role != 'ceo' and CALL_POLICY.get(tool) != 'read':
        raise PermissionError('Only CEO can change the world or purchase research')


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
