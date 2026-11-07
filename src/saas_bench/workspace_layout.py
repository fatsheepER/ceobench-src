"""Explicit business-file layout; actor authority remains role-bound."""
from dataclasses import dataclass
from pathlib import PurePosixPath

LEGACY = 'role-v1'
SHARED = 'shared-v1'
MANAGED = ('docs', 'novamind-operation', 'documents')


@dataclass(frozen=True)
class SharedPlan:
    title: str
    purpose: str
    maintainer: str
    sections: tuple[str, ...]

    def template(self):
        return f'# {self.title}\n\n{self.purpose}\n\n' + ''.join(f'## {section}\n\n' for section in self.sections)


SHARED_PLANS = {
    'plans/commercial.md': SharedPlan('Commercial plan',
        'Customer groups, channels, prices, promotions, advertising and conversion or renewal assumptions.',
        'Growth', ('Current plan', 'Assumptions and evidence', 'Experiments and next actions')),
    'plans/service.md': SharedPlan('Service plan',
        'Model tiers, capacity, usage quotas, operations, development and service risks.',
        'OpsFinance', ('Current settings', 'Risks and business rationale', 'Next actions')),
    'plans/cash.md': SharedPlan('Cash plan',
        'Baseline and candidate forecasts, final CEO forecast references, assumptions and due forecast reviews.',
        'OpsFinance', ('Forecasts and sources', 'Due forecast reviews', 'Assumptions and next actions')),
    'plans/decisions.md': SharedPlan('Decisions',
        'Chosen actions, confirmed business results and references, pending work, owners and review dates.',
        'CEO', ('Chosen actions', 'Confirmed results', 'Pending work and review dates')),
}


def shared_work_instructions():
    lines = ['All roles share scripts/, data/, outputs/ and these plans:']
    lines.extend(f'{path}: {plan.purpose} Suggested maintainer: {plan.maintainer}.'
                 for path, plan in SHARED_PLANS.items())
    lines.extend(('Keep the plans current. Anyone may edit them.',
                  'Distinguish proposed actions from confirmed business results in your saved work.',
                  'Use focused questions to coordinate work when needed.',
                  'Keep scripts in scripts/ and larger tables or outputs in data/ or outputs/.',
                  'All roles may read and write memory/ceo.md, memory/growth.md and memory/ops_finance.md.',
                  'Only your own role memory is loaded automatically each week.'))
    return '\n'.join(lines) + '\n'


@dataclass(frozen=True)
class FileKey:
    namespace: str
    relative_path: str

    def __post_init__(self):
        path = PurePosixPath(self.relative_path)
        if not self.namespace or path.is_absolute() or '..' in path.parts or str(path) != self.relative_path:
            raise ValueError('Expected a normalized workspace file identity')


def evidence_namespace(metadata, fallback=None):
    return (metadata.get('workspace_namespace') if metadata.get('layer') in ('file_bytes', 'file_text') else None) or metadata.get('owner_role', fallback)


def workspace_names(layout):
    from .role_policy import ROLES
    if layout == SHARED:
        return ('shared',)
    if layout == LEGACY:
        return ROLES
    raise ValueError('Unknown workspace layout: ' + str(layout))


def managed_path(relative):
    parts = PurePosixPath(relative).parts
    return bool(parts) and parts[0] in MANAGED
