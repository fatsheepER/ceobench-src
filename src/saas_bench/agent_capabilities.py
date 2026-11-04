"""Explicit workspace capabilities, independent of history storage and scripts."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AgentCapabilities:
    text_tools: bool = True
    source_declarations: bool = True
    tool_notes: bool = True
    git_extra_saves: bool = True

    def __post_init__(self):
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError('Agent capabilities must be booleans')

    @classmethod
    def resolve(cls, value=None, *, text_registration=True, pf=False):
        if value is None:
            return cls(text_registration, text_registration, text_registration or pf, True)
        return value if isinstance(value, cls) else cls(**value)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def round_three(cls, mode):
        if mode not in ('git', 'pf'):
            raise ValueError('Third-round capabilities require git or pf mode')
        return cls(False, mode == 'pf', mode == 'pf', mode == 'pf')
