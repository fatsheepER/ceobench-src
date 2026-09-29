"""Declaration input shapes per group. Predicates are stored, never evaluated here."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence_handles import HANDLE_PATTERN


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


Text = Annotated[str, Field(min_length=1)]
Number = Annotated[float, Field(allow_inf_nan=False)]


class BusinessObject(Input):
    kind: Text
    id: Text


class ContentTime(Input):
    day: Annotated[int, Field(ge=0)] | None = None
    start_day: Annotated[int, Field(ge=0)] | None = None
    end_day: Annotated[int, Field(ge=0)] | None = None
    unknown: Text | None = None

    @model_validator(mode='after')
    def shape(self):
        if self.unknown is not None:
            valid = self.day is self.start_day is self.end_day is None
        elif self.day is not None:
            valid = self.start_day is self.end_day is None
        else:
            # start_day alone is open-ended: from that day until the text is revised or retired.
            valid = self.start_day is not None and (self.end_day is None or self.start_day <= self.end_day)
        if not valid:
            raise ValueError('Use {"day": 21}, {"start_day": 21, "end_day": 27}, {"start_day": 21} '
                             '(until revised or retired), or {"unknown": "reason"}')
        return self


class Selector(Input):
    row: dict[str, str | int | float | bool | None] | None = None
    col: Text | None = None
    path: str | None = None

    @model_validator(mode='after')
    def shape(self):
        if self.path is not None:
            if self.row is not None or self.col is not None or (self.path and not self.path.startswith('/')):
                raise ValueError('JSON path must be a JSON Pointer; do not mix path and row/col')
        elif self.col is None:
            raise ValueError('Table selection requires col; row uses equality keys, never a row number')
        elif self.row == {}:
            raise ValueError('row must contain at least one equality key')
        return self


class Tolerance(Input):
    type: Literal['tolerance']
    amount: Annotated[Number, Field(ge=0)]


class Threshold(Input):
    type: Literal['threshold']
    op: Literal['>', '>=', '<', '<=']
    value: Number


class Compare(Input):
    type: Literal['compare']
    left: Selector
    op: Literal['>', '>=', '<', '<=']
    right: Selector


def _reference_shape(self):
    if self.predicate is not None and self.predicate.type == 'compare':
        if self.select is not None:
            raise ValueError('compare uses left/right selectors, not select')
        if self.predicate.left.path is not None or self.predicate.right.path is not None:
            raise ValueError('compare requires two cells of one query result')
        if self.evidence.path or self.evidence.record:
            raise ValueError('compare requires a query view')
    elif self.predicate is not None and self.select is None:
        raise ValueError('A numeric predicate requires a selector')
    if self.evidence.record and (self.select or self.predicate):
        raise ValueError('Registered text supports whole-text equality only')
    return self


def _declaration_models(pf):
    """Git/prefix inputs expose only Git evidence; PF adds SQL, handles and compare."""
    class Evidence(Input):
        path: Text | None = None
        commit: Text | None = None
        if pf:
            sql: Text | None = None
        record: Annotated[str, Field(pattern=r'^r[1-9][0-9]*(\.[1-9][0-9]*)?$')] | None = None
        if pf:
            version: Annotated[str, Field(pattern=HANDLE_PATTERN)] | None = None
        unknown: Text | None = None

        @model_validator(mode='after')
        def shape(self):
            given = [getattr(self, k, None) for k in ('path', 'sql', 'record', 'version', 'unknown')]
            if sum(x is not None for x in given) != 1:
                raise ValueError('Specify exactly one of ' + ('path, sql, record, version, unknown' if pf
                                                               else 'path, record, unknown'))
            if self.commit is not None and self.path is None:
                raise ValueError('commit requires a path')
            return self

    Predicate = Tolerance | Threshold | Compare if pf else Tolerance | Threshold

    class Reference(Input):
        evidence: Evidence
        purpose: Literal['current', 'historical_only']
        select: Selector | None = None
        predicate: Annotated[Predicate, Field(discriminator='type')] | None = None
        note: str | None = None
        shape = model_validator(mode='after')(_reference_shape)

    class Create(Input):
        text: Text
        objects: Annotated[list[BusinessObject], Field(min_length=1)]
        references: list[Reference]
        applies_at: ContentTime
        reason: Text

    class Revise(Input):
        record: Annotated[str, Field(pattern=r'^r[1-9][0-9]*$')]
        reason: Text
        text: Text | None = None
        objects: Annotated[list[BusinessObject], Field(min_length=1)] | None = None
        references: list[Reference] | None = None
        applies_at: ContentTime | None = None

    return Create, Revise


class Retire(Input):
    record: Annotated[str, Field(pattern=r'^r[1-9][0-9]*$')]
    reason: Text


class ListTexts(Input):
    after: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1, le=100)] = 20


GIT_CREATE, GIT_REVISE = _declaration_models(pf=False)
PF_CREATE, PF_REVISE = _declaration_models(pf=True)
MODELS = dict(git=dict(create=GIT_CREATE, revise=GIT_REVISE, retire=Retire, list=ListTexts),
              pf=dict(create=PF_CREATE, revise=PF_REVISE, retire=Retire, list=ListTexts))
MODELS['prefix'] = MODELS['git']  # The prefix is the Git configuration, word for word.


def tool_definitions(pf=False):
    evidence = ('Evidence uses a workspace path you read or wrote (a bare path binds the version last sent to you; '
                'path@commit binds that committed file), a handle shown in results such as forecast.json@v3, '
                'query7@v2 or analyze.py.out@v4 (for a script\'s printed numbers, the 输出 handle after the command), '
                'exact SQL of raw query output you saw, or a registered text rN / rN.M.'
                if pf else
                'Evidence uses a workspace file path (a bare path binds the commit closing this week, shown as '
                'week-N; path@commit or commit selects an earlier commit by hex prefix or week-N) or a registered '
                'text rN / rN.M.')
    predicates = ('Optional predicates: tolerance amount around the cited value, threshold op/value, or compare '
                  'left/op/right within one query result. Predicates on current-purpose references are checked '
                  'by pf_dependencies and the weekly check.' if pf else
                  'Optional predicates: tolerance amount around the cited value, or threshold op/value. '
                  'Predicates are only stored.')
    descriptions = {
        'create': 'Register a hypothesis, forecast, plan, conclusion or counterevidence in registrations.json. '
                  'Supply explicit business objects and applicability time (applies_at: day, start_day with end_day, or '
                  'start_day alone for "until revised or retired"); use unknown with a reason when evidence '
                  'is unavailable. ' + evidence + ' purpose is current or historical_only. Optional select uses row '
                  'equality keys and col, or a JSON Pointer path. ' + predicates + ' Notes over 200 characters are '
                  'truncated. Returns rN and rN.M.',
        'revise': 'Append a revision with a reason. Omitted fields, including existing evidence bindings, stay unchanged. Supplied references replace the entire reference list and are validated anew. Old revisions remain in registrations.json.',
        'retire': 'Stop using a registered text. Append a retired revision with a reason and preserve all history.',
        'list': 'Page through current, active registered texts in creation order. Includes text and reference notes; does not expand references, find reverse links, or check staleness. Pass next_after as after for the next page.',
    }
    models = MODELS['pf' if pf else 'git']
    return [dict(name='text_' + name, description=descriptions[name], parameters=model.model_json_schema())
            for name, model in models.items()]


REGISTRATION_COMMON = '''

You may use text_create, text_revise, text_retire and text_list to preserve useful
hypotheses, forecasts, plans, conclusions and counterevidence across weeks.
Registrations are saved in registrations.json, which also persists across weeks
in addition to the items listed under Memory & Persistence. When you save what
matters in the weekly workflow, you can register or revise the items you expect
to rely on later, and keep the weekly summary and useful rN IDs in MEMORY.md.
Registrations are not automatically injected into your context; use text_list
to review them. Each week begins with a check, shown after the dashboard, that
lists the active texts whose cited evidence changed. Choose what to register;
missing registration never blocks business actions.
Distinguish acquisition time, the day/interval described by evidence, and when
you read it. Use an explicit unknown reason when applicability is unclear.
Dashboard normally reflects the previous weekly advance. After changing settings
within a week, use the corresponding public query to obtain current settings.
Registered texts can cite each other directly using rN.M, including before a Git
commit. A bare rN binds its current revision; later revisions do not change that
reference. Revise with omitted references to preserve the original bindings.
An unknown reference with a reason is always allowed.
'''

GIT_EVIDENCE_RULES = '''File references cite committed files. A bare path cites the file as this week's
closing commit stores it (shown as week-N), so edits later this week are included;
path@commit cites an earlier commit by a unique prefix or week-N.
The weekly check lists cited committed files and texts that differ from their
current versions. Example: before reusing last week's plan, git log -p on the files it cites shows
whether they changed since you registered it.
'''

PF_EVIDENCE_RULES = '''Cite what you actually saw or wrote. A bare file path binds the version
last sent to you or written by you; path@commit binds that committed file. After a command,
a footer such as [输出: v12 | q: v10 v11 | 写: plan.json v13] names its printed
output, the queries it ran and the files it wrote. To cite numbers a script printed,
cite its output (v12); the queries behind it are traced for you. Exact SQL works
only for raw query output you saw.
'''


def registration_prompt(pf=False):
    return REGISTRATION_COMMON + (PF_EVIDENCE_RULES if pf else GIT_EVIDENCE_RULES)
