"""Declaration input shapes per group. Predicates are stored, never evaluated here."""
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence_handles import VERSIONED


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


Text = Annotated[str, Field(min_length=1)]
Number = Annotated[float, Field(allow_inf_nan=False)]


class BusinessObject(Input):
    kind: Text
    id: Text


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


CITE_FORMS = dict(
    git='a workspace file path (the version this week\'s closing commit week-N will store), path@week-N or '
        'path@<commit prefix>, a registered text r4 or r4.2, or "unknown: <reason>"',
    pf='a handle from a [pf: ...] line such as scripts/an_w7d.py.out@v1 or MEMORY.md@v8, a file path (the '
       'version present in this request or written in this context), path@week-N or path@<commit prefix>, a registered text r4 or r4.2, '
       'or "unknown: <reason>"')
_RECORD = re.compile(r'(?:(?:ceo|growth|ops_finance):)?r[1-9][0-9]*(\.[1-9][0-9]*)?')
_SINGLE = re.compile(r'[a-z_]+[1-9][0-9]*')
_COMMIT = re.compile(r'week-[1-9][0-9]*|[0-9a-fA-F]{1,40}')
_UNKNOWN = re.compile(r'unknown(?:(?:\s*:|\s)(.*))?', re.I | re.S)


def parse_cite(cite, pf):
    """The evidence object a cite string names, e.g. "a.py@week-3" -> {"path": "a.py", "commit": "week-3"}."""
    cite = cite.strip()
    if m := _UNKNOWN.fullmatch(cite):
        if not (m.group(1) or '').strip():
            raise ValueError('Write "unknown: <reason>"')
        return dict(unknown=m.group(1).strip())
    if _RECORD.fullmatch(cite):
        return dict(record=cite)
    if pf and (VERSIONED.fullmatch(cite) or ('/' not in cite and '.' not in cite and _SINGLE.fullmatch(cite.split(':')[-1]))):
        return dict(version=cite)
    if '@' in cite:
        path, commit = cite.rsplit('@', 1)
        if path and _COMMIT.fullmatch(commit):
            return dict(path=path, commit=commit)
        raise ValueError('Cite a committed file as path@week-N or path@<commit prefix>' +
                         ('; a handle is NAME@vK' if pf else ''))
    return dict(path=cite)


def parse_applies(value):
    """"21", "21-27", "21-" (until revised or retired) or "unknown: reason" -> applies_at."""
    text = value.strip()
    if m := _UNKNOWN.fullmatch(text):
        if (m.group(1) or '').strip():
            return dict(unknown=m.group(1).strip())
    elif m := re.fullmatch(r'([0-9]+)(?:\s*-\s*([0-9]*))?', text):
        start = int(m.group(1))
        if m.group(2) is None:
            return dict(day=start)
        if not m.group(2):
            return dict(start_day=start)
        if start <= int(m.group(2)):
            return dict(start_day=start, end_day=int(m.group(2)))
    raise ValueError(APPLIES_ERROR)


def cite_text(evidence):
    """The cite string of a stored evidence object; the inverse of parse_cite."""
    if 'unknown' in evidence:
        return 'unknown: ' + evidence['unknown']
    if 'sql' in evidence:
        return 'SQL: ' + evidence['sql']
    if 'path' in evidence:
        return (evidence['owner'] + ':' if evidence.get('owner') else '') + evidence['path'] + ('@' + evidence['commit'] if evidence.get('commit') else '')
    return evidence.get('record') or evidence.get('version')


APPLIES_ERROR = 'applies must be "21", "21-27", "21-" (until revised or retired) or "unknown: <reason>"'


def applies_text(applies_at):
    """The applies string of a stored applies_at object; rejects shapes it never had."""
    day = lambda k: type(applies_at.get(k)) is int and applies_at[k] >= 0
    keys = set(applies_at)
    if keys == {'unknown'} and isinstance(applies_at['unknown'], str) and applies_at['unknown'].strip():
        return 'unknown: ' + applies_at['unknown']
    if keys == {'day'} and day('day'):
        return str(applies_at['day'])
    if keys == {'start_day'} and day('start_day'):
        return f"{applies_at['start_day']}-"
    if keys == {'start_day', 'end_day'} and day('start_day') and day('end_day') and \
            applies_at['start_day'] <= applies_at['end_day']:
        return f"{applies_at['start_day']}-{applies_at['end_day']}"
    raise ValueError(APPLIES_ERROR)


def _legacy(data, field, flat, convert):
    """Accept the earlier nested input shape without advertising it in the schema."""
    if isinstance(data, dict) and field in data and flat not in data and isinstance(data[field], dict):
        data = dict(data)
        value = data.pop(field)
        data[flat] = convert(value)
    return data


def _declaration_models(pf):
    """Git/prefix inputs expose only Git evidence; PF adds handles and compare."""
    Predicate = Tolerance | Threshold | Compare if pf else Tolerance | Threshold

    class Reference(Input):
        cite: Text
        purpose: Literal['current', 'historical_only'] = 'current'
        select: Selector | None = None
        predicate: Annotated[Predicate, Field(discriminator='type')] | None = None
        note: str | None = None

        @model_validator(mode='before')
        @classmethod
        def legacy(cls, data):
            if isinstance(data, dict) and isinstance(data.get('evidence'), dict):
                given = [k for k in ('path', 'sql', 'record', 'version', 'unknown') if data['evidence'].get(k) is not None]
                if len(given) != 1 or ('sql' in given and not pf) or ('version' in given and not pf) or \
                        set(data['evidence']) - {'path', 'commit', 'sql', 'record', 'version', 'unknown', 'owner'} or \
                        ('commit' in data['evidence'] and 'path' not in given) or \
                        not all(isinstance(v, str) and v for v in data['evidence'].values()):
                    raise ValueError('Specify exactly one of ' + ('path, sql, record, version, unknown' if pf
                                                                   else 'path, record, unknown'))
            return _legacy(data, 'evidence', 'cite', cite_text)

        @model_validator(mode='after')
        def shape(self):
            evidence = self.evidence
            if self.predicate is not None and self.predicate.type == 'compare':
                if self.select is not None:
                    raise ValueError('compare uses left/right selectors, not select')
                if self.predicate.left.path is not None or self.predicate.right.path is not None:
                    raise ValueError('compare requires two cells of one query result')
                if 'path' in evidence or 'record' in evidence:
                    raise ValueError('compare requires a query view')
            elif self.predicate is not None and self.select is None:
                raise ValueError('A numeric predicate requires a selector')
            if 'record' in evidence and (self.select or self.predicate):
                raise ValueError('Registered text supports whole-text equality only')
            return self

        @property
        def evidence(self):
            if self.cite.startswith('SQL: ') and pf:
                return dict(sql=self.cite[len('SQL: '):])
            return parse_cite(self.cite, pf)

    Applies = Annotated[str, Field(min_length=1)]

    def applies(data):
        data = _legacy(data, 'applies_at', 'applies', applies_text)
        if isinstance(data, dict) and isinstance(data.get('applies'), str):
            parse_applies(data['applies'])
        return data

    class Create(Input):
        text: Text
        objects: Annotated[list[BusinessObject], Field(min_length=1)]
        references: list[Reference]
        applies: Applies
        reason: Text

        @model_validator(mode='before')
        @classmethod
        def flat_applies(cls, data):
            return applies(data)

    class Revise(Input):
        record: Annotated[str, Field(pattern=r'^r[1-9][0-9]*$')]
        reason: Text
        text: Text | None = None
        objects: Annotated[list[BusinessObject], Field(min_length=1)] | None = None
        references: list[Reference] | None = None
        applies: Applies | None = None

        @model_validator(mode='before')
        @classmethod
        def flat_applies(cls, data):
            return applies(data)

    return Create, Revise


def internal(values, pf):
    """Validated flat input -> the stored declaration shape (evidence and applies_at objects)."""
    values = dict(values)
    if 'applies' in values:
        values['applies_at'] = parse_applies(values.pop('applies'))
    if 'references' in values:
        references = []
        for ref in values['references']:
            ref = dict(ref)
            cite = ref.pop('cite')
            ref['evidence'] = dict(sql=cite[len('SQL: '):]) if pf and cite.startswith('SQL: ') else parse_cite(cite, pf)
            references.append(ref)
        values['references'] = references
    return values


class Retire(Input):
    record: Annotated[str, Field(pattern=r'^r[1-9][0-9]*$')]
    reason: Text


class ListTexts(Input):
    after: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1, le=100)] = 20


class PFListTexts(ListTexts):
    review: Literal['pending'] | None = Field(default=None, description='Only texts with pending source checks; includes first and last verification days.')


GIT_CREATE, GIT_REVISE = _declaration_models(pf=False)
PF_CREATE, PF_REVISE = _declaration_models(pf=True)
MODELS = dict(git=dict(create=GIT_CREATE, revise=GIT_REVISE, retire=Retire, list=ListTexts),
              pf=dict(create=PF_CREATE, revise=PF_REVISE, retire=Retire, list=PFListTexts))
MODELS['prefix'] = MODELS['git']  # The prefix is the Git configuration, word for word.


def compact_schema(node):
    """Pydantic's JSON schema without titles or the null branch of optional fields.

    Optional fields are simply not required; spelling out {"type": "null"} and a null
    default for every one of them made the declaration schemas several times longer
    than the fields they describe.
    """
    if isinstance(node, list):
        return [compact_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    result = {}
    for key, value in node.items():
        if key == 'title' or (key == 'default' and value is None):
            continue
        if key in ('properties', '$defs'):
            result[key] = {name: compact_schema(item) for name, item in value.items()}
        else:
            result[key] = compact_schema(value)
    options = result.get('anyOf')
    if options and len(options) == 2 and {'type': 'null'} in options:
        other = next(o for o in options if o != {'type': 'null'})
        result = {k: v for k, v in result.items() if k != 'anyOf'} | other
    return result


def tool_definitions(pf=False):
    group = 'pf' if pf else 'git'
    checks = ('Optional select and predicate on a reference make the weekly check test a condition instead '
              'of reporting any change. select picks one cell: {"row": {"group_id": "S1"}, "col": "conv"} in a '
              'query result or CSV file (row uses equality keys), or {"path": "/a/b"} in a JSON file. predicate '
              'is {"type": "threshold", "op": ">=", "value": 0.5}, {"type": "tolerance", "amount": 5}, '
              'or {"type": "compare", "left": <select>, "op": "<", "right": <select>} within one query '
              'result; query handles are listed by pf depend <output> --detail. '
              'Script outputs and plain text support whole-content equality only.') if pf else (
              'The weekly check compares whole cited files and registered texts with their current versions. '
              'Optional select and predicate fields are stored but do not affect this check.')
    descriptions = {
        'create': 'Register a hypothesis, forecast, plan, conclusion or counterevidence (see Registered Texts). '
                  'Each reference is {"cite": ..., "purpose": "current" or "historical_only" (default current), '
                  '"note": what you use it for}; cite is ' + CITE_FORMS[group] + '. applies is "49", "49-55", '
                  '"49-" (until revised or retired) or "unknown: <reason>". ' + checks +
                  ' Notes over 200 characters are truncated. Returns rN and rN.M.',
        'revise': 'Append a revision with a reason. Omitted fields, including references, stay unchanged; '
                  'supplied references replace the whole list. Earlier revisions remain in registrations.json.',
        'retire': 'Stop using a registered text. Appends a retired revision with a reason and keeps all history.',
        'list': 'Page through your active registered texts in creation order, with their references and notes. '
                'Does not check them. Pass next_after as after for the next page.',
    }
    models = MODELS[group]
    if pf:
        descriptions['create'] += (' An explicit whole-version citation binds that captured version even when its body '
                                   'is absent from this request; reading scope is recorded separately. '
                                   'Selected fields require coverage in this request or your own current-context file write.')
        descriptions['list'] += ' review="pending" filters pending source checks; checks keeps their original verification dates.'
    return [dict(name='text_' + name, description=descriptions[name], parameters=compact_schema(model.model_json_schema()))
            for name, model in models.items()]
