"""The shared, per-target declaration protocol."""
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .registration_schema import Input, Text, compact_schema


class Missing(Input):
    description: Text
    reason: Text
    last_known_ref: Text | None = None


class Cite(Input):
    cite: Text


class Operation(Input):
    operation_ref: Annotated[str, Field(pattern=r'^(ceo|growth|ops_finance):[1-9][0-9]*$')]


class MissingRef(Input):
    missing: Missing


Endpoint = Cite | Operation


class Source(Input):
    ref: Endpoint | MissingRef
    note: str | None = None


class Create(Input):
    item_id: Text
    op: Literal['create']
    target: Endpoint
    sources: Annotated[list[Source], Field(min_length=1)]
    assertion: Literal['author_declared', 'inferred']
    reason: Text


class Revise(Input):
    item_id: Text
    op: Literal['revise']
    relation: Text
    expected_revision: Annotated[int, Field(ge=1)] | Text
    reason: Text
    target: Endpoint | None = None
    sources: list[Source] | None = None
    assertion: Literal['author_declared', 'inferred'] | None = None

    @model_validator(mode='before')
    @classmethod
    def omitted_is_not_null(cls, data):
        if isinstance(data, dict) and any(k in data and data[k] is None for k in ('target', 'sources', 'assertion')):
            raise ValueError('Omit fields to retain them; sources=[] explicitly clears the list')
        return data


class Retire(Input):
    item_id: Text
    op: Literal['retire']
    relation: Text
    expected_revision: Annotated[int, Field(ge=1)] | Text
    reason: Text


class Batch(Input):
    items: Annotated[list[Create | Revise | Retire], Field(min_length=1)]


MODELS = dict(create=Create, revise=Revise, retire=Retire)


def tool_definition(*, texts=True, git_extra_saves=True, pf=False):
    if pf:
        endpoints = ('Endpoints use {cite: "ROLE:HANDLE@vK"} for exact captured files or outputs. '
                     'Copy role-qualified handles from receipts. ')
        endpoints += ('A bare file path binds a version read in this request or written in this context. ' if texts else
                      'File/output declarations require an explicit @vK version. ')
    else:
        saving = ('saved file paths (snapshotted at registration)' if git_extra_saves else
                  'files you have committed yourself')
        endpoints = ('Endpoints use {cite: "ROLE:path@COMMIT"} for an exact saved file, or ' + saving + '. ')
    if texts:
        endpoints += 'Exact text revisions use {cite: ROLE:rN.K}. '
    endpoints += 'Use {operation_ref: growth:1} copied from a business receipt for an operation. '
    return dict(name='escalate', description=(
        'Optionally save source declarations for selected reusable work, one complete source set per item. '
        'Use create, revise, or retire. ' + endpoints +
        'Each source is {ref: endpoint, note?: text}, or {ref: {missing: {description, reason, '
        'last_known_ref?: text}}}. Missing targets are invalid. Known exact sources need no extra lookup. '
        'author_declared requires the target producer to be your role; use inferred for another '
        'role or unknown author. A source may be an exact version from a readable peer workspace. '
        'The host supplies identity. For revise/retire use your relation '
        'ID and expected_revision from its receipt. Omitted revision fields stay; supplied sources '
        'replace the whole set, including []. Changing target kind requires a new relation. '
        'Inspect each item result: ordinary rejection leaves other items eligible; saved with '
        'missing sources is incomplete evidence, and saved does not certify business success or validity. '
        'Declarations are optional, including for handoffs. History is readable in documents/relations.json.'),
        parameters=compact_schema(Batch.model_json_schema()))
