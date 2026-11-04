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


def tool_definition(*, texts=True, git_extra_saves=True):
    definition = dict(name='escalate', description=(
        'Save source declarations for several targets, one complete source set per item. '
        'Use create, revise, or retire. Endpoints are {cite: ROLE:rN.K}, saved file paths '
        '(snapshotted at registration), exact ROLE:path@COMMIT, or {operation_ref: growth:1} copied '
        'from a business receipt. PF also accepts exact captured ROLE:HANDLE@vK. '
        'Each source is {ref: endpoint, note?: text}, or {ref: {missing: {description, reason, '
        'last_known_ref?: text}}}. Missing targets are invalid. No lookup or predicate is required. '
        'author_declared requires the target producer to be your role; use inferred for another '
        'role or unknown author. The host supplies identity. For revise/retire use your relation '
        'ID and expected_revision from its receipt. Omitted revision fields stay; supplied sources '
        'replace the whole set, including []. Changing target kind requires a new relation. '
        'Inspect each item result: ordinary rejection leaves other items eligible; saved with '
        'missing sources is incomplete evidence, and saved does not certify business success or validity. '
        'History is readable in relations.json. Existing text references stay unchanged.'),
        parameters=compact_schema(Batch.model_json_schema()))
    if not texts:
        definition['description'] = definition['description'].replace(
            'Endpoints are {cite: ROLE:rN.K}, saved file paths ', 'Endpoints are {cite: saved file path}, ')
        definition['description'] = definition['description'].replace(
            ' Existing text references stay unchanged.', ' New and revised declarations accept only files, captured outputs and business operations; registered texts are disabled. Historical records remain readable.')
    if not git_extra_saves:
        definition['description'] = definition['description'].replace(
            '(snapshotted at registration)', '(commit files yourself before citing them)')
    return definition
