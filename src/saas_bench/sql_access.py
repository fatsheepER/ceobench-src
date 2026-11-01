"""Saved SQLite authorization observations, with a public-only projection."""
import sqlite3

from .database import TABLE_DOCS


FORMAT = 'pf.sql_access.v1'
PUBLIC_COLUMNS = {table: tuple(doc['columns']) for table, doc in TABLE_DOCS.items()}
SEMANTICS = ('SQLite prepare/reprepare authorization observations; an upper bound that can '
             'include view expansion and unused projections. Not row reads, output columns, '
             'model delivery, source adoption or business causality.')


def new_record():
    return dict(format=FORMAT, status='not_started', raw_reads=[], public_reads=[])


def public_read(row):
    table, column, database = row['table'], row['column'], row['database']
    if (row.get('decision') != 'allowed' or table not in PUBLIC_COLUMNS or
            database not in ('main', 'temp', None) or
            column != '' and column not in PUBLIC_COLUMNS[table]):
        return None
    source = row.get('source')
    source_state = 'none' if source is None else 'public_name' if source in PUBLIC_COLUMNS else 'redacted'
    basis = ('column_free_authorization' if column == '' else
             'public_view_reference' if database == 'temp' else
             'named_source_authorization' if source is not None else 'direct_authorization')
    return dict(table=table, column=column or None, database=database,
                source=source if source_state != 'redacted' else None, source_state=source_state,
                basis=basis)


def observe(record, action, table, column, database, source, decision):
    row = dict(action=action, table=table, column=column, database=database, source=source,
               decision='allowed' if decision == sqlite3.SQLITE_OK else 'denied')
    record['raw_reads'].append(row)
    public = public_read(row)
    if public is not None and public not in record['public_reads']:
        record['public_reads'].append(public)


def projection(record):
    if not isinstance(record, dict) or record.get('format') != FORMAT:
        return dict(format=FORMAT, status='unknown', public_reads=[], semantics=SEMANTICS)
    reads = []
    for row in record.get('public_reads', []):
        # Saved projections are checked against today's public policy on every discovery.
        current = public_read(dict(row, column=row['column'] or '', decision='allowed',
                                   source='__redacted_context__' if row.get('source_state') == 'redacted' else row.get('source')))
        if current is not None:
            reads.append(current)
    return dict(format=FORMAT, status=record['status'], public_reads=reads, semantics=SEMANTICS)


def authorize_projection(record):
    if record and projection(record) != record:
        raise PermissionError('Saved SQL access observations are no longer public; start a new query')


def matches(record, table=None, column=None):
    if not table and not column:
        return True
    return any((table is None or row['table'] == table) and
               (column is None or row['column'] == column) for row in record['public_reads'])
