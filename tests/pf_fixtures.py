"""Captured evidence and declaration helpers shared by PF tests."""
from saas_bench.public_sql import PUBLIC_POLICY_VERSION, execute_query
from saas_bench.sql_evidence import encoded
from test_text_registry import captured, call, declaration, send


def record_sql(store, sql, body, *, parent=None, execution=None):
    event = store.begin(encoded({'sql': sql}), PUBLIC_POLICY_VERSION, parent)
    store.finish(event, 200, '', encoded(body), execution or {})
    store.delivered(event, 'internal')
    return event + ':public_response'


def cite(store, registry, executor, version, **fields):
    handle = registry.resolver.handle(version)
    send(store, executor.execute('pf_read', {'target': {'version': handle}}))
    return call(registry, 'create', **declaration(references=[dict(
        evidence={'version': handle}, purpose='current', **fields)]))


def forward(executor, record='r2', **args):
    args.setdefault('detail', True)
    return executor.pf_queries.answer('pf_dependencies', dict(target={'record': record}, purpose='current', **args))


def chain(workspace, tmp_path, server, sql='SELECT amount FROM ledger', **fields):
    store, registry, executor = captured(workspace, tmp_path)
    server.sql_evidence = store
    version = record_sql(store, sql, execute_query(server, sql))
    cite(store, registry, executor, version, **fields)
    send(store, executor.execute('text_list', {}))
    call(registry, 'create', **declaration({'record': 'r1'}))
    return store, registry, executor
