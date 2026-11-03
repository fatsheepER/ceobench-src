"""Deliberate PF contract errors and public framework diagnostic codes."""


class PublicPFError(ValueError):
    pass


class PublicPFPermissionError(PermissionError):
    pass


_ERROR_CODES = frozenset(('timeout', 'execution_boundary_closed', 'cancelled'))
_ERROR_KINDS = frozenset(('timeout', 'query', 'denied', 'service'))
_ERROR_TYPES = frozenset(('TimeoutError', 'PermissionError', 'ValueError', 'KeyError',
    'OperationalError', 'ProgrammingError', 'DatabaseError', 'QueryDenied', 'QueryResultTooLarge'))
_FAILURE_STAGES = frozenset(('request', 'snapshot', 'connect', 'execute', 'fetch', 'serialize',
    'response', 'lock', 'setup', 'cleanup', 'public_reference_allocation'))
_INTERRUPTION_REASONS = frozenset(('bash_client_timeout', 'python_timeout_kill', 'python_boundary_interruption'))
_CAPTURE_GAPS = frozenset(('unobserved_internal_file_reads', 'unobserved_intermediate_file_versions',
    'unobserved_pipe_streams', 'unobserved_program_data_dependencies', 'pf_pipeline_source_ranges_unknown',
    'stdout_bytes_without_public_text', 'stderr_bytes_without_public_text'))


def public_exception_message(exc):
    if isinstance(exc, (PublicPFError, PublicPFPermissionError)):
        return str(exc)
    if isinstance(exc, TimeoutError):
        return 'PF call cancelled' if str(exc) == 'PF call cancelled' else 'PF call time limit exceeded'
    if isinstance(exc, PermissionError):
        return 'Evidence is inaccessible'
    if isinstance(exc, KeyError):
        return 'Captured reference is unavailable'
    from .public_refs import PublicReferenceUnavailable
    if isinstance(exc, PublicReferenceUnavailable):
        return 'Public reference history is unavailable; manual recovery is required'
    return 'PF query failed; host diagnostics retained'


def public_diagnostics(value):
    result = {}
    for name, allowed in (('error', _ERROR_CODES), ('error_kind', _ERROR_KINDS),
                          ('error_type', _ERROR_TYPES), ('failure_stage', _FAILURE_STAGES),
                          ('reason', _INTERRUPTION_REASONS)):
        if name in value:
            code = value[name]
            result[name] = code if code is None or isinstance(code, str) and code in allowed else 'details_retained_privately'
    for name in ('capture_gaps', 'gaps'):
        if name in value:
            result[name] = sorted({code if isinstance(code, str) and code in _CAPTURE_GAPS else
                'unspecified_capture_gap' for code in value[name]})
    return result
