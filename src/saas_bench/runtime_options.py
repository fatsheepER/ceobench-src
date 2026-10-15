"""Launch settings shared by the driver, controller and worker processes."""
import argparse
import math
import os


def nonnegative_number(value):
    if str(value).lower() in ('none', 'max', 'infinity', 'unlimited'):
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('Use a nonnegative number, or unlimited') from None
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError('Use a finite nonnegative number, or unlimited')
    return result


def nonnegative_integer(value):
    result = nonnegative_number(value)
    if not result.is_integer():
        raise argparse.ArgumentTypeError('Use a nonnegative integer, or unlimited')
    return int(result)


def retry_delays(value):
    try:
        values = tuple(nonnegative_number(part) for part in str(value).split(','))
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError('Use comma-separated nonnegative retry intervals') from None
    return ','.join(f'{part:g}' for part in values)


def positive_integer(value):
    result = nonnegative_integer(value)
    if not result:
        raise argparse.ArgumentTypeError('The model provider requires a positive output token limit')
    return result


def bash_memory_bytes(value):
    from .systemd_guard import memory_bytes
    if str(value).lower() in ('0', 'none', 'max', 'infinity', 'unlimited'):
        return 0
    result = memory_bytes(str(value))
    if result < 32 * 1024**2:
        raise argparse.ArgumentTypeError('Bash memory limit must be at least 32 MiB, or unlimited')
    return result


OPTIONS = {
    'bash_memory_max': ('CEOBENCH_BASH_MEMORY_MAX_BYTES', bash_memory_bytes, 'Bash address space, default 8G; unlimited disables this limit'),
    'model_request_attempts': ('CEOBENCH_MODEL_REQUEST_ATTEMPTS', nonnegative_integer, 'Network attempts per logical request, default 5; 0 is unlimited'),
    'model_retry_delays': ('CEOBENCH_MODEL_RETRY_DELAYS', retry_delays, 'Retry seconds, default 10,20,40,80; the last interval repeats'),
    'model_attempt_seconds': ('CEOBENCH_MODEL_ATTEMPT_SECONDS', nonnegative_number, 'First response byte wait, default 600 seconds; 0 is unlimited'),
    'invalid_generation_attempts': ('CEOBENCH_INVALID_GENERATION_ATTEMPTS', nonnegative_integer, 'Invalid generations per act, default 4; 0 is unlimited'),
    'advance_timeout': ('CEOBENCH_ADVANCE_TIMEOUT_SECONDS', nonnegative_number, 'Week advancement timeout, default 0 (warning only)'),
    'advance_warn_after': ('CEOBENCH_ADVANCE_WARN_AFTER_SECONDS', nonnegative_number, 'Slow week warning interval, default 600 seconds'),
    'public_python_timeout': ('CEOBENCH_PUBLIC_PYTHON_TIMEOUT_SECONDS', nonnegative_number, 'Public Python script timeout, default 1800 seconds; 0 is unlimited'),
    'pf_timeout': ('CEOBENCH_PF_TIMEOUT_SECONDS', nonnegative_number, 'PF call timeout, default 600 seconds; 0 is unlimited'),
    'sql_timeout': ('CEOBENCH_SQL_TIMEOUT_SECONDS', nonnegative_number, 'Public SQL deadline, default 600 seconds; 0 is unlimited'),
    'sql_response_timeout': ('CEOBENCH_SQL_RESPONSE_TIMEOUT_SECONDS', nonnegative_number, 'SQL response write deadline, default 120 seconds'),
    'checkpoint_wait': ('CEOBENCH_CHECKPOINT_WAIT_SECONDS', nonnegative_number, 'Checkpoint completion wait, default 600 seconds'),
    'completion_wait': ('CEOBENCH_COMPLETION_WAIT_SECONDS', nonnegative_number, 'Public request completion wait, default 30 seconds'),
    'sql_row_limit': ('CEOBENCH_SQL_ROW_LIMIT', nonnegative_integer, 'SQL returned row limit, default 20000; 0 is unlimited'),
    'sql_column_limit': ('CEOBENCH_SQL_COLUMN_LIMIT', nonnegative_integer, 'SQL column limit, default 512'),
    'sql_cell_limit_bytes': ('CEOBENCH_SQL_CELL_LIMIT_BYTES', nonnegative_integer, 'SQL/value byte limit, default 8388608'),
    'sql_result_limit_bytes': ('CEOBENCH_SQL_RESULT_LIMIT_BYTES', nonnegative_integer, 'SQL result byte limit, default 33554432'),
    'sql_response_limit_bytes': ('CEOBENCH_SQL_RESPONSE_LIMIT_BYTES', nonnegative_integer, 'SQL JSON byte limit, default 67108864'),
    'bash_output_max_chars': ('CEOBENCH_BASH_OUTPUT_MAX_CHARS', nonnegative_integer, 'Bash model return characters, default 100000; 0 is unlimited'),
    'search_file_limit': ('CEOBENCH_SEARCH_FILE_LIMIT', nonnegative_integer, 'Search candidate files, default 1000; 0 is unlimited'),
    'search_match_limit': ('CEOBENCH_SEARCH_MATCH_LIMIT', nonnegative_integer, 'Search matches, default 2000; 0 is unlimited'),
    'glob_limit': ('CEOBENCH_GLOB_LIMIT', nonnegative_integer, 'Glob returned paths, default 2000; 0 is unlimited'),
    'memory_max_chars': ('CEOBENCH_MEMORY_MAX_CHARS', nonnegative_integer, 'MEMORY prompt characters, default 100000; 0 is unlimited'),
    'model_max_output_tokens': ('CEOBENCH_MODEL_MAX_OUTPUT_TOKENS', positive_integer, 'Model output tokens, default 32768 (Anthropic 128000)'),
}


def add_runtime_arguments(parser):
    group = parser.add_argument_group('runtime limits')
    for name, (_, parse, help_text) in OPTIONS.items():
        group.add_argument('--' + name.replace('_', '-'), type=parse, default=None, help=help_text)


def runtime_environment(args):
    return {env: str(value) for name, (env, _, _) in OPTIONS.items()
            if (value := getattr(args, name, None)) is not None}


def runtime_value(name, default):
    env, parse, _ = OPTIONS[name]
    return parse(os.environ[env]) if env in os.environ else default


def timeout_value(name, default):
    return runtime_value(name, default) or None
