"""PF argument parsing. The old command parser is retained only for historical logs."""
import re
import shlex

from .evidence_handles import RAW_VERSION, RECORD, SINGLE, VERSIONED
from .pf_render import _HEREDOC

USAGE = '''usage: pf <command> ...
  pf log <object>                      versions of a file, an output or a text, newest first
  pf show <handle> [--full]            one saved version
  pf diff <object> | <old> [<new>]     what changed; one object compares its last two versions
  pf blame <file>                      the version and day each line was written
  pf depend <object> [--detail] [--history]   what it cites; reruns and checks current references
  pf rdepend <object> [--detail] [--all]      registered texts that cite it
  pf search <id> [--kind KIND] [--all]        texts, business writes and outputs about S1, t10_2, ...
  pf search --text "term"              literal text in public captured history, newest first
  pf more <cursor>                     the next page of an earlier result
Objects: a handle (MEMORY.md@v8, scripts/a.py.out@v2, query7@v1), a file path or output name
(the latest version) or a text (r4, r4.2). Shell combinations, pipes and redirection work normally.'''

VERBS = ('log', 'show', 'diff', 'blame', 'depend', 'rdepend', 'search', 'more', 'help')
_INVOCATION = re.compile(r'(?:^|&&|\|\||[;|\n(])\s*pf(?:\s+(?:' + '|'.join(VERBS) + r')\b|\s*$)', re.M)


class Usage(ValueError):
    """A pf invocation that cannot run; its message is the whole agent-facing reply."""


def mentions_pf(command):
    """Whether a command invokes pf anywhere outside heredoc bodies."""
    return bool(_INVOCATION.search(_HEREDOC.sub('\n', command)))


def target(token):
    if RECORD.fullmatch(token):
        return {'record': token}
    if RAW_VERSION.fullmatch(token):
        return {'version': token}
    if VERSIONED.fullmatch(token) or (SINGLE.fullmatch(token.split(':')[-1]) and not re.fullmatch(r'(cmd|query|read|call)[1-9][0-9]*', token.split(':')[-1])):
        return {'version': token}
    return {'path': token}


def _flags(words, allowed):
    flags, rest, i = {}, [], 0
    while i < len(words):
        word = words[i]
        if word.startswith('--'):
            name = word[2:]
            if name not in allowed:
                raise Usage(f'pf: unknown option {word}\n' + USAGE)
            if allowed[name]:
                if i + 1 >= len(words):
                    raise Usage(f'pf: {word} needs a value\n' + USAGE)
                flags[name] = words[i + 1]
                i += 1
            else:
                flags[name] = True
        else:
            rest.append(word)
        i += 1
    return flags, rest


def _view(words):
    """`head -N`, `head -n N`, `tail -N` or `tail -n N` -> ('head', N)."""
    text = ' '.join(words)
    match = re.fullmatch(r'(head|tail)(?:\s+-n)?\s+-?([1-9][0-9]*)|(head|tail)', text)
    if not match:
        raise Usage('pf: only | head -N and | tail -N can follow pf\n' + USAGE)
    return (match.group(1) or match.group(3), int(match.group(2) or 10))


def parse(command, roots):
    """(operation, arguments, view) for a standalone pf command, else None."""
    text = command.strip()
    match = re.match(r'cd\s+(\S+)\s*(?:&&|;|\n)\s*', text)
    if match and match.group(1).strip('\'"') in roots:
        text = text[match.end():]
    text = re.sub(r'\s+2>&1(?=\s*(?:\||$))', '', text).strip()
    if not re.match(r'pf(\s|$)', text):
        if mentions_pf(command):
            raise Usage('pf: run pf on its own in a bash call, not together with other commands\n' + USAGE)
        return None
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError as exc:
        raise Usage(f'pf: {exc}\n' + USAGE) from exc
    segments, current = [], []
    for token in tokens:
        if token == '|':
            segments.append(current)
            current = []
        elif token and set(token) <= set('|&;<>()'):
            raise Usage('pf: run pf on its own in a bash call, not together with other commands\n' + USAGE)
        else:
            current.append(token)
    segments.append(current)
    if len(segments) > 2 or not all(segments):
        raise Usage('pf: only | head -N and | tail -N can follow pf\n' + USAGE)
    view = _view(segments[1]) if len(segments) == 2 else None
    words = segments[0][1:]
    if not words or words[0] in ('help', '--help', '-h'):
        raise Usage(USAGE)
    return (*parse_argv(words), view)


def parse_argv(words):
    """Parse argv already expanded by Bash; never interpret shell syntax here."""
    if not words or words[0] in ('help', '--help', '-h'):
        return 'pf_help', {}
    verb, words = words[0], words[1:]
    if verb not in VERBS:
        raise Usage(f'pf: unknown command {verb}\n' + USAGE)
    options = dict(show={'full': False}, depend={'detail': False, 'history': False},
                   rdepend={'detail': False, 'all': False}, search={'all': False, 'kind': True, 'text': False}).get(verb, {})
    flags, rest = _flags(words, options)
    expected = dict(diff=(1, 2)).get(verb, (1, 1))
    if not expected[0] <= len(rest) <= expected[1]:
        raise Usage(f'pf {verb}: wrong number of arguments\n' + USAGE)
    first = rest[0]
    if verb == 'more':
        if not re.fullmatch(r'c[1-9][0-9]*', first):
            raise Usage('pf more: give the cursor from a previous result, e.g. pf more c2\n' + USAGE)
        return 'pf_more', {'cursor': first}
    if verb == 'search':
        if flags.get('text'):
            if 'kind' in flags or 'all' in flags:
                raise Usage('pf search --text cannot be combined with --kind or --all')
            return 'pf_search', {'text': first}
        return 'pf_search', dict(object=dict(id=first, **({'kind': flags['kind']} if 'kind' in flags else {})),
                                 all=bool(flags.get('all')))
    if verb == 'log':
        return 'pf_log', {'target': target(first)}
    if verb == 'blame':
        return 'pf_blame', {'target': target(first)}
    if verb == 'show':
        return 'pf_read', {'target': target(first), 'mode': 'content', 'full': bool(flags.get('full'))}
    if verb == 'diff':
        if len(rest) == 2:
            return 'pf_read', {'target': target(rest[1]), 'mode': 'diff', 'baseline': target(first)}
        return 'pf_diff', {'target': target(first)}
    if verb == 'depend':
        return 'pf_dependencies', {'target': target(first), 'detail': bool(flags.get('detail')),
                                   'purpose': 'historical_only' if flags.get('history') else 'current'}
    return 'pf_dependents', {'target': target(first), 'detail': bool(flags.get('detail')),
                             'current_only': not flags.get('all')}


def apply_view(text, view):
    if not view:
        return text
    lines = str(text).splitlines(keepends=True)
    kind, count = view
    return ''.join(lines[:count] if kind == 'head' else lines[-count:])
