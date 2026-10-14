"""PF argument parsing. The old command parser is retained only for historical logs."""
import re
import shlex

from .evidence_handles import RAW_VERSION, RECORD, SINGLE, VERSIONED
from .pf_render import _HEREDOC

USAGE = '''usage: pf <command> ...
  pf help | pf --help | pf -h          complete command help and examples
  pf log <object>                      versions of a file, an output or a text, newest first
  pf show <handle> [--full]            one saved version
  pf diff <object> | <old> [<new>]     what changed; one object compares its last two versions
  pf blame <file>                      the version and day each line was written
  pf depend <object> [--detail] [--history]   what it cites; reruns and checks current references
  pf rdepend <object> [--detail] [--all]      registered texts that cite it
  pf search <id> [--kind KIND] [--all]        texts, business writes and outputs about an object
  pf search --text "term"              literal text in public captured history, newest first
  pf more <cursor>                     the next page of an earlier result
Objects: exact handles (MEMORY.md@v8, query7@v1, growth:r4.2), or names for the latest
version (MEMORY.md, query7, growth:r4). Bare r4 is local. Copy returned role/version handles.
Bare pf also shows help. Every subcommand accepts --help or -h.
Use -- before an object name beginning with a dash, e.g. pf show -- -h.'''

HELP = USAGE + '''

Examples below use synthetic names. Replace them with handles copied from actual
receipts, including the role and version. Do not guess a peer's version number.

1. Read exact peer evidence
When a handoff cites a saved file, read that exact version:
```bash
pf show growth:notes.md@v1 --full
```
This reads saved content, not a current verification. A bare growth:notes.md would
read the latest version, which may differ from the cited one. Check the returned
version and reading scope. Stop when the cited content answers the question;
use depend only if a current dependency check is needed.

2. Review an old conclusion
When deciding whether to reuse a registered conclusion, read its saved revision,
then check its current dependencies if needed:
```bash
pf show r1.1 --full
pf depend r1.1 --detail
```
For historical inspection without a current check:
```bash
pf depend r1.1 --detail --history
```
depend replays permitted SQL and public read-only API calls, never analyst scripts.
A SQL query with fixed dates keeps those dates when replayed. Read the checked day,
expanded scope and per-reference results. Historical-only references are not checked.
Zero checked sources and cannot_check provide no passing condition. Stop when the
needed scope has a result; if a source cannot be checked, repair or replace that
reference, obtain other evidence or keep the uncertainty explicit.

3. Find versions and revise references
When evidence changed, inspect its history and compare versions of the same object:
```bash
pf log growth:notes.md
pf diff growth:notes.md@v1 growth:notes.md@v2
```
Use handles returned by log; show reads either saved version. A diff reports content
changes, not whether the conclusion remains valid. If revising your own conclusion,
text_revise takes the complete intended reference list. For example, after reading
the actual query handle, this tool request replaces all earlier references:
```json
{"record": "r1", "text": "The revised conclusion uses the reviewed query.", "reason": "Use the reviewed evidence set.", "references": [{"cite": "query1@v1", "purpose": "current"}]}
```
Omitting references inherits the old list; references: [] clears it. Use a local rN
for text_revise, then share the exact role-qualified revision from its receipt.
Stop once the conclusion and its references match the evidence you intend to retain.

4. Register a selected condition
When a conclusion depends on one value or threshold, read the saved query:
```bash
pf show query1@v1 --full
```
Use the complete JSON request in the text_create tool description. Replace its
synthetic handle, group, metric, threshold and days with the observed evidence and
the intended claim. Selected fields must be covered in your current model input
or your own current-context file write; a truncated or partial view may omit them.
purpose defaults to current. historical_only records past evidence without checking
it against the current world. With no select/predicate, checks compare whole content;
select alone checks equality of the selected value. A threshold check reports holds,
fails or cannot_check. Holds applies only to that declared condition and checked
scope; it does not verify the whole conclusion. Stop after registration accepts the
intended condition. A missing selected field needs an adequate read, not a guessed value.

Other boundaries: blame locates the version and day each saved file line was written.
rdepend follows declared references, not every consumer or inferred business impact.
more takes the actual cursor returned by a paginated result and continues that frozen
snapshot. It does not refresh the check. Preserve scope headers when reading results;
pipes and redirection can leave only a partial view.
'''

VERBS = ('log', 'show', 'diff', 'blame', 'depend', 'rdepend', 'search', 'more', 'help')
_INVOCATION = re.compile(r'(?:^|&&|\|\||[;|\n(])\s*pf(?:\s+(?:' + '|'.join(VERBS + ('--help', '-h')) + r')\b|\s*$)', re.M)


class Usage(ValueError):
    """A pf invocation that cannot run; its message is the whole agent-facing reply."""

    def __init__(self, message):
        super().__init__(message if message == USAGE else message + '\nRun pf help for usage and complete examples.')


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
        if word == '--':
            rest.extend(words[i + 1:])
            break
        if word.startswith('--') or word == '-h':
            name = 'help' if word == '-h' else word[2:]
            if name not in allowed:
                raise Usage(f'pf: unknown option {word}')
            if allowed[name]:
                if i + 1 >= len(words):
                    raise Usage(f'pf: {word} needs a value')
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
        raise Usage('pf: only | head -N and | tail -N can follow pf')
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
            raise Usage('pf: run pf on its own in a bash call, not together with other commands')
        return None
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError as exc:
        raise Usage(f'pf: {exc}') from exc
    segments, current = [], []
    for token in tokens:
        if token == '|':
            segments.append(current)
            current = []
        elif token and set(token) <= set('|&;<>()'):
            raise Usage('pf: run pf on its own in a bash call, not together with other commands')
        else:
            current.append(token)
    segments.append(current)
    if len(segments) > 2 or not all(segments):
        raise Usage('pf: only | head -N and | tail -N can follow pf')
    view = _view(segments[1]) if len(segments) == 2 else None
    words = segments[0][1:]
    operation, arguments = parse_argv(words)
    if operation == 'pf_help':
        raise Usage(USAGE)
    return operation, arguments, view


def parse_argv(words):
    """Parse argv already expanded by Bash; never interpret shell syntax here."""
    if not words or words[0] in ('help', '--help', '-h'):
        return 'pf_help', {}
    verb, words = words[0], words[1:]
    if verb not in VERBS:
        raise Usage(f'pf: unknown command {verb}')
    options = dict(show={'full': False}, depend={'detail': False, 'history': False},
                   rdepend={'detail': False, 'all': False}, search={'all': False, 'kind': True, 'text': True}).get(verb, {})
    flags, rest = _flags(words, dict(options, help=False))
    if flags.get('help'):
        return 'pf_help', {}
    expected = (0, 0) if 'text' in flags else dict(diff=(1, 2)).get(verb, (1, 1))
    if not expected[0] <= len(rest) <= expected[1]:
        raise Usage(f'pf {verb}: wrong number of arguments')
    first = flags['text'] if 'text' in flags else rest[0]
    if verb == 'more':
        if not re.fullmatch(r'c[1-9][0-9]*', first):
            raise Usage('pf more: give the cursor from a previous result, e.g. pf more c2')
        return 'pf_more', {'cursor': first}
    if verb == 'search':
        if 'text' in flags:
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
