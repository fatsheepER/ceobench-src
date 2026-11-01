"""PF argument parsing. The old command parser is retained only for historical logs."""
import re
import shlex

from .evidence_handles import RAW_VERSION, RECORD, SINGLE, VERSIONED
from .pf_render import _HEREDOC

USAGE = '''usage: pf <command> ...
  pf help | pf --help | pf -h          complete command help and examples
  pf log <object>                      versions of a file, an output or a text, newest first
  pf show <handle> [--full]            one saved version
  pf show <output> --captures         latest execution: responses, output and files
  pf diff <object> | <old> [<new>]     what changed; one object compares its last two versions
  pf blame <file>                      the version and day each line was written
  pf depend <object> [--detail]        saved sources of this exact target
  pf rdepend <object> [--detail]       saved declarations citing this source
      [--all-revisions] [--include-candidates] [--depth N]
      depend --history excludes candidates; rdepend --all includes old revisions
  pf search <id> [--kind KIND] [--all]        texts, business writes and outputs about an object
  pf search --text "KEYWORD"           literal text in public captured history, newest first
  pf candidates [--role ROLE|all] [--history] [--day-from N] [--day-to N]
                [--kind KIND] [--text TEXT] [--event EVENT] [--detail]
                                      captured objects across calls, 20 per page
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
use depend when you need its saved source relationships.

2. Review an old conclusion
When deciding whether to reuse a registered conclusion, read its saved revision,
then inspect its saved source relationships if needed:
```bash
pf show r1.1 --full
pf depend r1.1 --detail
```
To follow declared references without automatic capture relationships:
```bash
pf depend r1.1 --detail --history
```
depend reads captured versions and relationships. It does not replay SQL or APIs,
evaluate conditions, or decide whether a judgment is valid. Read the source dates
and coverage. Run new public queries or your calculations when current evidence is
needed. Both directions default to one hop and the latest active declaration
revision across visible roles and all saved days. A forward path to an exact old
text follows that revision's saved references; reverse historical paths require
--all-revisions. --detail expands complete saved fields, not depth. Use --depth N
for more hops. --all-revisions includes old and
withdrawn revisions; a current revision does not certify current validity.
--history keeps its original meaning: follow declarations without automatic
capture relationships. It does not select old revisions.

Files, outputs, exact text revisions and concrete op: operation references can
be targets or sources. Copy the complete operation_ref from its business receipt.
Git file references keep their exact commit and storage role. Author declarations,
other authors' inferences and automatic execution observations remain separate.
--include-candidates adds saved derivation/range links and execution membership.
Membership names a captured object and its execution, stops at that event, and
provides a candidates command for its recorded members. It does not connect every
co-occurring input and output or prove source use. Missing materials retain their descriptions without a
fabricated readable version. Each declaration keeps its author, record and revision.
Use pf more to continue the frozen page or read its full details. Known sources
can be registered directly with relation_batch, without a candidates query.

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
Before registering selected fields, run pf show alone, without a pipe.
purpose defaults to current. historical_only marks a reference as past evidence.
select and predicate store your declared condition without evaluating it.
A missing selected field needs an adequate read, not a guessed value.

5. Locate evidence
Replace KEYWORD with the words you need to find; this works before any registration:
```bash
pf search --text "KEYWORD"
```
For a known business object, replace ACTUAL_ID with its real ID:
```bash
pf search ACTUAL_ID
```
Literal search scans captured content; object search uses recorded object IDs.
Read the relevant returned version with show.

6. Find sources across calls and weeks
```bash
pf candidates
pf candidates --role all --history --kind sql --text "KEYWORD"
pf candidates --day-from 7 --day-to 13 --kind file
```
The default is executions initiated by your role in the current simulated week,
20 captured objects per page. The header labels this role basis explicitly.
--role all includes currently visible business material; --history removes the week
filter. Explicit day bounds replace the week filter (an omitted bound is open).
Kinds: file, sql, api, stdout, stderr, output, code, text, script, dashboard, execution.
Each row keeps its execution event even when content has the same version handle.
Use the returned pf more cursor to continue the frozen snapshot or expand this
page's details. For a copied EVENT, --event EVENT includes its captured children;
--detail pages exact requests and receipt metadata. Source bodies use pf show.
Details include content_ref, an exact acquisition accepted by pf show, separately
from the reusable content handle. Use the content handle in relation_batch.
Captured, delivered to a model, and declared used are different. Model delivery
is shown for your reader role and the originating request before the query. Unknown
does not mean unread. Candidate summaries do not deliver the full source body.
Text filters are case-insensitive literals in source names, captured UTF-8 content,
and non-file requests (up to 2,000 characters). Details retain exact long handles.
Capture gaps include unobserved internal reads, intermediate writes and pipe streams.
Binary script streams without a public text representation are marked explicitly.
File reads that reuse saved bytes keep their own acquisition event and text
representation; same-event byte/text duplicates and SQL display projections are
represented by their source version. Interpreter cache files, session logs and
private model bodies are excluded from candidates.
No query replay or validity judgment occurs. Known sources can be registered directly.

Other boundaries: blame locates the version and day each saved file line was written.
rdepend navigates saved declarations and inferences, not inferred business impact.
more takes the actual cursor returned by a paginated result and continues that frozen
snapshot. Preserve scope headers when reading results;
pipes and redirection can leave only a partial view.
'''

VERBS = ('log', 'show', 'diff', 'blame', 'depend', 'rdepend', 'search', 'candidates', 'more', 'help')
_INVOCATION = re.compile(r'(?:^|&&|\|\||[;|\n(])\s*pf(?:\s+(?:' + '|'.join(VERBS + ('--help', '-h')) + r')\b|\s*$)', re.M)


class Usage(ValueError):
    """A pf invocation that cannot run; its message is the whole agent-facing reply."""

    def __init__(self, message):
        super().__init__(message if message == USAGE else message + '\nRun pf help for usage and complete examples.')


def mentions_pf(command):
    """Whether a command invokes pf anywhere outside heredoc bodies."""
    return bool(_INVOCATION.search(_HEREDOC.sub('\n', command)))


def target(token):
    if re.fullmatch(r'op:[0-9a-f]{32}', token):
        return {'operation_ref': token}
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
    options = dict(show={'full': False, 'captures': False}, depend={'detail': False, 'history': False, 'all-revisions': False, 'include-candidates': False, 'depth': True},
                   rdepend={'detail': False, 'all': False, 'all-revisions': False, 'include-candidates': False, 'depth': True}, search={'all': False, 'kind': True, 'text': True},
                   candidates={'role': True, 'history': False, 'day-from': True, 'day-to': True,
                               'kind': True, 'text': True, 'event': True, 'detail': False}).get(verb, {})
    flags, rest = _flags(words, dict(options, help=False))
    if flags.get('help'):
        return 'pf_help', {}
    if verb == 'candidates':
        if rest:
            raise Usage('pf candidates: use filters, not positional arguments')
        for name in ('day-from', 'day-to'):
            if name in flags:
                if not re.fullmatch(r'[0-9]+', flags[name]):
                    raise Usage('pf candidates: --' + name + ' requires a nonnegative simulated day')
                flags[name] = int(flags[name])
        return 'pf_candidates', {name.replace('-', '_'): value for name, value in flags.items()}
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
        if flags.get('captures') and flags.get('full'):
            raise Usage('Use --captures or --full, not both')
        return 'pf_read', {'target': target(first), 'mode': 'captures' if flags.get('captures') else 'content', 'full': bool(flags.get('full'))}
    if verb == 'diff':
        if len(rest) == 2:
            return 'pf_read', {'target': target(rest[1]), 'mode': 'diff', 'baseline': target(first)}
        return 'pf_diff', {'target': target(first)}
    if flags.get('history') and flags.get('include-candidates'):
        raise Usage('pf depend: --history excludes candidates; omit --include-candidates')
    if 'depth' in flags and not re.fullmatch(r'(?:[1-9]|10)', flags['depth']):
        raise Usage('pf: --depth requires an integer from 1 to 10')
    options = dict(target=target(first), detail=bool(flags.get('detail')))
    if 'depth' in flags:
        options['depth'] = int(flags['depth'])
    if flags.get('all-revisions'):
        options['all_revisions'] = True
    if flags.get('include-candidates'):
        options['include_candidates'] = True
    if verb == 'depend':
        return 'pf_dependencies', dict(options, purpose='historical_only' if flags.get('history') else 'current')
    return 'pf_dependents', dict(options, current_only=not flags.get('all'))


def apply_view(text, view):
    if not view:
        return text
    lines = str(text).splitlines(keepends=True)
    kind, count = view
    return ''.join(lines[:count] if kind == 'head' else lines[-count:])
