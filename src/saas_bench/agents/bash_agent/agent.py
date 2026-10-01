"""Bash agent for SaaS Bench.

A Claude Code-style agent that uses bash and file tools to interact with the
NovaMind SaaS simulator via the novamind_api Python library and CLI.

Supports OpenAI-compatible APIs (OpenAI, xAI) and Anthropic APIs (direct, Bedrock).
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

from ..base import BaseAgent
from ...environment import Action
from ...model_usage import ModelUsage, usage_values


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: Any  # str or list (Anthropic content blocks)
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    # DeepSeek thinking mode: returned reasoning must be sent back verbatim after tool calls.
    reasoning_content: Optional[str] = None


# Regex to detect dashboard in bash output (day advancement)
_DASHBOARD_RE = re.compile(r'=== (?:Day (\d+) Dashboard|Week \d+ Dashboard \(Day (\d+)\)) ===')


class BashAgent(BaseAgent):
    """Bash agent for SaaS Bench — Claude Code-style.

    Uses bash, read_file, write_file, edit_file, search_files, glob_files
    tools. Interacts with the simulator via novamind_api Python library
    and ./novamind-operation CLI.

    After calling `./novamind-operation next-week`, context is refreshed:
    the conversation is cleared and rebuilt with system prompt + MEMORY.md
    contents + the new dashboard.
    """

    def __init__(
        self,
        tool_descriptions: List[Dict[str, Any]],
        client,
        model: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        max_turns_per_day: int = 0,  # 0 = no limit
        response_callback: Optional[callable] = None,
        reasoning_effort: Optional[str] = None,
        tool_result_callback: Optional[callable] = None,
        workspace_path: Optional[Path] = None,
        total_days: int = 3650,
        anthropic_fallback_model: Optional[str] = None,
        usage_recorder: Optional[ModelUsage] = None,
        text_registration: bool = False,
        pf: bool = False,
    ):
        if not tool_descriptions:
            raise ValueError('BashAgent requires tools; an empty list cannot produce a valid action')
        super().__init__(tool_descriptions)
        self.client = client
        self.usage_recorder = usage_recorder or ModelUsage(None, 'agent')
        self.usage_recorder.attach(client)
        self.model = model
        self.max_turns_per_day = max_turns_per_day
        self.response_callback = response_callback
        self.reasoning_effort = reasoning_effort
        self.tool_result_callback = tool_result_callback
        self.workspace_path = workspace_path or Path('.')
        self.total_days = total_days
        self.anthropic_fallback_model = anthropic_fallback_model

        # Detect client type
        client_type = type(client).__name__
        self.use_anthropic = client_type in ('Anthropic', 'AnthropicBedrock')
        self.use_portkey = client_type == 'Portkey'

        # Detect if the endpoint supports OpenAI Responses API.
        # Only OpenAI's own endpoint implements /v1/responses. Google's OpenAI-compat
        # and Together AI only expose /v1/chat/completions. Portkey AI Gateway
        # forwards to OpenAI for gpt-* models so /v1/responses works there too —
        # in fact, gpt-5.x with reasoning_effort + tools REQUIRES /v1/responses.
        base_url = str(getattr(client, 'base_url', '') or '')
        _non_responses_hosts = (
            'generativelanguage.googleapis.com',
            'api.together.xyz',
            'api.deepseek.com',
            'opencode.ai',
        )
        self.supports_responses_api = not any(h in base_url for h in _non_responses_hosts)

        # Build system prompt
        self.system_prompt = system_prompt or self._default_system_prompt()
        if pf and not text_registration:
            raise ValueError('PF requires text registration')
        self.text_registration, self.pf = text_registration, pf
        if text_registration:
            # Registration groups integrate their sections into the original prompt.
            from saas_bench.registration_prompt import integrate
            self.system_prompt = integrate(self.system_prompt, pf)

        # Agent state
        self.conversation: List[Message] = []
        self.current_day: int = -1
        self.turns_today: int = 0
        self._pending_tool_calls: List[Dict] = []
        self._last_observation: str = ""
        self.total_turns: int = 0
        self._day_advanced: bool = False
        self._new_dashboard: str = ""
        self._consecutive_errors: int = 0

        # Token usage tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_reasoning_tokens: int = 0
        self.total_cache_creation_tokens = 0
        self.last_input_tokens = None
        self.last_output_tokens = None
        self.last_cached_tokens = None
        self.last_reasoning_tokens = None
        self.last_cache_creation_tokens = None
        self.last_serving_model: str = model
        self.last_anthropic_fallback_used: bool = False
        self.last_anthropic_fallbacks: List[Dict[str, str]] = []
        self.total_anthropic_fallbacks: int = 0

        # Conversation snapshot — persisted after each LLM call so a mid-day
        # crash can restore the exact accumulated context on resume. Path is
        # set by run_test.py after the session_id is known. Snapshot includes
        # the conversation, _pending_tool_calls, current_day, and turns_today.
        self._snapshot_path: Optional[Path] = None
        # When True, the next observation will skip _refresh_context so a
        # restored mid-day conversation isn't wiped. Cleared after one act().
        self._skip_next_refresh: bool = False

    def _default_system_prompt(self) -> str:
        """Build the default system prompt.

        Loads the bash_agent system_prompt.md and fills in
        simulator instructions and configured days, weeks and years.

        ORACLE MODE: when env var ORACLE_MODE=1, prepend system_prompt_oracle.md
        as a preamble. The oracle preamble explicitly overrides the "hidden
        column / not accessible" caveats in the standard prompt, lists the
        hidden tables and customer_state columns, and points the agent at the
        read-only simulator source tree under /data/saas-bench/src.
        """
        base_dir = Path(__file__).parent

        # Load simulator instructions — strip the {tool_list} placeholder
        # (bash_agent has its own tool reference in system_prompt.md)
        simulator_file = base_dir.parent / "simulator_instructions.md"
        with open(simulator_file, 'r') as f:
            sim_text = f.read()

        sim_text = sim_text.replace('{tool_list}\n', '')
        sim_text = sim_text.replace('{tool_list}', '')

        template_file = base_dir / "system_prompt.md"
        with open(template_file, 'r') as f:
            template = f.read()

        prompt = template.replace('{simulator_instructions}', sim_text)

        # Fill duration consistently for all experiment groups.
        total_years = self.total_days / 365
        years_str = f"{total_years:.0f}" if total_years == int(total_years) else f"{total_years:.1f}"
        prompt = prompt.replace('{total_days}', str(self.total_days))
        prompt = prompt.replace('{total_weeks}', str((self.total_days + 6) // 7))
        prompt = prompt.replace('{total_years}', years_str)

        if os.environ.get("ORACLE_MODE") == "1":
            oracle_file = base_dir / "system_prompt_oracle.md"
            if oracle_file.exists():
                with open(oracle_file, 'r') as f:
                    oracle_preamble = f.read()
                prompt = oracle_preamble + "\n\n" + prompt
        return prompt

    @property
    def evidence_store(self):
        return getattr(getattr(self, 'usage_recorder', None), 'evidence_store', None)

    def _get_system_prompt_with_memory(self) -> str:
        """Read MEMORY.md once when building a new week's system prompt."""
        prompt = self.system_prompt
        memory_path = self.workspace_path / 'MEMORY.md'
        if not memory_path.resolve().is_relative_to(self.workspace_path.resolve()):
            raise ValueError('MEMORY.md must stay within the agent workspace')
        if memory_path.exists():
            try:
                original_memory = memory_path.read_bytes()
                from saas_bench.execution_capture import ExecutionCapture, CapturedText, decoded, origin
                memory_text = decoded(original_memory)
                memory_content = memory_text.strip()
                memory_origin = memory_version = None
                if self.evidence_store:
                    capture = ExecutionCapture(self.evidence_store)
                    capture.begin('memory_read', {'path': 'MEMORY.md'})
                    memory_version = capture.file('MEMORY.md', original_memory, memory_text)
                    if memory_version and memory_content:
                        begin = len(memory_text) - len(memory_text.lstrip())
                        memory_origin = origin(memory_version, memory_text, begin, begin + min(40000, len(memory_content)))
                    if capture.event:
                        capture.safe(capture.store.complete, capture.event)
                if memory_content:
                    max_memory_chars = 40_000
                    if len(memory_content) > max_memory_chars:
                        memory_content = memory_content[:max_memory_chars] + (
                            "\n\n--- MEMORY.md TRUNCATED ---\n"
                            f"Showing first {max_memory_chars:,} of {len(memory_content):,} characters. "
                            "Use the read_file tool to see the full contents if needed."
                        )
                    if getattr(self, 'text_registration', False):
                        from saas_bench.registration_prompt import MEMORY_HEADER
                        history = self._memory_history(memory_version)
                        prompt += MEMORY_HEADER + (history + '\n' if history else '') + '\n' + memory_content
                    else:
                        prompt += (
                            "\n\n## Your MEMORY.md (auto-loaded)\n\n"
                            "The following is the contents of your MEMORY.md file. "
                            "This is automatically loaded into your context at the start of every day.\n\n"
                            f"{memory_content}"
                        )
                    if memory_origin:
                        length = memory_origin['request_range'][1]
                        offset = len(prompt) - len(memory_content)
                        memory_origin['request_range'] = [offset, offset + length]
                        prompt = CapturedText(prompt, [memory_origin])
            except Exception as exc:
                if self.evidence_store:
                    self.evidence_store.fail(exc)
        return prompt

    def _memory_history(self, version):
        """The registration groups' line on MEMORY.md's history: PF names the loaded version,
        Git the last weekly commit that changed the file."""
        from saas_bench import registration_prompt
        if self.pf:
            return registration_prompt.pf_memory_line(self.evidence_store, version) if version else None
        return registration_prompt.git_memory_line(self.workspace_path)

    def _context_system_prompt(self) -> str:
        """Reuse the frozen prompt, including its private source ranges."""
        for message in self.conversation:
            if message.role == 'system':
                return message.content
        # Legacy snapshots may have no system message. Freeze once on migration.
        prompt = self._get_system_prompt_with_memory()
        self.conversation.insert(0, Message(role='system', content=prompt))
        return prompt

    def reset(self):
        """Reset agent state for a new episode."""
        self.conversation = []
        self.current_day = -1
        self.turns_today = 0
        self._pending_tool_calls = []
        self._last_observation = ""
        self._observation_recorded = False
        self._day_advanced = False
        self._new_dashboard = ""
        self._skip_next_refresh = False

    def _refresh_context(self, dashboard: str, new_day: int):
        """Refresh conversation context for a new day.

        Clears conversation, inserts system prompt + dashboard.
        The agent reads its own files via tools when it needs context.
        """
        self.conversation = []
        if self.evidence_store:
            import uuid
            self.usage_recorder.context_id = uuid.uuid4().hex
        self._pending_tool_calls = []
        self._observation_recorded = False

        self._context_system_prompt()

    def check_day_advanced(self, bash_output: str) -> bool:
        """Check if bash output contains a dashboard (day advanced).

        Returns True if a new day dashboard was detected.
        Stores the dashboard text for context refresh.
        """
        match = _DASHBOARD_RE.search(bash_output)
        if match:
            new_day = int(match.group(1) or match.group(2))
            if new_day > self.current_day:
                self._day_advanced = True
                # Extract dashboard from the output (everything from === Day N ===)
                dashboard_start = match.start()
                self._new_dashboard = bash_output[dashboard_start:]
                if hasattr(bash_output, 'origins'):
                    from saas_bench.execution_capture import CapturedText, slice_origins
                    self._new_dashboard = CapturedText(self._new_dashboard,
                        slice_origins(bash_output.origins, dashboard_start, len(bash_output)))
                return True
        return False

    @property
    def day_advanced(self) -> bool:
        """Whether the last bash command advanced the day."""
        return self._day_advanced

    @property
    def new_dashboard(self) -> str:
        """The dashboard text from the last day advancement."""
        return self._new_dashboard

    def clear_day_advanced(self):
        """Clear the day-advanced flag after the runner processes it."""
        self._day_advanced = False
        self._new_dashboard = ""

    def act(self, observation: str, reward: float, done: bool, info: Dict[str, Any]) -> Optional[Action]:
        """Choose an action based on the observation.

        The agent processes tool outputs and decides the next action.
        After day advancement is detected, context is refreshed.
        """
        if done:
            return None

        self._last_observation = observation

        # Check if this is a new day (context refresh)
        current_day = info.get('day', 0)
        if current_day > self.current_day:
            if self._skip_next_refresh:
                # Mid-day resume: conversation was restored from snapshot.
                # Don't wipe it; just sync the day counter and clear the flag.
                self.current_day = current_day
                self._skip_next_refresh = False
            else:
                self._refresh_context(observation, current_day)
                self.current_day = current_day
                self.turns_today = 0

        # Safety: force next_week if too many turns (0 = no limit)
        if self.max_turns_per_day > 0 and self.turns_today >= self.max_turns_per_day:
            return Action(tool='bash', arguments={'command': './novamind-operation next-week'})

        # If we have pending tool call results to process, add them
        if getattr(self, '_observation_recorded', False):
            self._observation_recorded = False
        else:
            self.record_tool_result(observation)
            self._observation_recorded = False

        # Call LLM
        if self._pending_tool_calls:
            raise RuntimeError('Finish the current tool batch before requesting another model response')
        self._llm_attempt = 0
        action = self._call_llm()
        self.turns_today += 1

        # Persist conversation snapshot so a mid-day crash can be resumed
        # with the exact accumulated context. Best-effort: failure here must
        # not derail the run, just log.
        self._save_conversation_snapshot()

        return action

    def _request_model(self, api, request, invoke):
        self._llm_attempt = getattr(self, '_llm_attempt', 0) + 1
        try:
            response = self.usage_recorder.call(api, request, invoke, day=self.current_day,
                                                turn=self.total_turns + 1, outer_attempt=self._llm_attempt)
        finally:
            for field in ('input_tokens', 'output_tokens', 'cached_tokens', 'cache_creation_tokens', 'reasoning_tokens'):
                setattr(self, 'total_' + field, self.usage_recorder.summary['known'][field] or 0)
        for field, value in usage_values(response, api).items():
            setattr(self, 'last_' + field, value)
        return response

    def next_tool_action(self):
        """Next call from the current response, in provider order; no new model request."""
        if not self._pending_tool_calls:
            return None
        call = self._pending_tool_calls[0]
        return Action(tool=call['name'], arguments=call.get('arguments', {}))

    def record_tool_result(self, observation, call_id=None):
        """Complete exactly one call; a batch must finish before checkpoint publication."""
        if self._pending_tool_calls:
            call = self._pending_tool_calls[0]
            if call_id is not None and call_id != call['id']:
                raise ValueError('Tool results must follow response order')
            if self.use_anthropic:
                result = dict(type='tool_result', tool_use_id=call['id'], content=observation)
                if (self.conversation and self.conversation[-1].role == 'user'
                        and isinstance(self.conversation[-1].content, list)
                        and all(item.get('type') == 'tool_result' for item in self.conversation[-1].content)):
                    self.conversation[-1].content.append(result)
                else:
                    self.conversation.append(Message(role='user', content=[result]))
            else:
                self.conversation.append(Message(role='tool', content=observation,
                                                  tool_call_id=call['id'], name=call['name']))
            self._pending_tool_calls.pop(0)
        else:
            self.conversation.append(Message(role='user', content=observation))
        self._last_observation = observation
        self._observation_recorded = True

    def _serialize_content_item(self, item: Any) -> Any:
        """Convert a single content item to a JSON-safe form.

        Assistant messages from the OpenAI Responses API store raw pydantic
        output items (ResponseReasoningItem, ResponseFunctionToolCall, etc.)
        in `content`. These must round-trip via `.model_dump()` so reasoning
        summaries + function_calls survive the snapshot. Plain dicts and
        primitives pass through unchanged.
        """
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json", exclude_none=False, by_alias=True)
        if isinstance(item, dict):
            return item
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        # Anthropic content blocks may be objects without model_dump; fall back
        # to repr so we never crash. They won't be replay-correct, but the
        # snapshot is best-effort and Anthropic uses a different code path.
        return repr(item)

    def _serialize_message(self, m: "Message") -> Dict[str, Any]:
        content = m.content
        if isinstance(content, list):
            content = [self._serialize_content_item(x) for x in content]
        data = {
            "role": m.role,
            "content": content,
            "tool_calls": m.tool_calls,
            "tool_call_id": m.tool_call_id,
            "name": m.name,
        }
        if m.reasoning_content is not None:
            data["reasoning_content"] = m.reasoning_content
        return data

    def _save_conversation_snapshot(self, strict=False) -> None:
        """Atomically write self.conversation + minimal turn state to disk.

        Overwrites the same file each call (single snapshot, not append-only).
        On Modal/NFS, rename() is atomic — readers see either the old or new
        complete file, never a partial write.

        Pydantic Responses-API output items (reasoning summaries, function
        calls) are converted via `.model_dump()` so they survive the JSON
        round-trip — see _serialize_content_item.
        """
        if self._snapshot_path is None:
            return
        try:
            payload = {
                "conversation": [self._serialize_message(m) for m in self.conversation],
                "pending_tool_calls": list(self._pending_tool_calls),
                "current_day": self.current_day,
                "turns_today": self.turns_today,
                "total_turns": self.total_turns,
                "last_observation": self._last_observation,
                "observation_recorded": getattr(self, "_observation_recorded", False),
                "saved_at": time.time(),
            }
            tmp_path = self._snapshot_path.with_suffix(self._snapshot_path.suffix + ".tmp")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, self._snapshot_path)
            if self.evidence_store:
                from saas_bench.execution_capture import text_sources
                from saas_bench.sql_evidence import digest
                self.evidence_store.save_state('agent_sources',
                    dict(snapshot_sha256=digest(self._snapshot_path.read_bytes()),
                         context_id=self.usage_recorder.context_id, sources=text_sources(payload)))
        except Exception as e:
            if self.evidence_store:
                self.evidence_store.fail(e)
            if strict:
                raise
            # Per-turn best effort files are not published checkpoints.
            print(f"[snapshot] WARN failed to save conversation snapshot: {e}")

    def load_conversation_snapshot(self, path: Path) -> bool:
        """Restore self.conversation + turn state from a snapshot file.

        Rejects unresolved tool calls. Completed tool results are kept and
        the same-week observation is not inserted a second time.

        Returns True if restoration succeeded, False otherwise.
        """
        if not path.exists():
            return False
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            if self.evidence_store:
                from saas_bench.execution_capture import restore_sources
                from saas_bench.sql_evidence import digest
                state = self.evidence_store.load_state('agent_sources')
                if not state or state['snapshot_sha256'] != digest(path.read_bytes()):
                    raise ValueError('Missing or mismatched private source state')
                restore_sources(payload, state['sources'])
                self.usage_recorder.context_id = state['context_id']
            raw_msgs = payload.get("conversation", [])
            msgs: List[Message] = []
            for m in raw_msgs:
                msgs.append(Message(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                    name=m.get("name"),
                    reasoning_content=m.get("reasoning_content"),
                ))

            if payload.get('pending_tool_calls'):
                raise ValueError('Snapshot has a tool with an unknown outcome')
            self.conversation = msgs
            self._pending_tool_calls = []
            self.current_day = int(payload.get("current_day", 0) or 0)
            self.turns_today = int(payload.get("turns_today", 0) or 0)
            self._skip_next_refresh = False
            self._last_observation = payload['last_observation']
            self._observation_recorded = payload['observation_recorded']
            if not any(message.role == 'system' for message in self.conversation):
                print('[snapshot] Legacy context has no system prompt; freezing current system/MEMORY once.')
                self._context_system_prompt()
            return True
        except Exception as e:
            print(f"[snapshot] WARN failed to load conversation snapshot: {e}")
            return False

    def _wants_reasoning(self) -> bool:
        """True when a non-empty reasoning effort other than 'none' is set."""
        return bool(self.reasoning_effort) and self.reasoning_effort != 'none'

    def _call_llm(self) -> Optional[Action]:
        """Call the LLM and parse the response into an action."""
        if self.use_anthropic:
            return self._call_anthropic()
        elif self._wants_reasoning() and self.supports_responses_api:
            return self._call_openai_responses()
        else:
            return self._call_openai()

    def _call_openai(self) -> Optional[Action]:
        """Call OpenAI-compatible API and parse the response."""
        self._context_system_prompt()
        import time as _time
        import traceback
        import signal
        import openai

        LLM_WALL_CLOCK_TIMEOUT = 600  # 10min hard wall-clock limit per LLM call

        class LLMTimeoutError(Exception):
            pass

        def _llm_timeout_handler(signum, frame):
            raise LLMTimeoutError(f"LLM call exceeded {LLM_WALL_CLOCK_TIMEOUT}s wall-clock timeout")

        while True:
            messages = []
            for msg in self.conversation:
                m = {'role': msg.role, 'content': msg.content or ''}
                if msg.tool_call_id:
                    m['tool_call_id'] = msg.tool_call_id
                if msg.name:
                    m['name'] = msg.name
                if msg.tool_calls:
                    m['tool_calls'] = msg.tool_calls
                # Omitting returned reasoning after a tool call makes DeepSeek answer 400.
                if msg.reasoning_content is not None:
                    m['reasoning_content'] = msg.reasoning_content
                messages.append(m)

            tools = [
                {
                    'type': 'function',
                    'function': {
                        'name': t['name'],
                        'description': t['description'],
                        'parameters': t['parameters']
                    }
                }
                for t in self.tool_descriptions
            ]

            try:
                _base_url = str(getattr(self.client, 'base_url', '') or '')
                _is_together = 'api.together.xyz' in _base_url
                _is_together_deepseek = _is_together and 'deepseek' in self.model.lower()
                _is_deepseek_compat = (
                    'api.deepseek.com' in _base_url or 'opencode.ai' in _base_url
                )
                api_kwargs = {
                    'model': self.model,
                    'messages': messages,
                    'tools': tools,
                    'tool_choice': 'auto',
                    'temperature': 1.0,
                }
                # OpenAI-compat gateways (DeepSeek, OpenCode Go) reject
                # max_completion_tokens; native OpenAI accepts both.
                if _is_deepseek_compat:
                    api_kwargs['max_tokens'] = 16384
                else:
                    api_kwargs['max_completion_tokens'] = 16384
                if self._wants_reasoning():
                    api_kwargs['reasoning_effort'] = self.reasoning_effort
                if _is_together_deepseek:
                    # Together's DeepSeek-V4 thinking is gated on the chat-template flag,
                    # not on `reasoning_effort` (which Together rejects with 400 for max,
                    # 500 for medium, and silently ignores for low/high). The flag below
                    # produces real chain-of-thought in `message.reasoning`.
                    api_kwargs['extra_body'] = {'chat_template_kwargs': {'thinking': True}}
                elif _is_deepseek_compat:
                    # V4 Flash/Pro default thinking ON. Smoke runs disable it.
                    # Official DeepSeek honors extra_body; OpenCode Go also needs
                    # reasoning_effort=none or it still emits reasoning_content.
                    thinking_type = 'enabled' if self._wants_reasoning() else 'disabled'
                    api_kwargs['extra_body'] = {'thinking': {'type': thinking_type}}
                    if not self._wants_reasoning():
                        api_kwargs['reasoning_effort'] = 'none'

                # Set hard wall-clock timeout via signal.alarm
                old_handler = signal.signal(signal.SIGALRM, _llm_timeout_handler)
                signal.alarm(LLM_WALL_CLOCK_TIMEOUT)
                try:
                    response = self._request_model('chat', api_kwargs,
                                                   lambda: self.client.chat.completions.create(**api_kwargs))
                finally:
                    signal.alarm(0)  # Cancel alarm
                    signal.signal(signal.SIGALRM, old_handler)  # Restore handler
                self.total_turns += 1
                self._consecutive_errors = 0

                if self.response_callback:
                    self.response_callback(
                        turn=self.total_turns,
                        day=self.current_day,
                        messages=messages,
                        raw_response=response.model_dump() if hasattr(response, 'model_dump') else str(response),
                    )

                assistant_msg = response.choices[0].message

                # Log reasoning_content if present (e.g. GLM-5 reasoning model)
                reasoning_content = getattr(assistant_msg, 'reasoning_content', None)
                if not reasoning_content:
                    extras = getattr(assistant_msg, 'model_extra', {}) or {}
                    reasoning_content = extras.get('reasoning_content')
                if reasoning_content and self.tool_result_callback:
                    self.tool_result_callback(
                        self.total_turns, self.current_day, '_reasoning', {},
                        reasoning_content
                    )

                # Validate tool_call arguments are parseable JSON BEFORE appending
                # to conversation. Storing a tool_call with invalid-JSON args poisons
                # the history: some OpenAI-compat servers (e.g. Together) reject every
                # subsequent request with 400 "Input validation error" on replay.
                json_validation_error = None
                if assistant_msg.tool_calls:
                    for tc in assistant_msg.tool_calls:
                        if tc.function.arguments:
                            try:
                                json.loads(tc.function.arguments)
                            except json.JSONDecodeError as je:
                                json_validation_error = (
                                    tc.function.name,
                                    str(je),
                                    tc.function.arguments[:300],
                                )
                                break

                if json_validation_error:
                    name, err, preview = json_validation_error
                    print(f"  Invalid JSON in tool_call `{name}`: {err}. Feeding error back to LLM and regenerating.")
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            f"Your previous response contained invalid JSON in the `{name}` tool_call arguments.\n"
                            f"JSON decode error: {err}\n"
                            f"Arguments started with: {preview}...\n\n"
                            f"Valid JSON escape sequences are limited to: \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX. "
                            f"Shell-style escapes like \\$ or \\! are NOT valid JSON. "
                            f"Please re-emit the tool call with valid JSON."
                        )
                    ))
                    continue

                tool_calls_data = None
                if assistant_msg.tool_calls:
                    tool_calls_data = []
                    for tc in assistant_msg.tool_calls:
                        tc_dict = {
                            'id': tc.id,
                            'type': 'function',
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments
                            }
                        }
                        # Preserve Gemini thought_signature (required by Gemini
                        # OpenAI-compat endpoint — must be echoed back on replay).
                        tc_extras = getattr(tc, 'model_extra', None) or {}
                        extra_content = tc_extras.get('extra_content')
                        if extra_content:
                            tc_dict['extra_content'] = extra_content
                        tool_calls_data.append(tc_dict)

                self.conversation.append(Message(
                    role='assistant',
                    content=assistant_msg.content or '',
                    tool_calls=tool_calls_data,
                    # DeepSeek-compatible thinking requires the field on every later assistant turn.
                    reasoning_content=(reasoning_content or '') if (_is_deepseek_compat and self._wants_reasoning()) else None,
                ))

                if not assistant_msg.tool_calls:
                    # LLM emitted no tool_call — feed feedback and retry.
                    print("  LLM returned no tool_call. Feeding feedback and regenerating.")
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            "You must call a tool to proceed. If you have nothing else to do this week, "
                            "call `./novamind-operation next-week <cash_1wk> <cash_4wk> <cash_12wk>` via bash to advance."
                        )
                    ))
                    continue

                self._pending_tool_calls = [dict(id=tc.id, name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or '{}')) for tc in assistant_msg.tool_calls]
                return self.next_tool_action()

            except Exception as e:
                if self.evidence_store:
                    self.evidence_store.assert_healthy(quiescent=False)
                status = getattr(e, 'status_code', 0) or 0
                is_retryable = isinstance(e, openai.APIStatusError) and (status >= 500 or status == 429)
                if not is_retryable:
                    is_retryable = isinstance(e, (openai.APIConnectionError, openai.APITimeoutError, LLMTimeoutError))
                if not is_retryable:
                    is_retryable = any(code in str(e) for code in ('429', '500', '502', '503', '504', '529'))
                print(f"OpenAI LLM call error (retryable={is_retryable}, status={status}): {e}")
                if is_retryable:
                    # Retry with exponential backoff — keep trying forever until
                    # the endpoint comes back. Never fall back to next-week.
                    self._consecutive_errors = getattr(self, '_consecutive_errors', 0) + 1
                    wait_time = min(120, 10 * (2 ** min(self._consecutive_errors - 1, 3)))
                    print(f"  Server error ({self._consecutive_errors}), retrying in {wait_time}s...")
                    _time.sleep(wait_time)
                    # Free memory before retry (messages/tools rebuilt at top of loop)
                    del messages, tools
                    continue  # Loop back to retry
                else:
                    # Non-retryable error (4xx other than 429). No next-week fallback —
                    # feed the error message back to the LLM as a user turn and regenerate.
                    print(f"  Non-retryable error — feeding back to LLM for regeneration.")
                    print(f"Traceback: {traceback.format_exc()}")
                    self._consecutive_errors = getattr(self, '_consecutive_errors', 0) + 1
                    wait_time = min(60, 5 * self._consecutive_errors)
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            f"The previous API request failed with a non-retryable error:\n"
                            f"{type(e).__name__}: {e}\n\n"
                            f"Please re-emit your response. If the error mentions input validation, "
                            f"check your tool_call arguments are valid JSON. "
                            f"If the error mentions context length, produce a shorter response."
                        )
                    ))
                    _time.sleep(wait_time)
                    del messages, tools
                    continue

    def _call_openai_responses(self) -> Optional[Action]:
        """Call OpenAI Responses API (required for reasoning models with tools)."""
        import time as _time
        import traceback
        import signal
        import openai

        LLM_WALL_CLOCK_TIMEOUT = 600

        class LLMTimeoutError(Exception):
            pass

        def _llm_timeout_handler(signum, frame):
            raise LLMTimeoutError(f"LLM call exceeded {LLM_WALL_CLOCK_TIMEOUT}s wall-clock timeout")

        while True:
            # Build input array from conversation
            input_items = []
            for msg in self.conversation:
                if msg.role == 'system':
                    continue  # System prompt goes in instructions parameter
                elif msg.role == 'user':
                    input_items.append({'role': 'user', 'content': msg.content or ''})
                elif msg.role == 'assistant':
                    if isinstance(msg.content, list):
                        # Raw response.output items from previous Responses API call
                        input_items.extend(msg.content)
                    else:
                        input_items.append({'role': 'assistant', 'content': msg.content or ''})
                elif msg.role == 'tool':
                    input_items.append({
                        'type': 'function_call_output',
                        'call_id': msg.tool_call_id,
                        'output': msg.content or '',
                    })

            # Build tools (Responses API format — no nested function wrapper)
            tools = [
                {
                    'type': 'function',
                    'name': t['name'],
                    'description': t['description'],
                    'parameters': t['parameters'],
                }
                for t in self.tool_descriptions
            ]

            try:
                api_kwargs = {
                    'model': self.model,
                    'input': input_items,
                    'tools': tools,
                    'tool_choice': 'auto',
                    'max_output_tokens': 16384,
                    'instructions': self._context_system_prompt(),
                }
                if self.reasoning_effort:
                    api_kwargs['reasoning'] = {'effort': self.reasoning_effort, 'summary': 'auto'}

                # Set hard wall-clock timeout via signal.alarm
                old_handler = signal.signal(signal.SIGALRM, _llm_timeout_handler)
                signal.alarm(LLM_WALL_CLOCK_TIMEOUT)
                try:
                    response = self._request_model('responses', api_kwargs,
                                                   lambda: self.client.responses.create(**api_kwargs))
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                self.total_turns += 1
                self._consecutive_errors = 0

                if self.response_callback:
                    self.response_callback(
                        turn=self.total_turns,
                        day=self.current_day,
                        messages=input_items,
                        raw_response=response.model_dump() if hasattr(response, 'model_dump') else str(response),
                    )

                # Log reasoning content if present
                for item in response.output:
                    if getattr(item, 'type', '') == 'reasoning' and self.tool_result_callback:
                        reasoning_text = ''
                        for summary in getattr(item, 'summary', []) or []:
                            reasoning_text += getattr(summary, 'text', '') + '\n'
                        if reasoning_text.strip():
                            self.tool_result_callback(
                                self.total_turns, self.current_day, '_reasoning', {},
                                reasoning_text.strip()
                            )

                # Find function_call items
                function_calls = [item for item in response.output
                                  if getattr(item, 'type', '') == 'function_call']

                # Validate each function_call's arguments JSON BEFORE storing. An
                # invalid-JSON tool_call poisons the conversation (server-side
                # validators reject every subsequent request on replay).
                json_validation_error = None
                for fc in function_calls:
                    if fc.arguments:
                        try:
                            json.loads(fc.arguments)
                        except json.JSONDecodeError as je:
                            json_validation_error = (
                                fc.name,
                                str(je),
                                fc.arguments[:300],
                            )
                            break

                if json_validation_error:
                    name, err, preview = json_validation_error
                    print(f"  Invalid JSON in function_call `{name}`: {err}. Feeding error back to LLM and regenerating.")
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            f"Your previous response contained invalid JSON in the `{name}` tool_call arguments.\n"
                            f"JSON decode error: {err}\n"
                            f"Arguments started with: {preview}...\n\n"
                            f"Valid JSON escape sequences are limited to: \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX. "
                            f"Shell-style escapes like \\$ or \\! are NOT valid JSON. "
                            f"Please re-emit the tool call with valid JSON."
                        )
                    ))
                    continue

                # Store raw output items for conversation history reconstruction
                self.conversation.append(Message(
                    role='assistant',
                    content=list(response.output),
                ))

                if not function_calls:
                    # LLM emitted no tool_call — feed feedback and retry.
                    print("  LLM returned no function_call. Feeding feedback and regenerating.")
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            "You must call a tool to proceed. If you have nothing else to do this week, "
                            "call `./novamind-operation next-week <cash_1wk> <cash_4wk> <cash_12wk>` via bash to advance."
                        )
                    ))
                    continue

                self._pending_tool_calls = [dict(id=fc.call_id, name=fc.name,
                    arguments=json.loads(fc.arguments or '{}')) for fc in function_calls]
                return self.next_tool_action()

            except Exception as e:
                if self.evidence_store:
                    self.evidence_store.assert_healthy(quiescent=False)
                status = getattr(e, 'status_code', 0) or 0
                is_retryable = isinstance(e, openai.APIStatusError) and (status >= 500 or status == 429)
                if not is_retryable:
                    is_retryable = isinstance(e, (openai.APIConnectionError, openai.APITimeoutError, LLMTimeoutError))
                if not is_retryable:
                    is_retryable = any(code in str(e) for code in ('429', '500', '502', '503', '504', '529'))
                print(f"OpenAI Responses API error (retryable={is_retryable}, status={status}): {e}")
                if is_retryable:
                    self._consecutive_errors = getattr(self, '_consecutive_errors', 0) + 1
                    wait_time = min(120, 10 * (2 ** min(self._consecutive_errors - 1, 3)))
                    print(f"  Server error ({self._consecutive_errors}), retrying in {wait_time}s...")
                    _time.sleep(wait_time)
                    del input_items, tools
                    continue  # Loop back to retry
                else:
                    # Non-retryable error. No next-week fallback —
                    # feed the error message back to the LLM and regenerate.
                    print(f"  Non-retryable error — feeding back to LLM for regeneration.")
                    print(f"Traceback: {traceback.format_exc()}")
                    self._consecutive_errors = getattr(self, '_consecutive_errors', 0) + 1
                    wait_time = min(60, 5 * self._consecutive_errors)
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            f"The previous API request failed with a non-retryable error:\n"
                            f"{type(e).__name__}: {e}\n\n"
                            f"Please re-emit your response. If the error mentions input validation, "
                            f"check your tool_call arguments are valid JSON. "
                            f"If the error mentions context length, produce a shorter response."
                        )
                    ))
                    _time.sleep(wait_time)
                    del input_items, tools
                    continue

    _ANTHROPIC_VALID_EFFORTS = frozenset({'low', 'medium', 'high', 'xhigh', 'max'})
    _ANTHROPIC_HAIKU_BUDGET_BY_EFFORT = {
        'low': 4096,
        'medium': 16000,
        'high': 32000,
        'xhigh': 48000,
        'max': 64000,
    }

    def _uses_native_128k_output(self, model: Optional[str] = None) -> bool:
        """Return True for models whose 128K output is not beta-gated."""
        model_name = (model or self.model).lower()
        return 'fable' in model_name or 'mythos' in model_name

    def _anthropic_extra_headers(self) -> Dict[str, str]:
        """Return model-specific Anthropic beta headers."""
        if self._uses_native_128k_output():
            return {}
        return {'anthropic-beta': 'output-128k-2025-02-19'}

    def _anthropic_betas(self) -> List[str]:
        """Return beta flags for Anthropic beta Messages requests."""
        betas = []
        if self.anthropic_fallback_model:
            betas.append('server-side-fallback-2026-06-01')
        if (
            not self._uses_native_128k_output()
            or (
                self.anthropic_fallback_model
                and not self._uses_native_128k_output(self.anthropic_fallback_model)
            )
        ):
            betas.append('output-128k-2025-02-19')
        return betas

    def _apply_anthropic_reasoning_params(self, api_kwargs: Dict[str, Any]) -> None:
        """Add Anthropic thinking/effort params for models that support them."""
        if self.reasoning_effort not in self._ANTHROPIC_VALID_EFFORTS:
            return

        if 'haiku' in self.model.lower():
            api_kwargs['thinking'] = {
                'type': 'enabled',
                'budget_tokens': self._ANTHROPIC_HAIKU_BUDGET_BY_EFFORT[self.reasoning_effort],
            }
            return

        # Fable/Mythos and the recent Opus/Sonnet models use adaptive thinking
        # with output_config.effort instead of fixed budget tokens.
        api_kwargs['thinking'] = {'type': 'adaptive'}
        api_kwargs['output_config'] = {'effort': self.reasoning_effort}

    def _anthropic_content_text(self, content: Any) -> str:
        """Best-effort text extraction from Anthropic content blocks."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get('text') or block.get('content')
            else:
                text = getattr(block, 'text', None)
            if text:
                parts.append(str(text))
        return "\n".join(parts)

    def _anthropic_response_dict(self, response: Any) -> Dict[str, Any]:
        """Convert an Anthropic response object to a JSON-safe dict."""
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json", exclude_none=False, by_alias=True)
        if isinstance(response, dict):
            return response
        return {}

    def _record_anthropic_response_metadata(self, response: Any) -> None:
        """Track served model and server-side fallback hops for logging."""
        response_dict = self._anthropic_response_dict(response)
        self.last_serving_model = str(response_dict.get('model') or self.model)
        self.last_anthropic_fallbacks = []

        for block in response_dict.get('content') or []:
            if not isinstance(block, dict) or block.get('type') != 'fallback':
                continue
            from_model = ((block.get('from') or {}).get('model') or '')
            to_model = ((block.get('to') or {}).get('model') or '')
            self.last_anthropic_fallbacks.append({
                'from': from_model,
                'to': to_model,
            })

        self.last_anthropic_fallback_used = bool(self.last_anthropic_fallbacks)
        if self.last_anthropic_fallback_used:
            self.total_anthropic_fallbacks += len(self.last_anthropic_fallbacks)

    def _anthropic_no_tool_feedback(self, response: Any, assistant_content: Any) -> str:
        """Feedback used when Anthropic returns text instead of a tool."""
        preview = self._anthropic_content_text(assistant_content).strip()
        if len(preview) > 1200:
            preview = preview[:1200] + "..."

        return (
            "You must call a tool to proceed. If you need context, use read_file, "
            "search_files, or bash. If you have nothing else to do this week, call "
            "`./novamind-operation next-week <cash_1wk> <cash_4wk> <cash_12wk>` via bash. "
            f"Previous non-tool response preview: {preview or '(no text)'}"
        )

    def _call_anthropic(self) -> Optional[Action]:
        """Call Anthropic/Bedrock API and parse the response."""
        import copy

        no_tool_retries = 0

        while True:
            messages = []
            for msg in self.conversation:
                if msg.role == 'system':
                    continue
                messages.append({'role': msg.role, 'content': copy.deepcopy(msg.content)})

            # Strip any leftover cache_control from previous messages, then add
            # a single breakpoint on the last message. Combined with the system
            # prompt and tools breakpoints this stays within the 4-breakpoint limit.
            def _strip_cache_control(content):
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and 'cache_control' in block:
                            del block['cache_control']

            for msg in messages:
                _strip_cache_control(msg.get('content'))

            # Add cache_control to the last message so the entire conversation
            # prefix is cached between consecutive turns.
            if messages:
                last_msg = messages[-1]
                content = last_msg.get('content', '')
                if isinstance(content, str) and content:
                    last_msg['content'] = [
                        {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                    ]
                elif isinstance(content, list) and content:
                    last_block = content[-1]
                    if isinstance(last_block, dict):
                        last_block['cache_control'] = {"type": "ephemeral"}

            system_text = self._context_system_prompt()
            system_content = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

            tools = [dict(name=t['name'], description=t['description'], input_schema=t['parameters'])
                     for t in self.tool_descriptions]
            if tools:
                tools[-1]['cache_control'] = {"type": "ephemeral"}

            api_kwargs = {
                'model': self.model,
                'max_tokens': 128000,
                'system': system_content,
                'messages': messages,
                'tools': tools,
            }
            anthropic_messages = self.client.messages
            if self.anthropic_fallback_model:
                anthropic_messages = self.client.beta.messages
                api_kwargs['fallbacks'] = [{'model': self.anthropic_fallback_model}]
                api_kwargs['betas'] = self._anthropic_betas()
            elif (extra_headers := self._anthropic_extra_headers()):
                # output-128k beta lets Sonnet/Opus 4.x emit up to 128K output
                # tokens. Fable/Mythos expose 128K output without this beta.
                api_kwargs['extra_headers'] = extra_headers

            # Anthropic SDK refuses non-streaming when max_tokens implies > 10min
            # budget (raised in _calculate_nonstreaming_timeout). Always stream
            # for max_tokens > 64000.
            use_streaming = api_kwargs['max_tokens'] > 64000
            if self.reasoning_effort in self._ANTHROPIC_VALID_EFFORTS:
                self._apply_anthropic_reasoning_params(api_kwargs)
                use_streaming = True

            try:
                def invoke():
                    if use_streaming:
                        with anthropic_messages.stream(**api_kwargs) as stream:
                            return stream.get_final_message()
                    return anthropic_messages.create(**api_kwargs)
                response = self._request_model('messages', dict(api_kwargs, stream=use_streaming), invoke)

                self.total_turns += 1
                self._consecutive_errors = 0
                self._record_anthropic_response_metadata(response)

                if self.response_callback:
                    self.response_callback(
                        turn=self.total_turns,
                        day=self.current_day,
                        messages=messages,
                        raw_response=self._anthropic_response_dict(response) or str(response),
                    )

                if self.last_anthropic_fallback_used and self.tool_result_callback:
                    self.tool_result_callback(
                        self.total_turns,
                        self.current_day,
                        '_anthropic_fallback',
                        {
                            'requested_model': self.model,
                            'served_model': self.last_serving_model,
                            'fallbacks': self.last_anthropic_fallbacks,
                        },
                        '',
                    )

                assistant_content = response.content
                self.conversation.append(Message(
                    role='assistant',
                    content=assistant_content
                ))

                tool_use_blocks = [block for block in assistant_content if block.type == 'tool_use']
                if not tool_use_blocks:
                    no_tool_retries += 1
                    stop_reason = getattr(response, 'stop_reason', '') or 'no_tool_use'
                    if self.tool_result_callback:
                        self.tool_result_callback(
                            self.total_turns,
                            self.current_day,
                            '_anthropic_no_tool',
                            {'stop_reason': stop_reason, 'attempt': no_tool_retries},
                            self._anthropic_content_text(assistant_content),
                        )
                    if no_tool_retries > 3:
                        raise RuntimeError(
                            "Anthropic response did not include a tool_use block after "
                            f"{no_tool_retries} attempts (last stop_reason={stop_reason!r})."
                        )
                    print(
                        f"  Anthropic returned no tool_use "
                        f"(stop_reason={stop_reason!r}); feeding feedback and regenerating."
                    )
                    self.conversation.append(Message(
                        role='user',
                        content=self._anthropic_no_tool_feedback(response, assistant_content),
                    ))
                    continue

                self._pending_tool_calls = [dict(id=block.id, name=block.name,
                    arguments=block.input or {}) for block in tool_use_blocks]
                return self.next_tool_action()

            except Exception as e:
                if self.evidence_store:
                    self.evidence_store.assert_healthy(quiescent=False)
                import traceback
                if str(e).startswith("Anthropic response did not include a tool_use block"):
                    raise
                error_msg = f"Anthropic LLM call error: {e}"
                tb = traceback.format_exc()
                print(f"\n{'='*60}")
                print(f"ERROR in BashAgent._call_anthropic()")
                print(f"{'='*60}")
                print(error_msg)
                print(f"Traceback:\n{tb}")
                print(f"{'='*60}\n")

                self._consecutive_errors += 1
                if self._consecutive_errors <= 3:
                    wait = 2 ** self._consecutive_errors
                    print(f"  Retrying in {wait}s (attempt {self._consecutive_errors}/3)...")
                    time.sleep(wait)
                    return self._call_anthropic()

                raise RuntimeError(
                    f"LLM failed {self._consecutive_errors} consecutive times. "
                    f"Last error: {e}"
                ) from e
