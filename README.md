<p align="center">
  <img src="assets/mascot.png" alt="CEO-Bench mascot" width="220"/>
</p>

<h1 align="center">CEO-Bench: Can Agents Play the Long Game?</h1>

<p align="center">
  <a href="https://tonychen.xyz/">Haozhe Chen</a>,
  <a href="https://www.cs.princeton.edu/~karthikn/">Karthik Narasimhan</a>,
  <a href="https://www.cs.princeton.edu/~zhuangl/">Zhuang Liu</a>
</p>

<p align="center">Princeton University</p>

<p align="center">
  <a href="https://ceobench.com">🌐 Website</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/pdf/2606.18543">📄 Paper</a> &nbsp;|&nbsp;
  <a href="https://ceobench.com/trajectory-viewer/">📊 Trajectory Viewer</a>
</p>



## 📊 Overview

<p align="center">
  <img src="assets/teaser.png" alt="CEO-Bench teaser" width="100%"/>
</p>

CEO-Bench evaluates general long-horizon agent capabilities by simulating a
startup over 500 days in a realistic and challenging environment. The agent
operates through a programmable interface with access to business databases,
company management tools, and social media. Outcomes are driven by a partially
observable, noisy, and evolving market with delayed and coupled consequences.




## 🚀 Running CEO-Bench

### 🔑 Setup: Environment variables

CEO-Bench has three LLM roles:

- the benchmarked agent model
- the social/macro post simulator model, Haiku 4.5 by default
- the enterprise customer simulator model, Sonnet 4.5 by default

Pick one provider family and use provider-specific model identifiers in
`src/saas_bench/config.py`.

**Option A: Amazon Bedrock for all models**

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-2"
```

```python
agent_llm_provider: str = "bedrock"
agent_llm_model: str = "anthropic.claude-fable-5"  # Claude Fable 5
# or another Bedrock model id, e.g. "us.anthropic.claude-sonnet-4-6"

social_post_llm_provider: str = "bedrock"
social_post_llm_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

enterprise_llm_provider: str = "bedrock"
enterprise_llm_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
```

**Option B: Anthropic direct API for all models**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
agent_llm_provider: str = "anthropic"
agent_llm_model: str = "claude-fable-5"  # Claude Fable 5

social_post_llm_provider: str = "anthropic"
social_post_llm_model: str = "claude-haiku-4-5"

enterprise_llm_provider: str = "anthropic"
enterprise_llm_model: str = "claude-sonnet-4-5"
```

The LLM config fields are:

- `agent_llm_provider`, `agent_llm_model`, `agent_llm_reasoning_effort`
- `social_post_llm_provider`, `social_post_llm_model`
- `enterprise_llm_provider`, `enterprise_llm_model`

The bash-agent CLI `--provider`, `--model`, and `--reasoning-effort` flags only
override the benchmarked agent for ad hoc runs. Simulator social/macro and
enterprise LLMs use the simulator config and do not reuse the agent-only
`--api-key`. If you change a simulator provider, also set the corresponding
model to the identifier expected by that provider; model names are not
translated automatically.

For Claude Fable 5 agent runs, use `--provider anthropic --model claude-fable-5`
for the direct Anthropic API, or `--provider bedrock --model anthropic.claude-fable-5`
for Amazon Bedrock.


### 🎯 Option A: Evaluate any coding agent easily

We built CEO-Bench into a single executable and docs that any coding agent can just download the game and start playing.

The executable is hosted at **[zlab-princeton/run-ceobench](https://github.com/zlab-princeton/run-ceobench)**

If you want to evaluate a coding agent with terminal and internet access, prompt it

```
Download this, read instructions, and finish 500 day gameplay. https://github.com/zlab-princeton/run-ceobench
```



### ⚙️ Option B: Customize the configuration

All tunable simulator constants live in **`src/saas_bench/config.py`**: pricing,
customer groups, ad-channel productivity, R&D speed, competitor difficulty, etc.
After editing, rebuild the public bundle. 

```bash
uv sync                                  # one-time install
uv run python scripts/build_public.py    # rebuild public/ artifact
```

Then generated `public/` directory would play the same role as the same way as **[zlab-princeton/run-ceobench](https://github.com/zlab-princeton/run-ceobench)** in Option A

**Tuning difficulty** You can modify configuration in `config.py` to adjust difficulty.

An important difficulty is competitor strength. Competitor keeps track of a unreleased_dev_bank. Each agent's research and development quality improvement is added to this variable. At each competitor event, competitor draws `u ~ U(competitor_feedback_u_min, competitor_feedback_u_max)`, raises customer expectations by u × unreleased_dev_bank, and subtract this amount from unreleased_dev_bank. Larger competitor_feedback_u_min and competitor_feedback_u_max leads to stronger competitor and higher quality pressure. The default config value is (0.2,0.5). 



### 🤖 Option C: Replicate the bash-agent baseline

The paper's baseline gives an LLM a sandboxed bash shell plus the public CLI and
runs the full 500-day loop with checkpointing and logging. The full process:

**1. Install dependencies** (one-time):

```bash
uv sync
uv run python scripts/build_agent_runtime.py   # Linux: Python runtime for the agent sandbox
```

On Linux the agent's commands run under bubblewrap. The sandbox shows the agent
workspace at `/workspace`, read-only system directories, and the runtime built
above at `/opt/python` (this interpreter plus numpy, pandas and scikit-learn at the
installed versions). Host paths, `/proc` and the harness-owned `sessions/`
directory are not visible. Rebuild the runtime after changing the environment.

**2. Set provider credentials** in a `.env` file at the repo root. Which keys you
need depends on the agent model; for a Bedrock run:

```bash
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
AWS_REGION="us-east-2"
```

Other providers read `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
`XAI_API_KEY`, `TOGETHER_API_KEY`, `DEEPSEEK_API_KEY`, `OPENCODE_API_KEY`, `OPENDESIGN_API_KEY`,
or `MODAL_TOKEN_*`. No `NMDB_KEY` is needed: the SQLCipher key is embedded
in the engine.

If you configure simulator LLMs to use direct `anthropic` or `openai`, export
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in the shell before running. The
simulator does not receive the agent-only `--api-key`. DeepSeek, OpenCode,
and OpenDesign agent runs select the same provider and model for the simulator,
with reasoning disabled, unless `CEOBENCH_SIMULATOR_LLM_PROVIDER` and
`CEOBENCH_SIMULATOR_LLM_MODEL` override that selection.

**3. Run.** `public/` ships prebuilt, so there is no build step:

```bash
uv run python -m saas_bench.agents.bash_agent.run_test \
    --model us.anthropic.claude-sonnet-4-6 \
    --provider bedrock \
    --reasoning-effort max \
    --seed 42 \
    --days 500 \
    --workspace bash_agent_runs
```

Select any of the three Chat providers using `--provider`. New runs default to
`high` for the agent and disabled thinking for the simulator. These commands make
paid model requests; `--days` must cover at least one whole week.

| Provider | Default model | API base URL | Credential variable |
|---|---|---|---|
| `deepseek` | `deepseek-flash` | `https://api.deepseek.com/` | `DEEPSEEK_API_KEY` |
| `opencode` | `deepseek-v4.1-flash` | `https://opencode.ai/inference/go/openai/v1/` | `OPENCODE_API_KEY` |
| `opendesign` | `deepseek-v4.1-flash-eco` | `https://amr-link.open-design.ai/v1/` | `OPENDESIGN_API_KEY` |

```bash
uv run python -m saas_bench.agents.bash_agent.run_test \
    --provider deepseek --days 7 --seed 42 --workspace deepseek_runs
uv run python -m saas_bench.agents.bash_agent.run_test \
    --provider opencode --days 7 --seed 42 --workspace opencode_runs
uv run python -m saas_bench.agents.bash_agent.run_test \
    --provider opendesign --days 7 --seed 42 --workspace opendesign_runs
```

`--model`, `--base-url`, and `--api-key` explicitly override agent settings.
`DEEPSEEK_BASE_URL`, `OPENCODE_BASE_URL`, and `OPENDESIGN_BASE_URL` override only
the named provider's default address. Model IDs were checked against authenticated
catalogs on 2026-11-09; see the official [DeepSeek docs](https://api-docs.deepseek.com/),
[OpenCode Go docs](https://opencode.ai/docs/console/go/), and
[OpenDesign API guide](https://open-design.ai/help/cloud-api/).

A new standalone simulator follows the agent unless selected independently:

```bash
CEOBENCH_SIMULATOR_LLM_PROVIDER=deepseek \
  uv run python -m saas_bench.agents.bash_agent.run_test \
    --provider opendesign --days 7 --workspace mixed_provider_runs
```

Changing only the simulator provider selects that provider's default model.
`CEOBENCH_SIMULATOR_LLM_MODEL` overrides its model, and
`CEOBENCH_SIMULATOR_LLM_BASE_URL` overrides its address, taking precedence over
the provider's address variable. The simulator needs its own credential variable;
agent `--api-key` is not passed to it.

Same-provider recovery preserves the exact checkpoint model and endpoint.
Explicitly changing `--provider` selects the new provider's defaults for omitted
model, endpoint and reasoning options. The simulator retains its independent
checkpoint settings unless explicitly overridden with its environment variables.
Old standalone OpenCode checkpoints without a serialized simulator address keep
their historical `/zen/go/v1/` route.

For direct teams, generate credential-free contracts with
`make_chat_models('opendesign', simulator_provider='deepseek')` from
`saas_bench.team_models`, and pass them as `model_configs`. This factory keeps
agent and simulator choices independent; simulator thinking is disabled.
`MultiAgentRuntime.restore(..., model_configs=new_models)` explicitly changes the
new attempt's providers, budgets and price configuration while preserving the
selected snapshot. Client factories must use each selected contract's `base_url`
and `credential_env`; simulator factories receive the updated `BenchmarkConfig`.
Unknown prices remain `null` rather than inheriting another provider's price.
Unprofiled `engineering` batch manifests accept the same JSON through
`python -m saas_bench.team_batch freeze --category engineering --model-configs models.json`
with the usual required D0/output arguments. `prepare(live=False)` and
`team_driver.py --check` perform zero-request preparation; `start`/`run` start workers.
Frozen R4/R5 profiles and their model contracts remain unchanged.

OpenDesign sends `max_tokens` and `reasoning_effort=high|none`, without a
`thinking` body. Its agent uses streaming with `include_usage=true`: actual
nonstream responses split commentary and a tool call into separate choices,
whereas streaming returned one complete choice and native `reasoning_content`.
The adapter requires `[DONE]` plus successful `stop` or `tool_calls` before
accepting text or executing tools, and preserves reasoning through tool results
and conversation restore. The simulator uses nonstream `none` replies and requires
one nonempty, non-refusal text choice with `finish_reason=stop`. Invalid generations
use the existing bounded correction path; network failures use the existing retry
and independent STOP paths. The provider's `insufficient_balance` wallet refusal
is permanent and enters HOLD for manual recovery
([provider issue](https://github.com/nexu-io/open-design/issues/5246)).
Token usage and served models are recorded. Missing tokenizer mappings, cache
fields and billed costs remain unknown; OpenCode Go's visible allowance/list-cost
fields do not establish a cash invoice. Transport probes establish protocol
compatibility rather than reasoning quality or benchmark performance.

**4. Output.** Each run lands at `bash_agent_runs/run_<id>/`: `world.nmdb`
(encrypted ledger), `config.json`, `checkpoint.json`, `agent_workspace/` (the
agent's sandbox, a fresh git repo with weekly commits), and `logs/` containing
per-turn `raw_responses_<id>.jsonl` (model thinking + tool calls),
`tool_results_<id>.jsonl` (tool calls + their outputs), and
`timing_<id>.jsonl`. To score and analyze the run, see
[docs/analyze_trajectory.md](docs/analyze_trajectory.md).

If you edit `src/saas_bench/config.py`, rebuild the bundle the agent sees with
`uv run python scripts/build_public.py` before launching.



## 📈 Analyzing agent trajectory

Every finished run leaves a single artifact: an encrypted `world.nmdb` ledger
(SQLCipher, page-level AES-256). It is the complete record of the run: cash,
subscriptions, customers, competitor events, and every action the agent took.

The decryption key is fixed and bundled into the published `novamind-operation`
zipapp at build time; see `KEYS.md` in this repo for the value, or import it
from the compiled `saas_bench._embedded_key` module. To decrypt and query:

```bash
KEY=$(grep _NMDB_KEY KEYS.md | head -1 | cut -d'"' -f2)
sqlcipher path/to/world.nmdb \
  "PRAGMA key = '$KEY';" \
  "SELECT day, category, amount FROM ledger ORDER BY day, id LIMIT 10;"
```

For the database schema, analysis recipes, and notes on keeping the agent from
cheating, see **[docs/analyze_trajectory.md](docs/analyze_trajectory.md)**.



## 📁 Repo layout

```
ceobench-src/
├── README.md                          ← this file
├── docs/
│   └── analyze_trajectory.md          ← decrypt, schema + analysis guide
├── public_sources/                    ← human-written inputs to the public build
│   ├── README.md, requirements.txt
│   └── examples/{autoplay_loop,basic_strategy}.py
├── scripts/
│   ├── build_public.py                ← canonical public-repo builder
│   ├── build_agent_runtime.py         ← agent sandbox Python runtime
│   ├── start_fresh_sonnet_bash.sh     ← bash-agent launcher (Bedrock Sonnet)
│   ├── start_fresh_gpt_bash.sh        ← bash-agent launcher (OpenAI GPT)
│   └── resume_run.sh                  ← resume bash agent from checkpoint
└── src/saas_bench/                    ← simulator + bash agent
    ├── simulation.py, environment.py, shocks.py, event_logger.py
    ├── config.py                      ← all tunable constants
    ├── customer_llm.py, personas.py, enterprise.py
    ├── database.py, db_protection.py
    ├── api_server.py, server_entry.py, tools.py
    ├── novamind_api/, novamind_cli.py, _public_cli.py
    └── agents/bash_agent/             ← canonical baseline harness
```



## 📜 Citation

```bibtex
@misc{chen2026ceobenchagentsplaylong,
  title={CEO-Bench: Can Agents Play the Long Game?},
  author={Haozhe Chen and Karthik Narasimhan and Zhuang Liu},
  year={2026},
  eprint={2606.18543},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2606.18543},
}
```
